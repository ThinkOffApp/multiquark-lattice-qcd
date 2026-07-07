import Metal
import Foundation

// su2gen_gpu.swift — GPU-resident SU(2) pure-gauge heatbath generator (Metal).
//
// The gauge field lives in a unified-memory MTLBuffer as one float4 per link
// (quaternion rep: U = a0*I + i*(a1*s1 + a2*s2 + a3*s3), stored (a0,a1,a2,a3)).
// A sweep = 8 sequential kernel dispatches (2 checkerboard parities x 4 mu),
// each updating half the links of one direction via the Kennedy-Pendleton
// heatbath, exactly mirroring gpt's su2_heat_bath conventions:
//   staple S(x,mu) = sum_{nu!=mu} [ U_nu(x) U_mu(x+nu) U_nu^dag(x+mu)
//                                 + U_nu^dag(x-nu) U_mu(x-nu) U_nu(x+mu-nu) ]
//   (gpt scales S by beta/N with N=2; here the scale enters only via
//    alpha = beta * |quat(U*S^dag)| which is identical)
//   KP accept loop (niter=20): x1=-ln(r1)/alpha, x2=-ln(r2)/alpha,
//   c=cos^2(2 pi r3), d=x2+x1*c, accept if r0^2 < 1-d/2.
//   a0=1-d, |a| uniform direction; U_new = normalize(Vhat^dag * A * U_old).
// RNG: philox4x32-10, counter = (site, event_lo, event_hi, purpose),
// event = sweepCounter*8 + parity*4 + mu, so every dispatch draws from a
// fresh, reproducible stream keyed on the run seed.
//
// C ABI (ctypes from python):
//   su2gen_init(L0,L1,L2,L3, seed) -> 0/-1
//   su2gen_load(ptr)   float[nsites*4*4]  (site-major, then mu, then quat)
//   su2gen_store(ptr)
//   su2gen_sweep(nsweeps, beta, muMask) -> sweeps done
//   su2gen_plaquette() -> mean plaquette, gpt normalization (ReTr P / 2)
//   su2gen_set_counter(c) / su2gen_get_counter()

let metalSrc = """
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint L0, L1, L2, L3;
    uint parity;
    uint mu;
    uint eventLo, eventHi;   // per-dispatch RNG event counter
    uint seedLo, seedHi;
    float beta;
    uint niter;
};

// ---- quaternion SU(2): q = (a0, a1, a2, a3) == a0*I + i*(a.s) ----
inline float4 qmul(float4 A, float4 B) {
    float a0 = A.x; float3 av = A.yzw;
    float b0 = B.x; float3 bv = B.yzw;
    float c0 = a0 * b0 - dot(av, bv);
    float3 cv = a0 * bv + b0 * av - cross(av, bv);
    return float4(c0, cv);
}
inline float4 qdag(float4 A) { return float4(A.x, -A.yzw); }
inline float qnorm2(float4 A) { return dot(A, A); }        // == det(matrix)
inline float qretr(float4 A) { return 2.0f * A.x; }        // Re Tr

// ---- philox4x32-10 ----
inline uint mulhilo(uint a, uint b, thread uint &hi) {
    ulong p = ulong(a) * ulong(b);
    hi = uint(p >> 32);
    return uint(p);
}
inline uint4 philox4x32(uint4 ctr, uint2 key) {
    const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
    const uint W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;
    for (int r = 0; r < 10; r++) {
        uint hi0, hi1;
        uint lo0 = mulhilo(M0, ctr.x, hi0);
        uint lo1 = mulhilo(M1, ctr.z, hi1);
        ctr = uint4(hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0);
        key.x += W0; key.y += W1;
    }
    return ctr;
}
// uniform in (0,1), 24-bit, never 0 or 1 (safe for log)
inline float u01(uint x) { return (float(x >> 8) + 0.5f) * (1.0f / 16777216.0f); }

// ---- lattice indexing: site = x + L0*(y + L1*(z + L2*t)) ----
inline uint4 siteCoord(uint s, constant Params &p) {
    uint x = s % p.L0; s /= p.L0;
    uint y = s % p.L1; s /= p.L1;
    uint z = s % p.L2; s /= p.L2;
    return uint4(x, y, z, s);
}
inline uint coordSite(uint4 c, constant Params &p) {
    return c.x + p.L0 * (c.y + p.L1 * (c.z + p.L2 * c.w));
}
inline uint4 hop(uint4 c, uint d, int sgn, constant Params &p) {
    uint L[4] = { p.L0, p.L1, p.L2, p.L3 };
    uint4 r = c;
    uint v = c[d];
    r[d] = (sgn > 0) ? ((v + 1u) % L[d]) : ((v + L[d] - 1u) % L[d]);
    return r;
}
inline float4 link(device const float4 *U, uint site, uint mu) { return U[site * 4u + mu]; }

inline float4 stapleSum(device const float4 *U, uint4 c, uint site, uint mu, constant Params &p) {
    float4 S = float4(0.0f);
    uint4 cpmu = hop(c, mu, +1, p);
    uint spmu = coordSite(cpmu, p);
    for (uint nu = 0; nu < 4; nu++) {
        if (nu == mu) continue;
        uint4 cpnu = hop(c, nu, +1, p);
        uint4 cmnu = hop(c, nu, -1, p);
        uint4 cpmu_mnu = hop(cpmu, nu, -1, p);
        // forward: U_nu(x) U_mu(x+nu) U_nu^dag(x+mu)
        float4 f = qmul(qmul(link(U, site, nu), link(U, coordSite(cpnu, p), mu)),
                        qdag(link(U, spmu, nu)));
        // backward: U_nu^dag(x-nu) U_mu(x-nu) U_nu(x+mu-nu)
        uint smnu = coordSite(cmnu, p);
        float4 b = qmul(qmul(qdag(link(U, smnu, nu)), link(U, smnu, mu)),
                        link(U, coordSite(cpmu_mnu, p), nu));
        S += f + b;
    }
    return S;
}

kernel void hb_update(device float4 *U [[buffer(0)]],
                      constant Params &p [[buffer(1)]],
                      uint gid [[thread_position_in_grid]])
{
    uint nsites = p.L0 * p.L1 * p.L2 * p.L3;
    if (gid >= nsites) return;
    uint4 c = siteCoord(gid, p);
    if (((c.x + c.y + c.z + c.w) & 1u) != p.parity) return;

    uint mu = p.mu;
    float4 Uold = link(U, gid, mu);
    float4 S = stapleSum(U, c, gid, mu, p);

    float4 M = qmul(Uold, qdag(S));      // V = U * S^dag (unscaled)
    float k2 = qnorm2(M);
    if (k2 < 1e-24f) return;             // degenerate staple: keep link
    float k = sqrt(k2);
    float4 Vhat = M / k;
    float alpha = p.beta * k;            // = 2 * |quat(U * (beta/2 S)^dag)|

    uint2 key = uint2(p.seedLo, p.seedHi);

    // Kennedy-Pendleton rejection loop
    float d = 0.0f;
    bool accepted = false;
    for (uint it = 0; it < p.niter && !accepted; it++) {
        uint4 r = philox4x32(uint4(gid, p.eventLo, p.eventHi, it), key);
        float r0 = u01(r.x), r1 = u01(r.y), r2 = u01(r.z), r3 = u01(r.w);
        float x1 = -log(r1) / alpha;
        float x2 = -log(r2) / alpha;
        float cc = cospi(2.0f * r3);
        float dt = x2 + x1 * cc * cc;
        if (r0 * r0 < 1.0f - 0.5f * dt) { d = dt; accepted = true; }
    }
    if (!accepted) return;               // keep old link (matches gpt)

    float a0 = 1.0f - d;
    float amag = sqrt(fabs(1.0f - a0 * a0));
    uint4 r = philox4x32(uint4(gid, p.eventLo, p.eventHi, 100u), key);
    float phi2 = 2.0f * u01(r.x);        // phi / pi
    float cost = 2.0f * u01(r.y) - 1.0f;
    float sint = sqrt(fabs(1.0f - cost * cost));
    float4 A = float4(a0,
                      amag * sint * cospi(phi2),
                      amag * sint * sinpi(phi2),
                      amag * cost);

    float4 Unew = qmul(qmul(qdag(Vhat), A), Uold);
    Unew /= sqrt(qnorm2(Unew));          // reunitarize (project defect)
    U[gid * 4u + mu] = Unew;
}

// staple debug output for parity validation: writes S(x,mu) for all sites
kernel void staple_dump(device const float4 *U [[buffer(0)]],
                        constant Params &p [[buffer(1)]],
                        device float4 *out [[buffer(2)]],
                        uint gid [[thread_position_in_grid]])
{
    uint nsites = p.L0 * p.L1 * p.L2 * p.L3;
    if (gid >= nsites) return;
    uint4 c = siteCoord(gid, p);
    out[gid] = stapleSum(U, c, gid, p.mu, p);
}

// per-site sum over mu<nu of ReTr(P_munu)/2  (gpt plaquette normalization
// divides by nsites*6 on the host)
kernel void plaq_sum(device const float4 *U [[buffer(0)]],
                     constant Params &p [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     uint gid [[thread_position_in_grid]])
{
    uint nsites = p.L0 * p.L1 * p.L2 * p.L3;
    if (gid >= nsites) return;
    uint4 c = siteCoord(gid, p);
    float acc = 0.0f;
    for (uint mu = 0; mu < 4; mu++) {
        uint4 cpmu = hop(c, mu, +1, p);
        uint spmu = coordSite(cpmu, p);
        for (uint nu = mu + 1; nu < 4; nu++) {
            uint4 cpnu = hop(c, nu, +1, p);
            // P = U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
            float4 pq = qmul(qmul(link(U, gid, mu), link(U, spmu, nu)),
                             qmul(qdag(link(U, coordSite(cpnu, p), mu)),
                                  qdag(link(U, gid, nu))));
            acc += 0.5f * qretr(pq);
        }
    }
    out[gid] = acc;
}
"""

// ---- host state ----
final class Gen {
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let hbPipe: MTLComputePipelineState
    let stPipe: MTLComputePipelineState
    let plPipe: MTLComputePipelineState
    let uBuf: MTLBuffer
    let plBuf: MTLBuffer
    let stBuf: MTLBuffer
    let L: [UInt32]
    let nsites: Int
    let seed: UInt64
    var sweepCounter: UInt64 = 0

    init?(L0: Int32, L1: Int32, L2: Int32, L3: Int32, seed: UInt64) {
        guard let d = MTLCreateSystemDefaultDevice(), let q = d.makeCommandQueue() else { return nil }
        dev = d; queue = q
        guard let lib = try? d.makeLibrary(source: metalSrc, options: nil),
              let hbF = lib.makeFunction(name: "hb_update"),
              let stF = lib.makeFunction(name: "staple_dump"),
              let plF = lib.makeFunction(name: "plaq_sum"),
              let hbP = try? d.makeComputePipelineState(function: hbF),
              let stP = try? d.makeComputePipelineState(function: stF),
              let plP = try? d.makeComputePipelineState(function: plF) else { return nil }
        hbPipe = hbP; stPipe = stP; plPipe = plP
        L = [UInt32(L0), UInt32(L1), UInt32(L2), UInt32(L3)]
        nsites = Int(L0) * Int(L1) * Int(L2) * Int(L3)
        self.seed = seed
        guard let ub = d.makeBuffer(length: nsites * 4 * 16, options: .storageModeShared),
              let pb = d.makeBuffer(length: nsites * 4, options: .storageModeShared),
              let sb = d.makeBuffer(length: nsites * 16, options: .storageModeShared) else { return nil }
        uBuf = ub; plBuf = pb; stBuf = sb
    }

    struct Params {
        var L0: UInt32; var L1: UInt32; var L2: UInt32; var L3: UInt32
        var parity: UInt32; var mu: UInt32
        var eventLo: UInt32; var eventHi: UInt32
        var seedLo: UInt32; var seedHi: UInt32
        var beta: Float; var niter: UInt32
    }

    func params(parity: UInt32, mu: UInt32, event: UInt64, beta: Float) -> Params {
        Params(L0: L[0], L1: L[1], L2: L[2], L3: L[3],
               parity: parity, mu: mu,
               eventLo: UInt32(truncatingIfNeeded: event),
               eventHi: UInt32(truncatingIfNeeded: event >> 32),
               seedLo: UInt32(truncatingIfNeeded: seed),
               seedHi: UInt32(truncatingIfNeeded: seed >> 32),
               beta: beta, niter: 20)
    }

    func dispatch(_ enc: MTLComputeCommandEncoder, _ pipe: MTLComputePipelineState, _ p: inout Params) {
        enc.setComputePipelineState(pipe)
        enc.setBuffer(uBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        let w = min(pipe.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: nsites, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    func sweep(n: Int32, beta: Float, muMask: UInt32) -> Int32 {
        for _ in 0..<n {
            guard let cb = queue.makeCommandBuffer() else { return -1 }
            for parity in 0..<UInt32(2) {
                for mu in 0..<UInt32(4) {
                    if (muMask & (1 << mu)) == 0 { continue }
                    guard let enc = cb.makeComputeCommandEncoder() else { return -1 }
                    let event = sweepCounter * 8 + UInt64(parity) * 4 + UInt64(mu)
                    var p = params(parity: parity, mu: mu, event: event, beta: beta)
                    dispatch(enc, hbPipe, &p)
                    enc.endEncoding()
                }
            }
            cb.commit()
            cb.waitUntilCompleted()
            if cb.status == .error { return -1 }
            sweepCounter += 1
        }
        return n
    }

    func plaquette() -> Double {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return -1 }
        var p = params(parity: 0, mu: 0, event: 0, beta: 0)
        enc.setComputePipelineState(plPipe)
        enc.setBuffer(uBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        enc.setBuffer(plBuf, offset: 0, index: 2)
        let w = min(plPipe.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: nsites, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        let ptr = plBuf.contents().bindMemory(to: Float.self, capacity: nsites)
        var s = 0.0
        for i in 0..<nsites { s += Double(ptr[i]) }
        return s / (Double(nsites) * 6.0)
    }

    func stapleDump(mu: UInt32, out: UnsafeMutablePointer<Float>) {
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return }
        var p = params(parity: 0, mu: mu, event: 0, beta: 0)
        enc.setComputePipelineState(stPipe)
        enc.setBuffer(uBuf, offset: 0, index: 0)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        enc.setBuffer(stBuf, offset: 0, index: 2)
        let w = min(stPipe.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreads(MTLSize(width: nsites, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        memcpy(out, stBuf.contents(), nsites * 16)
    }
}

var gen: Gen? = nil

@_cdecl("su2gen_init")
public func su2gen_init(_ L0: Int32, _ L1: Int32, _ L2: Int32, _ L3: Int32, _ seed: UInt64) -> Int32 {
    gen = Gen(L0: L0, L1: L1, L2: L2, L3: L3, seed: seed)
    return gen == nil ? -1 : 0
}

@_cdecl("su2gen_device_name")
public func su2gen_device_name(_ out: UnsafeMutablePointer<CChar>, _ cap: Int32) -> Int32 {
    guard let g = gen else { return -1 }
    let name = g.dev.name
    let bytes = Array(name.utf8.prefix(Int(cap) - 1)) + [0]
    bytes.withUnsafeBufferPointer { memcpy(out, $0.baseAddress!, $0.count) }
    return 0
}

@_cdecl("su2gen_load")
public func su2gen_load(_ ptr: UnsafePointer<Float>) -> Int32 {
    guard let g = gen else { return -1 }
    memcpy(g.uBuf.contents(), ptr, g.nsites * 4 * 16)
    return 0
}

@_cdecl("su2gen_store")
public func su2gen_store(_ ptr: UnsafeMutablePointer<Float>) -> Int32 {
    guard let g = gen else { return -1 }
    memcpy(ptr, g.uBuf.contents(), g.nsites * 4 * 16)
    return 0
}

@_cdecl("su2gen_sweep")
public func su2gen_sweep(_ n: Int32, _ beta: Float, _ muMask: UInt32) -> Int32 {
    guard let g = gen else { return -1 }
    return g.sweep(n: n, beta: beta, muMask: muMask)
}

@_cdecl("su2gen_plaquette")
public func su2gen_plaquette() -> Double {
    guard let g = gen else { return -1 }
    return g.plaquette()
}

@_cdecl("su2gen_staple_dump")
public func su2gen_staple_dump(_ mu: UInt32, _ out: UnsafeMutablePointer<Float>) -> Int32 {
    guard let g = gen else { return -1 }
    g.stapleDump(mu: mu, out: out)
    return 0
}

@_cdecl("su2gen_set_counter")
public func su2gen_set_counter(_ c: UInt64) -> Int32 {
    guard let g = gen else { return -1 }
    g.sweepCounter = c
    return 0
}

@_cdecl("su2gen_get_counter")
public func su2gen_get_counter() -> UInt64 {
    return gen?.sweepCounter ?? 0
}
