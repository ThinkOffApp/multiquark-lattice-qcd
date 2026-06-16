import Metal
import Foundation

// Standalone parity validator for the Wilson-loop path-product kernel
// (gpu-metal-wip/su3_wilson_loop.metal). Validates GPU Re Tr of an ordered
// product of N pre-aligned SU(3) link fields (with a dagger pattern) against
// an independent CPU reference, over random matrices packed in the NEON
// Nsimd=2 layout (float4 = 2 complex lanes). This proves the GPU path-product
// arithmetic is correct for loops of arbitrary size; the physical-geometry
// Cshift alignment is the cgpt integration step done separately on the mini.

let metalPath = CommandLine.arguments.count > 1
    ? CommandLine.arguments[1]
    : "/tmp/su3_wilson_loop.metal"
let src = try String(contentsOfFile: metalPath, encoding: .utf8)
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: src, options: nil)
let fn = lib.makeFunction(name: "WilsonLoopPath")!
let pipe = try dev.makeComputePipelineState(function: fn)

let nV = 4096            // vSites (each = 2 lane-sites)
let floatsPerMat = 9 * 4 // 9 complex * float4(2 lanes)

// xorshift RNG (deterministic)
var s: UInt64 = 0x9e3779b97f4a7c15
func rnd() -> Float { s ^= s<<13; s ^= s>>7; s ^= s<<17; return Float(Double(s % 20000)/10000.0 - 1.0) }

// Random UNITARY field: each vSite/lane gets an independent unitary 3x3 via
// Gram-Schmidt. Real Wilson-loop links are unitary, so the path product stays
// bounded (ReTr in [-3,3]). Validating with non-unitary random matrices makes
// the product magnitude (and thus absolute error) explode with loop length,
// which is not a meaningful parity metric — ether's review asks for error
// measured at the real target loop sizes, which requires unitary links.
func makeField() -> [Float] {
    typealias CC = (Float, Float)
    func mul2(_ x: CC, _ y: CC) -> CC { (x.0*y.0 - x.1*y.1, x.0*y.1 + x.1*y.0) }
    func conj2(_ x: CC) -> CC { (x.0, -x.1) }
    func gramSchmidt() -> [[CC]] {        // 3 orthonormal rows
        var rows = [[CC]]()
        for _ in 0..<3 {
            var v: [CC] = [(rnd(),rnd()), (rnd(),rnd()), (rnd(),rnd())]
            for u in rows {
                var p: CC = (0,0)         // p = <u,v> = sum conj(u_i) v_i
                for i in 0..<3 { let t = mul2(conj2(u[i]), v[i]); p = (p.0+t.0, p.1+t.1) }
                for i in 0..<3 { let t = mul2(p, u[i]); v[i] = (v[i].0-t.0, v[i].1-t.1) } // v -= p*u
            }
            var nrm: Float = 0
            for i in 0..<3 { nrm += v[i].0*v[i].0 + v[i].1*v[i].1 }
            nrm = nrm.squareRoot()
            for i in 0..<3 { v[i] = (v[i].0/nrm, v[i].1/nrm) }
            rows.append(v)
        }
        return rows
    }
    var a = [Float](repeating: 0, count: nV*floatsPerMat)
    for vs in 0..<nV {
        for lane in 0..<2 {
            let M = gramSchmidt()
            for r in 0..<3 { for c in 0..<3 {
                let base = vs*floatsPerMat + (r*3+c)*4 + (lane==0 ? 0 : 2)
                a[base] = M[r][c].0
                a[base+1] = M[r][c].1
            }}
        }
    }
    return a
}
func buf(_ a: [Float]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: a.count*4, options: .storageModeShared)! }

// ---- CPU reference: ordered product of links[0..<N] with dagger pattern ----
typealias C = (Float, Float)
func cm(_ x: C, _ y: C) -> C { (x.0*y.0 - x.1*y.1, x.0*y.1 + x.1*y.0) }
func cadd(_ x: C, _ y: C) -> C { (x.0+y.0, x.1+y.1) }
func cconj(_ x: C) -> C { (x.0, -x.1) }
typealias M = [[C]]
func ident() -> M { (0..<3).map { r in (0..<3).map { c in r==c ? (Float(1),Float(0)) : (Float(0),Float(0)) } } }
func mul(_ X: M, _ Y: M) -> M {
    var R = ident(); for i in 0..<3 { for j in 0..<3 { var su:C=(0,0); for k in 0..<3 { su = cadd(su, cm(X[i][k], Y[k][j])) }; R[i][j]=su } }; return R
}
func dag(_ X: M) -> M { var R = ident(); for i in 0..<3 { for j in 0..<3 { R[i][j]=cconj(X[j][i]) } }; return R }
func mat(_ f: [Float], _ vsite: Int, _ lane: Int) -> M {
    var m = ident()
    for r in 0..<3 { for c in 0..<3 {
        let base = vsite*floatsPerMat + (r*3+c)*4
        m[r][c] = lane==0 ? (f[base], f[base+1]) : (f[base+2], f[base+3])
    }}
    return m
}
func cpuReTr(_ fields: [[Float]], _ dagger: [UInt8], _ vsite: Int, _ lane: Int) -> Float {
    var acc = ident()
    for k in 0..<fields.count {
        let u = mat(fields[k], vsite, lane)
        acc = dagger[k] != 0 ? mul(acc, dag(u)) : mul(acc, u)
    }
    return acc[0][0].0 + acc[1][1].0 + acc[2][2].0
}

// ---- run one case (N links, per-link dagger flags) on the GPU ----
// Validates EVERY vSite/lane (not a sample): full-site coverage catches buffer
// packing/indexing bugs, per ether's review.
func runCase(_ name: String, _ fields: [[Float]], _ dagger: [UInt8]) -> Float {
    let nLink = fields.count
    // pack into one buffer: links[k*nV + id]
    var packed = [Float](repeating: 0, count: nLink*nV*floatsPerMat)
    for k in 0..<nLink {
        let off = k*nV*floatsPerMat
        for i in 0..<(nV*floatsPerMat) { packed[off+i] = fields[k][i] }
    }
    let bLinks = buf(packed)
    let bDag = dev.makeBuffer(bytes: dagger, length: dagger.count, options: .storageModeShared)!
    let outBuf = dev.makeBuffer(length: nV*2*4, options: .storageModeShared)!
    var n = UInt32(nLink), nvs = UInt32(nV)

    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(bLinks, offset: 0, index: 0)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    enc.setBytes(&n, length: 4, index: 2)
    enc.setBuffer(bDag, offset: 0, index: 3)
    enc.setBytes(&nvs, length: 4, index: 4)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    let grp = MTLSize(width: (nV+255)/256, height: 1, depth: 1)
    enc.dispatchThreadgroups(grp, threadsPerThreadgroup: tg)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    let out = outBuf.contents().bindMemory(to: Float.self, capacity: nV*2)
    var maxErr: Float = 0
    for vs in 0..<nV {            // ALL sites
        for lane in 0..<2 {
            let gpu = out[vs*2 + lane]
            let cpu = cpuReTr(fields, dagger, vs, lane)
            maxErr = max(maxErr, abs(gpu - cpu))
        }
    }
    print(String(format: "  %-16@ N=%2d  max|gpu-cpu|=%.3e  (all %d sites)", name as NSString, nLink, maxErr, nV*2))
    return maxErr
}

// Per-link dagger flags for an R x T loop: forward mu (R), forward nu (T),
// backward mu (R, dag), backward nu (T, dag) -> last R+T links daggered.
func loopDagger(_ R: Int, _ T: Int) -> [UInt8] {
    let n = 2*R + 2*T
    return (0..<n).map { $0 >= (R+T) ? UInt8(1) : UInt8(0) }
}
func loopFields(_ R: Int, _ T: Int) -> [[Float]] { (0..<(2*R+2*T)).map { _ in makeField() } }
func runLoop(_ R: Int, _ T: Int) -> Float {
    runCase("rect \(R)x\(T)", loopFields(R,T), loopDagger(R,T))
}

print("Wilson-loop path-product kernel — standalone parity (float32, all sites):")
var worst: Float = 0
// 1x1 == plaquette (sanity: general kernel subsumes su3_plaquette.metal)
worst = max(worst, runLoop(1,1))
worst = max(worst, runLoop(2,1))
worst = max(worst, runLoop(2,2))
worst = max(worst, runLoop(3,3))
worst = max(worst, runLoop(6,6))    // 24 links
worst = max(worst, runLoop(12,6))   // 36 links — exceeds old 32-bit mask; ether's target shape
worst = max(worst, runLoop(12,12))  // 48 links — stress float32 accumulation
print(String(format: "MAX |gpu-cpu| over all cases = %.3e (float32)", worst))
print(worst < 1e-3 ? "PARITY PASS" : "PARITY FAIL")
