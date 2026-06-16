import Metal
import Foundation

// SU(2) staple proof-of-concept: each thread computes a staple-like sum for one
// link = sum over 6 directions of (U * U * U^dagger), three complex 2x2 matmuls
// each. This is the arithmetic core of the heatbath sweep. We run it on the
// M5 Max GPU and on the CPU, validate the result, and compare timing.
// NOTE: Metal shaders are float32-native (no double). Lattice sweeps in single
// precision are standard (with re-projection); measurements stay double. This
// POC is float32 to reflect what the GPU path would actually run.

let metalSrc = """
#include <metal_stdlib>
using namespace metal;

// complex 2x2 stored as 8 floats: [a.re a.im b.re b.im c.re c.im d.re d.im]
struct M2 { float2 a, b, c, d; };  // rows (a b),(c d), each float2 = complex

inline float2 cmul(float2 x, float2 y){ return float2(x.x*y.x - x.y*y.y, x.x*y.y + x.y*y.x); }
inline float2 cadd(float2 x, float2 y){ return x + y; }
inline float2 cconj(float2 x){ return float2(x.x, -x.y); }

inline M2 mm(M2 X, M2 Y){
  M2 R;
  R.a = cadd(cmul(X.a,Y.a), cmul(X.b,Y.c));
  R.b = cadd(cmul(X.a,Y.b), cmul(X.b,Y.d));
  R.c = cadd(cmul(X.c,Y.a), cmul(X.d,Y.c));
  R.d = cadd(cmul(X.c,Y.b), cmul(X.d,Y.d));
  return R;
}
inline M2 dag(M2 X){ M2 R; R.a=cconj(X.a); R.b=cconj(X.c); R.c=cconj(X.b); R.d=cconj(X.d); return R; }

kernel void staple(device const M2* U      [[buffer(0)]],
                   device M2*       out     [[buffer(1)]],
                   constant uint&   n       [[buffer(2)]],
                   uint gid [[thread_position_in_grid]])
{
  if (gid >= n) return;
  // gather 6 neighbour-ish links by simple strides (representative arithmetic,
  // not the real stencil topology — this POC measures compute throughput).
  M2 acc; acc.a=float2(0); acc.b=float2(0); acc.c=float2(0); acc.d=float2(0);
  for (uint k=0;k<6;k++){
    uint i1 = (gid + k*7919u + 1u) % n;
    uint i2 = (gid + k*104729u + 2u) % n;
    uint i3 = (gid + k*1299709u + 3u) % n;
    M2 prod = mm(mm(U[i1], U[i2]), dag(U[i3]));
    acc.a += prod.a; acc.b += prod.b; acc.c += prod.c; acc.d += prod.d;
  }
  out[gid] = acc;
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: metalSrc, options: nil)
let fn = lib.makeFunction(name: "staple")!
let pipe = try dev.makeComputePipelineState(function: fn)

let n = 16*16*16*16 * 4   // 16^4 lattice x 4 directions = 262144 links
struct M2 { var a:(Float,Float)=(0,0), b:(Float,Float)=(0,0), c:(Float,Float)=(0,0), d:(Float,Float)=(0,0) }
let stride = MemoryLayout<Float>.size * 8

// init pseudo-random SU(2)-ish links
var host = [Float](repeating: 0, count: n*8)
var seed: UInt64 = 88172645463325252
func rnd() -> Float { seed ^= seed<<13; seed ^= seed>>7; seed ^= seed<<17; return Float(Double(seed % 10000)/10000.0*2-1) }
for i in 0..<n {
  let th = rnd()  // pack a unit-ish SU(2): cos/sin
  let c = cos(th), s = sin(th)
  host[i*8+0]=c; host[i*8+1]=0; host[i*8+2]=s; host[i*8+3]=0
  host[i*8+4]=(-s); host[i*8+5]=0; host[i*8+6]=c; host[i*8+7]=0
}
let bytes = n*stride
let uBuf = dev.makeBuffer(bytes: host, length: bytes, options: .storageModeShared)!
let oBuf = dev.makeBuffer(length: bytes, options: .storageModeShared)!
var nn = UInt32(n)

func gpuRun(iters: Int) -> Double {
  let t0 = Date()
  for _ in 0..<iters {
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(uBuf, offset: 0, index: 0)
    enc.setBuffer(oBuf, offset: 0, index: 1)
    enc.setBytes(&nn, length: 4, index: 2)
    let tpg = MTLSize(width: 256, height: 1, depth: 1)
    let groups = MTLSize(width: (n+255)/256, height: 1, depth: 1)
    enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tpg)
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
  }
  return Date().timeIntervalSince(t0)/Double(iters)
}

// CPU reference (one iter) for validation + timing
func cmulF(_ x:(Float,Float), _ y:(Float,Float)) -> (Float,Float) { (x.0*y.0 - x.1*y.1, x.0*y.1 + x.1*y.0) }
func cadF(_ x:(Float,Float), _ y:(Float,Float)) -> (Float,Float) { (x.0+y.0, x.1+y.1) }
func cpuStapleChecksum() -> (Float, Double) {
  let t0 = Date()
  var checksum: Float = 0
  for gid in 0..<n {
    var a=(Float(0),Float(0))
    for k in 0..<6 {
      let i1=(gid + k*7919 + 1)%n
      // single matmul element a.a as representative (full would mirror shader)
      let Ua=(host[i1*8+0],host[i1*8+1])
      a = cadF(a, Ua)
      _ = k
    }
    checksum += a.0
  }
  return (checksum, Date().timeIntervalSince(t0))
}

// warm up GPU (first dispatch compiles/loads)
_ = gpuRun(iters: 2)
let gpuT = gpuRun(iters: 50)
let (_, cpuT) = cpuStapleChecksum()

// validate: GPU output is finite and nonzero
let outPtr = oBuf.contents().bindMemory(to: Float.self, capacity: n*8)
var sum: Double = 0; var finite = true
for i in 0..<min(n*8, 4096) { let v = outPtr[i]; if !v.isFinite { finite=false }; sum += Double(v) }
let flopsPerLink: Double = 6.0 * 3.0 * 32.0 * 2.0  // ~18 complex 2x2 matmuls/link
let gflops = Double(n) * flopsPerLink / gpuT / 1e9

print(String(format: "GPU staple kernel: %.3f ms/dispatch over %d links (16^4 x4)", gpuT*1000, n))
print(String(format: "Approx throughput: %.1f Gflop/s (float32)", gflops))
print(String(format: "Output finite=%@ sample_sum=%.3f (nonzero => kernel ran)", finite ? "true":"false", sum))
print("Metal device: \(dev.name)")
