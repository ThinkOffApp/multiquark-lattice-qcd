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

func makeField() -> [Float] {
    var a = [Float](repeating: 0, count: nV*floatsPerMat)
    for i in 0..<(nV*floatsPerMat) { a[i] = rnd() }
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
func cpuReTr(_ fields: [[Float]], _ daggerMask: UInt32, _ vsite: Int, _ lane: Int) -> Float {
    var acc = ident()
    for k in 0..<fields.count {
        let u = mat(fields[k], vsite, lane)
        acc = (daggerMask & (1 << k)) != 0 ? mul(acc, dag(u)) : mul(acc, u)
    }
    return acc[0][0].0 + acc[1][1].0 + acc[2][2].0
}

// ---- run one case (N links, dagger pattern) on the GPU ----
func runCase(_ name: String, _ fields: [[Float]], _ daggerMask: UInt32) -> Float {
    let nLink = fields.count
    // pack into one buffer: links[k*nV + id]
    var packed = [Float](repeating: 0, count: nLink*nV*floatsPerMat)
    for k in 0..<nLink {
        let off = k*nV*floatsPerMat
        for i in 0..<(nV*floatsPerMat) { packed[off+i] = fields[k][i] }
    }
    let bLinks = buf(packed)
    let outBuf = dev.makeBuffer(length: nV*2*4, options: .storageModeShared)!
    var n = UInt32(nLink), mask = daggerMask, nvs = UInt32(nV)

    let cb = q.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pipe)
    enc.setBuffer(bLinks, offset: 0, index: 0)
    enc.setBuffer(outBuf, offset: 0, index: 1)
    enc.setBytes(&n, length: 4, index: 2)
    enc.setBytes(&mask, length: 4, index: 3)
    enc.setBytes(&nvs, length: 4, index: 4)
    let tg = MTLSize(width: 256, height: 1, depth: 1)
    let grp = MTLSize(width: (nV+255)/256, height: 1, depth: 1)
    enc.dispatchThreadgroups(grp, threadsPerThreadgroup: tg)
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

    let out = outBuf.contents().bindMemory(to: Float.self, capacity: nV*2)
    var maxErr: Float = 0
    for vs in [0, 1, 17, 100, 2000, 4095] {
        for lane in 0..<2 {
            let gpu = out[vs*2 + lane]
            let cpu = cpuReTr(fields, daggerMask, vs, lane)
            maxErr = max(maxErr, abs(gpu - cpu))
        }
    }
    print(String(format: "  %-28@ N=%d mask=0x%02X  max|gpu-cpu|=%.3e", name as NSString, nLink, daggerMask, maxErr))
    return maxErr
}

// Build the dagger mask for an R x T loop: forward mu (R), forward nu (T),
// backward mu (R, dag), backward nu (T, dag) -> top R+T bits set.
func loopMask(_ R: Int, _ T: Int) -> UInt32 {
    var m: UInt32 = 0
    for k in (R+T)..<(2*R+2*T) { m |= (1 << k) }
    return m
}
func loopFields(_ R: Int, _ T: Int) -> [[Float]] { (0..<(2*R+2*T)).map { _ in makeField() } }

print("Wilson-loop path-product kernel — standalone parity (float32):")
var worst: Float = 0
// 1x1 == plaquette (sanity: general kernel subsumes su3_plaquette.metal)
worst = max(worst, runCase("plaquette 1x1", loopFields(1,1), loopMask(1,1)))
worst = max(worst, runCase("rect 2x1",      loopFields(2,1), loopMask(2,1)))
worst = max(worst, runCase("rect 2x2",      loopFields(2,2), loopMask(2,2)))
worst = max(worst, runCase("rect 3x2",      loopFields(3,2), loopMask(3,2)))
worst = max(worst, runCase("rect 3x3",      loopFields(3,3), loopMask(3,3)))
print(String(format: "MAX |gpu-cpu| over all cases = %.3e (float32)", worst))
print(worst < 1e-3 ? "PARITY PASS" : "PARITY FAIL")
