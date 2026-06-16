import Metal
import Foundation

// Validate the SU(3) plaquette kernel arithmetic: GPU Re Tr(A B C^dag D^dag)
// vs an independent CPU reference, over random complex 3x3 matrices packed in
// the NEON Nsimd=2 layout (float4 = 2 complex lanes).

let src = try String(contentsOfFile: "/tmp/su3_plaquette.metal", encoding: .utf8)
let dev = MTLCreateSystemDefaultDevice()!
let q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: src, options: nil)
let fn = lib.makeFunction(name: "PlaquettePlane")!
let pipe = try dev.makeComputePipelineState(function: fn)

let nV = 4096           // vSites (each = 2 lane-sites)
let floatsPerMat = 9*4  // 9 complex * float4(2 lanes)

// xorshift RNG
var s: UInt64 = 0x9e3779b97f4a7c15
func rnd() -> Float { s ^= s<<13; s ^= s>>7; s ^= s<<17; return Float(Double(s % 20000)/10000.0 - 1.0) }

func makeField() -> [Float] {
    var a = [Float](repeating: 0, count: nV*floatsPerMat)
    for i in 0..<(nV*floatsPerMat) { a[i] = rnd() }
    return a
}
let A = makeField(), B = makeField(), C = makeField(), D = makeField()

func buf(_ a: [Float]) -> MTLBuffer { dev.makeBuffer(bytes: a, length: a.count*4, options: .storageModeShared)! }
let bA = buf(A), bB = buf(B), bC = buf(C), bD = buf(D)
let outBuf = dev.makeBuffer(length: nV*2*4, options: .storageModeShared)!
var n = UInt32(nV)

let cb = q.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipe)
enc.setBuffer(bA,offset:0,index:0); enc.setBuffer(bB,offset:0,index:1); enc.setBuffer(bC,offset:0,index:2)
enc.setBuffer(bD,offset:0,index:3); enc.setBuffer(outBuf,offset:0,index:4); enc.setBytes(&n,length:4,index:5)
let tg = MTLSize(width: 256, height: 1, depth: 1)
let grp = MTLSize(width: (nV+255)/256, height: 1, depth: 1)
enc.dispatchThreadgroups(grp, threadsPerThreadgroup: tg)
enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()

// CPU reference for a chosen vSite + lane
func cpuReTr(_ vsite: Int, _ lane: Int) -> Float {
    // unpack complex 3x3 for one lane: element (r,c) = data[(r*3+c)] float4,
    // lane0 -> (.x,.y), lane1 -> (.z,.w)
    func mat(_ f: [Float]) -> [[(Float,Float)]] {
        var m = [[(Float,Float)]](repeating: [(Float,Float)](repeating:(0,0),count:3),count:3)
        for r in 0..<3 { for c in 0..<3 {
            let base = vsite*floatsPerMat + (r*3+c)*4
            m[r][c] = lane==0 ? (f[base],f[base+1]) : (f[base+2],f[base+3])
        }}
        return m
    }
    func cm(_ x:(Float,Float),_ y:(Float,Float))->(Float,Float){ (x.0*y.0-x.1*y.1, x.0*y.1+x.1*y.0) }
    func cadd(_ x:(Float,Float),_ y:(Float,Float))->(Float,Float){ (x.0+y.0,x.1+y.1) }
    func cconj(_ x:(Float,Float))->(Float,Float){ (x.0,-x.1) }
    func mul(_ X:[[(Float,Float)]],_ Y:[[(Float,Float)]])->[[(Float,Float)]]{
        var R=[[(Float,Float)]](repeating:[(Float,Float)](repeating:(0,0),count:3),count:3)
        for i in 0..<3 { for j in 0..<3 { var su=(Float(0),Float(0)); for k in 0..<3 { su=cadd(su,cm(X[i][k],Y[k][j])) }; R[i][j]=su }}
        return R
    }
    func dag(_ X:[[(Float,Float)]])->[[(Float,Float)]]{
        var R=[[(Float,Float)]](repeating:[(Float,Float)](repeating:(0,0),count:3),count:3)
        for i in 0..<3 { for j in 0..<3 { R[i][j]=cconj(X[j][i]) }}
        return R
    }
    let ma=mat(A), mb=mat(B), mc=mat(C), md=mat(D)
    let p = mul(mul(mul(ma,mb), dag(mc)), dag(md))
    return p[0][0].0 + p[1][1].0 + p[2][2].0
}

let out = outBuf.contents().bindMemory(to: Float.self, capacity: nV*2)
var maxErr: Float = 0
for vs in [0, 1, 17, 100, 2000, 4095] {
    for lane in 0..<2 {
        let gpu = out[vs*2 + lane]
        let cpu = cpuReTr(vs, lane)
        let err = abs(gpu - cpu)
        maxErr = max(maxErr, err)
        if vs == 0 { print(String(format: "vSite %d lane %d: gpu=%.5f cpu=%.5f err=%.2e", vs, lane, gpu, cpu, err)) }
    }
}
print(String(format: "MAX |gpu-cpu| over sampled sites = %.3e (float32)", maxErr))
print(maxErr < 1e-3 ? "PARITY PASS" : "PARITY FAIL")
