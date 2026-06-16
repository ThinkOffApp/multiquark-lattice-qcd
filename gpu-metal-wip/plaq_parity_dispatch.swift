// Dispatch su3_plaquette.metal PlaquettePlane on real gpt-extracted link fields.
// Reads 4 packed field files (float32, nVSite*9*4 each, layout = SU3Matrix[9]
// float4 = (lane0_re,lane0_im,lane1_re,lane1_im)) + writes per-vSite Re Tr (float2).
// Usage: plaq_parity_dispatch <nVSite> <A.bin> <B.bin> <C.bin> <D.bin> <out.bin>
import Metal
import Foundation

let args = CommandLine.arguments
let nV = Int(args[1])!
func load(_ p: String) -> [Float] {
    let d = FileManager.default.contents(atPath: p)!
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}
let A = load(args[2]), B = load(args[3]), C = load(args[4]), D = load(args[5])

let src = try String(contentsOfFile: "/tmp/su3_plaquette.metal", encoding: .utf8)
let dev = MTLCreateSystemDefaultDevice()!
print("AcceleratorMetalInit: device = \(dev.name)")
let q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: src, options: nil)
let pipe = try dev.makeComputePipelineState(function: lib.makeFunction(name: "PlaquettePlane")!)

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

let out = outBuf.contents().bindMemory(to: Float.self, capacity: nV*2)
let outArr = Array(UnsafeBufferPointer(start: out, count: nV*2))
let data = outArr.withUnsafeBytes { Data($0) }
try data.write(to: URL(fileURLWithPath: args[6]))
print("wrote \(nV*2) floats to \(args[6])")
