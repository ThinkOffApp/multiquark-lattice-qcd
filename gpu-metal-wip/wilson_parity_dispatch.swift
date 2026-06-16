// Dispatch su3_wilson_loop.metal WilsonLoopPath on real gpt-extracted link
// fields, for a general rectangular Wilson loop. Generalizes
// plaq_parity_dispatch.swift to N pre-aligned link fields + per-link dagger.
//
// Usage: wilson_parity_dispatch <nVSite> <nLink> <daggerBits> <out.bin> <f0.bin> <f1.bin> ...
//   daggerBits: string of '0'/'1', length nLink (1 => that link enters as U^dag).
//   each fk.bin: float32, nVSite*9*4, layout SU3Matrix[9] float4=(l0re,l0im,l1re,l1im).
//   out.bin: per-vSite Re Tr (float2 -> nVSite*2 float32).
import Metal
import Foundation

let args = CommandLine.arguments
let nV = Int(args[1])!
let nLink = Int(args[2])!
let daggerBits = Array(args[3]).map { UInt8($0 == "1" ? 1 : 0) }
precondition(daggerBits.count == nLink, "daggerBits length must equal nLink")
let outPath = args[4]
let fieldPaths = Array(args[5..<(5+nLink)])

func load(_ p: String) -> [Float] {
    let d = FileManager.default.contents(atPath: p)!
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

let floatsPerMat = 9 * 4
// Pack all fields into one buffer: links[k*nV + id]
var packed = [Float](repeating: 0, count: nLink * nV * floatsPerMat)
for k in 0..<nLink {
    let f = load(fieldPaths[k])
    precondition(f.count == nV * floatsPerMat, "field \(k) wrong size")
    let off = k * nV * floatsPerMat
    for i in 0..<f.count { packed[off + i] = f[i] }
}

let src = try String(contentsOfFile: "/tmp/su3_wilson_loop.metal", encoding: .utf8)
let dev = MTLCreateSystemDefaultDevice()!
print("AcceleratorMetalInit: device = \(dev.name)")
let q = dev.makeCommandQueue()!
let lib = try dev.makeLibrary(source: src, options: nil)
let pipe = try dev.makeComputePipelineState(function: lib.makeFunction(name: "WilsonLoopPath")!)

let bLinks = dev.makeBuffer(bytes: packed, length: packed.count*4, options: .storageModeShared)!
let bDag = dev.makeBuffer(bytes: daggerBits, length: daggerBits.count, options: .storageModeShared)!
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
let outArr = Array(UnsafeBufferPointer(start: out, count: nV*2))
let data = outArr.withUnsafeBytes { Data($0) }
try data.write(to: URL(fileURLWithPath: outPath))
print("wrote \(nV*2) floats to \(outPath)")
