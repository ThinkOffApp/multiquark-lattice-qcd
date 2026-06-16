// In-process Metal dispatcher for the SU(3) gauge measurement, exposed as a
// C-ABI dylib so gpt's own python process can call it via ctypes. This removes
// the file-roundtrip + per-call subprocess of the parity harness: the Metal
// device, command queue and compute pipeline are built once and kept warm.
//
// Same kernel (WilsonLoopPath) and NEON Nsimd=2 / 9x float4 layout as the proven
// su3_wilson_loop.metal. This is the python-side in-process step; the deeper
// zero-copy-on-Grid-buffers version lives in cgpt C++ (KERNEL_CALLNB).
//
// Build: swiftc -O -emit-library -o libcwgauge.dylib gpu_gauge_inproc.swift
import Metal
import Foundation

final class GaugeDispatcher {
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let pipe: MTLComputePipelineState

    init?(metalSourcePath: String) {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue(),
              let src = try? String(contentsOfFile: metalSourcePath, encoding: .utf8),
              let lib = try? d.makeLibrary(source: src, options: nil),
              let fn = lib.makeFunction(name: "WilsonLoopPath"),
              let p = try? d.makeComputePipelineState(function: fn)
        else { return nil }
        dev = d; queue = q; pipe = p
    }

    // links: nLink*nV*36 float32 (SU3Matrix[9] float4 per vSite, packed).
    // dagger: nLink uint8. out: nV*2 float32 (per-vSite Re Tr, 2 lanes).
    func dispatch(_ links: UnsafeRawPointer, _ nLink: Int,
                  _ dagger: UnsafeRawPointer, _ nV: Int,
                  _ out: UnsafeMutableRawPointer) {
        let floatsPerMat = 36
        let bLinks = dev.makeBuffer(bytes: links, length: nLink*nV*floatsPerMat*4, options: .storageModeShared)!
        let bDag   = dev.makeBuffer(bytes: dagger, length: nLink, options: .storageModeShared)!
        let outBuf = dev.makeBuffer(length: nV*2*4, options: .storageModeShared)!
        var n = UInt32(nLink), nvs = UInt32(nV)
        let cb = queue.makeCommandBuffer()!
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
        memcpy(out, outBuf.contents(), nV*2*4)
    }
}

private var GDISP: GaugeDispatcher? = nil

@_cdecl("cw_gauge_init")
public func cw_gauge_init(_ path: UnsafePointer<CChar>) -> Int32 {
    GDISP = GaugeDispatcher(metalSourcePath: String(cString: path))
    return GDISP == nil ? -1 : 0
}

@_cdecl("cw_wilson_dispatch")
public func cw_wilson_dispatch(_ links: UnsafeRawPointer, _ nLink: Int32,
                               _ dagger: UnsafeRawPointer, _ nV: Int32,
                               _ out: UnsafeMutableRawPointer) -> Int32 {
    guard let g = GDISP else { return -1 }
    g.dispatch(links, Int(nLink), dagger, Int(nV), out)
    return 0
}
