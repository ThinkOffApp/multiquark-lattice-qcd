# In-process GPU gauge measurement — results

Implements the in-process step petrus asked for ("the full qcd gpu gauge
measurement"): remove the file-roundtrip parity harness and dispatch the Metal
gauge kernel in gpt's own process. MacBook Apple M5 Max, 2026-06-16.
Log: `logs/gauge_inproc_bench_m5max.txt`.

## What this is
- `gpu_gauge_inproc.swift` -> `libcwgauge.dylib`: a Swift C-ABI dylib holding a
  warm Metal device / queue / `WilsonLoopPath` pipeline. gpt's python process
  loads it via **ctypes** and dispatches in-process: no subprocess spawn, no
  `/tmp` file roundtrip (the parity harness used both).
- `gauge_inproc_bench.py`: builds a real gpt SU(3) config, computes the full
  6-plane plaquette in-process on the GPU, checks it, and times GPU vs CPU.

## Results (16^4, 65536 sites, float32)
```
  GPU in-process plaquette = 0.0004936306
  CPU numpy plaquette      = 0.0004936306
  gpt-native plaquette     = 0.0004936323
  |GPU - gpt| = 1.7e-9     |GPU - CPU| = 3.1e-11      -> CORRECT
  in-process GPU:  ~32 ms/plaquette
  CPU numpy     :  ~40 ms/plaquette
  speedup       :  ~1.25x   (numpy repack included in GPU time)
```

## Honest proof / speedup levels
- **PROVEN:** in-process GPU gauge measurement is correct (matches gpt-native to
  1.7e-9) with NO file roundtrip and NO subprocess. This is the in-process
  dispatch working end to end inside the gpt process.
- **Modest speedup only (~1.25x vs naive numpy double).** The Metal kernel itself
  is microseconds; the wall time is dominated by marshalling: repacking the gpt
  field into the NEON `9 x float4` layout and copying it into a fresh MTLBuffer
  every call. The CPU baseline here is naive numpy, not Grid's optimized SIMD
  plaquette, so this is NOT yet a production speedup claim.
- **Path to real speedup (next):** the zero-copy cgpt route. Grid lattice fields
  already live in unified-memory MTLBuffers registered in
  `acceleratorMetalBufferMap`; the dslash dispatches on them directly
  (`KERNEL_CALLNB` in `WilsonKernelsImplementation.h`). Adding the gauge kernels
  to `default.metallib` and dispatching on Grid's own buffers (no repack, no
  copy) removes the marshalling that bounds the number above. That is a cgpt C++
  change and is the engineering still to do.

## Reproduce
```bash
swiftc -O -emit-library -o /tmp/libcwgauge.dylib gpu-metal-wip/gpu_gauge_inproc.swift
PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 gpu-metal-wip/gauge_inproc_bench.py
```
