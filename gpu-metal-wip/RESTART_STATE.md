# Metal GPU gauge-measurement WIP — pause state (2026-06-13, before MacBook OS update)

Work-in-progress toward GPU-accelerated SU(3) gauge measurement on the M5 Max,
preserved here because /tmp is wiped on reboot.

## Artifacts in this dir
- `su3_plaquette.metal` — SU(3) plaquette compute kernel (MSL), antigravity's NEON
  Nsimd=2 conventions (9x float4 link, per-lane complex mul/conj, 3x3 mul+dag+ReTr).
  Local per-site (neighbours pre-aligned by CPU Cshift; no in-kernel stencil).
- `su3_plaq_validate.swift` — parity validator: GPU vs CPU Re Tr(A B C^dag D^dag)
  over random SU(3). PASSED at max err 9.5e-7 (float32).
- `su2_metal_poc.swift` — earlier SU(2) staple throughput POC (~675 Gflop/s).
- `su2_mb_bench.py`, `su2_mb_bench16.py` — CPU SU(2) heatbath benchmarks (8^4, 16^4).
- `p9804004.txt` — extracted text of hep-lat/9804004 (NOT committed; copyright).

## Verified state before pause
- gpt runs on this MacBook via **python3.12** (cgpt built cpython-312 ABI).
- cgpt.so needed `codesign --force --sign -` (Gatekeeper rejected linker-signed adhoc).
  Path: `gpt/lib/cgpt/build/cgpt.cpython-312-darwin.so`.
- Run env: `PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 ...`
- SU(3) plaquette GPU kernel: parity PASS. NERSC save/load/measure round-trip: PASS.

## Post-OS-update restart checklist (ether's list)
1. `which python3.12` still present; `PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 -c "import gpt"` works.
2. `xcrun metal --version` still works (Metal toolchain survives update?).
3. cgpt codesign still valid: `codesign -v gpt/lib/cgpt/build/cgpt.cpython-312-darwin.so`
   — if Gatekeeper re-flags after update, re-run `codesign --force --sign -` on it.
4. GPU plaquette test still passes: `cd gpu-metal-wip && xcrun -sdk macosx metal -c su3_plaquette.metal -o /tmp/k.air && xcrun -sdk macosx metallib /tmp/k.air -o /tmp/k.metallib && swiftc -O su3_plaq_validate.swift -o /tmp/v && /tmp/v`

## Next steps (not yet done)
- Integrate the GPU plaquette into cgpt: CPU Cshift align 4 fields per plane ->
  dispatch via antigravity's KERNEL_CALLNB harness -> sum-reduce -> parity vs
  g.qcd.gauge.plaquette on a real config.
- Obtain a small public SU(3) ensemble (24^3x64 / 32^3x64) — needs download access
  (registration/email; gated on Petrus).
