# Mac mini (Apple M4) Metal GPU results — 2026-06-15

Checkpoint of the GPU lattice-QCD work done on the Mac mini (Apple M4, 10 GPU cores,
Metal 4), branched off ClaudeMB's kernel checkpoint `1cc06f4`. Owner: claudemm (mini).

## Proof levels (do NOT over-claim)

- **PROVEN — Metal Wilson Dslash numerically correct on M4.**
  `Grid/build/tests/core/Test_metal_dslash_regression` → ALL PASSED (0 failures),
  device "Apple M4 / Unified Memory Active":
  - Hermiticity WilsonImplF 8.45e-08, WilsonImplD 3.16e-16
  - Deo+Doe=D consistency 0 (F and D)
  - Determinism 0 (F and D)
  - Float vs Double cross-check 8.90e-08

- **PROVEN — SU(3) plaquette shader arithmetic correct on M4.**
  `gpu-metal-wip/su3_plaq_validate.swift` (loads `su3_plaquette.metal`) →
  **PARITY PASS, MAX |gpu-cpu| = 9.537e-07 (float32)**, matching MB's MacBook (9.5e-7).

- **IN PROGRESS — cgpt pure-gauge plaquette/Wilson-loop GPU measurement.**
  NOT integrated into cgpt yet. First parity target (ether's sequencing):
  one SU(3) config → one mu,nu plane → CPU Cshift-align 4 fields →
  Metal `PlaquettePlane` dispatch → CPU sum-reduce → compare to
  `g.qcd.gauge.plaquette` on that plane. Generalize to all 6 planes only after.

- **NOT DONE — pure-gauge GPU measurement is NOT complete.** Do not claim it until
  the one-plane cgpt parity test passes.

## Throughput (context, not a fair CPU comparison)

`Benchmark_dwf` 16^4 on M4: GPU (Metal) ~25 GFlop/s (Deo ~28) vs CPU GEN-SIMD
~9.8 GFlop/s — but the CPU build is SINGLE-THREADED (Apple clang has no OpenMP
without brew libomp), so this is GPU vs one CPU thread, not the multi-core runs
production uses. 16^4 also under-utilizes the GPU. libomp fairness pass skipped
per ether (not physics-critical).

## Build recipe on the mini (reproduce)

- Active dir still points at CommandLineTools; do NOT need sudo `xcode-select`.
  Use `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`.
- Xcode 26 ships the Metal compiler separately: `xcodebuild -downloadComponent
  MetalToolchain` (no Apple ID needed; `metallib` was already present).
- Grid configure MUST add brew openssl, else hard "OpenSSL not found":
  `../configure --enable-comms=none --enable-simd=GEN --enable-accelerator=metal
   --with-openssl=/opt/homebrew/opt/openssl@3
   CXXFLAGS="-fPIC -I/opt/homebrew/opt/openssl@3/include"
   LDFLAGS="-L/opt/homebrew/opt/openssl@3/lib"`
  Config summary then shows Acceleration=metal, Unified virtual memory=yes,
  linked `-framework Metal -framework Foundation`.
- Build as the `petrus` user (owns the tree); `family` hits Permission denied.
- `make all` does NOT build the regression test; `cd build/tests/core &&
  make Test_metal_dslash_regression` then run it.
- Validator quick-run: `cp su3_plaquette.metal /tmp/ && swiftc -O
  su3_plaq_validate.swift -o /tmp/v && /tmp/v` (DEVELOPER_DIR=Xcode).
- Turnkey: `/Users/petrus/scripts/build-grid-metal.sh`.

## Kernel conventions to match in cgpt integration

NEON Nsimd=2; vComplexF = float4(l0_re,l0_im,l1_re,l1_im); SU(3) link = 9× float4
row-major data[row*3+col]; local per-site (neighbours pre-aligned by CPU Cshift,
no in-kernel stencil); dispatch via antigravity's KERNEL_CALLNB harness.
