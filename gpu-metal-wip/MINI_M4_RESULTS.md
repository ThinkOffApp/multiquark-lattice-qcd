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

- **PROVEN (2026-06-16) — SU(3) plaquette GPU measurement parity vs gpt, on real config.**
  `plaq_parity.py` + `plaq_parity_dispatch.swift`: builds an SU(3) gpt config,
  extracts/Cshift-aligns the 4 link fields per plane, packs into the kernel layout,
  dispatches `PlaquettePlane` on the M4, reduces. Results on 4^4:
  - ladder step 1 (one plane (0,1)): GPU vs numpy max per-site |ReTr| diff 2.66e-07.
  - ladder step 2 (all 6 planes): GPU full plaquette = gpt `g.qcd.gauge.plaquette`
    to **|GPU-gpt| = 6.9e-12**; worst per-site |GPU-CPU| ReTr = 2.7e-07 (float32).
  - RESULT: FULL PLAQUETTE PARITY PASS.
  Convention self-check: numpy 6-plane average == gpt plaquette to 9e-10 (validates
  Cshift directions A=Umu(x), B=Unu(x+mu), C=Umu(x+nu), D=Unu(x)).

- **STILL IN PROGRESS — performance + Wilson loops.** This proves correctness of the
  GPU plaquette measurement, not yet an end-to-end speedup (the file-roundtrip Swift
  dispatch here is a parity harness, not the in-process KERNEL_CALLNB path). Ladder
  step 3 (Wilson loops) is ClaudeMB's on the MacBook. Heatbath/HMC on GPU not started.

- **gpt note:** `g.qcd.gauge.random` is broken on the mini (numpy-version bug in
  `gpt/core/matrix/exp.py`: `int(np.log2(n/maxn))` on an array). Worked around by
  building SU(3) configs via numpy QR + det normalization and `U[mu][:] = arr`.
  `unit`, `plaquette`, `cshift`, numpy export/assign all work fine.

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
