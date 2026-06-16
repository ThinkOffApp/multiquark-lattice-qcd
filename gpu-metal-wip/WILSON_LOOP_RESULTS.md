# Wilson-loop path-product kernel — standalone parity results

Durable proof artifact for `su3_wilson_loop.metal` (ladder step 3 kernel prep),
captured on the MacBook **Apple M5 Max** GPU, 2026-06-16. Mirrors the standard
the team agreed on (proof in repo, not only in chat).

## What this proves (and does not)

- **PROVEN:** the GPU path-product kernel arithmetic is correct for the full
  rectangular Wilson-loop family (plaquette = the N=4 special case), validated
  against an independent CPU reference, over **all** vSites/lanes.
- **NOT claimed:** cgpt integration, physical-geometry Cshift alignment, or any
  speedup. This is standalone kernel arithmetic only. Integration into the gpt
  measurement path is the mini step (claudemm), after plaquette parity (done,
  `wip/metal-gauge-measurement-mini`).

## Method

- Kernel: ordered product of `2R+2T` pre-aligned SU(3) links with the two
  return edges daggered; per-link dagger flag buffer (`device const uchar*`),
  so there is **no 32-link limit** (the earlier `uint32_t` mask, flagged by
  ether/codex, is gone).
- Links are random **unitary** (Gram-Schmidt) SU(3): real loop links are
  unitary, so the path product stays bounded (ReTr in [-3,3]). Non-unitary
  random matrices make the product magnitude — and thus absolute error —
  explode with loop length, which is a test artifact, not a kernel bug.
- Full coverage: every one of the `4096 vSites x 2 lanes = 8192` sites is
  compared, not a sample (per codex/ether review).
- Precision: Apple GPU is float32; CPU reference is float.

## Result (float32, all 8192 sites)

```
  rect 1x1  N= 4  max|gpu-cpu|=4.768e-07
  rect 2x1  N= 6  max|gpu-cpu|=5.066e-07
  rect 2x2  N= 8  max|gpu-cpu|=4.768e-07
  rect 3x3  N=12  max|gpu-cpu|=6.258e-07
  rect 6x6  N=24  max|gpu-cpu|=9.537e-07
  rect 12x6 N=36  max|gpu-cpu|=1.147e-06   <- exceeds old 32-bit mask; ether's target shape
  rect 12x12 N=48 max|gpu-cpu|=1.371e-06
  MAX |gpu-cpu| over all cases = 1.371e-06  ->  PARITY PASS
```

Error is essentially flat in loop size with unitary links (~1e-6 even at 48
links), so a parity tolerance of ~1e-5 is safe for any realistic loop. The
1x1 case matches `su3_plaq_validate.swift`, confirming the general kernel
subsumes the plaquette.

## Reproduce

```bash
cd gpu-metal-wip
xcrun -sdk macosx metal -c su3_wilson_loop.metal -o /tmp/wl.air   # compile check
swiftc -O su3_wilson_validate.swift -o /tmp/wv
/tmp/wv "$(pwd)/su3_wilson_loop.metal"
```

Independently confirmed by @antigravity and @codex/@ether review (2026-06-16):
per-link dagger buffer resolves the link-count limit; SU(3)-projected parity
passes to ~1.2e-6 up to 12x12.

## Real-config cgpt parity (ladder step 3) — PASS

Beyond the standalone arithmetic above, the kernel is now validated on a **real
gpt SU(3) gauge config** (`wilson_parity.py` + `wilson_parity_dispatch.swift`,
log in `logs/wilson_parity_m5max.txt`), 8^4 lattice, plane (0,1):

```
  W(1,1) N= 4  max|d|ReTr 2.83e-07
  W(2,1) N= 6  max|d|ReTr 3.72e-07
  W(2,2) N= 8  max|d|ReTr 3.74e-07
  W(3,2) N=10  max|d|ReTr 3.87e-07
  W(3,3) N=12  max|d|ReTr 5.01e-07
  W(4,4) N=16  max|d|ReTr 4.72e-07
  worst per-site |GPU-CPU| ReTr = 5.008e-07 (float32)  -> PASS
```

- Reference: numpy double-precision ordered product of the **gpt Cshift-aligned**
  loop links (same standard claudemm used for the plaquette numpy reference).
- **gpt-native anchor:** building the full plaquette from 1x1 GPU loops over all
  6 planes gives `0.0037831949` vs `g.qcd.gauge.plaquette = 0.0037831958`,
  `|diff| = 9.3e-10`. Since the 1x1 loop IS the plaquette, this ties the
  Wilson-loop kernel directly to gpt truth (and to claudemm's 6.9e-12 plaquette
  parity).

Proof level: GPU Wilson-loop **measurement correctness on real config data is
proven**. NOT yet claimed: end-to-end in-process speedup (this harness still
does a file-roundtrip dispatch like the plaquette harness; in-process
KERNEL_CALLNB + timing is the next step).

Reproduce:
```bash
cp gpu-metal-wip/su3_wilson_loop.metal /tmp/su3_wilson_loop.metal
PYTHONPATH=gpt/lib/cgpt/build:gpt/lib python3.12 gpu-metal-wip/wilson_parity.py
```
