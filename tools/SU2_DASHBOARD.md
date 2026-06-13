# SU(2) Dashboard

The run script writes two live files in the output directory:

- `progress_<seed>.json`: phase, progress bars, ETA, current scalar values
- `live_<seed>.json`: live measurement history for charts

This workflow is supported on both Linux and macOS.

## Advanced Analytical Features

The dashboard provides real-time physics interpretation beyond simple progress tracking:

### 1. Robust Physics Extraction
- **Cornell Potential Fitting**: Automatically fits $V(R) = \sigma R - e/R + V_0$.
- **Model Comparison**: Performs **AIC (Akaike Information Criterion)** testing to determine if the data favors a free Coulomb term or the theoretical **Lüscher term** ($\pi/12$).
- **Sommer Scale**: Solves for $r_0$ and $r_1$ (via $r^2 F(r) = C$) to provide lattice-independent scale setting.
- **Flux Tube Roughening**: Fits transverse width growth $w^2(R) = A + B \ln R$ to verify logarithmic broadening.

### 2. Statistical Health Monitoring
- **Autocorrelation Analysis**: Calculates $\tau_{int}$ for the plaquette to ensure configuration independence and suggests optimal `nskip` values.
- **Signal-to-Noise (SNR)**: Real-time evaluation of signal emergence for Wilson loops and flux profiles, including stability and drift checks.
- **Center Symmetry ($Z_2$)**: Visualizes $\text{Re } P_\mu$ sectors to detect the confinement phase vs. deconfined transitions.

### 3. Direct Physical Comparison
- **Benchmark Alignment**: Converts measured $\sigma$, $\sqrt{\sigma}$, and $r_0$ into physical units (GeV/fm) for direct comparison with literature benchmarks (e.g., SU(3) $\sqrt{\sigma} \approx 440$ MeV).

### 4. Navigation
- **Run picker**: a dropdown at the top of the controls, populated from the dashboard server's `/api/runs` endpoint, lists every run under `results/` (newest first; per-seed remeasure files are collapsed to the `remeasure_total`). Selecting an entry fills both JSON path inputs and reloads, so switching between seeds/runs no longer needs hand-typed paths. Manual path entry still works.
- **Sticky status bar**: once you scroll past the inline metric cards, a compact bar pins the headline values (phase, N_meas, plaquette, ETA, elapsed) to the top of the viewport. The dashboard is several screens tall, so this keeps the live numbers in view while reading the charts. It stays hidden until real data has loaded.

## Start dashboard

The recommended way is the bundled server, which adds the `/events` SSE stream, the JSONL merge for `live_*.json`, and the `/api/runs` endpoint that powers the run picker:

```bash
cd /path/to/multiquark-lattice-qcd
python3 tools/su2_dashboard_server.py --host 127.0.0.1 --port 8001 --root .
```

Open:

`http://localhost:8001/tools/su2_dashboard.html`

A plain static server (`python3 -m http.server 8000`) also serves the page, but without the run picker, the SSE stream, or the JSONL merge.

The run picker fills the path inputs for you. To set them manually, point at any run under `results/`, for example:

- `results/su2_signal_scan_v2/progress_petrus-su2-signal-v2.json`
- `results/su2_signal_scan_v2/live_petrus-su2-signal-v2.json`

## Run command example

The example below reproduces the genuine-SU(2) production run (`petrus-su2-signal-v2`,
plaquette $0.629958 \pm 0.000023$, on the literature value). Note the explicit
SU(2) gauge group from PR #18 — without it the script silently defaults to
SU(3) fundamental links, which is why the older `petrus-su2-signal` archive is
actually SU(3) strong-coupling data and needs regeneration.

```bash
cd /path/to/multiquark-lattice-qcd/gpt
source lib/cgpt/build/source.sh
python3 applications/hmc/su2_2q_signal_scan.py \
  --seed petrus-su2-signal-v2 \
  --L 16,16,16,16 \
  --beta 2.4 \
  --ntherm 200 \
  --nmeas 200 \
  --nskip 5 \
  --R 1,2,3,4,5,6,7,8,12 \
  --T 1,2,3,4,5,6 \
  --flux-r 6 \
  --flux-t 4 \
  --flux-rperp-max 6 \
  --out /path/to/multiquark-lattice-qcd/results/su2_signal_scan_v2
```

`R=1` anchors the short-distance Coulomb part of the Cornell fit (it is
dominated by lattice artifacts and self-energy, not the string tension);
`R=5,7` fill the gaps the earlier run left as `n/a`.
