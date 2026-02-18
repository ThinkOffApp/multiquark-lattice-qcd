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

## Start dashboard

From the project root (`/path/to/multiquark-lattice-qcd`):

```bash
cd /path/to/multiquark-lattice-qcd
python3 -m http.server 8000
```

Open:

`http://localhost:8000/grid-gpt/tools/su2_dashboard.html`

Then set the two file paths in the dashboard UI, for example:

- `results/su2_signal_scan/progress_petrus-su2-signal.json`
- `results/su2_signal_scan/live_petrus-su2-signal.json`

## Run command example

```bash
cd /path/to/multiquark-lattice-qcd/gpt
source lib/cgpt/build/source.sh
python3 applications/hmc/su2_2q_signal_scan.py \
  --seed petrus-su2-signal \
  --L 16,16,16,16 \
  --beta 2.4 \
  --ntherm 200 \
  --nmeas 200 \
  --nskip 5 \
  --R 2,3,4,6,8,12 \
  --T 2,3,4,5,6 \
  --flux-r 6 \
  --flux-t 4 \
  --flux-rperp-max 6 \
  --out /path/to/multiquark-lattice-qcd/results/su2_signal_scan
```
