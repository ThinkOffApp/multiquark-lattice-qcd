# SU(2) Dashboard

The run script writes two live files in the output directory:

- `progress_<seed>.json`: phase, progress bars, ETA, current scalar values
- `live_<seed>.json`: live measurement history for charts

## Start dashboard

From the project root:

```bash
cd /Users/petrus/AndroidStudioProjects/ThinkOff
python3 -m http.server 8000
```

Open:

`http://localhost:8000/grid-gpt/tools/su2_dashboard.html`

Then set the two file paths in the dashboard UI, for example:

- `results/su2_signal_scan/progress_petrus-su2-signal.json`
- `results/su2_signal_scan/live_petrus-su2-signal.json`

## Run command example

```bash
cd /Users/petrus/AndroidStudioProjects/ThinkOff/grid-gpt/gpt
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
  --out /Users/petrus/AndroidStudioProjects/ThinkOff/results/su2_signal_scan
```
