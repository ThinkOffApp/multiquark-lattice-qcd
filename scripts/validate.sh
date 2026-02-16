#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

LIVE_JSON="$TMP_DIR/live_baseline.json"
OUT_JSON="$TMP_DIR/postprocess_baseline.json"

python3 - "$LIVE_JSON" <<'PY'
import json
import math
import sys

live_path = sys.argv[1]
Rs = [2, 3, 4]
Ts = [2, 3, 4, 5, 6]
V_true = {2: 0.31, 3: 0.44, 4: 0.57}
A_true = {2: 0.95, 3: 0.72, 4: 0.55}
n_cfg = 24

measurements = []
for i in range(n_cfg):
    plaq = 0.603 + 0.002 * math.sin(0.37 * i)
    loops = {}
    for r in Rs:
        for t in Ts:
            amp = A_true[r]
            v = V_true[r]
            noise = 1.0 + 0.02 * math.sin(0.23 * i + 0.19 * r + 0.11 * t)
            val = amp * math.exp(-v * t) * noise
            loops[f"R{r}_T{t}"] = {"re": val, "im": 0.0}
    flux = [0.08 * math.exp(-0.45 * rp) + 0.001 * math.sin(0.2 * i + rp) for rp in range(7)]
    measurements.append(
        {
            "plaquette": plaq,
            "loops": loops,
            "flux_profile_r_perp": flux,
        }
    )

payload = {
    "meta": {"seed": "baseline-synth", "R": Rs, "T": Ts},
    "measurements": measurements,
}
with open(live_path, "w", encoding="utf-8") as f:
    json.dump(payload, f)
PY

python3 "$ROOT_DIR/tools/su2_signal_postprocess.py" \
  --live "$LIVE_JSON" \
  --out "$OUT_JSON" \
  --bin-size 4 \
  --fit-tmin 2 \
  --fit-tmax 6 \
  --min-fit-points 3 \
  --flux-vacuum-tail 2 >/dev/null

baseline="$(python3 - "$OUT_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    out = json.load(f)

potential = {int(x["R"]): x for x in out.get("potential", [])}
if 2 not in potential:
    raise SystemExit("validation failed: missing R=2 fit")

v = float(potential[2]["V"])
if not (0.29 <= v <= 0.33):
    raise SystemExit(f"validation failed: baseline V(R=2) out of range: {v:.6f}")

print(f"{v:.6f}")
PY
)"

echo "baseline_V_R2=${baseline}"
echo "validate: OK"
