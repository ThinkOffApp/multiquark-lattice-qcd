#!/usr/bin/env bash
set -e

echo "====================================="
echo "Running SU(2) Measurement Smoke Test"
echo "====================================="

if [ ! -d ".venv" ]; then
    echo "Creating python environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -r requirements.txt -q

SU2_OUT_DIR="$(mktemp -d -t su2_smoke.XXXXXX)"
export SU2_OUT_DIR
export PYTHONPATH="$(pwd)/gpt/lib/cgpt/build:$(pwd)/gpt/lib:${PYTHONPATH:-}"
trap 'rm -rf "$SU2_OUT_DIR"' EXIT

echo "Executing tiny 4^4 lattice run in $SU2_OUT_DIR ..."
python3 gpt/applications/hmc/su2_2q_signal_scan.py \
    --L 4,4,4,4 \
    --R 1,2 --T 1,2 \
    --ntherm 1 --nmeas 1 --nskip 1 \
    --pipeline-label cpu \
    --seed smoke \
    --skip_flux 1 \
    --out "$SU2_OUT_DIR" \
    || { echo "Smoke test failed!"; exit 1; }

# Assert the driver actually produced a progress/live file for the seed.
if ! ls "$SU2_OUT_DIR"/progress_smoke.json >/dev/null 2>&1 \
    && ! ls "$SU2_OUT_DIR"/live_smoke.json >/dev/null 2>&1; then
    echo "Smoke test failed: no progress_smoke.json or live_smoke.json produced in $SU2_OUT_DIR"
    ls -la "$SU2_OUT_DIR" || true
    exit 1
fi

echo "Smoke test complete (output verified)."
