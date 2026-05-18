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
pip install -r requirements.txt -q

export SU2_OUT_DIR="./tmp_smoke_test"
export PYTHONPATH="$(pwd)/gpt/lib/cgpt/build:$(pwd)/gpt/lib:${PYTHONPATH:-}"
mkdir -p "$SU2_OUT_DIR"

echo "Executing dummy 2^4 lattice run..."
python3 gpt/applications/hmc/su2_2q_signal_scan.py --L 2,2,2,2 --ntherm 1 --nmeas 1 --backend cpu --seed smoke --out "$SU2_OUT_DIR" || { echo "Smoke test failed!"; exit 1; }
echo "Python execution engine loaded successfully and ran a tiny lattice."

echo "Cleaning up..."
rm -rf "$SU2_OUT_DIR"
echo "Smoke test complete."
