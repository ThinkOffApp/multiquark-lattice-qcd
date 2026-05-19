#!/usr/bin/env bash
set -e

echo "====================================="
echo "Running SU(2) Measurement Smoke Test"
echo "====================================="

# The cgpt native extension has a fixed CPython ABI tag (e.g.
# cgpt.cpython-312-darwin.so). If the smoke test runs under a different
# Python version it gets the cgpt Python wrapper but no native bindings,
# which fails later inside gpt as `AttributeError: module 'cgpt' has no
# attribute 'time'`. Detect the required interpreter from the built .so.
CGPT_BUILD_DIR="$(pwd)/gpt/lib/cgpt/build"
CGPT_SO="$(ls "$CGPT_BUILD_DIR"/cgpt.cpython-*-darwin.so "$CGPT_BUILD_DIR"/cgpt.cpython-*-linux-gnu.so 2>/dev/null | head -n1 || true)"
if [ -z "$CGPT_SO" ]; then
    echo "Smoke test failed: no built cgpt native extension found in $CGPT_BUILD_DIR." >&2
    echo "Build cgpt first via the project's normal build steps." >&2
    exit 2
fi
# cgpt.cpython-312-darwin.so -> 3.12
CGPT_PY_TAG="$(basename "$CGPT_SO" | sed -nE 's/^cgpt\.cpython-([0-9])([0-9]+)-.*\..*$/\1.\2/p')"
PY="python${CGPT_PY_TAG}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Smoke test failed: cgpt was built for Python $CGPT_PY_TAG (${CGPT_SO##*/})" >&2
    echo "  but '$PY' is not on PATH. Install Python $CGPT_PY_TAG or rebuild cgpt." >&2
    exit 2
fi
echo "Using $PY ($(command -v "$PY")) to match cgpt ABI ($CGPT_PY_TAG)"

if [ ! -d ".venv-smoke" ]; then
    echo "Creating Python $CGPT_PY_TAG venv at .venv-smoke ..."
    "$PY" -m venv .venv-smoke
fi
# shellcheck source=/dev/null
source .venv-smoke/bin/activate
python -m pip install -r requirements.txt -q

SU2_OUT_DIR="$(mktemp -d -t su2_smoke.XXXXXX)"
export SU2_OUT_DIR
export PYTHONPATH="$CGPT_BUILD_DIR:$(pwd)/gpt/lib:${PYTHONPATH:-}"
trap 'rm -rf "$SU2_OUT_DIR"' EXIT

echo "Executing tiny 4^4 lattice run in $SU2_OUT_DIR ..."
python gpt/applications/hmc/su2_2q_signal_scan.py \
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
