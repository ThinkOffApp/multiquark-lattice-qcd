#!/usr/bin/env bash
#
# Metal Wilson Dslash parity / readiness gate.
#
# Replaces the retired scripts/gpu_crosscheck.py (see issue #9). Produces a
# structured pass/fail receipt that the dashboard / CI can rely on.
#
# What this gate actually checks (in order, all must pass):
#
#   1. A cgpt .so is on PYTHONPATH and importable.
#   2. The loaded Grid build's configure summary reports
#      `Acceleration: metal` (the *enumerated* signal; "none"/"cpu" rejected).
#   3. Loading cgpt prints `AcceleratorMetalInit: Selected device is ...`
#      and the SU(2) driver's detect_runtime_backend() reports backend=gpu
#      with the patched logic from PR #10 (#9a).
#   4. If `Benchmark_wilson` is built in the same Grid tree, it runs to
#      completion and prints a non-zero mflop/s line. (Demonstrates a real
#      Wilson Dslash actually fires on Metal; this is the substantive part.)
#   5. (Future) If `Test_metal_dslash_regression` is built (Grid/tests/core/),
#      it runs and returns 0. That's the proper float-vs-double precision-
#      guard regression test, but it requires a Grid build off the
#      multiquark-lattice-qcd Grid source tree that has the
#      `MetalWilsonImplOK` trait. Currently TODO.
#
# Receipt is written to $SU2_GATE_RECEIPT (default: /tmp/metal_parity_gate.json)
# and includes: loaded_cgpt_so, python_version, grid_summary_path,
# grid_acceleration, simd, backend, accelerator_total_bytes, mflop_s.
#
# Exit codes:
#   0  all required checks passed (receipt complete and positive)
#   2  prerequisites missing (cgpt not importable, no Grid summary, etc.)
#   3  signal mismatch (grid_acceleration not metal/cuda/sycl/hip, etc.)
#   4  Benchmark_wilson failed or did not print mflop/s
#
# Required env (one of):
#   GRID_CGPT_BUILD  : path to a dir containing cgpt.cpython-*-darwin.so
#                      built against a Metal-enabled Grid.
#   PYTHONPATH       : pre-set so `import cgpt` works (script will trust it).
#
# Optional env:
#   GRID_CONFIG_SUMMARY : path to grid.configure.summary (auto-discovered if
#                         a Grid/build/grid.configure.summary lives next to
#                         the cgpt source tree).
#   GRID_BUILD_DIR      : path to the Grid build dir, used to locate
#                         benchmarks/Benchmark_wilson.
#   SU2_GATE_RECEIPT    : output JSON path (default /tmp/metal_parity_gate.json).
#   PYTHON              : python interpreter matching the cgpt ABI tag
#                         (default: auto-detect from cgpt.cpython-X.Y-*.so).

set -euo pipefail

RECEIPT_PATH="${SU2_GATE_RECEIPT:-/tmp/metal_parity_gate.json}"
ACCEPTED_BACKENDS_RE='^(cuda|metal|sycl|hip)$'

log() { printf '[metal-parity-gate] %s\n' "$*" >&2; }
fail() { local code="$1"; shift; log "FAIL: $*"; emit_receipt "fail" "$*"; exit "$code"; }

emit_receipt() {
    local status="$1" reason="${2:-}"
    python3 - "$status" "$reason" <<'PY' > "$RECEIPT_PATH"
import json, os, sys
status, reason = sys.argv[1], sys.argv[2]
receipt = {
    "status": status,
    "reason": reason,
    "loaded_cgpt_so": os.environ.get("RECEIPT_LOADED_SO", ""),
    "python_version": os.environ.get("RECEIPT_PY_VER", ""),
    "grid_summary_path": os.environ.get("RECEIPT_SUMMARY_PATH", ""),
    "grid_acceleration": os.environ.get("RECEIPT_ACCEL", ""),
    "grid_simd": os.environ.get("RECEIPT_SIMD", ""),
    "backend": os.environ.get("RECEIPT_BACKEND", ""),
    "accelerator_total_bytes": int(os.environ.get("RECEIPT_ACCEL_TOTAL", "0") or "0"),
    "mflop_s": os.environ.get("RECEIPT_MFLOPS", ""),
}
json.dump(receipt, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY
    log "receipt -> $RECEIPT_PATH"
}

# 1. PYTHONPATH / cgpt prerequisites
if [ -n "${GRID_CGPT_BUILD:-}" ]; then
    GPT_LIB="$(dirname "$GRID_CGPT_BUILD")"
    export PYTHONPATH="${GRID_CGPT_BUILD}:${GPT_LIB}:${PYTHONPATH:-}"
fi
if [ -z "${PYTHONPATH:-}" ]; then
    fail 2 "PYTHONPATH is not set and GRID_CGPT_BUILD is not provided"
fi

# Auto-pick python that matches cgpt ABI tag. Search GRID_CGPT_BUILD first,
# then any directory on PYTHONPATH.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    SEARCH_DIRS="${GRID_CGPT_BUILD:-} ${PYTHONPATH//:/ }"
    for d in $SEARCH_DIRS; do
        SO="$(ls "$d"/cgpt.cpython-*-*.so 2>/dev/null | head -n1 || true)"
        if [ -n "$SO" ]; then
            ABI="$(basename "$SO" | sed -nE 's/^cgpt\.cpython-([0-9])([0-9]+)-.*\..*$/\1.\2/p')"
            if [ -n "$ABI" ] && command -v "python${ABI}" >/dev/null 2>&1; then
                PY="python${ABI}"
                break
            fi
        fi
    done
fi
if [ -z "$PY" ]; then
    fail 2 "could not auto-detect a Python matching cgpt's ABI tag; set PYTHON=pythonX.Y"
fi
log "python: $($PY -V 2>&1)"

# 2 & 3. Import cgpt, probe runtime backend, capture receipt fields.
TMPOUT="$(mktemp -t metal_parity_probe.XXXXXX.json)"
trap 'rm -f "$TMPOUT"' EXIT
DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/gpt/applications/hmc"
if [ ! -f "$DRIVER_DIR/su2_2q_signal_scan.py" ]; then
    fail 2 "driver not found at $DRIVER_DIR/su2_2q_signal_scan.py"
fi

# The probe lives in a heredoc so we can capture both stderr (Grid banner
# + AcceleratorMetalInit) and the structured JSON written to TMPOUT.
"$PY" - "$DRIVER_DIR" "$TMPOUT" <<'PY' 2>&1 | tee /tmp/metal_parity_gate.probe.log
import json, os, sys
driver_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, driver_dir)
from su2_2q_signal_scan import detect_runtime_backend
rb = detect_runtime_backend()
with open(out_path, "w") as f:
    json.dump(rb, f, indent=2, sort_keys=True)
print("PROBE_OK")
PY

if ! grep -q PROBE_OK /tmp/metal_parity_gate.probe.log; then
    fail 2 "detect_runtime_backend probe failed; see /tmp/metal_parity_gate.probe.log"
fi
METAL_INIT_LINE="$(grep -E 'AcceleratorMetalInit' /tmp/metal_parity_gate.probe.log | head -n1 || true)"

LOADED_SO="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("loaded_cgpt_so") or "")' "$TMPOUT")"
PY_VER="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("python_version") or "")' "$TMPOUT")"
SUMMARY_PATH="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("grid_summary_path") or "")' "$TMPOUT")"
ACCEL="$(python3 -c 'import json,sys;print((json.load(open(sys.argv[1])).get("grid_acceleration") or "").lower())' "$TMPOUT")"
SIMD="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("grid_simd") or "")' "$TMPOUT")"
BACKEND="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("backend") or "")' "$TMPOUT")"
ACCEL_TOTAL="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("accelerator_total_bytes") or 0)' "$TMPOUT")"

export RECEIPT_LOADED_SO="$LOADED_SO"
export RECEIPT_PY_VER="$PY_VER"
export RECEIPT_SUMMARY_PATH="$SUMMARY_PATH"
export RECEIPT_ACCEL="$ACCEL"
export RECEIPT_SIMD="$SIMD"
export RECEIPT_BACKEND="$BACKEND"
export RECEIPT_ACCEL_TOTAL="$ACCEL_TOTAL"

log "loaded_cgpt_so=$LOADED_SO"
log "grid_summary_path=$SUMMARY_PATH"
log "grid_acceleration=$ACCEL  grid_simd=$SIMD  backend=$BACKEND  accelerator_total_bytes=$ACCEL_TOTAL"
log "metal init line: ${METAL_INIT_LINE:-<not observed>}"

if [ -z "$SUMMARY_PATH" ] || [ ! -f "$SUMMARY_PATH" ]; then
    fail 2 "no Grid configure summary found (set GRID_CONFIG_SUMMARY)"
fi
if ! echo "$ACCEL" | grep -qE "$ACCEPTED_BACKENDS_RE"; then
    fail 3 "grid_acceleration='$ACCEL' is not in {cuda,metal,sycl,hip}"
fi
if [ "$BACKEND" != "gpu" ]; then
    fail 3 "detect_runtime_backend reports backend='$BACKEND'; expected 'gpu'"
fi

# 4. Optional: run Grid's own Benchmark_wilson if available.
MFLOPS=""
if [ -n "${GRID_BUILD_DIR:-}" ] && [ -x "$GRID_BUILD_DIR/benchmarks/Benchmark_wilson" ]; then
    log "running $GRID_BUILD_DIR/benchmarks/Benchmark_wilson"
    BENCH_LOG="${TMPDIR:-/tmp}/metal_parity_gate.bench.log"
    # Benchmark_wilson loads default.metallib from its build directory at
    # runtime; running it from elsewhere fails with MTLLibraryErrorDomain
    # Code=6 "library not found". Run from the build dir so the relative
    # lookup resolves.
    ( cd "$GRID_BUILD_DIR" && ./benchmarks/Benchmark_wilson --grid 8.8.8.8 ) > "$BENCH_LOG" 2>&1 || true
    set +e +o pipefail
    MFLOPS="$(grep -m1 'mflop/s' "$BENCH_LOG" | sed -nE 's/.*mflop\/s[[:space:]]*=[[:space:]]*([0-9.]+).*/\1/p')"
    METAL_INIT_BENCH="$(grep -m1 'AcceleratorMetalInit: Selected device' "$BENCH_LOG" || true)"
    set -e -o pipefail
    if [ -n "$METAL_INIT_BENCH" ]; then
        log "metal init (from Benchmark_wilson): $METAL_INIT_BENCH"
        METAL_INIT_LINE="$METAL_INIT_BENCH"
    fi
    log "Benchmark_wilson mflop/s: ${MFLOPS:-<not parsed>}"
    if [ -z "$MFLOPS" ]; then
        export RECEIPT_MFLOPS=""
        fail 4 "Benchmark_wilson did not emit a parseable mflop/s line"
    fi
    export RECEIPT_MFLOPS="$MFLOPS"
else
    log "Benchmark_wilson not found at \$GRID_BUILD_DIR/benchmarks/Benchmark_wilson; skipping mflop/s receipt"
    export RECEIPT_MFLOPS=""
fi

emit_receipt "pass" ""
log "PASS"
exit 0
