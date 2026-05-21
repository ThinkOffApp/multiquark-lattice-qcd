#!/usr/bin/env bash
#
# Metal Wilson readiness gate (issue #9b).
#
# Scope (explicit): this is a Metal *readiness* gate, not a Metal *parity*
# gate. It proves the build IS Metal and that a Metal Wilson Dslash actually
# dispatches on this box. It does NOT yet run a Wilson/Dslash correctness
# regression against a CPU reference. That second piece (Wilson Dslash
# CPU-single vs Metal-single parity via Grid's Test_metal_dslash_regression)
# is the follow-up that finally closes #9.
#
# Replaces the retired scripts/gpu_crosscheck.py (see issue #9 for history).
# Produces a structured JSON receipt the dashboard / CI can rely on.
#
# What this gate REQUIRES (every item must hold; missing or weak signal -> fail):
#
#   1. A cgpt .so is on PYTHONPATH and importable.
#   2. Grid configure summary reports EXACTLY `Acceleration: metal`. Other
#      enumerated accelerators (cuda/sycl/hip) and any non-enumerated value
#      ("none"/"cpu"/""/unknown) FAIL this gate. This is a Metal gate.
#   3. The patched detect_runtime_backend() (PR #10) reports backend=gpu.
#   4. GRID_BUILD_DIR is set and $GRID_BUILD_DIR/benchmarks/Benchmark_wilson
#      exists and is executable.
#   5. Benchmark_wilson runs to completion (from its build dir so
#      default.metallib resolves) and its output includes
#      `AcceleratorMetalInit: Selected device is ...`. Absence of that
#      line FAILS the gate.
#   6. Benchmark_wilson emits a parseable, strictly positive mflop/s line.
#
# Receipt is written to $SU2_GATE_RECEIPT (default: /tmp/metal_readiness_gate.json)
# and includes: loaded_cgpt_so, python_version, grid_summary_path,
# grid_acceleration, grid_simd, backend, accelerator_total_bytes, mflop_s,
# metal_init_line, status, reason.
#
# Exit codes:
#   0  all required checks passed (receipt complete and positive)
#   2  prerequisites missing (cgpt not importable, no Grid summary,
#      GRID_BUILD_DIR missing, Benchmark_wilson binary missing, etc.)
#   3  signal mismatch (grid_acceleration != "metal", or backend != "gpu")
#   4  Benchmark_wilson failed, did not print AcceleratorMetalInit, or
#      did not emit a strictly positive mflop/s value
#
# Required env:
#   GRID_CGPT_BUILD     : path to a dir containing cgpt.cpython-*-darwin.so
#                         built against a Metal-enabled Grid.
#                         Alternatively, pre-set PYTHONPATH so `import cgpt`
#                         already works.
#   GRID_BUILD_DIR      : path to the Grid build dir containing
#                         benchmarks/Benchmark_wilson and default.metallib.
#                         No fallback; required.
#
# Optional env:
#   GRID_CONFIG_SUMMARY : path to grid.configure.summary. If unset, the
#                         driver auto-discovers <repo>/Grid/build/grid.configure.summary.
#   SU2_GATE_RECEIPT    : output JSON path (default /tmp/metal_readiness_gate.json).
#   PYTHON              : python interpreter matching the cgpt ABI tag
#                         (default: auto-detect from cgpt.cpython-X.Y-*.so).

set -euo pipefail

RECEIPT_PATH="${SU2_GATE_RECEIPT:-/tmp/metal_readiness_gate.json}"
# Hard-coded to "metal" since this is the Metal gate. CUDA/SYCL/HIP builds
# need their own readiness gates (#9a's detector already supports them).
REQUIRED_ACCELERATION="metal"

log() { printf '[metal-readiness-gate] %s\n' "$*" >&2; }
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
    "metal_init_line": os.environ.get("RECEIPT_METAL_INIT", ""),
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
TMPOUT="$(mktemp -t metal_readiness_probe.XXXXXX.json)"
trap 'rm -f "$TMPOUT"' EXIT
DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/gpt/applications/hmc"
if [ ! -f "$DRIVER_DIR/su2_2q_signal_scan.py" ]; then
    fail 2 "driver not found at $DRIVER_DIR/su2_2q_signal_scan.py"
fi

# The probe lives in a heredoc so we can capture both stderr (Grid banner
# + AcceleratorMetalInit) and the structured JSON written to TMPOUT.
"$PY" - "$DRIVER_DIR" "$TMPOUT" <<'PY' 2>&1 | tee /tmp/metal_readiness_gate.probe.log
import json, os, sys
driver_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, driver_dir)
from su2_2q_signal_scan import detect_runtime_backend
rb = detect_runtime_backend()
with open(out_path, "w") as f:
    json.dump(rb, f, indent=2, sort_keys=True)
print("PROBE_OK")
PY

if ! grep -q PROBE_OK /tmp/metal_readiness_gate.probe.log; then
    fail 2 "detect_runtime_backend probe failed; see /tmp/metal_readiness_gate.probe.log"
fi
METAL_INIT_LINE="$(grep -E 'AcceleratorMetalInit' /tmp/metal_readiness_gate.probe.log | head -n1 || true)"

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
if [ "$ACCEL" != "$REQUIRED_ACCELERATION" ]; then
    fail 3 "grid_acceleration='$ACCEL'; this is the Metal gate, requires exactly 'metal'"
fi
if [ "$BACKEND" != "gpu" ]; then
    fail 3 "detect_runtime_backend reports backend='$BACKEND'; expected 'gpu'"
fi

# 4. GRID_BUILD_DIR + Benchmark_wilson are REQUIRED (no longer optional).
if [ -z "${GRID_BUILD_DIR:-}" ]; then
    fail 2 "GRID_BUILD_DIR is not set; the gate needs Benchmark_wilson + default.metallib"
fi
BENCH_BIN="$GRID_BUILD_DIR/benchmarks/Benchmark_wilson"
if [ ! -x "$BENCH_BIN" ]; then
    fail 2 "Benchmark_wilson not found / not executable at $BENCH_BIN"
fi
log "running $BENCH_BIN"
BENCH_LOG="${TMPDIR:-/tmp}/metal_readiness_gate.bench.log"
# Benchmark_wilson loads default.metallib from its build directory at
# runtime; running it from elsewhere fails with MTLLibraryErrorDomain
# Code=6 "library not found". Run from the build dir so the relative
# lookup resolves.
( cd "$GRID_BUILD_DIR" && ./benchmarks/Benchmark_wilson --grid 8.8.8.8 ) > "$BENCH_LOG" 2>&1
BENCH_RC=$?
log "Benchmark_wilson exit code: $BENCH_RC"
if [ "$BENCH_RC" -ne 0 ]; then
    fail 4 "Benchmark_wilson exited non-zero ($BENCH_RC); see $BENCH_LOG"
fi
set +e +o pipefail
MFLOPS="$(grep -m1 'mflop/s' "$BENCH_LOG" | sed -nE 's/.*mflop\/s[[:space:]]*=[[:space:]]*([0-9.]+).*/\1/p')"
METAL_INIT_BENCH="$(grep -m1 'AcceleratorMetalInit: Selected device' "$BENCH_LOG" || true)"
set -e -o pipefail

# 5. AcceleratorMetalInit must be observed. Without it, Metal didn't init
# even though the Grid build claims metal acceleration. That's exactly the
# fall-through case @ether asked us to gate on.
if [ -z "$METAL_INIT_BENCH" ]; then
    export RECEIPT_METAL_INIT=""
    export RECEIPT_MFLOPS="${MFLOPS:-}"
    fail 4 "AcceleratorMetalInit line not observed in Benchmark_wilson output ($BENCH_LOG)"
fi
log "metal init (from Benchmark_wilson): $METAL_INIT_BENCH"
export RECEIPT_METAL_INIT="$METAL_INIT_BENCH"

# 6. mflop/s must be parseable AND strictly positive.
if [ -z "$MFLOPS" ]; then
    export RECEIPT_MFLOPS=""
    fail 4 "Benchmark_wilson did not emit a parseable mflop/s line"
fi
# Strictly positive numeric check (awk handles the float compare portably).
if ! awk -v m="$MFLOPS" 'BEGIN { exit (m+0 > 0) ? 0 : 1 }'; then
    export RECEIPT_MFLOPS="$MFLOPS"
    fail 4 "Benchmark_wilson mflop/s='$MFLOPS' is not strictly positive"
fi
log "Benchmark_wilson mflop/s: $MFLOPS"
export RECEIPT_MFLOPS="$MFLOPS"

emit_receipt "pass" ""
log "PASS"
exit 0
