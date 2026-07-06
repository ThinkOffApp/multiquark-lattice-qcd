#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <seed> [pipeline:auto|cpu|gpu]"
  exit 2
fi

SEED="$1"
PIPELINE_RAW="${2:-${SU2_WORKER_PIPELINE:-auto}}"
PIPELINE="$(echo "$PIPELINE_RAW" | tr '[:upper:]' '[:lower:]')"
if [[ "$PIPELINE" != "auto" && "$PIPELINE" != "cpu" && "$PIPELINE" != "gpu" ]]; then
  echo "invalid pipeline '$PIPELINE_RAW' (expected auto|cpu|gpu)"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ROOT="${SU2_ROOT:-$DEFAULT_ROOT}"
GPT_DIR="${SU2_GPT_DIR:-$ROOT/gpt}"
# Track whether the operator explicitly set SU2_OUT_DIR before we apply the default.
if [[ -n "${SU2_OUT_DIR-}" ]]; then
  OUT_DIR_EXPLICIT=1
else
  OUT_DIR_EXPLICIT=0
fi
OUT_DIR="${SU2_OUT_DIR:-$ROOT/results/su2_signal_scan}"
CGPT_SOURCE="${SU2_CGPT_SOURCE:-$GPT_DIR/lib/cgpt/build/source.sh}"
LOG_FILE="${SU2_LOG_FILE:-$OUT_DIR/log_${SEED}.txt}"

# OUT_DIR safety guardrail:
#   - If the operator explicitly set SU2_OUT_DIR, allow it but print a banner so
#     external paths are visible in the worker log.
#   - Otherwise (default OUT_DIR), refuse to start when the resolved path is a
#     symlink or escapes the repo root, unless SU2_ALLOW_EXTERNAL_OUT_DIR=1.
out_parent="$(dirname "$OUT_DIR")"
mkdir -p "$out_parent"
out_parent_real="$(cd "$out_parent" && pwd -P)"
out_dir_real="$out_parent_real/$(basename "$OUT_DIR")"
root_real="$(cd "$ROOT" && pwd -P)"

if [[ "$OUT_DIR_EXPLICIT" == "1" ]]; then
  echo "[start_su2_worker] external OUT_DIR: $OUT_DIR"
else
  is_symlink=0
  if [[ -L "$OUT_DIR" ]]; then
    is_symlink=1
  fi
  # Also flag any ancestor symlink (e.g. results -> /Volumes/...).
  probe="$OUT_DIR"
  while [[ "$probe" != "/" && "$probe" != "." ]]; do
    if [[ -L "$probe" ]]; then
      is_symlink=1
      break
    fi
    probe="$(dirname "$probe")"
  done
  outside_repo=0
  case "$out_dir_real/" in
    "$root_real"/*) ;;
    *) outside_repo=1 ;;
  esac
  if (( is_symlink == 1 )) || (( outside_repo == 1 )); then
    if [[ "${SU2_ALLOW_EXTERNAL_OUT_DIR:-0}" != "1" ]]; then
      echo "[start_su2_worker] refusing to start: default OUT_DIR ($OUT_DIR)"
      if (( is_symlink == 1 )); then
        echo "  reason: path or an ancestor is a symlink"
      fi
      if (( outside_repo == 1 )); then
        echo "  reason: resolves outside repo root ($root_real -> $out_dir_real)"
      fi
      echo "  set SU2_OUT_DIR explicitly, or export SU2_ALLOW_EXTERNAL_OUT_DIR=1 to bypass."
      exit 5
    else
      echo "[start_su2_worker] external OUT_DIR allowed via SU2_ALLOW_EXTERNAL_OUT_DIR=1: $OUT_DIR"
    fi
  fi
fi

mkdir -p "$OUT_DIR"
cd "$GPT_DIR"

if [[ ! -f "$CGPT_SOURCE" ]]; then
  echo "missing cgpt source file: $CGPT_SOURCE"
  exit 3
fi
source "$CGPT_SOURCE"

# cgpt is built against the cpython-312 ABI; a bare python3 (e.g. the system
# 3.9 after a reboot resets PATH) imports a stale/incompatible cgpt.
PYTHON_BIN="${SU2_PYTHON:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python interpreter '$PYTHON_BIN' not found (set SU2_PYTHON)"
  exit 6
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

REQUIRE_ACCEL=0
if [[ "$PIPELINE" == "gpu" ]]; then
  REQUIRE_ACCEL=1
fi

if [[ "$PIPELINE" == "gpu" ]]; then
  probe_out="$("$PYTHON_BIN" - <<'PY'
import gpt as g
try:
    info = g.mem_info() or {}
except Exception:
    info = {}
total = int(float(info.get("accelerator_total") or 0.0))
print(f"__ACCEL_TOTAL__={total}")
PY
)"
  accel_total="$(printf '%s\n' "$probe_out" | sed -n 's/^__ACCEL_TOTAL__=//p' | tail -n 1)"
  if [[ ! "$accel_total" =~ ^[0-9]+$ ]] || (( accel_total <= 0 )); then
    echo "gpu pipeline requested but accelerator_total == 0 in current Grid/GPT build"
    exit 4
  fi
fi

R_VALUES="${SU2_R_VALUES:-1,2,3,4,6,8,12}"
T_VALUES="${SU2_T_VALUES:-1,2,3,4,5,6}"

cmd=(
  "$PYTHON_BIN" applications/hmc/su2_2q_signal_scan.py
  --seed "$SEED"
  --out "$OUT_DIR"
  --L "${SU2_LATTICE:-24,24,24,24}"
  --beta "${SU2_BETA:-2.4}"
  --ntherm "${SU2_NTHERM:-200}"
  --nmeas "${SU2_NMEAS:-200}"
  --nskip "${SU2_NSKIP:-5}"
  --R "$R_VALUES"
  --T "$T_VALUES"
  --flux-r "${SU2_FLUX_R:-6}"
  --flux-t "${SU2_FLUX_T:-4}"
  --flux-rperp-max "${SU2_FLUX_RPERP_MAX:-6}"
  --precision "${SU2_PRECISION:-double}"
  --resume "${SU2_RESUME:-1}"
  --resume-force "${SU2_RESUME_FORCE:-1}"
  --progress-every "${SU2_PROGRESS_EVERY:-1}"
  --checkpoint-every "${SU2_CHECKPOINT_EVERY:-1}"
  --multilevel-blocks "${SU2_ML_BLOCKS:-8}"
  --multilevel-sweeps "${SU2_ML_SWEEPS:-4}"
  --multihit-samples "${SU2_MH_SAMPLES:-2}"
  --multihit-temporal-sweeps "${SU2_MH_TEMP_SWEEPS:-1}"
  --pipeline-label "$PIPELINE"
  --require-accelerator "$REQUIRE_ACCEL"
)

if [[ -n "${SU2_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${SU2_EXTRA_ARGS} )
  cmd+=("${extra[@]}")
fi

"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
