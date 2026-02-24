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

ROOT="${SU2_ROOT:-/Users/petrus/AndroidStudioProjects/ThinkOff}"
GPT_DIR="${SU2_GPT_DIR:-$ROOT/grid-gpt-public/gpt}"
OUT_DIR="${SU2_OUT_DIR:-$ROOT/results/su2_signal_scan}"
CGPT_SOURCE="${SU2_CGPT_SOURCE:-$GPT_DIR/lib/cgpt/build/source.sh}"
LOG_FILE="${SU2_LOG_FILE:-$OUT_DIR/log_${SEED}.txt}"

mkdir -p "$OUT_DIR"
cd "$GPT_DIR"

if [[ ! -f "$CGPT_SOURCE" ]]; then
  echo "missing cgpt source file: $CGPT_SOURCE"
  exit 3
fi
source "$CGPT_SOURCE"

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
  probe_out="$(python3 - <<'PY'
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
  python3 applications/hmc/su2_2q_signal_scan.py
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
