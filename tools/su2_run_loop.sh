#!/usr/bin/env bash
# su2_run_loop.sh <seed> — supervisor for the SU(2) GPU-generation runs.
#
# The measurement phase leaks C++ memory (macOS compressor filled to ~76 GB
# and memorystatus SIGKILLed the workers on Jul 5-6 2026), and every
# measurement is checkpointed. So: run the worker in a loop; the driver exits
# cleanly every SU2_MAX_MEAS_PER_RUN measurements (leak containment) and this
# loop resumes it; a SIGKILL (137) or crash also just resumes. Stop when the
# progress file reports done, or on repeated immediate startup failures.
set -u

SEED="${1:?usage: su2_run_loop.sh <seed>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${SU2_OUT_DIR:-$ROOT/results/su2_signal_scan}"
PROGRESS="$OUT_DIR/progress_${SEED}.json"
MAX_MEAS="${SU2_MAX_MEAS_PER_RUN:-5}"
LOOP_LOG="$OUT_DIR/runloop_${SEED}.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOOP_LOG"; }

fast_fail_count=0
log "supervisor start (seed=$SEED, max_meas_per_run=$MAX_MEAS)"
while :; do
  done_flag=$(python3 -c "
import json,sys
try: print(json.load(open('$PROGRESS')).get('done'))
except Exception: print('unknown')
" 2>/dev/null)
  if [ "$done_flag" = "True" ]; then
    log "run complete (done=true); supervisor exiting"
    break
  fi

  start_ts=$(date +%s)
  log "launching worker (resume)"
  SU2_EXTRA_ARGS="--gauge-gpu 1 --max-meas-per-run $MAX_MEAS ${SU2_EXTRA_EXTRA_ARGS:-}" \
    "$ROOT/tools/start_su2_worker_gpu.sh" "$SEED"
  code=$?
  dur=$(( $(date +%s) - start_ts ))
  log "worker exited code=$code after ${dur}s"

  if [ "$dur" -lt 60 ]; then
    fast_fail_count=$((fast_fail_count + 1))
    if [ "$fast_fail_count" -ge 3 ]; then
      log "3 consecutive fast failures; giving up (check config/build)"
      exit 1
    fi
  else
    fast_fail_count=0
  fi
  sleep 5
done
