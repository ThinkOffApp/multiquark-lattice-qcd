#!/usr/bin/env bash
# compressor_watchdog.sh — kill a thrashing su2 worker before it wedges.
#
# The measurement-phase leak fills the macOS memory compressor; once it passes
# ~60-70GB the worker spends all CPU on page decompression and makes ~zero
# progress (observed Jul 7 2026: a single measurement ran 2h wedged at 78GB
# compressor). This watchdog polls the compressor every INTERVAL seconds and
# SIGKILLs the worker when it crosses THRESHOLD_GB, so the supervisor
# (su2_run_loop.sh) resumes from the last checkpoint within seconds instead of
# losing hours. It also logs a compressor/RSS trace so the leak stays visible.
set -u

SEED="${1:-9101}"
THRESHOLD_GB="${SU2_COMP_THRESHOLD_GB:-62}"
INTERVAL="${SU2_COMP_INTERVAL:-60}"
LOG="/Users/petrus/multiquark-lattice-qcd/results/su2_signal_scan/compwatch_${SEED}.log"

comp_gb() {
  python3 -c "
import subprocess
o=subprocess.check_output(['vm_stat']).decode()
for l in o.splitlines():
    if 'occupied by compressor' in l:
        print(int(''.join(c for c in l if c.isdigit()))*16384/1e9); break
"
}

echo "[$(date '+%F %T')] compressor watchdog start seed=$SEED threshold=${THRESHOLD_GB}GB" | tee -a "$LOG"
while true; do
  sleep "$INTERVAL"
  c=$(comp_gb 2>/dev/null)
  [ -z "$c" ] && continue
  pid=$(pgrep -f "su2_2q_signal_scan.py --seed $SEED" | head -1)
  rss=$( [ -n "$pid" ] && ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f", $1/1048576}' || echo "-" )
  echo "[$(date '+%F %T')] comp=${c}GB rss=${rss}GB pid=${pid:-none}" >> "$LOG"
  over=$(python3 -c "print(1 if float('$c') > $THRESHOLD_GB else 0)")
  if [ "$over" = "1" ] && [ -n "$pid" ]; then
    echo "[$(date '+%F %T')] THRASH: compressor ${c}GB > ${THRESHOLD_GB}GB, killing worker $pid for clean resume" | tee -a "$LOG"
    kill -9 "$pid" 2>/dev/null
  fi
done
