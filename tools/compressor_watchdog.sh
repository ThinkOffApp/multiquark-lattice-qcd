#!/usr/bin/env bash
# compressor_watchdog.sh — kill a THRASHING su2 worker, not a merely-heavy one.
#
# The measurement leak fills the macOS compressor; once a worker is thrashing
# (all CPU on page decompression) it makes ~zero progress and must be killed so
# the supervisor resumes from checkpoint. BUT a healthy heavy measurement can
# also touch high memory transiently while still advancing. Killing on
# compressor alone (v1) created a kill-loop: it killed measurements that were
# progressing fine at low memory during a system-wide compressor spike.
#
# v2 discriminator = THRASH is high compressor AND frozen progress. We kill the
# worker only when compressor > THRESHOLD_GB AND meas_cfg_substep_done has NOT
# advanced for STALL_SEC. A progressing measurement is never killed regardless
# of memory; a true wedge (frozen substep + ballooning compressor) dies fast.
set -u

SEED="${1:-9101}"
THRESHOLD_GB="${SU2_COMP_THRESHOLD_GB:-55}"
STALL_SEC="${SU2_STALL_SEC:-120}"
INTERVAL="${SU2_COMP_INTERVAL:-30}"
OUT="/Users/petrus/multiquark-lattice-qcd/results/su2_signal_scan"
PROG="$OUT/progress_${SEED}.json"
LOG="$OUT/compwatch_${SEED}.log"

comp_gb(){ python3 -c "
import subprocess
o=subprocess.check_output(['vm_stat']).decode()
for l in o.splitlines():
    if 'occupied by compressor' in l:
        print(int(''.join(c for c in l if c.isdigit()))*16384/1e9); break
"; }
substep(){ python3 -c "
import json
try: print(json.load(open('$PROG')).get('meas_cfg_substep_done') or -1)
except Exception: print(-1)
" 2>/dev/null; }

echo "[$(date '+%F %T')] watchdog v2 start seed=$SEED thresh=${THRESHOLD_GB}GB stall=${STALL_SEC}s" | tee -a "$LOG"
last_sub=$(substep); last_advance=$(date +%s)
while true; do
  sleep "$INTERVAL"
  c=$(comp_gb 2>/dev/null); [ -z "$c" ] && continue
  sub=$(substep); now=$(date +%s)
  [ "$sub" != "$last_sub" ] && { last_sub=$sub; last_advance=$now; }
  frozen=$(( now - last_advance ))
  pid=$(pgrep -f "su2_2q_signal_scan.py --seed $SEED" | head -1)
  rss=$( [ -n "$pid" ] && ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.1f",$1/1048576}' || echo "-" )
  echo "[$(date '+%F %T')] comp=${c}GB rss=${rss}GB sub=${sub} frozen=${frozen}s pid=${pid:-none}" >> "$LOG"
  over=$(python3 -c "print(1 if float('$c')>$THRESHOLD_GB else 0)")
  if [ "$over" = "1" ] && [ "$frozen" -ge "$STALL_SEC" ] && [ -n "$pid" ]; then
    echo "[$(date '+%F %T')] THRASH: compressor ${c}GB AND substep frozen ${frozen}s -> killing $pid for clean resume" | tee -a "$LOG"
    kill -9 "$pid" 2>/dev/null; last_advance=$(date +%s)
  fi
done
