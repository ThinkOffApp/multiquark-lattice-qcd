#!/usr/bin/env bash
# port_linux_cpu.sh: build and validate the SU(2) lattice pipeline on a Linux x86-64 host
# with no Metal (Bosgame M5 / Strix Halo first). One command:
#
#   tools/port_linux_cpu.sh            # build Grid (AVX512) + cgpt, then smoke-test
#   tools/port_linux_cpu.sh --smoke-only   # skip the build (also works on macOS with the existing build)
#
# The heat bath falls back to the driver's gpt CPU path (no gpu-metal-heatbath here);
# the Wilson-loop measurement is Grid on the CPU as on the MacBook.
# Nothing here starts a production seed: the smoke run is 8^4, 20 thermalisation
# sweeps, 2 measurements, into results/port_smoke, and prints sweep/measurement timings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_PREFIX="${GRID_PREFIX:-$HOME/grid-install}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
SIMD="${GRID_SIMD:-AVX512}"
SMOKE_ONLY=0
[[ "${1:-}" == "--smoke-only" ]] && SMOKE_ONLY=1

log() { printf '[port] %s\n' "$*"; }

if (( ! SMOKE_ONLY )); then
  [[ "$(uname -s)" == "Linux" ]] || { log "build stage is for Linux; on macOS use --smoke-only"; exit 2; }
  if [[ "$SIMD" == "AVX512" ]] && ! grep -q avx512f /proc/cpuinfo; then
    log "CPU has no avx512f; set GRID_SIMD=AVX2 and rerun"; exit 2
  fi
  for t in g++ make python3 python3-config; do command -v "$t" >/dev/null || { log "missing $t"; exit 3; }; done
  python3 -c "import numpy" 2>/dev/null || { log "python3 needs numpy (pip install numpy)"; exit 3; }
  command -v autoreconf >/dev/null || { log "missing autoconf/automake/libtool (apt install autoconf automake libtool)"; exit 3; }

  log "Grid: bootstrap + configure --enable-simd=$SIMD --enable-comms=none --prefix=$GRID_PREFIX (jobs=$JOBS)"
  ( cd "$ROOT/Grid" && [[ -x configure ]] || ./bootstrap.sh
    mkdir -p build && cd build
    ../configure --enable-simd="$SIMD" --enable-comms=none --prefix="$GRID_PREFIX" > configure.log 2>&1 \
      || { tail -30 configure.log; exit 4; }
    make -j"$JOBS" > make.log 2>&1 || { tail -40 make.log; exit 4; }
    make install > install.log 2>&1 )
  export PATH="$GRID_PREFIX/bin:$PATH"
  command -v grid-config >/dev/null || { log "grid-config not on PATH after install"; exit 4; }

  log "cgpt: make against $GRID_PREFIX"
  ( cd "$ROOT/gpt/lib/cgpt" && ./make "$GRID_PREFIX" > make.log 2>&1 || { tail -40 make.log; exit 5; } )
fi

CGPT_SOURCE="${SU2_CGPT_SOURCE:-$ROOT/gpt/lib/cgpt/build/source.sh}"
[[ -f "$CGPT_SOURCE" ]] || { log "no cgpt build at $CGPT_SOURCE"; exit 5; }
# shellcheck disable=SC1090
source "$CGPT_SOURCE"
PY="${SU2_PYTHON:-python3}"
"$PY" -c "import gpt, numpy; print('[port] gpt import OK, numpy', numpy.__version__)" || { log "import gpt failed"; exit 5; }

# Smoke: tiny lattice on the CPU pipeline through the real launcher.
SEED="${SMOKE_SEED:-portsmoke}"
OUT="$ROOT/results/port_smoke"
mkdir -p "$OUT"
rm -f "$OUT"/*"$SEED"* 2>/dev/null || true
export SU2_OUT_DIR="$OUT" SU2_PYTHON="$PY" SU2_ALLOW_EXTERNAL_OUT_DIR=1
export SU2_LATTICE="${SMOKE_LATTICE:-8,8,8,8}" SU2_NTHERM="${SMOKE_NTHERM:-20}" SU2_NMEAS="${SMOKE_NMEAS:-2}" SU2_RESUME=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$JOBS}"
log "smoke: L=$SU2_LATTICE ntherm=$SU2_NTHERM nmeas=$SU2_NMEAS pipeline=cpu OMP_NUM_THREADS=$OMP_NUM_THREADS"
t0=$(date +%s)
"$ROOT/tools/start_su2_worker.sh" "$SEED" cpu > "$OUT/smoke_$SEED.log" 2>&1 || { tail -40 "$OUT/smoke_$SEED.log"; exit 6; }
t1=$(date +%s)

"$PY" - "$OUT/progress_$SEED.json" "$OUT/live_$SEED.json" "$((t1 - t0))" <<'PY'
import json, sys
p = json.load(open(sys.argv[1])); live = json.load(open(sys.argv[2])); wall = int(sys.argv[3])
meta = live.get("meta", {})
plaq = p.get("last_plaquette")
sweeps = p.get("sweeps_done") or 0
meas = p.get("meas_done") or 0
print(f"[port] backend={meta.get('gauge_update_backend')} compute={meta.get('compute_backend')} L={meta.get('L')} beta={meta.get('beta')}")
print(f"[port] plaquette={plaq} sweeps_done={sweeps} meas_done={meas} wall={wall}s")
# SU(2) beta=2.4 plaquette on the MacBook 24^4 production run: 0.6295 (Sep 7 2026, seed 9101).
# A tiny 8^4 lattice with 20 sweeps sits near it but not on it; the window only catches a broken build.
ok = plaq is not None and 0.58 <= float(plaq) <= 0.68 and meas >= 1
print("[port] PASS" if ok else "[port] FAIL: plaquette outside [0.58, 0.68] or no measurement")
sys.exit(0 if ok else 7)
PY
