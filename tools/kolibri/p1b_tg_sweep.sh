#!/usr/bin/env bash
# tools/kolibri/p1b_tg_sweep.sh - task 7 (P1b) measurement: the CPU expert pool's tg/pp by worker count, on the
# ExpertPool row-split path, pinned to the GPU's socket.  This is the fixed form of p1b_pool_gate.sh's
# sections 1-2.
#
# Why a second file: p1b_pool_gate.sh's bench legs were silent no-ops.  `timeout 2400 pin ...` cannot exec a
# bash FUNCTION (timeout execs an external command), so every run died with "timeout: failed to run command
# 'pin'"; the message went to the stderr log, and the pipeline's exit status was sed's (0), so the script
# reported success.  Fixed here: timeout runs INSIDE the pin function, and pipefail makes the real status
# visible.  The old script is left untouched because bash re-reads a script as it executes - editing a running
# script corrupts the run (a rule this project already learned the hard way).
#
# Gates (plan task 7): pool standalone tg >= 25-30 tok/s.  Reference: pool_bench measured 6 experts/layer at
# 24 workers pinned to node 0 = 47.7 GB/s = 71.6 tok/s pool-only; 48 workers on one socket collapse to
# 4.5 GB/s, so the 48 row is expected to be the collapse, not a tuning miss.
#
# Usage:  tools/kolibri/p1b_tg_sweep.sh [OUTDIR]        # default /tmp/kolibri_p1b
#   NUMA=0|1|none   placement (default 0 = the T4's socket)
#   SWEEP="6 12 24 48"
#   WAIT_GPU=1      wait for the T4 to be free before starting (default 1)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build/strata-kolibri"
MODEL="${KOLIBRI_MODEL:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
IDS512="${KOLIBRI_IDS512:-/tmp/bench_ids512.txt}"
IDS32K="${KOLIBRI_IDS32K:-/tmp/bench_ids32k.txt}"
OUT="${1:-/tmp/kolibri_p1b}"; mkdir -p "$OUT"
NUMA="${NUMA:-0}"
SWEEP="${SWEEP:-6 12 24 48}"
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1
PY=/data2/develop/Strata/.venv/bin/python
[ -x "$PY" ] || PY=python3

for p in "$BIN" "$MODEL" "$IDS512"; do [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }; done

# timeout INSIDE the function: `timeout N <function>` cannot work, timeout execs a program.
pin() { local t="$1"; shift
        if [ "$NUMA" = none ]; then timeout "$t" "$@"
        else timeout "$t" numactl --cpunodebind="$NUMA" --membind="$NUMA" "$@"; fi; }

# Require the T4 to be empty for 6 consecutive samples (30 s) before starting.  ONE empty sample is not
# enough: bench.py leaves a gap between its pp and tg subprocesses, and this sweep started in exactly that
# gap (measured: it ran concurrently with the KL leg it was waiting behind - two contexts on the T4 - which
# invalidates every timing number it produces).
if [ "${WAIT_GPU:-1}" = 1 ]; then
  clear=0
  for i in $(seq 1 1440); do
    if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; then
      clear=$((clear + 1)); [ $clear -ge 6 ] && { echo "GPU idle 30s; starting after $((i*5))s"; break; }
    else
      clear=0
    fi
    sleep 5
  done
fi

echo "== P1b tg sweep  ($(date -Is))  NUMA=$NUMA  sweep='$SWEEP'"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  gpu pid: /'

run_bench() {
  local label="$1"; shift
  echo "--- $label"
  local rc=0
  pin 2400 "$PY" -u "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$MODEL" \
      --ids512 "$IDS512" --ids32k "$IDS32K" --case short --extra="$*" 2>>"$OUT/counters.log" | sed 's/^/  /' || rc=$?
  [ $rc -ne 0 ] && echo "  (FAILED rc=$rc: $*; see $OUT/counters.log)"
  return 0
}

for T in $SWEEP; do run_bench "pool --cpu-threads $T" "--cpu-threads $T --counters"; done
run_bench "slots 4G + pool 24" "--expert-vram 4 --cpu-threads 24 --counters"
run_bench "slots 4G (no pool)" "--expert-vram 4 --counters"
run_bench "plain (no pool, no slots)" "--counters"

echo; echo "== counters"
grep '^CNT' "$OUT/counters.log" 2>/dev/null | sed 's/^/  /' || echo "  (no CNT lines)"
echo "== done: $OUT"
