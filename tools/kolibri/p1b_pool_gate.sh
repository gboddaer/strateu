#!/usr/bin/env bash
# tools/kolibri/p1b_pool_gate.sh - task 7 (P1b) of the speed plan: the CPU expert pool on the library's
# ExpertPool (every expert's rows split across ALL workers) replacing the engine's own shard-per-expert pool.
#
# Gates from docs/superpowers/plans/2026-10-17-kolibri1-speed.md (task 7):
#   * pool standalone tg >= 25-30 tok/s
#   * pool path KL < 0.02 vs llama.cpp
#   * the 48-worker collapse reproduced (pool_bench measured 4.5 GB/s on one socket vs 47.7 at 24)
#   * counters: expert PCIe bytes/token ~0 on the pool path (the pool reads the pinned table in place)
#
# The reference for every number is pool_bench (src/kernels/cpu/pool_bench.cpp): cold blobs, 6 experts/layer,
# 24 workers pinned to the GPU's socket = 47.7 GB/s = 71.6 tok/s pool-only.  The engine's token also pays
# attention + dense on the GPU and the per-layer D2H/H2D (0.79 ms/token), so the engine's tg sits under that.
#
# Usage:  tools/kolibri/p1b_pool_gate.sh [OUTDIR]
#   NUMA=0|1|none   placement for the whole run (default 0 = the T4's socket; see pool_bench's table)
#   SWEEP="6 12 24 48"  thread counts to sweep
#   SKIP_KL=1       skip the KL leg (it re-runs 6 prompts x ~65 steps; ~25 min on this box)
set -u
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

[ -x "$BIN" ] || { echo "FATAL: missing binary $BIN"; exit 1; }
[ -s "$MODEL" ] || { echo "FATAL: missing model $MODEL"; exit 1; }
[ -s "$IDS512" ] || { echo "FATAL: missing ids512 $IDS512 (tools/kolibri/bench.py builds it)"; exit 1; }

# Pin the whole process tree (bench.py -> strata-kolibri) to one socket.  pool_bench measured placement as
# worth more than thread count: node-0 24 workers 71.6 tok/s vs interleave-all 48 workers 48.0.
pin() { if [ "$NUMA" = none ]; then "$@"; else numactl --cpunodebind="$NUMA" --membind="$NUMA" "$@"; fi; }

echo "== P1b pool gate  ($(date -Is))  NUMA=$NUMA  sweep='$SWEEP'  bin=$BIN"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  gpu pid: /'

echo; echo "== 1. pool standalone tg/pp by worker count (case short: pp512 + tg128)"
for T in $SWEEP; do
  echo "--- pool --cpu-threads $T"
  local_rc=0
  timeout 2400 pin "$PY" -u "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$MODEL" \
      --ids512 "$IDS512" --ids32k "$IDS32K" --case short --extra="--cpu-threads $T --counters" \
      2>>"$OUT/counters.log" | sed 's/^/  /' || local_rc=$?
  [ $local_rc -ne 0 ] && echo "  (FAILED rc=$local_rc: --cpu-threads $T)"
done

echo; echo "== 2. slots + pool (the P2 preview: slots cover the hot experts, the pool takes the misses)"
for ARGS in "--expert-vram 4 --cpu-threads 24 --counters" "--expert-vram 4 --counters"; do
  echo "--- $ARGS"
  rc=0
  timeout 2400 pin "$PY" -u "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$MODEL" \
      --ids512 "$IDS512" --ids32k "$IDS32K" --case short --extra="$ARGS" \
      2>>"$OUT/counters.log" | sed 's/^/  /' || rc=$?
  [ $rc -ne 0 ] && echo "  (FAILED rc=$rc: $ARGS)"
done

echo; echo "== 3. counters from the runs above (expert PCIe bytes/token must be ~0 on the pool path)"
grep '^CNT' "$OUT/counters.log" 2>/dev/null | sed 's/^/  /' || echo "  (no CNT lines)"

echo; echo "== 4. KL gate on the pool path (mean KL < 0.02 vs llama.cpp, p00-p05)"
if [ "${SKIP_KL:-0}" = 1 ]; then
  echo "  skipped (SKIP_KL=1)"
else
  PARITY="${KL_PARITY:-/tmp/kl_s3_parity}"
  CACHE="${KL_CACHE:-/tmp/kl_s3_cache}"
  [ -d "$PARITY" ] || { echo "  FATAL: parity dir $PARITY missing (run tools/kolibri/parity.py first)"; exit 1; }
  LOCK=""; [ "$NUMA" != none ] && LOCK="numactl --cpunodebind=$NUMA --membind=$NUMA"
  "$PY" -u "$ROOT/tools/kolibri/kl.py" --ref "$BIN" --ref-args "--cpu-threads 24" \
      --dump "$OUT/dump" --model "$MODEL" --parity "$PARITY" --out "$OUT/kl" --cache "$CACHE" \
      --lock "$LOCK" 2>&1 | tail -25 | sed 's/^/  /'
fi

echo; echo "== done: $OUT (bench stdout above; counters.log; kl/)"
