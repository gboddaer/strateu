#!/usr/bin/env bash
# tools/kolibri/s5_flag_matrix.sh - task 8 (S5): the flag-combination matrix, with the CUDA-graph state per
# combination.  The plan's clause is "flag-combo matrix records graph state per combo"; the state is not a
# guess, it is a code condition:
#
#     e.graphs = grouped > 0 && !no_graph && slots == 0 && cpu_threads == 0     (kolibri_engine.cu)
#
# so any of --expert-vram / --cpu-threads, or --no-graph, turns capture off.  Each leg records the derived
# state, the engine's own capture line when it prints one, the per-token branch placement (CNT branches), and
# the pp/tg rate.  The measured capture line for `--grouped` is "25 / 50 layers captured as CUDA graphs (the
# rest use the mmvq loop)" - 25, not 50, because the Q6_K-down layers cannot be captured.
#
# Why the matrix matters now: the speed work moved the expert path to the CPU pool, and the pool path runs
# WITHOUT graphs by that same condition.  So the graph-state column is the record of which configurations are
# graph-decoded and which are launch-per-kernel, and the tg column shows what that costs (measured: graphs cost
# 0.6 tok/s on the pure-GPU grouped path, and the graph path is numerically divergent from eager - K28).
#
# Usage: tools/kolibri/s5_flag_matrix.sh [OUTDIR]     # default /tmp/kolibri_s5
#   NUMA=0|1|none   placement (default 0 = the T4's socket; cross-socket interleave measured worse)
#   WAIT_GPU=1      require the T4 idle for 30 s before starting (default 1)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build/strata-kolibri"
MODEL="${KOLIBRI_MODEL:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
IDS512="${KOLIBRI_IDS512:-/tmp/bench_ids512.txt}"
IDS32K="${KOLIBRI_IDS32K:-/tmp/bench_ids32k.txt}"
OUT="${1:-/tmp/kolibri_s5}"; mkdir -p "$OUT"
NUMA="${NUMA:-0}"
export STRATA_DUMP_CTX=8192
PY=python3
[ -x "$PY" ] || PY=python3

for p in "$BIN" "$MODEL" "$IDS512"; do [ -e "$p" ] || { echo "FATAL: missing $p"; exit 1; }; done

# timeout INSIDE the function: `timeout N <function>` cannot work, timeout execs a program.
pin() { local t="$1"; shift
        if [ "$NUMA" = none ]; then timeout "$t" "$@"
        else timeout "$t" numactl --cpunodebind="$NUMA" --membind="$NUMA" "$@"; fi; }

# Require the T4 empty for 6 consecutive samples (30 s): one empty sample is not enough, bench.py leaves a gap
# between its pp and tg subprocesses and a sweep started in that gap runs two contexts on the card, which
# invalidates every timing number it produces.
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

echo "== S5 flag matrix ($(date -Is))  NUMA=$NUMA"
printf '%-28s | %-22s | %-34s | %s\n' "combo" "graphs (derived)" "rate" "branch placement / pool"

# label|extra flags|derived graph state
COMBOS=(
  "plain|| OFF (grouped=0)"
  "grouped|--grouped| ON, 25/50 layers (measured capture line)"
  "grouped --no-graph|--grouped --no-graph| OFF (--no-graph)"
  "slots 4G|--expert-vram 4| OFF (slots != 0)"
  "pool 12|--cpu-threads 12| OFF (cpu_threads != 0)"
  "slots 4G + pool 12|--expert-vram 4 --cpu-threads 12| OFF (slots, cpu_threads)"
  "grouped + pool 12|--grouped --cpu-threads 12| OFF (cpu_threads != 0)"
  "slots 4G + grouped + pool 12|--expert-vram 4 --grouped --cpu-threads 12| OFF (slots, cpu_threads)"
)
: > "$OUT/counters.log"
for c in "${COMBOS[@]}"; do
  label="${c%%|*}"; rest="${c#*|}"; extra="${rest%%|*}"; gstate="${rest#*|}"
  rate=$(pin 900 "$PY" -u "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$MODEL" \
            --ids512 "$IDS512" --ids32k "$IDS32K" --case short --extra="$extra --counters" \
            2>>"$OUT/counters.log" | grep -E "^short" || echo "(FAILED)")
  br=$(grep "CNT branches" "$OUT/counters.log" | tail -1 | sed 's/^CNT branches per tok: //')
  pl=$(grep "CNT pool workers" "$OUT/counters.log" | tail -1 | sed 's/^CNT pool workers=/pool /')
  printf '%-28s | %-22s | %-34s | %s %s\n' "$label" "$gstate" "${rate#short   : }" "$br" "$pl"
done
echo "== done: $OUT (counters.log)"
