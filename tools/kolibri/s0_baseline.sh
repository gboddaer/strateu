#!/usr/bin/env bash
# tools/kolibri/s0_baseline.sh - the S0 "standing facts" gate of the speed plan
# (docs/superpowers/plans/2026-10-17-kolibri1-speed.md, task S0).
#
# One named, re-runnable command that produces every number the plan's floors depend on:
#   A environment   NUMA placement, node sizes, RAM, GPU state
#   B expert kernel native_expert_bench GB/s per quant type
#   C tg/pp matrix  per flag combo, via bench.py (serve protocol)
#   D counters      slot hit rate, expert PCIe bytes/token, D2H/H2D bytes, reset_state ms (--counters)
#   E prefill gate  --prefill vs plain greedy ids at 514 and 2048 tokens (the K1 causality gate)
#
# Usage:  tools/kolibri/s0_baseline.sh [OUTDIR]        # default OUTDIR=/tmp/kolibri_s0
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build/strata-kolibri"
MODEL="${KOLIBRI_MODEL:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
IDS512="${KOLIBRI_IDS512:-/tmp/bench_ids512.txt}"
IDS32K="${KOLIBRI_IDS32K:-/tmp/bench_ids32k.txt}"
OUT="${1:-/tmp/kolibri_s0}"; mkdir -p "$OUT"
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1
PY=python3
[ -x "$PY" ] || PY=python3

require_bin() { [ -x "$1" ] || { echo "FATAL: missing binary $1"; exit 1; }; }
require_file() { [ -s "$1" ] || { echo "FATAL: missing data file $1"; exit 1; }; }
require_bin "$BIN"; require_file "$MODEL"; require_file "$IDS512"; require_file "$IDS32K"

echo "== A environment =="
echo "T4 pci numa_node: $(cat /sys/bus/pci/devices/0000:03:00.0/numa_node 2>/dev/null || echo '?')"
numactl --hardware 2>/dev/null | sed -n '1,6p'
free -g | awk 'NR<=2{printf "mem %s\n", $0}'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/gpu pid: /'

echo; echo "== B native_expert_bench =="
# NOT APPLICABLE to Kolibri geometry: n_ff 14400 is not a whole number of Q4_K 256-element blocks, so every
# layer reports "expert geometry is not whole blocks" (measured 2026-10-18, layers 0-5, 24-26, 40, 48, 49).
# Expert bandwidth is therefore DERIVED: expert PCIe MB/token (from --counters) / tg seconds/token.
echo "  n/a on this geometry (n_ff 14400 % 256 != 0); derived from counters + tg below"

echo; echo "== C tg/pp per flag combo (pp512/tg128 short, pp32768/tg128 after-32K) =="
run_bench() {
  local label="$1"; shift
  echo "--- $label"
  local cs=short; [ "$STAGE" = long ] && cs=both
  # --extra=VALUE (not "--extra VALUE"): argparse rejects a value that starts with --.
  local rc=0
  timeout 4000 "$PY" "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$MODEL" \
      --ids512 "$IDS512" --ids32k "$IDS32K" --case "$cs" --extra="$*" 2>>"$OUT/counters.log" \
    | sed 's/^/  /' || rc=$?
  [ $rc -ne 0 ] && echo "  (FAILED rc=$rc: $*)"
}
STAGE="${S0_STAGE:-short}"
if [ "$STAGE" = short ]; then
  # Short context only.  The plain path runs pp32768 at ~1.3 tok/s = ~7 h per case, so the after-32K cases
  # live in the long stage and only for prefill-capable configs.
  run_bench "plain                " "--counters"
  run_bench "prefill              " "--prefill --counters"
  run_bench "pool 48 threads      " "--cpu-threads 48 --counters"
  run_bench "slots ~4 GiB         " "--expert-vram 4 --counters"
  run_bench "slots 4G + pool 48   " "--expert-vram 4 --cpu-threads 48 --counters"
  run_bench "grouped              " "--grouped --counters"
  run_bench "grouped + no-graph   " "--grouped --no-graph --counters"
else
  run_bench "after32K prefill       " "--prefill --counters"
  run_bench "after32K slots+prefill " "--prefill --expert-vram 4 --counters"
  run_bench "after32K pool+prefill  " "--prefill --cpu-threads 48 --counters"
  run_bench "after2K plain (probe)  " "--counters --ids512 /tmp/kolibri_s0/ids_2048.txt"
fi

echo; echo "== D counters (from the runs above) =="
grep '^CNT' "$OUT/counters.log" | sed 's/^/  /' || echo "  (no CNT lines)"

echo; echo "== E --prefill vs plain greedy ids (the K1 gate; expect FAIL before S0a) =="
for N in 514 $([ "${S0_STAGE:-short}" = short ] && echo "" || echo "2048") ; do
  "$PY" - "$IDS32K" "$OUT/ids_$N.txt" "$N" <<'PYEOF'
import sys
ids = open(sys.argv[1]).read().strip().replace(" ", "").split(",")
n = int(sys.argv[3])
open(sys.argv[2], "w").write(",".join(ids[:n]))
PYEOF
  IDS=$(cat "$OUT/ids_$N.txt")
  HASH=$(sha256sum "$OUT/ids_$N.txt" | cut -c1-16)
  echo "--- n=$N  ids=$(tr ',' '\n' < "$OUT/ids_$N.txt" | wc -l)  sha256=$HASH"
  timeout 3000 "$BIN" --model "$MODEL" --tokens "$IDS" --n 16 --ctx 8192            > "$OUT/plain_$N.out" 2>/dev/null
  timeout 3000 "$BIN" --model "$MODEL" --tokens "$IDS" --n 16 --ctx 8192 --prefill  > "$OUT/prefill_$N.out" 2>/dev/null
  "$PY" - "$OUT/plain_$N.out" "$OUT/prefill_$N.out" "$N" <<'PYEOF'
import sys
def gen(p):
    for l in open(p):
        if l.startswith("gen:"):
            return l.split()[1:]
    return []
a, b = gen(sys.argv[1]), gen(sys.argv[2])
div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1)
print(f"  plain n={len(a)}  prefill n={len(b)}  identical={a == b}  first_divergent_index={div}"
      + (f" (plain={a[div]} prefill={b[div]})" if div >= 0 else ""))
sys.exit(0 if a == b else 3)
PYEOF
  rc=$?
  [ $rc -eq 3 ] && echo "  -> MISMATCH (expected until S0a lands; S0a's gate requires identical=1)"
done
echo; echo "artifacts in $OUT"
