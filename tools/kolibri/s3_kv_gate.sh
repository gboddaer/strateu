#!/usr/bin/env bash
# tools/kolibri/s3_kv_gate.sh - the S3 gates (plan 2026-10-17-kolibri1-speed.md, fp16 KV at the source).
#
#   ./s3_kv_gate.sh OUTDIR [BIN]        # BIN defaults to build/strata-kolibri
#
# Rates come from tools/kolibri/bench.py (the same tool S0 used, so the numbers are comparable): it drives
# --serve, parses the PP/DONE lines, and feeds prompts as files - the engine's --tokens cannot carry 32768
# ids because Linux caps ONE argv entry at 128 KB.
#
# Gates:
#  1. KV bytes/token at 32K, 1.34 GB -> 0.67 GB.  Decode attention reads the full-attention layers' prefix
#     rows once per token: 10 layers x (pos+1) rows x 4 KV heads x 128 dim x 2 (K,V) x dtype.  fp32 at
#     32768 = 1.342 GB/token, fp16 = 0.671 GB/token.  The 40 sliding layers add 84 MB (fp32) / 42 MB (fp16);
#     the per-token write is 0.20 / 0.10 MB.  Confirmed by the measured tg-after-32K ratio (gate 4).
#  2. Prefill KV traffic LINEAR in prefix length.  The deleted k_kv_f16_full converted the WHOLE prefix per
#     512-token chunk, so per-token prefill cost grew with the prefix (quadratic total).  pp is timed at
#     512 / 2048 / 8192 / 32768; the per-token prompt cost must stay flat.
#  3. reset_state < 2.5 ms at 32K - measured by tools/kolibri/s0_numa_reset.sh, recorded in the ledger.
#  4. pp-after-32K >= 100 tok/s (the floor) and the tg-after-32K before/after ratio vs the fp32-KV binary.
set -uo pipefail
OUT=${1:-/tmp/kolibri_s3}
BIN=${2:-build/strata-kolibri}
M=${M:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
PY=${PY:-/data2/develop/Strata/.venv/bin/python}
CTX=${CTX:-33024}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1
mkdir -p "$OUT"

for try in 1 2 3 4 5 6; do
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  [ -z "$BUSY" ] && break
  echo "  GPU held by $BUSY - waiting (try $try/6)"; sleep 10
done
if [ -n "$BUSY" ]; then echo "GPU busy: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >&2; exit 3; fi

echo "== S3 gate: bin=$BIN ctx=$CTX $(date +%H:%M:%S)"
for N in 512 2048 8192 32768; do
  [ -s "$OUT/ids_$N.txt" ] || python3 -c "open('$OUT/ids_$N.txt','w').write(','.join(str(1000+(i*7919)%100000) for i in range($N)))"
done

# Gate 2 (linearity) + gate 4 (after-32K), in two bench.py passes:
#   pass A: 512 at ctx 8192 (short case) + 32768 at ctx $CTX (the 32K case)
#   pass B: 2048 at ctx 8192 + 8192 at ctx $CTX
echo "--- pass A: 512 short, 32768 after-32K"
timeout 6000 "$PY" "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$M" \
    --ids512 "$OUT/ids_512.txt" --ids32k "$OUT/ids_32768.txt" --ctx "$CTX" --case both --extra="--prefill" \
    2>&1 | sed 's/^/  /' | tee "$OUT/passA.txt"
echo "--- pass B: 2048 short, 8192 after-32K"
timeout 6000 "$PY" "$ROOT/tools/kolibri/bench.py" --bin "$BIN" --model "$M" \
    --ids512 "$OUT/ids_2048.txt" --ids32k "$OUT/ids_8192.txt" --ctx "$CTX" --case both --extra="--prefill" \
    2>&1 | sed 's/^/  /' | tee "$OUT/passB.txt"

echo "--- per-token prompt cost (flat == the quadratic conversion is gone)"
python3 - "$OUT" <<'PY'
import re, sys
out = sys.argv[1]
pts = {}
for f in ("passA.txt", "passB.txt"):
    try:
        txt = open(f"{out}/{f}").read()
    except OSError:
        continue
    for m in re.finditer(r"(short|after32K)\s*: pp (\d+) tok in (\d+) ms = ([\d.]+) tok/s", txt):
        pts[int(m.group(2))] = float(m.group(4))
for n in sorted(pts):
    print(f"  prefix {n:6d}: pp {pts[n]:7.1f} tok/s   prompt cost {1000/pts[n]:8.2f} ms/token")
if len(pts) >= 2:
    small = min(pts); big = max(pts)
    print(f"  cost growth {pts[big]/pts[small]:.2f}x from prefix {small} to {big} "
          f"(1.00x == linear; the pre-S3 quadratic conversion showed up as growth)")
PY
echo "== done $(date +%H:%M:%S)"
