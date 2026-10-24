#!/usr/bin/env bash
# tools/kolibri/s3_passb.sh - the S3 pp-linearity pass (prefix 2048 short + 8192 after-32K).
#
# Waits for the gate runner to release the GPU, then runs bench.py directly.  It deliberately does NOT
# re-run s3_kv_gate.sh (that would repeat the 45-min 32K pass), and it does not modify that script: bash
# re-reads a script file as it executes, so editing a running script corrupts the run (observed
# 2026-10-18 - a phantom "syntax error near '('" and a skipped pass).
set -uo pipefail
cd $(dirname "$0")/../..
M=${M:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
PY=python3
OUT=/tmp/kolibri_s3
CTX=${CTX:-33024}
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1

# 1. wait for the fp32 comparison pass to finish
while pgrep -f 's3_run_gat[e]s.sh' > /dev/null 2>&1; do sleep 30; done
for try in 1 2 3 4 5 6; do
  BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
  [ -z "$BUSY" ] && break
  echo "GPU held by $BUSY - waiting ($try/6)"; sleep 10
done
[ -n "$BUSY" ] && { echo "GPU busy, aborting"; exit 3; }

# 2. linearity: 2048 at ctx 8192 (short case) + 8192 at ctx $CTX (the 32K case slot)
for N in 2048 8192; do
  [ -s "$OUT/ids_$N.txt" ] || python3 -c "open('$OUT/ids_$N.txt','w').write(','.join(str(1000+(i*7919)%100000) for i in range($N)))"
done
timeout 4000 "$PY" -u tools/kolibri/bench.py --bin build/strata-kolibri --model "$M" \
    --ids512 "$OUT/ids_2048.txt" --ids32k "$OUT/ids_8192.txt" --ctx "$CTX" --case both --extra="--prefill" \
    > "$OUT/passB.txt" 2>&1
echo "PASSB_EXIT=$?" >> "$OUT/passB.txt"

# 3. the linearity verdict from all passes collected so far
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
print("per-token prompt cost (flat == the quadratic conversion is gone):")
for n in sorted(pts):
    print(f"  prefix {n:6d}: pp {pts[n]:7.1f} tok/s   prompt cost {1000/pts[n]:8.2f} ms/token")
if len(pts) >= 2:
    s, b = min(pts), max(pts)
    print(f"  cost growth {pts[s]/pts[b]:.2f}x per token from prefix {s} to {b} (1.00x == linear)")
PY
echo PASSB_DONE >> "$OUT/passB.txt"
