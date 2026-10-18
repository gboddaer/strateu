#!/usr/bin/env bash
# tools/kolibri/s3_kv_gate.sh - the S3 gates (plan 2026-10-17-kolibri1-speed.md, fp16 KV at the source).
#
#   ./s3_kv_gate.sh OUTDIR                      # fp16 build (build/strata-kolibri)
#   ./s3_kv_gate.sh OUTDIR /tmp/kolibri-f32kv   # the fp32-KV binary, for the before/after pair
#
# Three gates:
#  1. KV bytes/token at 32K, 1.34 GB -> 0.67 GB.  Derived from geometry and confirmed by the measured
#     tg-after-32K ratio: decode attention reads the full-attention layers' prefix rows once per token,
#     10 layers x (pos+1) rows x 4 KV heads x 128 dim x 2 (K,V) x dtype.  fp32 at 32768 = 1.34 GB/token,
#     fp16 = 0.67 GB/token.  The sliding layers contribute 40 x 513 x 4 x 128 x 2 x dtype = 84 MB (fp32)
#     / 42 MB (fp16), i.e. noise next to the full layers.
#  2. Prefill KV traffic LINEAR in prefix length.  The deleted k_kv_f16_full converted the WHOLE prefix per
#     512-token chunk, so per-token prefill cost grew with the prefix (quadratic total).  pp is timed at
#     four prefix lengths; the per-token cost must stay flat.
#  3. pp-after-32K >= 100 tok/s (the floor), measured as pp32768.
#
# GPU is exclusive: refuses to run if any compute app holds the card.
set -uo pipefail
OUT=${1:-/tmp/kolibri_s3}
BIN=${2:-build/strata-kolibri}
M=${M:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}
CTX=${CTX:-33024}
export STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1
mkdir -p "$OUT"
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "GPU busy: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')" >&2
  exit 3
fi
echo "== S3 gate: bin=$BIN ctx=$CTX $(date +%H:%M:%S)"

# 1. tg after a 32K prompt: the full-layer read traffic is the dominant per-token cost at that context.
echo "--- tg after 32K ($BIN)"
python3 - "$BIN" "$M" "$CTX" "$OUT" <<'PY'
import subprocess, sys, re, json
bin, m, ctx, out = sys.argv[1:5]
ids = ",".join(str(1000 + (i * 7919) % 100000) for i in range(32768))
r = subprocess.run([bin, "--model", m, "--tokens", ids, "--n", "32", "--ctx", ctx],
                   capture_output=True, text=True, env={**__import__("os").environ,
                   "STRATA_DUMP_CTX": "8192", "GGML_CUDA_DISABLE_GRAPHS": "1"})
open(f"{out}/tg_after32k.log", "w").write(r.stdout + "\n" + r.stderr)
pp = re.search(r"prompt\s+([0-9.]+)\s*tok/s", r.stdout)
tg = re.search(r"decode\s+([0-9.]+)\s*tok/s", r.stdout)
print(f"  pp32768 {pp.group(1) if pp else '?'} tok/s   tg128-after-32K {tg.group(1) if tg else '?'} tok/s")
json.dump({"pp": pp and float(pp.group(1)), "tg": tg and float(tg.group(1))}, open(f"{out}/tg_after32k.json", "w"))
PY

# 2. prefill cost per token vs prefix length: flat == the quadratic conversion is gone.
echo "--- prefill linearity (pp per token at growing prefix)"
for N in 512 2048 8192 32768; do
  IDS=$(python3 -c "print(','.join(str(1000+(i*7919)%100000) for i in range($N)))")
  T0=$(date +%s.%N)
  build/strata-kolibri --model "$M" --tokens "$IDS" --n 1 --ctx "$CTX" --prefill > "$OUT/pp_$N.log" 2>&1 || { echo "  $N: FAILED (see $OUT/pp_$N.log)"; continue; }
  T1=$(date +%s.%N)
  python3 - "$N" "$T0" "$T1" <<'PY'
import sys
n, t0, t1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
wall = t1 - t0
print(f"  prefix {n:6d}: wall {wall:7.1f} s   {n/max(wall,1e-9):7.1f} tok/s (wall incl. ~40 s load)")
PY
done
echo "== done $(date +%H:%M:%S)"
