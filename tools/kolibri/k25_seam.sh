#!/usr/bin/env bash
# K25 accuracy gate for the two-pass (tiled online softmax) attention path.
#
# Why this instrument: tools/kolibri/kl.py runs prompts p00-p05, longest 4147 tokens, and the two-pass branch
# only runs at n_pos > the seam, so the harness never executes the changed code -- a KL pass there says nothing
# about K25.  The valid test is the SAME prompt at the SAME position through BOTH paths, which is why the seam
# is a knob (--attn-2pass-max) and not a constant.
#
# An earlier version of this script compared n=11000 against n=11001.  That is invalid: the two runs then differ
# by one INPUT token, so the next-token distribution legitimately changes and a top-1 difference proves nothing.
# Recorded because the invalid design is the kind that gets rebuilt.
#
# --dump-logits writes 4 raw float32 per step: {argmax_id, logit[argmax], runnerup_id, logit[runnerup]}.
#
# Usage: tools/kolibri/k25_seam.sh [model] [n_positions]
set -u
M="${1:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
N="${2:-11001}"
BIN=build/strata-kolibri
CTX=12288
EXTRA="--cpu-threads 12"
F=/tmp/ids$N.txt
[ -s "$F" ] || { echo "missing fixture $F (comma-separated ids)"; exit 1; }
for MODE in two single; do
  # two: seam below N -> the tiled online softmax runs.  single: seam at N -> the smem path runs.
  K=$([ "$MODE" = two ] && echo $((N - 1)) || echo "$N")
  echo "=== n=$N path=$MODE (--attn-2pass-max $K)"
  env STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1 numactl --cpunodebind=0 --membind=0 \
    "$BIN" --model "$M" --ctx $CTX --tokens-file "$F" --n 1 $EXTRA --attn-2pass-max "$K" \
    --dump-logits /tmp/seam_${N}_$MODE.txt > /tmp/seam_${N}_$MODE.log 2>&1
  echo "exit=$? ; gen: $(grep -m1 '^gen:' /tmp/seam_${N}_$MODE.log | cut -c1-40)"
done
/data2/develop/Strata/.venv/bin/python - <<PY
import numpy as np
def last(f): return np.fromfile(f, dtype=np.float32)[-4:]
a = last('/tmp/seam_${N}_single.txt'); b = last('/tmp/seam_${N}_two.txt')
print(f"n=$N single-pass: top1={a[0]:.0f} l1={a[1]:.4f} top2={a[2]:.0f} l2={a[3]:.4f} margin={a[1]-a[3]:.4f}")
print(f"n=$N two-pass   : top1={b[0]:.0f} l1={b[1]:.4f} top2={b[2]:.0f} l2={b[3]:.4f} margin={b[1]-b[3]:.4f}")
print(f"top-1 ids {'AGREE' if a[0]==b[0] else 'DIFFER'}; |dlogit(top1)|={abs(a[1]-b[1]):.4f}; "
      f"margin delta={abs((a[1]-a[3])-(b[1]-b[3])):.4f}")
print("reading: identical input and identical position, only the code path differs, so |dlogit| is a pure")
print("numerics measure.  A near-tie (small margin) makes a top-1 flip expected, not a defect - read the margin.")
PY
