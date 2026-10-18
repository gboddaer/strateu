#!/usr/bin/env bash
# K25 accuracy gate for the two-pass attention path.
#
# Why this instrument and not the KL harness: tools/kolibri/kl.py runs prompts p00-p05, longest 4147 tokens,
# and the two-pass branch only runs at n_pos > 11000 -- the harness never executes the changed code, so a KL
# pass there says nothing about K25.  What does say something: the same prompt at 11000 tokens (single-pass,
# n_vis <= 11000) and at 11001 (two-pass).  The two runs differ by ONE position, so a sound two-pass path must
# produce top-2 ids and margins that agree closely across the seam; a defect in the tiled online softmax shows
# up as a discontinuity at exactly that boundary.
#
# --dump-logits writes 4 floats per step: {argmax_id, logit[argmax], runnerup_id, logit[runnerup]}
# (kolibri_engine.cu).  The comparison is on the LAST step, where n_pos is at its largest.
#
# Usage: tools/kolibri/k25_seam.sh [model]
set -u
M="${1:-/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf}"
BIN=build/strata-kolibri
CTX=12288
EXTRA="--cpu-threads 12"
for N in 11000 11001; do
  F=/tmp/ids$N.txt
  [ -s "$F" ] || { echo "missing fixture $F (comma-separated ids)"; exit 1; }
  echo "=== n=$N (path: $([ "$N" -le 11000 ] && echo single-pass || echo two-pass))"
  env STRATA_DUMP_CTX=8192 GGML_CUDA_DISABLE_GRAPHS=1 numactl --cpunodebind=0 --membind=0 \
    "$BIN" --model "$M" --ctx $CTX --tokens-file "$F" --n 1 $EXTRA \
    --dump-logits /tmp/seam_$N.txt > /tmp/seam_$N.log 2>&1
  echo "exit=$? ; last gen: $(grep -m1 '^gen:' /tmp/seam_$N.log | cut -c1-60)"
done
/data2/develop/Strata/.venv/bin/python - <<'PY'
def last(f):
    v = [float(x) for x in open(f).read().split()]
    return v[-4:]
a, b = last('/tmp/seam_11000.txt'), last('/tmp/seam_11001.txt')
print(f"seam 11000 (single-pass): top1={a[0]:.0f} l1={a[1]:.3f} top2={a[2]:.0f} l2={a[3]:.3f} margin={a[1]-a[3]:.3f}")
print(f"seam 11001 (two-pass)   : top1={b[0]:.0f} l1={b[1]:.3f} top2={b[2]:.0f} l2={b[3]:.3f} margin={b[1]-b[3]:.3f}")
print(f"top-1 ids {'AGREE' if a[0]==b[0] else 'DIFFER'}; |dlogit(top1)|={abs(a[1]-b[1]):.4f}; "
      f"margin delta={abs((a[1]-a[3])-(b[1]-b[3])):.4f}")
print("reading: one extra position should not move the argmax with a comfortable margin; a near-tie at the seam "
      "is the T6 cascade signature, not a K25 defect - check the margin before calling it.")
PY
