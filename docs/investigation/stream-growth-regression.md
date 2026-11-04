# Served Kolibri prefill: a short first request disabled streaming

## Problem and scope

On a Tesla T4, a short greeting followed by a longer prompt caused expert
streaming to refuse the larger request. Its slots were allocated for the first
request's selection count. The engine reused that context without checking
whether a later request needed more space. Default mode then permanently fell
back to the CPU pool; `STRATA_KOLIBRI_EXPERT_STREAM=1` exited with code 2.

This is separate from prefix/KV caching. This fix does not weaken zero-boundary
alignment or the stale-sliding-ring guard from `6c2ef8e4`. A previous proposed
alignment change was rejected in independent review: matching generated IDs
concealed the full-logit errors already recorded in
[the degeneration investigation](degeneration.md). `reuse_gate.py`'s historical
"bitwise" labels compare IDs, not complete logits.

## Fix

Track the context's selection capacity. When a request exceeds it, drain and
release the old context before creating the larger one. Requests that fit keep
the existing allocation; smaller requests do not shrink it. Allocation failure
retains the existing explicit failure/fallback policy, not a partially valid
context. Releasing before allocation avoids requiring both contexts' memory at
once. Expert arithmetic, routing and KV reuse rules are unchanged.

## Validation actually run

Host: Tesla T4 16 GB, driver 580.196.07, two E5-2690 v4 CPUs, 14 engine lanes,
Kolibri-1 Q4_K_M. Fresh CUDA Release build, SM75, Q6K experts enabled, based on
`main` at `15dac194` plus the capacity-only patch. No production server was
running during GPU validation; the user explicitly freed the machine.

- **Before:** `stream_growth_gate.py` against the retained pre-fix CUDA binary
  failed on short -> long with `expert streaming: this view has more selections
  than the slots hold`. The child closed stdout; no allocation/OOM claim was
  inferred from the Python server's generic error message.
- **After:** short/long/short/long (16/513/16/513 IDs) and repeated growth
  (16/257/513/257/513 IDs) passed. Streaming is demanded, KV reuse is disabled,
  and each long request is compared with a cold, long-first engine process.
  All **four complete 128000-value finite FP32 logit vectors were byte-identical**.
  Both down formats are traversed by the model's alternating layers.
- **After:** `degeneration_gate.py` passed all six finite full-logit cases with
  default streaming, and all six again with streaming explicitly off. The safe
  ring-capacity cases retained 512 reused positions; unsafe rewinds rebuilt cold.
- CPU instrument suite: **173 passed, 22 skipped**. The seven new checks verify
  the gate's request sequence, all-score comparison, finite-value rejection and
  malformed-dump rejection. These CPU checks are not GPU qualification.
- CUDA compilation passed in the isolated `build-review-cuda` directory.

Commands (set the paths to a matching build, model and new output directories):

```sh
python tools/kolibri/stream_growth_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/growth"
python tools/kolibri/degeneration_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/stream"
STRATA_KOLIBRI_EXPERT_STREAM=0 python tools/kolibri/degeneration_gate.py \
  --bin "$BIN" --model "$MODEL" --output "$OUT/pool"
python -m pytest tools/kolibri -q
```

Raw local evidence: `/data4/kolibri-sram/out/prefix-regression-review/`, including
baseline refusal, both growth matrices, complete logit dumps, both degeneration
runs, CPU test output, configure and build logs. No new speedup percentage is
claimed: this repairs loss of the existing streamed path, not its arithmetic.

## Limits

HIP and SYCL compilation/runtime were **not run**; those toolchains are absent.
Replacement-allocation failure was reviewed statically but not injected into the
full engine. Earlier full-model results do not establish broad coding quality,
long-context checkpoint reuse, or correctness on other hardware. Keep the PR a
draft until required backend checks and review are complete.
