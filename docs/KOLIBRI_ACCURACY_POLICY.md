# Kolibri accuracy policy

Accuracy takes priority over speed in Kolibri work. Permit justified floating-point rounding differences only
when qualified; do not knowingly trade accuracy for speed. This policy applies to CPU, GPU, prefill and decode
changes. The decode-specific gates remain in [KOLIBRI_DECODE.md](KOLIBRI_DECODE.md#gates-and-decision-rules).

## Choose the gate from the change

| Change | Requirement |
|---|---|
| Scheduling, prefetch, allocation or memory layout with unchanged arithmetic | Finite, bit-identical full logits on identical forced histories |
| Different accumulation order, or different CPU/GPU expert arithmetic | Separate numerical-fidelity and task-quality qualification; experimental and opt-in until approved |
| Lower precision, new quantization or approximate attention | Separate, higher-risk accuracy project; not part of routine performance work |

Moving computation between CPU and GPU may change accumulation order or activation quantization. Do not call
that a scheduling-only change. Equally, placement alone is not permission to waive bitwise checks if the
arithmetic is meant to stay unchanged.

Matching greedy tokens is not proof of correctness or accuracy. Different tokens can arise from harmless
near-ties, while a small average logit error can hide rare serious failures. No single error threshold proves
that a numerical change is safe.

## Qualification for numerical changes

Require all three:

1. **Numerical fidelity.** Compare complete output distributions on aligned, identical forced histories.
   Measure KL divergence, error percentiles and worst cases, and token-ranking changes relative to the
   reference's winning margin. Include long contexts, attention transitions and mixed expert formats. Do not
   compute cross-arm KL on divergent generated histories.
2. **Correctness and repeatability.** Reject non-finite values, stale results, races and unexplained same-arm
   variability. Check first use, replay, prefix reuse, reset and cancellation. Rounding differences do not
   excuse implementation bugs.
3. **Task quality.** Run paired evaluations of reasoning, coding, factual answers and long-context retrieval.
   Establish reference variability and evaluation sensitivity. Agree the maximum acceptable regression margin,
   sample size and statistical decision rule before seeing candidate results. Report uncertainty; an
   inconclusive result is not a pass. A nonzero margin requires explicit approval, not an assumed speed/quality
   trade-off.

The current engine is a regression reference, not proof of perfect model fidelity. Where practical, also use a
trusted higher-precision reference. Record model, build, hardware, configuration, prompts and raw results so
comparisons can be reproduced. Do not invent measurements or choose convenient thresholds after the run.

## Promotion and immediate scope

- Keep bit-identical full logits mandatory for the next unchanged-arithmetic CPU/prefetch and KV-placement
  experiments. Retain the existing full-logit, reuse and repeatability gates.
- GPU-resident experts with different arithmetic may enter an experimental, opt-in qualification track. They
  must not become a default based only on matching answers, greedy agreement or a single KL threshold.
- Preserve the existing pool and MMQ prefill expectations. Never overwrite them to make a candidate pass.
- Correctness failure rejects an optimization regardless of speed. Numerical acceptance and default promotion
  are separate decisions requiring actual results and explicit approval.
- Measure speed separately from accuracy, with an unprofiled same-day control. Keep load, prefill, capture and
  steady decode timings distinct. A qualified numerical change still needs a demonstrated performance gain.

No universal numerical tolerance is approved by this policy. Limits must be justified and agreed for the
specific change before its qualification results are examined.
