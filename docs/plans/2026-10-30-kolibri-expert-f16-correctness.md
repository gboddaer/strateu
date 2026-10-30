# Kolibri `EXPERT_F16` correctness on CUDA (sm_75 / Tesla T4)

Status: open, investigation not started. Branch `kolibri-expert-f16-correctness`, forked from `main` at `9373ef88`.

## The problem

`STRATA_KOLIBRI_EXPERT_F16=N` is the entire prefill lever on the Tesla T4: 52 tok/s without it, ~160 with it, ~182
with `PF_PARTIAL` added (`docs/findings-log.md` 0.8). It is also the configuration the shipped T4 setup uses, so the
prefill rates in the README and in `docs/KOLIBRI1.md` rest on it.

On the T4 it changes the numbers much more than the R9700 campaign recorded for the same knob:

| comparison (T4, 512-id prompt, 9 forced steps, full logits) | mean KL | p99 | max \|dlogit\| | top-1 |
|---|---:|---:|---:|---:|
| plain vs `EXPERT_F16=24` | 0.400 | 3.24 | **11.08** | 9/9 |
| plain vs `EXPERT_F16=24 PF_PARTIAL=1` | 0.168 | 1.26 | 5.47 | **7/9** |
| R9700, `PF_PARTIAL` (recorded 2026-10-27) | 0.00133 | — | — | 0.9725 |

Both T4 configurations are bitwise identical run to run, so this is rounding and accumulation order, not a race. Under
`docs/KOLIBRI_ACCURACY_POLICY.md` a change like this is "different CPU/GPU expert arithmetic": experimental and opt-in
until separately qualified, and it may not become a default on matching ids or on a single KL.

The open question is not "is the number big": it is **whether 11 logits is expected fp16 behaviour on this card or a
defect in the Turing path**. The answer decides whether this is a qualification project or a bug fix.

## Hypotheses, cheapest first

1. **The fp16 copy is not what the prefill GEMMs read on sm_75.** `kolibri_engine.cu:276` says the first N experts are
   dequantized to fp16 for the layer's prefill GEMMs; the fill is gated on
   `strata::kernels::dequant_bf16_supported(b.L.gu_type)`. Turing has no native bf16. Check whether the gate selects a
   conversion path there that loses more than fp16 rounding, and whether the Q6_K down projections of a Q4_K_M file
   (mixed types) take a different route than Q4_K.
2. **Accumulation, not just operand precision.** Compute one layer's prefill GEMM three ways on the same inputs (fp32
   from Q4_K, fp16 operands with fp32 accumulate, fp16 operands with fp16 accumulate) and compare against the plain
   path per layer. The worst-case step should name itself.
3. **My reference is too narrow.** Nine steps at the tail of a 512-token prompt is a flat-distribution region: a
   median KL of 0.0005 with a p99 of 3.24 is a heavy tail, and near-ties amplify. Re-run the comparison over longer
   forced histories, several prompts, and the mixed expert formats before treating 0.400 as the effect size.
4. **First-use / replay / chunk-boundary staleness.** Determinism across two runs does not exclude a stale fill that is
   reproduced identically. Exercise first use, chunk boundaries (256-token rule), prefix reuse and reset with the cache
   on, per the policy's repeatability clause.

## Gates to run before anything is called qualified

- **Numerical fidelity:** full logits on aligned forced histories; KL with median/p95/p99/worst case; top-1 and
  rank-change against the reference's winning margin; the 8191 KV-wrap boundary and an ~11000-token attention
  transition; both a 512-token and a long prompt; expert types Q4_K and Q6_K.
- **Reference:** `tools/kolibri/kl.py`, i.e. llama.cpp GPU as P and the llama.cpp CPU run as the noise floor. The T4
  work so far compared the engine against *itself* (`klcmp.py`), which measures a change but cannot say whether the
  plain path or the cache path is closer to the model.
- **Correctness and repeatability:** reject non-finite values, stale results, races, unexplained same-arm variability.
- **Task quality:** paired evaluations (reasoning, coding, factual, long-context retrieval) with the maximum acceptable
  regression, sample size and decision rule agreed *before* the candidate results are seen. An inconclusive result is
  not a pass.

## Deliverables

- A verdict on hypothesis 1/2: defect, or expected and quantified.
- If defect: the fix, with the plain path still byte-identical and the cache path brought to whatever fidelity the
  Turing hardware allows.
- If expected: the qualification record the policy requires, and only then a proposal to keep `EXPERT_F16` in the
  shipped T4 configuration. Until either lands, the accuracy-first T4 configuration stays the engine default
  (`EXPERT_F16=0`, 51.6/53.6 prefill, 30.5/30.5 decode, bitwise at 512/8191/11000).

## What this branch is not

Not a speed branch: the 182 tok/s is already measured, and the acceptance question is accuracy. Not a re-run of the
R9700 campaign: nothing measured there transfers to sm_75 without re-measurement, which is the whole point of the
11-logit finding.
