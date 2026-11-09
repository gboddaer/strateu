# Task 5 - redundant decode synchronization

## Trace (corrected after review 1)

The two targeted drains are the two unconditional `cudaStreamSynchronize(g_st)` calls in the slot decode path
(`src/kolibri/kolibri_engine.cu`, after gate/up/down work and after the overflow/pool result, before the shared expert).

- The first follows the slot gate/up/SwiGLU/down work and precedes the pool/overflow branch that produces `s.yc`.
- The second follows the overflow GEMVs or the pool result upload and precedes the shared-expert work; the source
  comment there is stale.
- Quantized `gemv()` submits to `g_st` (its default `st = 0`); SwiGLU in the same block runs on `st`. `g_st` is a
  created blocking stream, so it implicitly orders with the legacy default stream.
- `g_q8` is device scratch. `g_ids_host` is consumed synchronously to build weight pointers, and the selected MMVQ
  copies slots into launch parameters. The CPU pool does not read `ey2`; it produces `ey_h`/`ey_cpu`.
- Transfer reader/readiness waits and source-buffer fences stay in place. No other drain was removed.

## Retained change

`include/strata/kolibri/decode_order.hpp` admits the omission only for: opt-in, no-pin, `st == 0` (legacy/default
compute stream), no handoff, no graph, no CPU pool. `layer_step` keeps both drains in every other case.
Flag: `STRATA_KOLIBRI_ORDERED_DECODE=1`, default off. A one-time stderr trace names the skipped drains.

The removed host test checked the gate cases (the supported conjunction plus each single unsafe condition).

## Evidence

- 512/128 complete logits: off and on both SHA `6ffefa8f...8721` (compared against the Task 2 control).
- 8/32 cache8: both SHA `466f48f2...c3d6`.
- Short/long/hot/profile integration with the flag: 10/10 complete dumps byte-identical
  (`task5/integration-compare.txt`).
- First five 512/128 pairs: decode paired median +6.7% (3/5 positive; -6.6% to +12.9%), wall median
  139.0 -> 121.7 s. Cache8: paired median -1.0%, wall neutral (19.7/19.8 s).
- Second five 512/128 pairs (new run): decode paired median **-6.4%**, **0/5 positive** (-5.2% to -15.7%), wall
  median 144.1 -> 137.5 s. Combined ten-pair median is about **-5.9%**; the first series was not repeatable.
- Builds: HIP `strata-kolibri` OK; `kolibri_decode_order_test` passes.

The GPU test covers the blocking/default-stream dependency, not the full planned gate/up -> SwiGLU -> down -> shared ->
sum chain, and its negative control is timing-dependent. That limitation is recorded; it is not needed to remove. CUDA/SYCL NOT RUN. No server.

The dedicated GPU ordering test now passes: a created blocking stream orders with the legacy default stream, and a
non-blocking stream does not (negative control). Correctness of the narrow gate is supported.

Decision: **REJECT**. No repeatable speedup was demonstrated: the second series is uniformly slower and the combined median is negative. Remove the gate, engine wiring and both tests; keep the evidence. Not completed as a gain: no repeatable
speedup. CUDA/SYCL remain NOT RUN.
