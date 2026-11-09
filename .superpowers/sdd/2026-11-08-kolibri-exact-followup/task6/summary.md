# Task 6 - promotion fences per round

## First profile (existing runs)

The promotion-heavy runs logged every round's move count, bytes and wall time. Summed per run, promotion was
0.07-0.92% of whole-process wall (1222-1424 moves, 1.6-2.7 s; demand profile promoted nothing). The timer includes
packing and copy, so a fence-only change has a sub-1% ceiling there.

## Narrow ring-only batch, implemented and measured

`STRATA_KOLIBRI_PROMOTE_BATCH=1`, default off. When the transfer ring is present and the decode stream/handoff are
off, `slots_reselect` records one prior-reader event before the move list, runs every `acquire/pack/submit` without a
per-move fence, then records one readiness event for `g_st`. The no-ring `hstage` path and every unsupported mode keep
the per-move path.

RED/GREEN: `tests/kolibri/expert_transfer_test.cu` gained a batch-round test: one reader fence covers two delayed old
readers and both ring buffers; a following generation and a second consumer stream see only the new bytes; pageable
and pinned rings both pass. `kolibri_expert_transfer_test` passes.

Promotion-heavy measurement (`pipeline` profile, 8 slots/layer, hot per-layer map, 4097-token prefill, 389-move round):

| run | moves | promotion time | logits SHA |
|---|---:|---:|---|
| fence off | 389 | 487.8 ms | `c83db3e3...61a7` |
| fence on | 389 | 475.6 ms | `c83db3e3...61a7` |

The batch saves 12.2 ms on the round (2.5% of the promotion span). No end-to-end improvement was demonstrated; one
pair with process wall 259 s vs 132 s is load/cache-confounded and establishes neither a win nor its absence. The
round still packs and copies the same 890 MiB. This is not a completed performance qualification.

Decision: **REJECT**. Keep the evidence and the experimental diff, remove the engine gate and the test addition;
per-move fences stay. This is a performance rejection, not a correctness failure. The test's initialization join,
generation-boundary and mutation coverage were not qualified; no ordering defect was found in the narrow path. A
second consumer stream needs its own `wait_for` before a later overwrite; the engine gate required no decode stream. No server; CUDA/SYCL NOT RUN.
