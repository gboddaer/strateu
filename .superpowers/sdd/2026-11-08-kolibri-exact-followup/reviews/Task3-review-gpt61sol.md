## Decision: **REJECT the integration**

The evidence does not establish a repeatable gain, and there is no named retained consumer justifying KEEP-ENABLER. Remove the experimental integration as proposed; preserve the evidence. This is **not** proof that reuse can never help a later composition.

Read-only source/artifact review; I did not rerun tests or independently hash binary dumps.

## Findings by severity

### Medium — short host-pick fallback remains unsafe
**`src/kolibri/kolibri_engine.cu:2024–2027`, `:1972–1974`**

The helper detects `g_ids_host.size() < n_sel`, but falls back to `gemv()`, whose quantized loop unconditionally indexes `g_ids_host[sel]`. Thus:

- It preserves legacy behavior, including its existing out-of-bounds defect.
- It does **not** provide a safe short-array fallback.
- Current wired callers populate K picks (`:3130–3137`, `:3242`, `:3281`), so I found no normal-path trigger.

This is a pre-existing legacy defect exposed by the new guard, not evidence that the measured runs were corrupted. Record it separately even when removing Task 3.

### Medium — production regression protection and qualification are incomplete
**`tests/kolibri/quant_reuse_test.cpp:97–109`, `:135–169`;  
`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:24–27`**

The test meaningfully checks deterministic q8_1 bytes, finite bit-identical gate/up outputs, changed values at the same activation address, and contiguous versus separate down-input quantization.

However, it calls only existing primitives—not `gemv_pair_quantized()`, the selected branch, flags, projection offsets, or fallback logic. Every selection also uses the same weight matrix. Removing or breaking the production helper would leave this test passing.

The supplied Task 3 evidence also lacks the required **4097-input/8-step** comparison. The “long” integration run processes 1024 prefill rows, not 4096 (`task3/integration-both/long-prefill.log:10–11`). The integration runner is explicitly no-pin (`tools/kolibri/test_unpinned.py:23–25`), so it does not numerically qualify the pinned-table wiring.

These gaps block claiming full Task 3 qualification **if retained**. They are **not fatal to rejecting and removing the experiment**. Record the production RED and missing qualification as skipped, not passed.

## Correctness / bitwise assessment

For the current valid wired calls, I found **no new arithmetic or stale-reuse defect**:

- All three sites pass the same `s.h`, the same `gu` descriptor for gate/up, and projection bases separated by `b.L.up_off`: engine `:3244`, `:3290–3291`, `:3450`.
- Gate/up type equality is enforced during model loading (`:6756–6762`); helper type/shape mismatches return to the two legacy calls (`:2024–2027`).
- Each helper invocation freshly quantizes **one shared column**, then both consumers use that exact `g_q8` pointer (`:2036–2055`). There is no address-keyed cache or retained “valid” state.
- Selected dispatch is restricted to 2–8 selections; other positive counts use individual consumers (`:2041–2055`). Supported-format refusal also falls back.
- Resident, overflow/spill, and pinned bases retain the original indexing and stride. Duplicate/zero/spill slots are not reinterpreted.
- Down projections remain separate `gemv()` calls with per-selection inputs (`:3250`, `:3297`, `:3456`), producing fresh quantization through `:1955–1956`.

If gate-selected dispatch succeeds but up-selected dispatch refuses, fallback repeats gate before running up. That is redundant work, but same-stream overwriting with the same result—not a correctness race.

## Lifecycle / ordering assessment

No new allocation, ownership transfer, host-array DMA, or fence removal was introduced.

Quantization and both consumers share `qst` (`:2036–2055`), so later same-stream producers cannot overwrite `g_q8` before up finishes. The existing blocking `g_st`/legacy-stream ordering remains intact (`:1949–1953`, `:7033`). The slot paths preserve transfer readiness and reader synchronization (`:3237–3241`, `:3251–3255`).

I found no new lifecycle defect in the exercised scheduling mode. This is not blanket qualification of arbitrary concurrent-stream use of global scratch.

## Performance and enabler decision

`task3/summary.md:19–30` reports mixed-sign paired results, neutral cache8 behavior, and noisy selected-versus-both results. The 4.36 tok/s selected-arm outlier is real in the artifact (`task3/t3b-512-sel-4.json:48–51`). Those data support **“benefit unproven”**, not a reliable slowdown estimate.

That is sufficient to **REJECT under the plan’s retention rule** (`plan:35–37`).

Reuse could become a justified enabler later, but only with an actual named retained consumer and appropriate tests—or a measured composition benefit against the same composition **without reuse**. A hypothetical capture/synchronization benefit, or Task 2’s independent gain, does not justify retaining this integration now.
