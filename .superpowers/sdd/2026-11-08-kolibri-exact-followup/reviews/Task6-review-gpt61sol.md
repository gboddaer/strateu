**REJECT is not justified as a completed, plan-compliant Task 6.** The measurements support low priority, but the batching rationale is partly wrong, and the required bounded prototype and lifecycle tests were skipped. No current promotion-path race was established by this read-only review.

## Findings by severity

### Important — Existing transfer batching does not eliminate the proposed optimization

**`.superpowers/sdd/2026-11-08-kolibri-exact-followup/task6/summary.md:19–24`**

`ExpertTransfer` supports batching, but promotion currently does **not** use it:

- `submit()` records a buffer event and increments `generation` — `src/kolibri/expert_transfer.cu:54–56`.
- `wait_on()` enqueues a dependency on the latest submission; it does not record a generation event — `src/kolibri/expert_transfer.cu:58–62`.
- Every promotion calls `wait_for → submit → wait_on` — `src/kolibri/kolibri_engine.cu:5697–5700`, `5719–5722`.

Because each submission advances the generation, readiness-call deduplication does not collapse those per-move calls. The current path creates repeated copy/compute dependency chains.

The planned hoist can use the **existing two-buffer ring**, without additional staging memory: one prior-reader dependency, repeated ownership-preserving acquire/pack/submit operations, then one readiness dependency. The `hstage` path can remain the unsupported per-move fallback, exactly as requested by the plan.

### Important — Required experiment and acceptance gates were omitted

**`.superpowers/sdd/2026-11-08-kolibri-exact-followup/task6/summary.md:26–29`**  
**`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:182–186`**

The plan explicitly requires the delayed-reader RED test, bounded batch, fallback preservation and promotion-heavy profiling. Its common gate accepts **any repeatable positive gain**, not only gains exceeding 1% (`:33–35`).

Furthermore, the cited Task 4 off-runs use **326 slots/layer**, not a small cache, and lack the transfer-ring banner:

**`.superpowers/sdd/2026-11-08-kolibri-exact-followup/task4/t4c-4097-off.log:2–8`**

These measurements price the shared-`hstage` path, not the ring-enabled fence hoist. They cannot establish that ring fence overhead is negligible or that the batch is redundant.

**Disposition:** implement and evaluate the narrow ring-only batch, or explicitly record a scope deferral—not completed performance rejection.

### Medium — The ceiling is valid only for the measured denominator/configurations

**`.superpowers/sdd/2026-11-08-kolibri-exact-followup/task6/summary.md:14–24`**

Eliminating the entire measured promotion span would save under 1% of these **whole-process** runs. That bound is sound for those runs, but includes substantial startup time. For example:

- Startup: **100.88 s**
- Prefill: **71.52 s**
- Promotion: **1.6167 s**

Thus promotion is approximately **2.24% of prefill plus decode**, although only **0.924% of process wall**.

Evidence: **`.superpowers/sdd/2026-11-08-kolibri-exact-followup/task4/t4c-4097-off.log:10,19–21`**.

I found **no measured small-cache/long-prefix promotion-dominated counterexample**. An existing ring-enabled cache8/16K run records 342 promotions taking **428.4 ms / 525.98 s = 0.081%**:

- `.superpowers/sdd/2026-11-07-kolibri-overnight/runs/prefill-16385-cache8.log:2,9,46`
- Corresponding `.json:62`

That reinforces low expected value, but does not demonstrate a universal ceiling. Nor does timing pack+copy together establish that packing/copying *dominates* fence overhead.

### Medium — Required lifecycle coverage remains missing

**`tests/kolibri/expert_transfer_test.cu:28–52`**

Existing tests cover ring wraparound and one delayed old reader. However:

- The second consumer is tested **after the first consumer has synchronized**.
- Its assertion checks a wait counter, not reads racing unfinished uploads.
- The delayed-reader overwrite covers one destination, not a multi-layer promotion round and subsequent generation.

This is the exact coverage Task 6 required. It remains an unqualified lifecycle risk, **not a demonstrated current defect**.

## Checks confirmed

- **Extracted off-run shares are accurate:** 4K **0.626–0.924%**, 16K **0.562–0.632%**, 65K **0.125–0.127%**. The **0.07%** figure comes from an admission-**on** run with 548 moves, not the 1222–1424-move off-runs.
- **The timer includes packing and completed copies:** it encloses promotion calls and the final stream synchronization — `src/kolibri/kolibri_engine.cu:6019–6026`. It can also include pending reader work.
- **Current ownership protection appears sound statically:** ring buffers wait before reuse (`src/kolibri/expert_transfer.cu:43–46`); promotions retain reader/readiness dependencies. Shared `hstage` really is shared across layers (`src/kolibri/kolibri_engine.cu:6833–6834`), so removing its per-move synchronization without replacement ownership would be unsafe.

**Bottom line:** low expected payoff is credible; “already batched, requires new buffering, therefore REJECT” is not.
