**Yes—REJECT is justified as a retention decision, not as proof that the planner can never help.** The demand control shows no incremental traffic benefit on this workload and no observed speedup. There is no named retained consumer. It does not establish a correctness failure.

Read-only review; I did not rerun tests or independently hash dumps.

Paths below use `W = .superpowers/sdd/2026-11-08-kolibri-exact-followup`.

## Findings by severity

### Medium — The conclusion overstates what one demand pair proves
`W/task4/summary.md:61–68`

The defensible conclusion is:

> No incremental benefit demonstrated on cold-profile, auto-cache 65537/8 under demand; remove the integration because further retention is not justified.

It is **not** proof of no benefit generally:

- Zero evictions means demand exercised free-slot admission, not the planner’s distinguishing eviction behavior.
- Smaller caches, populated saved profiles, or changing expert demand could exercise evictions under demand too.
- “Only matters when the cache has no free slots” is too strong: the planner also ranks free-slot fills when capacity is insufficient.

**Baseline-only benefit:** real evidence, not something to erase retrospectively. The retention rule permits a scoped gain; it does not explicitly require beating demand in every configuration (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:35–36,163–166`). But the baseline results lack the required qualification, and no retained dependency needs them. Thus retention is not compelled; rejection is reasonable.

### Medium — “Prefill legacy admission” includes decode admission
`src/kolibri/kolibri_engine.cu:2987–2989,3137,5467–5469`

The legacy counter increments regardless of the `prefill` argument.

Consequently:

- Demand-on: **10,473 prefill admissions + 16 decode admissions**.
- Demand-off: **10,489 combined admissions**.

The totals reconcile exactly by expert count; the rounded byte totals reconcile too. This supports the comparison, but the label misstates phase scope.

These are padded expert-upload categories—not total process traffic. Startup seeding and hot promotions are outside them. In particular, baseline’s **19.88 GB** excludes its substantial startup population; it cannot be compared directly with demand’s **25.15 GB** as total upload volume.

### Medium — The execution ledger still says KEEP
`W/progress.md:31,140–141`

Update both entries to REJECT, record this control and review, and distinguish **proposed removal** from **completed removal**. The planner and integration remain present in the inspected tree.

### Minor — Demand prefill rates are incorrectly transcribed
`W/task4/summary.md:58–59`

The JSON records **52.73 / 52.02 tok/s**, not **48.8 / 47.8**:

- `W/task4/t4d-65537-demand-off.json:47`
- `W/task4/t4d-65537-demand-on.json:48`

Wall increased approximately **1.38%**. One pair does not establish a repeatable regression.

## Comparison validity

I found **no demonstrated confound invalidating the within-demand comparison**:

- Same recorded executable hash, context, forced history and reserve.
- Both logs show **334 slots/layer** and the same pinned transfer-ring configuration.
- The driver removes the output profile before launch unless an input profile is supplied (`tools/kolibri/bench_unpinned.py:165–167`); these are independent cold-profile runs.
- No staging line is credible here: its output is suppressed when the instrumented count is zero (`src/kolibri/kolibri_engine.cu:5470–5473`).

However, **baseline versus demand is not a planner-only comparison**: baseline had 326 slots/layer, different startup population, different transfer behavior, and a different executable hash. Demand also does not qualify transfer-ring **eviction** safety, because it recorded zero evictions.

## Alias check and counters

- **Alias check:** the original finding is fixed at `include/strata/kernels/kolibri_hot_policy.hpp:135`, with regression coverage at `tests/kolibri/hot_policy_test.cpp:249–254`. Removing the planner removes that specific hazard. There is no correctness requirement to retain a dead planner merely for its validation. A separate shared map-invariant check could be useful hardening, but no normal alias producer was demonstrated.
- **Counters:** useful diagnostics, not correctness dependencies. Keeping legacy-admission/staging counters separately is reasonable if accurately scoped. Removing new-planner counters is also reasonable.

## Required removal record

Record:

1. **Exact on measured histories; rejected for insufficient incremental value**, not correctness.
2. Preserve the genuine baseline gains and all raw evidence.
3. Mark balanced controls, repeated-pack qualification, five-pair timing gates and eviction-lifecycle qualification **not completed**, not passed.
4. Preserve existing demand free-slot admission and its readiness fences (`src/kolibri/kolibri_engine.cu:2975–2996,4077`).
5. Remove only Task4-specific tests/wiring—not existing seed, spill, hot-policy tests or unrelated staging counters—and verify the rebuilt flag-off path afterward.

**Bottom line: reject the integration, but narrow the written claim to “no demonstrated incremental gain in the tested demand configuration.”**
