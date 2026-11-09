## Verdict

**The dispatch fixes are sound by inspection; full evidence close-out remains incomplete.** No new production correctness or lifecycle defect found.

Read-only inspection. HEAD’s ref matches `a7574911`. Available tools did not permit a Git diff, test execution, or independent binary comparison/hashing.

Below, `T` means `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task2/`.

## Findings by severity

### Important — cross-flag integration equality is not auditable from the retained report

`T/summary.md:73–74` claims a SHA list exists in `integration-off/` and `integration-on/`. Neither directory contains such a list or comparison report.

The runner checks finiteness/size and **within-run** equivalences (`tools/kolibri/test_unpinned.py:34–42,63–65,78,89`), but never compares the off directory against the on directory. Matching `gen:` lines do not establish complete-logit equality.

The twenty dumps and corresponding logs exist, so this is **not evidence of a mismatch**. However, I cannot confirm “10/10 byte-identical” from the retained text evidence. Preserve a cross-directory comparison/finite-validation manifest, ideally including input, profile and executable identities.

**Prior Important finding:** execution coverage resolved; full-logit evidence closure **not fully resolved**.

### Minor — dispatch tests still depend on inherited environment

- `CMakeLists.txt:1345` correctly forces parity to `STRATA_MMVQ_WAVE=0`, but the refusal test at `:1346` inherits its wave setting. Launching CTest with that variable already set to `0` makes refusal fail on gfx906. Explicitly force its wave setting on.
- The packed-Q6 regression claims to cover an unset flag (`tests/kolibri/mmvq_selected_test.cpp:158`), but neither the test nor CMake establishes that precondition. With inherited `STRATA_Q6_PACKED=1`, the old env-only implementation could pass this sentinel. Force the flag off.
- The sentinel selects the same registered pointer twice (`:183–184`). A mixed selection whose **later** pointer alone is registered would additionally protect the per-pointer loop.

### Minor — cache8 reporting remains internally inconsistent

`T/summary.md:50` still says **+0.7%**, while `:53` correctly says approximately **+0.5%** for the ratio of medians.

The raw median throughputs are approximately 2.154015 and 2.164229 tok/s (`T/t2-c8-off-4.json:50`, `T/t2-c8-on-3.json:52`), giving approximately **+0.47%**. The paired median **+2.8%** and “inconclusive” classification are correct.

### Minor — integration watchdog violates the stated gate

The newly cited runner uses `timeout=300` (`tools/kolibri/test_unpinned.py:30`), versus the required minimum 10800 seconds (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:30`).

The logs show completed cases, not timeouts; this does not demonstrate incorrect outputs. It nevertheless remains an unrecorded procedural deviation.

## Requested confirmations

### 1. Per-pointer Q6 registry refusal: confirmed

`src/kernels/cuda/native_mmvq.cu:3483–3485` checks every selected weight pointer through `q6_packed_registered()`, whose lookup directly consults the registry (`:2805–2807`). No environment flag controls that lookup.

Refusal precedes all selected launches (`:3488` onward), so it cannot partially write earlier selections. The new sentinel checks refusal and unchanged output (`tests/kolibri/mmvq_selected_test.cpp:179–188`).

**`g_q6_bypass` is correct for applicable-dispatch equivalence:** independent Q6 calls also bypass packed dispatch while it is true (`native_mmvq.cu:2812`). Its current setters are internal load-time self-test code (`:3026–3029,3042–3051`). Thus “registered” here means *registration applicable to the reference dispatch*, not unconditional registry membership. No new supported-concurrency guarantee should be inferred.

**Prior Important finding: resolved.**

### 2. gfx906 registration and refusal mode: confirmed, with environment caveat above

Parity is forced off-wave, and refusal is registered only inside `if(STRATA_HIP_GFX906)` (`CMakeLists.txt:1343–1347`).

`--expect-refusal` supplies an otherwise supported Q4_K call (`tests/kolibri/mmvq_selected_test.cpp:223–225`), then checks false plus unchanged output (`:147–152`). That correctly targets the common wave refusal at `native_mmvq.cu:3459–3460`. Q4-only coverage is sufficient for that shared guard.

**Prior Important finding: resolved in source; gfx906 runtime qualification remains NOT RUN.**

### 3. Integration coverage: present; byte equality unconfirmed

All ten named cases have off/on artifacts. Candidate logs show selected-path activation in every case. Multi-chunk execution is explicit in `T/integration-on/long-prefill.log:9–12`; saved-profile replay is explicit in `T/integration-on/hot-warm.log:7,11`.

The runner constructs matching forced histories (`tools/kolibri/test_unpinned.py:22–24,49–55,68–74`). Logs show corresponding cache configurations and matching generated tokens.

No additional Task-2-specific shape is obviously missing: the original 512/128, cache8 and 4097/8 evidence remains. The 16385/65537 cases are conditional on cache/lifetime/long-prefix changes; the boundary case belongs to final composition (`plan:27–30`). The outstanding issue is recorded cross-flag byte validation, not missing named phases.

### 4. Alignment contract: resolved

The distinction is explicit in `include/strata/kernels/native_mmvq.hpp:160–162` and matches implementation (`native_mmvq.cu:3454,3479`). Tests cover base throwing (`mmvq_selected_test.cpp:195–208`) and derived-pointer refusal (`:253`).

Only `T/summary.md:9–10` remains imprecise: it still mentions the environment flag and broadly says “unaligned selected table.”

### 5–6. KEEP scope and new defects

The **local W7800 decode KEEP**, cache8 “inconclusive,” and explicit absence of an end-to-end wall-time claim are appropriately scoped (`T/summary.md:52–54,85`). Task 3 remains a prospective consumer, not demonstrated consumption.

No new kernel arithmetic, output-ordering, or lifetime defect found in these fixes. The engine still falls back to independent calls when selected dispatch refuses (`src/kolibri/kolibri_engine.cu:1959–1975`). The remaining issues are evidence completeness, test-environment isolation, and reporting precision.
