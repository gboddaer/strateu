## Verdict

**Conditional KEEP for the measured W7800 decode path; Task 2 is not ready to close.** The arithmetic matches the legacy kernels by inspection, but registered-Q6 rejection has a hole, the new test fails under the intentionally unsupported gfx906 wave dispatch, and required regression evidence is missing.

Read-only review; no files changed or tests rerun. Branch/HEAD match the brief. Hash statements below refer to recorded artifacts, not independently rehashed dumps.

## Correctness / bitwise risks

### Important: registered packed-Q6 dispatch can be bypassed

`src/kernels/cuda/native_mmvq.cu:3457` refuses Q6 based on `native_q6_k_packed_enabled()`, **not the selected pointers’ registrations**.

Those are not equivalent:

- `native_q6_k_pack()` can populate the registry without checking that environment flag (`:2970–2998`).
- Independent `native_mmvq()` calls consult that registry through `q6_packed_mmvq()` (`:2805–2810`, `:3416`).
- With the flag unset, a directly registered matrix can therefore take the selected GGUF-layout kernel instead of returning false.

For example, register an eligible Q6 matrix with the flag unset, then select it twice using `weight_stride=0`. The selected API currently launches. This violates the explicit registered-dispatch refusal rule, even if the two implementations happen to produce identical outputs.

**Required:** check every selected pointer’s applicable registration before launching; add a registered-Q6 sentinel/no-launch test.

### Arithmetic identity: no changed loop or reduction order found

The new bodies mechanically match the legacy bodies:

| Format | Legacy | Selected |
|---|---|---|
| Q4_K | `src/kernels/cuda/native_mmvq.cu:599–636` | `:780–819` |
| Q6_K | `:734–769` | `:821–860` |

Both preserve:

- `SmallK` row counts;
- thread-to-K-block mapping and iteration increments;
- the same `q4_q8_dot` / `q6_q8_dot` helpers;
- per-row accumulation order;
- ascending shared-partial additions followed by the same XOR warp reduction;
- valid-row guards and output-lane selection.

Dispatch thresholds also match: Q4_K uses fewer than eight blocks; Q6_K fewer than four (`:3390`, `:3433`, `:3485–3500`).

The selected kernels change SmallK’s launch-bounds occupancy hint from 1 to 4 (`:781`, `:822`). That is not a source arithmetic-order change, but bitwise qualification remains backend-specific.

The synthetic cases cover both branches and `n_out=37` tails (`tests/kolibri/mmvq_selected_test.cpp:39–43`). They share the actual quantized bytes with the independent calls (`:109–118`), use finite scale generation (`:54–57`), and check finiteness plus `memcmp` (`:127–131`).

Duplicate/permuted weights, shared/per-selection activations and zero weight stride are represented. Notably, the three-selection “permutation” produces duplicate slot 1; other tested widths produce actual permutations (`:85–87`).

### Minor: unaligned-base behavior needs clarification

An unaligned selected pointer derived from an aligned base returns false before launch (`src/kernels/cuda/native_mmvq.cu:3475`). But an unaligned **base**, including selected slot zero, throws through `validate_pointer(base)` (`:3449`, `:2402–2405`).

That is safe—nothing launches—but is not uniformly “unaligned table → false.” Clarify the contract or make the behavior uniform, and test both cases.

## Parameter safety and fallback

Apart from the packed-registry hole:

- The by-value block contains eight weight/output pointers (`src/kernels/cuda/native_mmvq.cu:775–778`).
- Selection count is bounded before host-array access (`:3446`).
- `grid.y` is exactly the selection count, keeping `p.weights[sel]` and `p.outputs[sel]` in bounds (`:3483–3500`).
- Shared input uses offset zero; per-selection input uses `sel * (n_in/32)` Q8_1 blocks (`:789`, `:830`, `:3482`).
- Output pointers use `s * n_out`, leaving duplicate weight selections with distinct output regions (`:3477`).
- Launches use the explicitly validated stream.
- Negative slots, multiplication overflow and base-address addition overflow are checked before the single launch (`:3463–3473`).
- Unsupported formats/counts, bad Q8 stride, gfx906 wave dispatch and misaligned derived pointers return before launch.

There is no incremental per-selection launch or write during validation. Allocation/table bounds remain the caller’s responsibility, as documented.

The new tests do **not** exercise packed-Q6, gfx906 refusal or unaligned-table refusal (`tests/kolibri/mmvq_selected_test.cpp:182–195`).

## Lifecycle / race risks

**No new reachable stale-pick or DMA-lifetime race found in the current no-pin CLI path.**

`g_ids_host` deliberately does **not** numerically equal device `ids` during slot execution: device IDs identify routed experts, while host arrays identify resident/zero/spill slots.

The current sequence is coherent:

1. Read current router IDs synchronously (`src/kolibri/kolibri_engine.cu:3080–3086`).
2. Construct slot mappings.
3. Assign `s.slot_ids` immediately before resident calls (`:3193–3197`).
4. Assign `s.ovf_ids` immediately before overflow calls (`:3228–3239`).

The selected launch materializes those host values into by-value pointers, so subsequent vector replacement cannot alter an outstanding launch.

The gate requires the opt-in flag, no-pin mode, non-null `ids` and 2–8 selections (`:1958–1962`). No-pin is initialized from the CLI (`:7002–7003`); load rejects CPU-pool/grouped combinations (`:6637–6646`). Scheduling arms that could skip the ordinary pick copy are disabled for this configuration (`:6960–6966`, `:7053–7055`).

Dense/shared calls have null IDs and cannot enter the selected path. All current non-null-ID call sites are expert calls.

**Residual design risk:** `g_ids_host.size() >= n_sel` proves neither freshness nor correspondence. The safety comes from current call-site sequencing, not the helper interface. Existing global `g_ids_host`/`g_q8` remain unsuitable for concurrent invocations without additional ownership.

## Performance evidence

The recorded evidence supports **local 512/128 decode KEEP**:

- All five pairs improve: **+7.9% to +34.7%**.
- Median paired improvement is **+20.9%**.
- Ratio of throughput medians is **+19.3%**—a different statistic.
- The recorded complete-logit hashes match for 512/128, cache8 and 4097/8.
- Candidate logs establish actual activation of the path, including six selections (`task2/on-512.log:14`).

The timings do **not** establish an end-to-end wall-time win; wall time regresses in the final two 512 pairs.

**Cache8 should remain “inconclusive,” not a gain.** Its paired spread includes −13.3% and +27.5%, with substantial warm-up/transfer variability.

There is also a reporting error: `task2/summary.md:52` says cache8’s median improvement is +0.1%. Its listed paired median is approximately **+2.8%**; the ratio of throughput medians is approximately **+0.7%** (`:50`). Neither establishes a reliable gain.

## Skipped steps / qualification gaps

1. **Important: required short/long/hot regression phases are not evidenced.**  
   The three benchmark shapes exist, but these are not the full regression matrix required by `docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:31–35`. Missing artifacts include the decode-only/cache-residency comparisons and saved-profile cold→warm replay exercised by `tools/kolibri/test_unpinned.py:44–97`. Benchmark runs delete their profile before starting (`tools/kolibri/bench_unpinned.py:167–169`), so repeated benchmark processes do not substitute for profile replay.

2. **Important: the registered test fails on default gfx906 wave mode.**  
   CMake registers it without a wave-mode override (`CMakeLists.txt:1334–1342`), while every parity case requires the selected API to return true (`tests/kolibri/mmvq_selected_test.cpp:119`). The API correctly returns false there (`src/kernels/cuda/native_mmvq.cu:3454–3455`). Add separate default-wave refusal coverage and CUDA-layout parity with `STRATA_MMVQ_WAVE=0`.

3. **RED and runtime test evidence are incomplete.**  
   The ledger does not record the intended synthetic RED failure. The summary reports GREEN, but the retained latest CTest log contains a missing-library failure, not the documented device-function failure (`build-kolibri-hip-tests/Testing/Temporary/LastTest.log:11`). Preserve explicit synthetic-pass and device-1 rerun logs.

4. **Builds are supported; cross-backend qualification is not.**  
   Ninja logs show main and standalone links, including both in the HIP test tree (`build-kolibri-hip-tests/.ninja_log:699–700`). CUDA/SYCL are correctly marked **NOT RUN**, not passed.

**Close-out recommendation:** retain the experiment provisionally, fix registered-Q6 refusal and dispatch-aware tests, complete the missing regression evidence, then finalize KEEP. Task 3 remains a prospective named consumer—not yet demonstrated consumption of this enabler.
