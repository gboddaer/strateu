## Verdict

**Narrow the task; do not mark Task4 KEEP/complete yet.** The baseline-profile results are promising, and I found no demonstrated valid-map arithmetic or overwrite race in the current execution path. However, the required demand comparison and common qualification gates are missing. Keeping this as an isolated, default-off experiment is reasonable; accepting the claimed scope is premature.

Read-only inspection only: I did not execute tests, recompute binary hashes, or obtain an executable `git diff` with the supplied tools.

## Findings by severity

### Important — Neither required profile control was measured

`src/kolibri/kolibri_engine.cu:321–333` makes an unset `STRATA_KOLIBRI_NOPIN_PROFILE` select **baseline**, not demand.

The recorded runs have empty inherited settings and either no override or only the admission flag:

- `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task4/t4c-65537-off.json:28–29`
- `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task4/t4c-65537-on.json:28–31`

Thus the supplied traffic ratios establish **baseline → baseline+admission**, not improvement over demand. Both demand and balanced controls are absent, although Task4 explicitly requires both (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:163`).

This matters because demand already changes startup population, transfer scheduling and repeated staging. The new path’s incremental benefit over that existing optimization remains unknown.

### Important — KEEP bypasses the written common gate

The summary explicitly acknowledges no five-pair alternating series (`task4/summary.md:42–45`), but still concludes KEEP (`:52–53`).

Missing qualification includes:

- Five alternating pairs on **512/128 and cache8**, required at plan `:32`; the supplied matrix’s four decode steps are not substitutes.
- Repeated-pack measurements required at plan `:163`.
- Evidence of the full HIP test-tree and main+standalone builds required at plan `:38`; the summary establishes only main build and host policy test (`task4/summary.md:49`).
- An observed failing Task4 RED test is not documented.

The 16K/65K results justify further qualification, not waiver of these gates. Eight decode steps also do not establish generally unchanged decode performance, particularly with the observed 16K regression.

### Medium — Transfer-ring eviction integration is unqualified

`src/kolibri/kolibri_engine.cu:2953–2969` supports the separate transfer stream, but the recorded timing configurations select baseline and therefore ring-off (`:335–344`). They exercise the synchronized `hstage` path, not admission through the ring.

The generic transfer implementation looks consistent with the proposed ordering, but the new integration lacks evidence covering ring pageable/pinned modes, delayed old readers, and repeated victim reuse. Either qualify those combinations or narrow the supported experimental configuration.

### Medium — Duplicate slot aliases silently violate the planner’s protection promise

`include/strata/kernels/kolibri_hot_policy.hpp:129–146` validates slot ranges but silently retains only the first occupant of a duplicated slot.

Counterexample:

```text
experts=3, slots=1
map={0,0,-1}
requested={1,2}
```

The planner treats expert 0 as the unrequested victim and admits expert 2 into slot 0. Expert 1 remains mapped to slot 0 despite being requested. Applying the plan at `src/kolibri/kolibri_engine.cu:2951–2955` leaves its map pointing at expert 2’s weights.

This requires an already-invalid map; I found no normal-path producer of that alias. Nevertheless, reject aliases or explicitly document the injective-map precondition and assert it. Do not describe protection as unconditional for arbitrary accepted maps.

### Medium — Counters are narrower than total traffic, and omit legacy admission

The new counters correctly count submitted padded blob bytes in their respective paths:

- Admission: `src/kolibri/kolibri_engine.cu:2962–2965`
- Nonresident staging: `:4089–4097`

But free-slot admission at `:2974–2990` is not counted by either. Startup population and hot promotions are also outside those counters. Consequently, their sum is **not a general total-prefill-upload counter**, especially for the missing demand comparison.

The reported 4.2×/9.4×/15.8× ratios are credible as these particular **staging-versus-admission categories**, not total process traffic reductions. Actual repeated-pack counts were not supplied.

### Minor — Planner tests overstate edge-case coverage

`tests/kolibri/hot_policy_test.cpp:227–230` calls K5 “no unprotected victim,” but requests only resident experts: there is no missing candidate to deny.

Also absent from case K:

- Invalid map slots, including values below `-1`.
- Missing candidates while every slot is protected.
- Mixed holes/victims with a small budget and slot reordering.
- Duplicate-slot-map rejection.

Budget zero returns before request/map validation (`kolibri_hot_policy.hpp:123`); that behavior should be explicitly understood rather than claiming universal out-of-range rejection.

## Technical checks that look sound

- **Planner, valid injective map:** Duplicate requests are counted; requested residents are protected; candidates are count-descending/expert-ascending; destinations are unique and capacity-bounded. **Sorting by slot is safe:** budget truncation occurs first (`kolibri_hot_policy.hpp:160–164`), and the engine executes the entire returned plan.
- **Reader lifetime:** The prefill caller drains `g_st` before admission (`kolibri_engine.cu:4071–4072`); nonempty plans additionally drain `st` and, if different, `g_st` (`:2946–2949`). In the supported single-owner execution, old readers finish before transfer-stream overwrites.
- **Weight readiness:** Ring uploads are fenced onto consumer streams (`:2967–2969`); `hstage` uploads complete before reuse (`:2957–2959`). Early host-map publication is safe under this serialized, fatal-error model—not a recoverable transactional update.
- **Layout and bounds:** Admission uses the existing `pack_blob` and stride. Planner destinations exclude zero/spill slots. Excess requests retain the explicit bounded staging check (`:4089`). Graph/table modes do not enter this new eviction branch.
- **Flag off:** The eviction branch is disabled. The unconditional prefill call still constructs a request vector and, with lazy caching enabled, performs legacy free-slot admission (`:2938–2939`, `:2974`). It is therefore correctness-harmless under existing ownership assumptions, but not universally a no-op or zero overhead. I cannot certify historical call-site equivalence without the base diff.
- **“Zero staging”:** Credible for the instrumented `t4c` runs: every no-pin staging copy crosses the increment at `:4091`, and output suppression at `:5462` explains the absent line. It means **zero fallback-staging copies**, not zero packing or H2D traffic.
- **Exactness:** The stored reports support the claimed full-logit comparisons, including ten finite complete integration dumps (`task4/integration-compare.txt:6–19`). This is encouraging but does not cover all supported transfer configurations.

**Bottom line:** retain the prototype provisionally, qualify demand/balanced and transfer-ring configurations, and complete the common gate before declaring KEEP. No evidence currently warrants rejecting the core admission algorithm outright.
