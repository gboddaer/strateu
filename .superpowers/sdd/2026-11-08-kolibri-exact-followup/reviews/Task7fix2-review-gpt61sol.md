**Not yet. BLOCKED is reasonable for capture/the proposed wait-free path, but the evidence does not justify closing Task 7 as plan-complete.** The prototype is useful progress; forced-hit/miss translation and the ownership contract remain unfinished.

Paths below are under `.superpowers/sdd/2026-11-08-kolibri-exact-followup/` unless otherwise stated.

### 1. Measurement: populated now, but still perturbed

The probe in `task7/device-slot-prototype.diff` correctly times the actual router-copy call without the immediately preceding synchronization. **0.567 ms/layer is therefore a real measured blocking interval**, unlike the previous unpopulated counter.

However:

- `src/kolibri/kolibri_engine.cu:2955` and `:3401` still synchronize at layer entry/exit under `DEC_PHASES`. The execution schedule remains instrumented.
- The copy waits for queued attention/router computation as well as transferring IDs. Its entire duration is **not removable copy overhead**: device dispatch must also wait for its GPU inputs.
- `dpt` is not advanced after the new measurement, so `d2h` is **nested within `post`**, not an additional phase.
- `task7/decphases3-512.log` records **8805.726 / 128 = 68.79 ms/token**, not ~77 ms/token. Using another run’s denominator does not establish a measured 36% share.

Correct wording: **“The instrumented router-copy call blocked for 0.567 ms/layer on average; potential end-to-end savings are unmeasured.”**

### 2. Important: the prototype does not satisfy forced-hit/miss dispatch

The diff demonstrates exact MMVQ **given an already selected device slot array**. That is a valid arithmetic/address-selection prototype.

It does **not** implement:

- GPU translation of routed expert IDs through an expert→slot map;
- resident/missing classification;
- zero/spill-slot dispatch for misses;
- packing/readiness/fallback integration.

The test uploads CPU-selected slots, then reverses and uploads them again. This establishes changed-slot consumption—not admission, eviction, generation safety, or miss handling. A missing-map sentinel such as `-1` cannot be passed directly to the prototype: its unsigned offset conversion would produce an invalid address.

Thus the synthetic exactness claim is supported in scope; **the plan’s forced-hit/miss requirement remains unmet**.

### 3. Important: partial residency is a constraint, not yet a demonstrated blocker

The memory argument convincingly rules out **all experts resident under the prescribed budget**. It does not rule out an offline partial-residency prototype.

The plan explicitly anticipates misses and permits leaving/refreshing capture. It does not require eliminating every host wait before testing dispatch.

Likewise, needing a generation/reset contract is unfinished design work—not evidence that prohibited server execution is necessary. Existing ownership points are accessible offline:

- `np_admit`: `src/kolibri/kolibri_engine.cu:2926`;
- promotion/map changes: `:5978–6034`;
- reset: `:5372`.

A bounded prototype can initially retain existing fences and host miss handling. Its contract should specify old-reader completion, weight readiness before map publication, admission/promotion updates, and drain/reset/destruction behavior. Request reset also needs an explicit decision about whether residency survives.

**An offline translation/miss/lifecycle test is still owed.** Full engine performance integration is necessary for KEEP, but neither server execution nor capture is mandatory merely to finish this bounded feasibility investigation.

### 4. Other evidence/reporting corrections

- The supplied Task-7 logs show **326 slots/layer**, not 334. Supply the separate 334-slot evidence or correct the statement.
- Preserve a Task-7-specific prototype test transcript and executable identity. A diff plus “passes” is weaker than an attributable run artifact.
- “Cannot be connected … offline” overstates what was demonstrated. **“Not connected or lifecycle-qualified”** is supported.
- The inspected source no longer contains the experimental API/probe identifiers. That supports removal, though this was static inspection—not a Git-diff certification.
- **Capture unattempted, CUDA/SYCL NOT RUN** are appropriate qualifications.

**Bottom line:** Do not approve BLOCKED as the completed Task-7 end state yet. Finish—or explicitly defer as unperformed—the ownership contract and bounded forced-hit/miss translation/lifecycle experiment. BLOCKED becomes justified if that work identifies a concrete prohibited dependency or unsafe required waiting mechanism; the current missing implementation alone does not establish one.
