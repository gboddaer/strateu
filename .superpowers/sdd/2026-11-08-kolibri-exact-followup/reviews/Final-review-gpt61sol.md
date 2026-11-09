**Experimental retention is justified, but I would not close the plan as written yet.** No retained-code conflict found; the remaining blockers are reporting and evidence bookkeeping.

Paths below are relative to `.superpowers/sdd/2026-11-08-kolibri-exact-followup/`.

### 1. Composition: correct, with two clarifications

- KEEP: Task 0 `a7574911`, Task 2 `33ed811a`, Task 10 `5aaee42b`.
- REJECT: 3/4/5/6/8/9; BLOCKED: 7.
- Task 9’s **standalone diagnostic probe remains retained**, despite rejecting a setting/engine change.
- Remove Task 2’s stale **KEEP-ENABLER for Task 3** designation from `progress.md:29,88`; Task 3 was rejected. Task 2’s standalone KEEP remains justified.

The final three-flag composition is correctly stated.

### 2. No Task 2/Task 10 conflict found

Static inspection supports safe separation:

- Selected MMVQ writes ordinary expert-output scratch.
- The zero buffer is separately allocated and only read as the skipped-overflow operand.
- `zero_used` is local and applied **after** the formerly problematic `yc = s.ey`.
- Consumption requires no-pin, enabled skipping, zero misses and poison disabled (`src/kolibri/kolibri_engine.cu:3232–3245`).

**No consumption path without enabled skip-zero was found.** However, skipping may be enabled through an existing profile without explicitly setting `SKIP_ZERO_OVERFLOW=1`. Explicit `=0` overrides that. `ZERO_BUFFER=1` alone still allocates memory, even when unused.

Dedicated miss→hit→miss, poison, mixed-format and flag-matrix tests remain unrun; disclose that in the final summary, not only `task10/summary.md`.

### 3. Build and long-context numbers: supported, narrowly scoped

The build logs and selected ctest transcript support the stated results: **full builds, three selected tests passed—not the entire runtime suite**.

The 131064/8 JSONs record the same executable hash, identical forced histories, the stated timings and identical full SHA:

`5c85e7f6569fea0c8a7162a9b84ee8e23e0dc26e283e539abe3cb605f000a322`

“+13.6% decode in one pair; no repeatable speed claim” is appropriate. These dumps contain **eight complete vocabulary distributions**, not logits for every prompt position. This is local regression identity, not independent model-quality qualification.

CUDA/SYCL NOT RUN and no default promotion are correctly limited.

### 4. Finalization remains incomplete

**Important:** `progress.md:35–38` still marks reviews pending; Task 2 still says its third review is requested. Review artifacts exist, but the required **per-finding resolutions and final review disposition are incomplete**.

Before closure:

- Record every finding as fixed/rejected/deferred with evidence or dependency.
- Record skipped qualification explicitly—especially Task 8 cold-4K/lifecycle coverage and Task 10 targeted tests.
- Correct Task 8’s unsupported IO-causality/coverage claims and Task 9’s contradictory SDMA ordering in the ledger.
- Preserve build/setup/ctest evidence somewhere durable, rather than relying only on `/tmp` and mutable build logs.
- Strengthen `final/integration/compare.txt` with per-case hashes, sizes and explicit reference paths/provenance.
- Save this final review and its resolution.

### 5. Unsupported summary wording

`final/summary.md:19–24` overclassifies failures:

- These are **three failing setup scripts**, not three individual environment failures. `test_setup_pins.py` has three assertion failures; its environmental cause is not established.
- Terminal includes a process-discovery assertion (`test_procs`), not solely missing component/server fixtures.

Use “failures outside the Kolibri engine tests; suite not green; causes not all established.” Avoid claiming baseline-confirmed unrelatedness without a baseline run.

**Verdict:** keep the retained code experimentally, default-off. **Not ready to close as-is; ready after the ledger, qualification-limit and evidence corrections above.** No default or release-readiness approval. Read-only review; no tests or dump hashes independently rerun.
