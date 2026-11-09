**KEEP is not yet justified under the plan’s acceptance rules.** The narrow implementation looks correct, but repeatable speedup is unproven and the required ordering test remains missing.

Read-only review; no tests rerun. Artifact paths below are relative to `.superpowers/sdd/2026-11-08-kolibri-exact-followup/`.

## Important

### 1. Performance evidence is inconclusive, not an established win
**`task5/summary.md:30–37`**

The paired decode gains are approximately **+12.9%, −6.1%, +6.8%, +6.7%, −6.6%**. The reported median is correct, but three positive pairs with this spread do not establish repeatability. Cache8 supplies no corroborating improvement.

The wall-time reduction is not independent support: much of the variation occurs outside the changed decode region. For example:

- `task5/t5-512-off-1.json:37–50`: startup **188.2 s**, decode **12.35 s**.
- `task5/t5-512-on-1.json:39–52`: startup **115.1 s**, decode **10.94 s**.

Removing decode drains cannot explain that startup improvement. Stabilize loading/cache conditions and counterbalance pair order before attributing the wall median to this change.

The plan explicitly says **noise is inconclusive**, and KEEP requires correctness **and** demonstrated speed (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:32–36`). These results neither prove a gain nor prove no gain.

### 2. Full-logit equality does not replace the required delayed-ordering test
**`task5/summary.md:34–35`; `tests/kolibri/decode_order_test.cpp:13–21`**

The host test checks eligibility, not execution dependencies. The ten finite, identical dumps are valuable regression evidence (`task5/integration-compare.txt:8–21`), but do not deliberately stress delayed producers or demonstrate fallback behavior at the actual drain sites.

Task5 explicitly requires delayed gate/up → SwiGLU → down → shared → sum checks (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:172`).

**Acceptable as a disclosed incomplete experiment; not acceptable as satisfying that acceptance step.** Add the GPU test or obtain an explicit waiver rather than treating equality as its replacement.

## Low

### 3. Trace semantics are corrected, but references and coverage wording remain stale
**`task5/summary.md:5–9,22`**

Current drain locations are **`src/kolibri/kolibri_engine.cu:3213–3214` and `:3255–3256`**, not 3202/3243. Shared work now runs at **`:3410`**, followed by the sum at **`:3411`**.

The source comment at `:3256` still incorrectly describes shared-expert work as already completed.

“Full gate table”/“exhaustive” overstates the test: it covers the supported case plus six single-condition exclusions, not all 64 Boolean combinations. This is not a defect in the simple conjunction.

## Correctness confirmations

- **The narrow gate is correct for the reviewed legacy-default-stream configuration.** The predicate is explicit at `include/strata/kolibri/decode_order.hpp:9–11`; runtime inputs are supplied at `src/kolibri/kolibri_engine.cu:3090–3092`. Every excluded condition makes both guarded drains execute.
- **The corrected lifetime explanation is accurate.** `g_q8` is device scratch (`:1910`); quantization/MMVQ use `qst` (`:1954–1977`). Legacy MMVQ constructs weight pointers synchronously; selected MMVQ constructs a by-value parameter block (`src/kernels/cuda/native_mmvq.cu:3465–3501`). Reassigning `g_ids_host` therefore needs no GPU-completion fence.
- **The pool does not consume `ey2`.** It reads activations/weights and writes `ey_h` (`src/kolibri/kolibri_engine.cu:2540–2552`), subsequently uploaded at `:3236`.
- **Reader/readiness and staging-source protections remain present**, including `:3158–3159`, `:3183–3202`, and `:3210–3211`.

**No new reachable unordered-consumer path found in the reviewed configuration.** `g_st` is created blocking (`:6984`); slot/overflow SwiGLU and the final sum use `st == 0` (`:3207`, `:3250`, `:3070–3073`). Legacy ordering also covers the zero-overflow memset (`:3244`) and shared-expert outputs.

That conclusion requires genuine **legacy** default-stream semantics—not merely a null handle under a per-thread-default-stream build. The reviewed HIP flags contain no such override (`build-kolibri-hip/build.ninja:1615–1619`); CUDA/SYCL remain unqualified.

**Bottom line: correctness looks sound for the narrow slice; KEEP as a completed, measured optimization is premature.**
