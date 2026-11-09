**REJECT is justified for this implementation—but the recorded explanation is wrong. The intended zero-buffer replacement was not actually tested.**

### 1. Genuine correctness failure: overwritten pointer

**Critical:** `task10/zero-buffer.diff:18` sets:
```cpp
if (s.zero_yc) yc = s.zero_yc;
```
But the unchanged code immediately downstream unconditionally executes:
```cpp
yc = s.ey;
```
See `src/kolibri/kolibri_engine.cu:3241–3244` in the reverted tree. The diff does not remove that assignment.

Consequently, enabling the gate removes the memset **but still passes `s.ey` to the sum**. The allocated zero buffer never reaches `s.yc`. Stale overflow outputs can therefore be added to resident outputs.

This is a concrete implementation bug consistent with deterministic divergence, not evidence of a measurement artifact. The JSON records show the same executable hash, identical forced continuations and unchanged fitted capacity. Differing printed `gen:` tokens do not imply differing input histories: the engine feeds the forced tokens (`kolibri_engine.cu:7504`).

### 2. No evidence of a dependency beyond the `s.yc` read

The summary’s “load-bearing beyond the value read through `s.yc`” conclusion is **unsupported**. The sum still reads the original, now uncleared buffer through `s.yc`; that alone explains the failure.

Correct the conclusion to:

> The candidate accidentally retained `s.ey` as the sum operand after removing its zeroing. This experiment rejects the implementation, not the immutable-zero-buffer approach.

Also replace “not a race” with “repeatable across two candidate runs.” Determinism alone does not exclude races, although this pointer bug supplies a straightforward explanation.

### 3. All-resident infeasibility is sound for this configuration

42.90 GiB is approximately **43,930 MiB**, already about **4,070 MiB above** the recorded 39,860 MiB free—before preserving the runtime reserve or allocating cache overhead.

The 326-of-384 slots/layer result corroborates that. Scope the statement to this device and tested configuration (`ctx=131072`, reserve 1536 MiB).

Clarify that **37.20 GiB includes zero/spill slots**; the log separately reports **36.42 GiB resident expert payload**. Also, skipping overflow requires all **current picks** to hit, not the entire expert table to be resident.

### 4. Preserve and correct the record

Keep:
- The exact diff **and surrounding unconditional `yc = s.ey` assignment**.
- All dumps, JSON, input fixtures, profiles, logs and executable/source identities.
- Correct skip counts: **off 6114; on 6026**. The repeat had statistics disabled.
- Cache8 as a **zero-skip negative control**, not qualification of the replacement.
- The effective skip/profile/poison settings; the driver’s environment allowlist currently omits `STRATA_KOLIBRI_SKIP_ZERO_OVERFLOW`.
- Explicitly unrun targeted lifecycle tests and CUDA/SYCL status.

Correct both `task10/summary.md` and `progress.md`. No performance claim or general rejection of zero buffers follows from these runs.

**Bottom line: remove this gate/allocation and REJECT the patch. Do not preserve the claimed “beyond `s.yc`” finding as established fact.**

*Read-only review; dump hashes/error magnitudes were inspected through recorded metadata, not independently recomputed from the binary dumps.*
