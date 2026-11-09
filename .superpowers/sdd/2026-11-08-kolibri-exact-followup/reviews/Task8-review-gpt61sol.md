**REJECT is justified as an early abandonment of this prototype: no demonstrated gain and no retained consumer. It is not proof that bounded packing cannot help cold 4K.** The evidence summary needs corrections.

Paths below are relative to `.superpowers/sdd/2026-11-08-kolibri-exact-followup/`.

### 1. Test adequacy — useful smoke test, incomplete lifetime qualification

**Medium — overstated coverage.** `task8/byte_pack_test.cpp:32–54` checks bytes, gaps, repetitions and lanes 1–4, but:

- The source overwritten after return is **`a`, always copied by caller lane 0**, not a worker (`:43`). It therefore does not directly stress worker-source lifetime.
- Every destruction occurs after synchronous `pack()` returns. This tests idle worker shutdown, **not destruction draining outstanding work**.
- No delayed-worker synchronization test or exception/failure injection exists.
- Only the last suffix byte is checked, rather than the entire suffix (`:41`).

For retention, add a deterministic delayed-worker test, overwrite/release worker-owned sources after return, and test construction failure cleanup. For removal, record these as **not qualified** rather than expand the test.

### 2. Performance — credible observation, unproven explanation

The four JSON/log pairs support a meaningful controlled comparison: same executable hash, pack counts, bytes, residency behavior and complete-logit hash. Wall time increased **10.6% and 8.4%** with packing enabled.

However:

- Cache/IO conditions changed substantially between runs; even off-arm startup packing doubled.
- All measurements used diagnostics, not the planned stats-off timing gate.
- Fault and input-block counts establish substantial filesystem activity, **not that contention caused the regression**. Synchronization overhead, scheduling and uneven lane work remain possible contributors.
- With two lanes, caller lane 0 copies **gate plus down**, while the worker copies up. This is not balanced three-way copying.

**Minor reporting error:** `task8/summary.md:24` says startup packing was “27–40% slower.” The actual increases are approximately **26.7% and 5.9%**.

Use: **“No benefit observed in two diagnostic 512/128 pairs under uncontrolled filesystem-cache conditions; mmap/IO contention is a hypothesis.”**

### 3. Cold 4K — explicitly planned, not completed

**Medium — plan deviation.** `docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:205` explicitly requires testing the cold 4K/source-fault workload. These 512-token runs do not satisfy it.

Also, `:203` specifies **two persistent workers plus caller**, whereas the measured two-lane configuration has **one worker plus caller**.

Therefore:

- To claim Task 8’s planned evaluation was completed, the cold-4K comparison is missing.
- **You can still reject/remove now**, provided the ledger explicitly records early termination and the untested workload/configuration.
- No 131K run is necessary merely to abandon the prototype.
- The wider integration matrix and five-pair stats-off gate were also not completed; do not imply otherwise.

### 4. Correctness risks worth preserving

These are static findings, not observed failures:

- **Important — constructor exception safety:** `task8/byte_pack_pool.hpp:24`. If worker creation/allocation throws after an earlier worker starts, unwinding destroys joinable `std::thread`s and can terminate the process. There is no cleanup/fallback.
- **Important if reused concurrently — single-caller contract:** `:49–58`. Concurrent `pack()` calls overwrite shared descriptors, generation and pending count. The mutex does not serialize entire operations. Document exclusive invocation.
- **Lifecycle limitation:** `:26–33,70`. Shutdown prioritizes `quit_` over unconsumed work. It is not a supported concurrent drain protocol; the `quit_` fallback in `pack()` cannot make calls during destruction safe.
- **Medium — sticky global configuration:** `task8/experimental-pack-pool.diff:65–73` creates the global pool for enabled loads but never clears it for a subsequent disabled load. Reload/off behavior is unqualified.
- **Minor — zero-length pointers:** empty-vector `data()` may be null; unconditional zero-byte `memcpy`/`memcmp` calls should be avoided for portable C++ correctness.

Under **one caller, valid nonoverlapping spans, and no concurrent destruction**, the worker-completion mutex/condition-variable ordering appears sufficient before DMA. I found no demonstrated normal-path byte corruption.

**Bottom line: remove the pool/wiring/test and preserve evidence. Record REJECT as “no demonstrated benefit; evaluation stopped early,” correct the percentage and coverage claims, and leave cold-4K effectiveness unresolved.** I reviewed files only; I did not rerun tests or recompute dump hashes.
