**Yes—KEEP as an experimental opt-in is justified. But Task 10 is not fully qualified or complete under the written plan.** The corrected code looks sound; targeted regression tests and stronger timing evidence remain outstanding.

### 1. Corrected pointer path: sound

In `src/kolibri/kolibri_engine.cu:3205–3249`:

- `zero_used` starts false for each layer invocation.
- It becomes true only inside the legally skipped overflow branch.
- The override **after** `yc = s.ey` now survives into `s.yc`, which `k_moe_sum2` consumes.
- Misses and CPU-pool work retain their original output pointers.
- Poison mode disables skipping.

One scoping clarification: consumption requires the skip path to be **enabled**, not necessarily an explicit `SKIP_ZERO_OVERFLOW=1`: `np_option()` also permits profile-based enablement. Also, `ZERO_BUFFER=1` allocates the buffer even when skipping is disabled; only its **use** is skip-scoped.

### 2. Evidence: encouraging, not conclusive

The corrected hash and recorded 10/10 finite, identical integration comparisons are strong regression evidence. Three positive timing pairs justify continuing a small, isolated experiment—not treating +5.6% as an established speedup. The +23.9% pair is conspicuously variable.

The plan explicitly requires five alternating pairs on **both** 512/128 and cache8. Those gates remain incomplete.

**Important evidence correction:** the summary’s **6026 on-arm skips belong to the broken implementation**:

- `task10/t10-512-on.json` records 6026 skips, hash `8da2091d…`, binary `48d0367e…`.
- `task10/t10-512-on-fixed.json` records the correct `6ffefa8f…` hash and binary `22403d46…`, but no skip statistics.

Do not attach that old counter to the corrected result. Capture corrected diagnostic counters separately from timing. Also record `SKIP_ZERO_OVERFLOW` in benchmark metadata; the current allowlist omits it.

### 3. Intended memset removal: yes, statically

`zero_yc` is a separate `cudaMalloc` allocation through `dalloc`, created before cache auto-fit. Its only subsequent use is as the read-only overflow operand. Writable expert outputs remain in `s.ey`/`s.ey2`.

Thus each qualifying skip avoids `cudaMemsetAsync(s.ey, …)` without aliasing writable output. `k_moe_sum2` and its operand order remain unchanged; both alternatives provide positive-zero bytes. I found no new lifecycle/race issue in this change.

Minor correction: initialization currently zeros twice—`dalloc()` already clears memory, then `init_scratch()` clears it again. That is startup redundancy, not a correctness problem.

### 4. Missing targeted tests

No dedicated zero-buffer regression test was found. Add:

1. **Miss → hit → miss:** write distinctive nonzero overflow outputs, take an all-hit skip, then miss again. Assert pointer selection, exact combined output, and unchanged zero-buffer bytes. Include cache-map changes and repeated layers/tokens. This should fail with the original patch.
2. **Poison-zero:** with both optimization flags enabled, assert poison disables skipping and the overflow kernels execute. Include a positive control proving poison reaches arithmetic; compare full outputs, not just greedy IDs.
3. **Mixed formats:** alternate Q4_K/Q6_K layers, including fallback paths, with hits and misses. Check finite, bit-identical outputs and zero-buffer immutability.
4. **Flag matrix:** ZERO_BUFFER off/on × skipping off/on, explicitly overriding profiles. Verify skipping-off never consumes the buffer.
5. **Mechanism check:** a diagnostic trace or counter should show one fewer memset per qualifying layer—not merely count skipped overflow work.

Also refresh `task10/zero-buffer.diff`: it still contains the broken pointer assignment.

**Bottom line:** retain it disabled by default, accurately labelled experimental. Do not claim an all-resident implementation, a proven +5.6% gain, or completed plan acceptance yet. This review inspected code and saved evidence; it did not rerun GPU tests.
