## Verdict

**No remaining Important finding or new functional defect found. Only a cosmetic reporting inconsistency remains.**

Read-only review; no tests rerun or binary hashes independently recomputed. Below, `T/` means `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task2/`.

### Minor — stale cache8 percentage

`T/summary.md:50` still says **+0.7%**, contradicting the corrected **+0.47%** at `:53`. Consequently, `progress.md:109` overstates reporting closure. Replace the stale percentage; the **+2.8% paired median** and **inconclusive** conclusion remain correct.

### Confirmed closures

1. **Integration evidence:** `T/integration-compare.txt:13–22` independently records all ten comparisons, each with 2,048,000 bytes, 512,000 floats, finiteness, SHA-256 and equality. Input/profile identities appear at `:23–28`; model/executable identities at `:3–6`.

   The historical executable hash is **defensible as recorded provenance**: `:3–5` explicitly distinguishes the integration binary from the later Q6-fix binary and explains applicability plus the current-binary 512/128 rerun. It does **not** establish that the whole matrix ran on the fixed binary. I cannot independently authenticate the historical hash with the available tools.

2. **Packed-Q6 sentinel:** `tests/kolibri/mmvq_selected_test.cpp:226` unsets the environment flag; `:180–186` registers only the later selection; `:188–191` asserts refusal and unchanged output.

3. **gfx906 isolation:** `CMakeLists.txt:1343–1347` forces parity to wave `0`, refusal to wave `1`, and registers refusal only under `STRATA_HIP_GFX906`. Runtime qualification remains explicitly NOT RUN (`T/summary.md:66–67`).

4. **Watchdog:** The historical 300-second deviation and absence of reruns are honestly recorded (`T/integration-compare.txt:9`, `T/summary.md:77–78`). The new **10800-second default** is correctly passed to each subprocess (`tools/kolibri/test_unpinned.py:18,31`).

5. **Scope:** Apart from the stale percentage, cache8 remains correctly inconclusive (`T/summary.md:52–54`). KEEP remains **local W7800 decode**, not an end-to-end or global promotion (`progress.md:88`; `T/summary.md:54,88–89`).

**Evidence closure is sufficient; fix the cosmetic percentage before calling the reporting fully consistent.**
