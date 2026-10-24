# Findings log — what was measured, and what was ruled out

This is the publishable half of the campaign ledger. Each entry below is one measurement campaign, replayed onto this
history from the private working ledger (which is not published, because it contains internal hostnames and machine
paths). The entries are kept in order because the *order* is the argument: several conclusions were corrected by later
measurements, and a summary would hide that.

Read `docs/strateu-campaign-report.md` first for the consolidated state; this log is the audit trail behind it.

## instruments cannot attribute pp-short: nsys inflates 15x on 15k launches and the phase marks double the pool's chunks; only wall-clock A/B survives

*Replayed from worktree commit `70a24da3`.*


## Instrumentation cannot attribute pp-short's remaining 1.33x - both instruments perturb it

Two attempts to split the 512-token wall, both self-defeating:

* **nsys** (`--trace=cuda`): 10,258 `dequant_wide_kernel` launches, 97% of them `grid=640` (one 2560x512 / 512x2560 matrix
  each), which *looked* like 1,087 ms = 60% of kernel time. But the traced run spanned **50.5 s against a 3.4 s wall** (15x
  inflation, because per-kernel overhead scales with the 15,590 launches), so those per-kernel times are launch-overhead
  artefacts, not compute.
* **the engine's own phase marks** (`STRATA_KOLIBRI_ATTN_PHASES=1`, which `cudaStreamSynchronize`s at every mark): per layer
  `pool=24.1 experts=15.1 post=4.7 dense=1.5 attn=1.0 ms`, i.e. the expert legs are 84% of the layer - but the same run reports
  `chunks=100 bytes=41 GB promotes=1986` against the un-instrumented `chunks=50 bytes=19 GB promotes=1004`, so enabling the marks
  **doubled the pool's work** by breaking the schedule it was measuring.

**What survives, because it is wall-clock A/B and nothing else:** the pool leg is **1.2 s of the 3.4 s wall** (measured
un-instrumented, 19 GB at 15.6 GB/s); `POOL_ALL=1` is 3.9x at this length; `RES_TOP=24` is optimal (8 is -20%, 56 is flat);
`DENSE_F16` is OOM-with-cache and exactly neutral without it; the router path is a null; and `ATTN_MB=1024` does not fit at 100K.

**Conclusion for the floor.** `pp short` is 150 tok/s against 200, and the remaining 1.33x is ~2.2 s of the wall that no
non-perturbing instrument currently attributes. The candidate levers are all kernel work of unknown size (a fast Q4_K-from-VRAM
GEMM, cheaper cache fills, fewer/bigger dequant launches), i.e. the kind of change that must be *built* and A/B-measured rather
than located by a profiler at this workload size.  That is the honest state, and it is the point at which continuing costs more
than the floor is worth without a direction from the goal owner.
