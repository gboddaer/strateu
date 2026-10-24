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

## per-layer promotion budgets: measured null at 512 tokens, reverted; five levers now dead at short context

*Replayed from worktree commit `86da7f9e`.*


## The fifth dead lever: per-layer promotion budgets do not pay (and the fill is not where the time is)

C0's 24%-98% per-layer coverage spread suggested a data-driven promotion rule: in `seed_hot`, stop promoting once an expert
would add less than a share of that layer's rows, so low-skew layers fill fewer slots and high-skew ones keep theirs.  Swept at
512 tokens, two runs each, against the stock uniform budget:

| promotion rule | wall | tok/s |
|---|---|---|
| stock (uniform `RES_TOP=24`) | 3,469 / 3,488 ms | 147.6 / 146.8 |
| marginal share >= 0.5% | 3,404 / 3,431 ms | 150.4 / 149.2 |
| marginal share >= 2% | 3,476 / 3,480 ms | 147.3 / 147.1 |

All inside the run-to-run spread.  Reverted rather than kept: a cache-policy knob that buys nothing is complexity with a
maintenance cost and a numerics surface (it changes which experts are resident, hence which arm serves a pick).  With this,
**five levers are measured dead** at short context - `DENSE_F16`, the router path, `RES_TOP` (both directions), the
marginal-share promotion rule, and re-scoping the overlap - and the only surviving wall-clock facts are the pool's 1.2 s of a
3.4 s wall and the 3.9x that defaulting `POOL_ALL` recovered.

`pp short` remains **150 tok/s against a floor of 200**.  Everything cheap has been tried and recorded; what remains is kernel
work whose size cannot even be located with the profilers available on this box, because both of them perturb the workload
they measure (nsys by 15x on a 15k-launch workload, the phase marks by doubling the pool's chunks).

## serve path at the pp-short shape: cold 150.6 / warm 176.2 tok/s, with cache_n 0 (one-time state, not KV reuse)

*Replayed from worktree commit `9f892bc8`.*


## The serve path at the floor's shape: cold 150.6, warm 176.2 tok/s

Three identical 415-token chat requests through `run-kolibri1.sh` (ctx 8192, shipping config), reading the server's own
`timings`:

| request | prompt_ms | prompt tok/s |
|---|---|---|
| 1 (cold, first after load) | 2,756.3 | **150.6** |
| 2 (warm) | 2,355.5 | **176.2** |
| 3 | 2,430.1 | 170.8 |

with `cache_n = 0` on all three, so this is **not** KV reuse - it is one-time process state (the expert-cache fill, cuBLAS
handles, the pinned-table warm-up) amortising.  That matters for how the pp-short floor is read: the benchmark is a cold
process, while the serve path - which is what the box is actually for - pays that once and then runs at ~176 tok/s.  Still
short of 200, but the gap in the *use case* is 12%, not 33%, and it is the same fill that C1's blob residency would shrink 3.5x.

Also worth recording: the `prompt_ids` field is not accepted as a prompt by `/v1/chat/completions` (a request carrying 500 ids was
answered as a 45-token chat prompt), so prompt-shape measurements through the API need real text.
