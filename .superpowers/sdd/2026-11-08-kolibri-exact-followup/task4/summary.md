# Task 4 - prefill cache admission

## Implementation

- `plan_prefill_admission(n_expert, slots_per_layer, map, requested, budget)` in
  `include/strata/kernels/kolibri_hot_policy.hpp`: counts duplicate requests, protects every current requested
  resident, ranks missing candidates by count then expert id, fills free slots first, then evicts only unrequested
  residents in slot order, caps at `budget`, and sorts by slot for a deterministic plan.
- `np_admit(..., prefill=true)` uses it behind `STRATA_KOLIBRI_PREFILL_ADMIT=1` when no-pin, hot, per-layer. It
  synchronizes readers before overwriting a victim, publishes the map, packs/copies each expert, and fences the
  copies before the layer consumes them. Without the flag the legacy free-slot-only path is unchanged.
- Counters: `CNT prefill admit experts/evictions/bytes` and `CNT prefill staged blobs/bytes`.

## Host tests

`kolibri_hot_policy_test` case K: duplicate counts, current-pick protection, requested set larger than capacity,
no unprotected victim, budget 0/2, deterministic expert/slot ties, out-of-range rejection. Passes.

## Full logits (complete dumps)

| input | off SHA | on SHA | staging -> admission |
|---|---|---|---|
| 4097 | `241969a0...791bf` | same | 19.78 GB -> 4.69 GB (4.2x) |
| 16385 | `2ccd3683...51abf` | same | 78.31 GB -> 8.35 GB (9.4x) |
| 65537 | `085e5575...0942e` | same | 313.35 GB -> 19.88 GB (15.8x) |

The on-runs recorded no staging counter, so every miss in these runs was admitted or hit. Integration
short/long/hot/profile with the flag: 10/10 complete 2,048,000-byte dumps byte-identical to the Task 2 matrix
(`task4/integration-compare.txt`).

## Wall and phase timings (paired, same binary and model, stats on)

| case | pair | wall off/on | prefill tok/s off/on | decode tok/s off/on |
|---|---|---|---|---|
| 4097 | 1 | 203.1 / 197.7 | 53.15 / 60.73 | 11.34 / 11.51 |
| 4097 | 2 | 174.9 / 259.0 | 57.27 / 54.91 | 11.78 / 11.16 |
| 16385 | 1 | 418.9 / 303.2 | 51.96 / 84.18 | 10.93 / 7.24 |
| 16385 | 2 | 314.1 / 297.2 | 77.55 / 86.60 | 11.45 / 10.55 |
| 65537 | 1 | 1509.3 / 1332.5 | 47.06 / 53.55 | 8.45 / 8.32 |
| 65537 | 2 | 1486.7 / 1342.6 | 47.28 / 53.50 | 8.34 / 8.25 |

The 4K wall result is noisy (one pair slower, one faster); admission traffic is still 4.2x lower. At 16K and 65K
the prefill rate rises consistently (11.7% to 62%) and wall falls in all four pairs (-5.4% to -27.6%), while decode
is unchanged within noise at 65K and can fall at 16K. No five-pair alternating series and no balanced-profile
control were run for this task; that is recorded as a gap, not a pass.

## Builds/tests

- HIP main build OK (`strata-kolibri`); host policy test passes.
- CUDA/SYCL NOT RUN. No server.

## Demand-profile control (decisive)

`STRATA_KOLIBRI_NOPIN_PROFILE=demand`, 65537/8:

| run | wall s | prefill tok/s | traffic | SHA |
|---|---|---|---|---|
| demand, flag off | 1246.0 | 52.73 | legacy admission 25.15 GB, no staging | `085e5575...0942e` |
| demand, flag on | 1263.2 | 52.02 | new admission 25.11 GB (0 evictions) + legacy 0.04 GB | same |

The demand profile already performs free-slot admission and reaches the same traffic result; wall differs by 1.38% in
one pair, not a repeatable regression. Zero evictions means this run qualifies free-slot admission, not the planner's
distinguishing eviction path. The defensible claim is **no demonstrated incremental gain in the tested cold-profile,
auto-cache, demand configuration**, not "no benefit anywhere". The plan forbids claiming balanced-to-demand savings as
a new gain; the same applies to baseline-to-demand.

Decision: **REJECT** the admission integration for insufficient incremental value, not for a correctness failure.
Remove the planner, engine wiring and Task4 tests; keep the evidence and the existing demand free-slot admission.
Not completed (not passed): balanced-profile controls, repeated-pack qualification, the five-pair alternating gates,
and eviction-lifecycle qualification (the demand runs made zero evictions). The 4K/16K/65K baseline reductions are a
genuine baseline-only benefit, recorded as reproducing what demand already provides, not as a new optimization.
