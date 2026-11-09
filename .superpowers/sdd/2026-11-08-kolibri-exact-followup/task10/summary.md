# Task 10 - zero-buffer fast path

## First attempt was wrong (review 1)

The first gate set `yc = s.zero_yc` inside the skip branch, but the unchanged downstream line still executed
`yc = s.ey`, so the zero buffer never reached `k_moe_sum2`; the run removed the memset and read stale `s.ey`.
Logits diverged from token 1 (`8da2091d...`). That was an implementation bug, not evidence against a zero buffer.

## Corrected implementation

`bool zero_used` is set only when the skip branch uses the dedicated buffer, and after the branch's unconditional
`yc = s.ey` the code applies `if (zero_used) yc = s.zero_yc`. `Scratch::zero_yc` is one K*N device allocation,
zeroed once at startup, never aliased with a writable expert output. Gate `STRATA_KOLIBRI_ZERO_BUFFER=1`, default off,
and it only takes effect with the existing `STRATA_KOLIBRI_SKIP_ZERO_OVERFLOW=1` skip path.

## Correctness

- 512/128, skip path active: both complete dumps SHA
  `6ffefa8f65b3cb8ca451ae77f23ee05ba197bd23e8e25febbf8121d806fa8721`; the corrected on-arm stats record 6114
  skips (the earlier 6026 counter belongs to the broken pointer run and is not used here).
- Cache8 (zero skips) also matched `466f48f2...`.
- Full integration with both flags, 10/10 short/long/hot/profile dumps byte-identical to the Task 2 control;
  `task10/integration-on/compare.txt`.

## Timing (512/128, five alternating pairs, skip path on in both arms)

| pair | off decode tok/s | zero-buffer decode tok/s | delta |
|---|---:|---:|---:|
| 1 | 13.82 | 17.13 | +23.9% |
| 2 | 13.79 | 14.14 | +2.5% |
| 3 | 13.87 | 14.65 | +5.6% |
| 4 | 13.59 | 14.14 | +4.0% |
| 5 | 12.83 | 14.13 | +10.1% |

Paired median +5.6%, 5/5 positive; ratio of medians 13.79 -> 14.14 tok/s (+2.5%). Cache8 (zero skips) was neutral:
paired median -0.1%, 2/5 positive, all hashes identical. The 512 gain is small and variable; keep opt-in, not promoted.

## All-resident scope

A fully resident expert table does not fit this 48 GB card (42.90 GiB experts vs 39,860 MiB free after
dense+KV+scratch; auto-fit 326/384 slots/layer, 1536 MiB reserve). The retained path is the existing skip-zero
path, not an all-resident table. It removes one K*N memset per skipped layer.

Missing targeted tests: no dedicated miss -> hit -> miss, poison-zero, mixed-format or explicit flag-matrix test.
The 10/10 integration covers short/long/hot and a cold/warm profile; cache8 is the zero-skip control. No mechanism
counter beyond the skip count was added. Decision: **KEEP opt-in**, experimental. CUDA/SYCL NOT RUN. No server.
