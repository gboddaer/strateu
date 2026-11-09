# Task 2 - exact multi-selection MMVQ

## Implementation

- `native_mmvq_selected(type, base, stride, host_slots, xq, xq_stride, y, n_in, n_out, selections, stream)` in
  `include/strata/kernels/native_mmvq.hpp` / `src/kernels/cuda/native_mmvq.cu`, Q4_K=12 and Q6_K=14, 2..8 selections.
- Host slots are copied into a by-value kernel parameter block (no DMA). `grid.y` selects weight table, activation
  column and output. Kernels mechanically preserve the legacy ncols=1 per-row loop, partials and warp reduction.
- Returns false for unsupported formats/counts, gfx906 wave layout, a registered packed-Q6 table, invalid q8 stride or an
  unaligned derived pointer; an unaligned base throws on negative/overflowing slots and invalid shapes/pointers.
- Engine `gemv` uses it only with `STRATA_KOLIBRI_MMVQ_SELECTED=1`, `--expert-no-pin`, host picks, 2..8 selections;
  otherwise the original per-selection loop runs. A one-time stderr trace names the first selected call.

## Full-logit gates (complete dumps, finite assertion in the driver)

- `off-512`: input 512, steps 128, 65536000 bytes, SHA `6ffefa8f65b3cb8ca451ae77f23ee05ba197bd23e8e25febbf8121d806fa8721`
- `on-512`: input 512, steps 128, 65536000 bytes, SHA `6ffefa8f65b3cb8ca451ae77f23ee05ba197bd23e8e25febbf8121d806fa8721`
- `off-cache8`: input 8, steps 32, 16384000 bytes, SHA `466f48f21dacab36cd4f7bcabb42d90c5bb32ceaba4aa784cfe9b6a8d027c3d6`
- `on-cache8`: input 8, steps 32, 16384000 bytes, SHA `466f48f21dacab36cd4f7bcabb42d90c5bb32ceaba4aa784cfe9b6a8d027c3d6`
- `t2-4097-off`: input 4097, steps 8, 4096000 bytes, SHA `241969a0f00f6f5e65adc2de3f37c87bcba723d6c52a1cfa378ebbccaee791bf`
- `t2-4097-on`: input 4097, steps 8, 4096000 bytes, SHA `241969a0f00f6f5e65adc2de3f37c87bcba723d6c52a1cfa378ebbccaee791bf`

The on-runs used `--compare` against the off dump and passed. `on-512.log` records
`selected MMVQ n_sel=6 n_in=2560 n_out=512`; `on-cache8.log` records one selected call as well.

## Timings (five alternating same-day pairs, stats off)

### 512

| pair | off decode tok/s | on decode tok/s | delta | off wall s | on wall s |
|---|---|---|---|---|---|
| 1 | 12.28 | 14.73 | +19.9% | 150.5 | 141.2 |
| 2 | 10.79 | 14.54 | +34.7% | 234.3 | 152.8 |
| 3 | 10.76 | 13.01 | +20.9% | 205.6 | 159.9 |
| 4 | 11.13 | 12.57 | +12.9% | 188.1 | 243.8 |
| 5 | 10.91 | 11.77 | +7.9% | 152.4 | 242.6 |

Decode medians: off 10.91 tok/s, on 13.01 tok/s (+19.3%).

### c8

| pair | off decode tok/s | on decode tok/s | delta | off wall s | on wall s |
|---|---|---|---|---|---|
| 1 | 1.22 | 1.55 | +27.5% | 39.5 | 29.9 |
| 2 | 1.87 | 1.62 | -13.3% | 23.9 | 26.4 |
| 3 | 2.16 | 2.16 | +0.1% | 19.9 | 19.7 |
| 4 | 2.15 | 2.21 | +2.8% | 19.7 | 19.3 |
| 5 | 2.15 | 2.22 | +3.1% | 19.8 | 19.4 |

Decode medians: off 2.15 tok/s, on 2.16 tok/s (+0.47%).

512/128: 5/5 pairs faster (7.9% to 34.7%, paired median +20.9%; ratio of medians +19.3%). cache8: 4/5 positive,
one -13.3%; paired median +2.8%, ratio of medians +0.47% - inconclusive, not a gain, because startup and expert
transfers dominate. Wall values vary with model-load time, so the 512/128 wall medians are not an end-to-end claim.

## Builds and tests

- HIP main build and full HIP test-tree build: OK.
- `kolibri_mmvq_selected`: Passed. Relevant host/kernel ctests pass.
- `kolibri_decode_wait_test` is registered with `HIP_VISIBLE_DEVICES=0`; on this host device 0 gives
  `doorbell_publish launch: invalid device function`. The same binary passes with device 1 (the campaign device).
  A rebuild without this change showed the same device-0 failure, so it is an environment mismatch, not this change.
- `mmvq_selected_test` also covers a registered packed-Q6 sentinel (only the later selection is registered), an
  unaligned derived table (false), an unaligned base (throw), negative/overflowing slots. The test unsets
  `STRATA_Q6_PACKED` so the sentinel cannot be satisfied by the environment flag.
- CUDA and SYCL: NOT RUN (no toolchain/hardware). gfx906 default-wave refusal is registered as a CMake test only in
  `STRATA_HIP_GFX906` builds; it cannot run on this gfx1100 host.

## Integration matrix (short/long/hot, profile replay)

`tools/kolibri/test_unpinned.py --phase all --hot` on the same binary, flag off then on:
cases `decode`, `prefill`, `prefill-repeat`, `prefill-spill`, `decode-short`, `decode-short-spill`,
`long-prefill`, `long-prefill-spill`, `hot-cold`, `hot-warm`.
All 10 complete 2,048,000-byte dumps are byte-identical between off and on, including multi-chunk prefill,
cache-residency changes and the cold-to-warm saved profile. The auditable manifest with per-case SHA-256, float
counts and finiteness is `task2/integration-compare.txt`; input/profile hashes and the run-time executable hash are
there too. Off run 22m19s, on run 24m12s (wall not used as a gain). These runs used the old 300 s per-case watchdog;
all completed. The driver now defaults to 10800 s, but these artifacts were not rerun.

## RED and test evidence

RED: the first build of `tests/kolibri/mmvq_selected_test.cpp` failed with
`error: use of undeclared identifier 'native_mmvq_selected'` (4 errors) before the API existed.
GREEN: `build-kolibri-hip-tests/mmvq_selected_test` -> exit 0, log in `task2/mmvq_selected_test.log`; ctest
`kolibri_mmvq_selected` passes. The default-wave refusal mode (`--expect-refusal`) fails on gfx1100 by design and is
registered only for gfx906 builds.

Decision: **KEEP** (repeatable 512/128 decode gain, bit-identical). It is not an enabler: Task 3, the only
consumer, was rejected and removed.
