# Task 3 - explicit activation quantization reuse

## Implementation (candidate, not retained)

- `gemv_pair_quantized()` quantizes the shared gate/up activation once into `g_q8`, then runs gate and up from
  those bytes; selected MMVQ is used when both flags are on, otherwise the original per-selection native_mmvq loop.
- Wired behind `STRATA_KOLIBRI_REUSE_Q8=1` at the three decode gate/up call sites (resident, overflow, pinned table).
  A one-time stderr trace records the reuse. The down projection (per-selection inputs) is untouched.
- Primitive contract test `tests/kolibri/quant_reuse_test.cpp`: three quantizations on the same input are byte-equal,
  gate/up outputs are bit-identical, a second different input reuses the same address without staleness, and
  per-selection contiguous quantization equals separate quantizations. It passes, but uses only existing APIs, so
  it is a contract test rather than a production RED; the flag-on trace + full logits are the integration gate.

## Full logits

- Four combinations on 512/128: off, selected only, reuse only, both - all SHA `6ffefa8f...8721`.
- Both flags on 8/32 cache8: SHA `466f48f2...c3d6`.
- Short/long/hot/profile integration with both flags: 10/10 complete dumps byte-identical to the flag-off matrix.
- Flag-off 512 with the current binary: same SHA `6ffefa8f...` as the Task 2 control.

## Timings (five alternating pairs, decode tok/s)

| case | off median | reuse median | paired deltas |
|---|---|---|---|
| 512 reuse | 12.30 | 11.04 | -10.2%, -15.8%, +1.2%, -0.1%, +16.6% |
| c8 reuse | 2.16 | 2.16 | +52.3%, +0.0%, -0.0%, -0.0%, +0.5% |
| 512 selected vs both | 13.01 | 12.90 | +16.3%, -25.1%, -1.4%, +246.0%, +16.4% |

Reuse alone: 512 paired median -0.1%, ratio of medians -10.2%; cache8 neutral. Both flags vs selected alone:
ratio of medians -0.8% with a 4.36 tok/s outlier in the selected arm; no repeatable gain.

Decision: **REJECT** the reuse integration. It is exact, but it shows no repeatable gain, cache8 is neutral, and it
is not consumed by a named retained path. Remove the helper, wiring, flag and test/CMake target; keep this evidence.
