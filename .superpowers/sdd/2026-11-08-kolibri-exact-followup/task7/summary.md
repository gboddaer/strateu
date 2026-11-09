# Task 7 - GPU cache-hit dispatch / capture feasibility

## Measurement (after two reviews)

The counter line counts only 128 full-logit D2H copies (65,536,000 B = 500 KiB each). It does **not** count the
per-layer router pick copy at `src/kolibri/kolibri_engine.cu:3080-3088`: 50 slot-path layers/token, 6,400 calls/run,
153,600 B total.

The first `DEC_PHASES` run printed `d2h=0.0` because the slot path never updated that counter; the pre-copy sync also
hid the wait. A temporary probe charged the actual router copy (without the pre-sync) to that counter:

```
CNT decode phases: pre=0.052 ms d2h=0.567 ms pool=0.000 ms post=1.276 ms (per layer)
```

So the instrumented router-copy call blocked for **0.567 ms/layer on average**, about 28 ms/token at 50 layers,
against a measured 68.79 ms/token in that run: a real target, but the end-to-end saving is unmeasured and the whole
duration is not removable copy overhead (the copy also waits for its GPU inputs). The recorded slot hit rate is
99.2% (38,107 / 293).

## Prototype

A device-slot form of the exact selected MMVQ was built in the experimental diff
(`task7/device-slot-prototype.diff`): the selected Q4_K/Q6_K kernels take a base, stride and device slot array and
index the weight table on the GPU, preserving the same per-row loop, partials and reduction. No host DMA.

The forced-map test compared it bit-for-bit with independent `native_mmvq` calls over all synthetic cases (Q4_K/Q6_K,
small/large K, tails, permutations, shared/per-selection activation, same/different weights), then changed the device
map and compared again. All cases identical.

A second test added GPU translation: ids -> device map -> hit slot, or zero slot plus a miss list/count via atomic
append. Slot 0's table bytes are zero, so a miss contributes exactly zero (fallback addressing and zero-output
semantics). Miss lists are compared as unordered contents because atomic append has no stable order. The test
compares every translated slot, miss count/list and complete device-MMVQ outputs with CPU translation and independent
`native_mmvq` calls, then repeats after a map update (misses become resident) and a map reset (every pick misses).
All stages pass; `task7/prototype-test.log` and the binary hash are preserved. This is a bounded sequential
hit -> miss -> update -> reset prototype, not concurrent generations or engine integration.

## Why BLOCKED

A device dispatch alone does not remove the host wait in the real cache. The layer still needs the picked ids on the
host for `np_admit` and for the 0.8% misses that pack from the GGUF map; a device miss list still needs a host wait
before packing, and the map must be updated on every promotion with a generation/reset contract owned by the cache
lifetime. A zero-miss all-resident run is not available on this 48 GB card: the expert blobs are 42.90 GiB against
39,860 MiB free after dense+KV+scratch, and the auto-fit stops at 326 slots/layer (37.20 GiB) with 1536 MiB reserve.
The prototype was not connected to, or lifecycle-qualified in, the engine; only the synthetic translation/MMVQ path
is qualified.

The explicit ownership contract is deferred: publication must follow weight readiness, old readers must finish
before a slot is overwritten, admission/promotion update the map only after publication, and request reset must state
whether residency survives. The synthetic test does not qualify those engine edges.

The experimental code and test changes were removed; the diff, transcript and binary hash are kept. No capture was
attempted. This is a best-effort BLOCKED end state, not a pass.

Missing gate: an engine integration that keeps host miss handling where needed while dispatching hits from a device
map, plus the cache-map generation/reset ownership contract. A completely host-free miss path is not required by
the plan. The bounded translation test covers only a synthetic map. CUDA/SYCL NOT RUN. No server.
