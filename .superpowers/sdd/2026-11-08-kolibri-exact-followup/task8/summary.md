# Task 8 - bounded CPU packing

## Implementation (removed after measurement)

`include/strata/kolibri/byte_pack_pool.hpp`: a bounded pool with 1-4 persistent lanes, caller = lane 0, one
`pack(spans, n)` call joins before return. Gate `STRATA_KOLIBRI_PACK_THREADS=2`, default 0. `pack_blob` splits its
three independent copies (gate, up, down) across the lanes; no arithmetic, queue or whole-model copy.

Unit test `tests/kolibri/byte_pack_test.cpp` (removed with the experiment; copy kept) covered 0/1/37/4099-byte
spans, three spans with prefix/gap/suffix sentinels, 20 repeats, a source overwritten after return, one-span layout
and pool destruction drain. All byte-exact.

## Measurement

Two alternating 512/128 pairs with `--stats`; complete logits were identical to `6ffefa8f...8721` in all four runs.

| run | startup pack ms | prefill pack ms | decode pack ms | wall s |
|---|---:|---:|---:|---:|
| off 1 | 70,804 | 6,807 | 938 | 146.0 |
| on 1 | 89,684 | 21,343 | 2,431 | 161.4 |
| off 2 | 147,967 | 25,782 | 2,041 | 258.1 |
| on 2 | 156,624 | 19,344 | 10,560 | 279.8 |

The two lanes are not faster: startup packing is 26.7% and 5.9% slower, prefill is mixed/slower, decode packing is
slower, and wall is worse in both pairs (10.6% and 8.4%). Mmap/IO contention is a hypothesis, not established;
synchronization, scheduling and uneven lane work remain possible causes. The spread also shows strong page-cache/IO
state. This is not a precision timing, but there is no positive signal to retain. Evaluation stopped early: cold
4K/source-fault, the planned two-worker-plus-caller configuration, delayed-worker lifetime, exception cleanup and the
stats-off five-pair gate were not run.

Decision: **REJECT**. Remove the pool, engine wiring and test; keep the diff, sources, logs and hashes. No concrete
retained dependency. CUDA/SYCL NOT RUN. No server.
