# Kolibri bit-exact GPU-streamed expert prefill — decisions and results

Plan: [2026-10-31-kolibri-gpu-streaming](../superpowers/plans/2026-10-31-kolibri-gpu-streaming.md).
Design: [2026-10-31-kolibri-gpu-streaming-design](../superpowers/specs/2026-10-31-kolibri-gpu-streaming-design.md).
This page is the campaign's ledger: what was decided, what was measured, and what failed. Negative results stay
here; they are not rewritten when a direction changes.

Nothing in this campaign changes a default, generates tokens through a new path, or starts a network server.
Every number below says what it was measured on: Tesla T4 (sm_75, 16 GB), 2 x E5-2690 v4, driver 580.196.07,
CUDA 12.8, Kolibri-1-Q4_K_M (47,454,113,472 bytes), engine built Release for sm_75 with native experts on.

## Task 1 — the CPU expert leg is captured and replayed, byte for byte

### What was built

| piece | what it does |
|---|---|
| `src/kolibri/expert_stream_plan.hpp` | grouping of a chunk's selections by expert and the exact slot capacity accounting; CUDA-free, host-testable |
| `tests/kolibri/expert_stream_plan_test.cpp` | pins the grouping order, every refusal, overflow before allocation, and the capacity numbers |
| `tools/kolibri/expert_stream_capture.hpp` | capture schema v1: framing, SHA-256, content-addressed blob store, writer and reader |
| `tools/kolibri/expert_stream_replay.cu` | replays a capture through the engine's own native kernels and shard fan-out; refuses to report a timing unless every output byte matches |
| `tools/kolibri/expert_stream_campaign.py` | sanitized run environment, the prefill timer read once, run manifests, an independent capture reader/validator |
| `tools/kolibri/test_expert_stream_campaign.py` | 7 harness tests, no GPU and no model |
| `src/kolibri/kolibri_engine.cu` | `STRATA_KOLIBRI_STREAM_CAPTURE=<new-dir>`: one record per (chunk, layer), written after the pool drains and before weighting |

The capture records, per layer-chunk: the published expert ids (`A.ids_h`), the already-quantized token
activations (`s.act_cur`), the unique touched expert blobs **in the order the pool dealt them**, the source stride,
the layer format field by field, and the unweighted expert rows (`A.ey_h`) before the router weight is applied.
Unset, the hook allocates nothing and opens no file.

### Capture v1

`<dir>/capture.bin` = magic `KSTRCAP1`, version, endianness, a length-delimited manifest with its own SHA-256, then
`[u32 length][payload][32-byte SHA-256]` records, then a trailer `KSTREND1`. `<dir>/blobs/<sha256>.bin` holds each
unique touched blob once. The reader refuses a bad version or byte order, a manifest checksum, a truncated record,
trailing bytes, a digest that does not match, an id out of range, a geometry outside the supported path
(Q4_K gate/up, Q4_K or Q6_K down, Q8_K activations), an NVFP4 tail or a rotated basis, and any payload whose length
disagrees with its own geometry.

A second, independent reader lives in the campaign harness (Python), so a capture is validated by two
implementations before a replay uses it.

### The CPU control reproduces the engine's own work

Both captures of each history produced **bit-identical expert output on every record**, and the second short
capture (written after the digest memo below was added) produced a **byte-identical `capture.bin`**, i.e. the same
digest for every record.

| history | captures | records | selections | blob store | outputs compared |
|---|---|---:|---:|---:|---|
| ids511, ctx 2048 | r1, r2, r3 | 100 (2 chunks x 50 layers) | 153,000 | 19 GB | 100 / 100 bit-identical (r1 vs r2, and r1 vs r3) |
| ids8191, ctx 16384 | r1, r2 | 1,600 (32 chunks x 50 layers) | 2,457,000 | 40-58 GB | 1,600 / 1,600 bit-identical |

The replay then reproduced **every one of those records bit for bit**, at
`/data4/kolibri-gpu-streaming/replay-cpu-short.json` and `replay-cpu-long.json`. A single differing float fails the
replay with the record, selection, row and both hex bit patterns named.

### The cross-check the plan asks for: replay wall against the engine's instrumented pool

Same day, same binary family, same binding (`numactl --cpunodebind=1 --membind=1`, `--cpu-threads 14`):

| distribution | engine `CNT prefill pool` (median of runs) | replay CPU control, median of 3 | replay / engine |
|---|---:|---:|---:|
| ids511, 2 chunks x 50 layers, 28.12 GB streamed | 6,060 / 6,155 ms (4.6 GB/s) | 6,858 ms | **1.11-1.13x** |
| ids8191, 32 chunks x 50 layers, 756.85 GB streamed | 85,373 ms (8.9 GB/s) | 92,898 ms | **1.09x** |

The replay's CPU arm is therefore 9-13 % slower than the engine's own pool for the same work, in the same
direction on both distributions. **This is the campaign's open item and it is not attributed with a measurement
yet.** What was ruled out by measurement, one variable at a time:

| candidate | measurement | verdict |
|---|---|---|
| blob buffer alignment (16 / 32 / 64 KiB / 4 KiB / 2 MiB) | 7.43 / 7.25 / 8.39 / 7.18 / 7.27 s | not the explanation (2 MiB best, chosen as the default) |
| one heap block per blob vs one contiguous table at the engine's stride | 8.37 s vs 7.18-7.47 s | part of it; the contiguous table is used |
| pinned (`cudaHostAlloc`) vs pageable blob memory | 8.43 s vs 7.18 s | pinned is *slower* here; pageable is used |
| lane pinning on / off, and `--membind` on / off | 7.18-7.64 s vs 7.50 s | 3-5 %, not the explanation |
| repeats 1..3 in one process | 8.72 / 7.96 / 7.18 s (pass-major) | the first pass is cold; record-major repeats remove the re-read |

What remains: the engine reads its blobs in place from the 42.9 GiB pinned table it allocated at load, while the
replay reads each record's blobs from the capture store into one pageable table (so every record touches two
buffers' worth of pages instead of one). That is the only structural difference left, and it is *not* reproduced by
`cudaHostAlloc`-ing the replay's table. Until it is attributed, the gate in Task 4 treats the replay's CPU number as
a slightly **optimistic** denominator and also records the engine's own pool wall, so a GO cannot rest on the replay
alone.

### Capture cost (instrumentation, never a timing)

| run | wall | prefill timer | note |
|---|---:|---:|---|
| ids511 capture | 5:08-5:39 | 264-274 s | 19 GB written at ~72 MB/s to ZFS |
| ids8191 capture, before the digest memo | 68:04 | 4,039,814 ms | 737 GB re-hashed: the same 16,604 blobs are passed 16 times |
| ids8191 capture, after the digest memo | 28:27 | 1,662,228 ms | 58 GB, digest memoised per source pointer |

The digest memo is worth recording as a decision: a 47 GB model read 16 times through a byte-at-a-time SHA-256 cost
an hour, and the engine's pinned table is immutable for the run's lifetime, so a digest is memoised per source
pointer. The measured check that this changes nothing: the memoised short capture is byte-identical to the
non-memoised one.

Reading a capture back is also expensive: 19 GB takes 153 s from ZFS cold (123 MB/s), 58 GB about an hour at 16k
files. Repeats are therefore **record-major** (each record's blobs are read once, then replayed `repeat` times), and
the loader's time is reported separately and excluded from `cpu_ms`, as the plan requires.

### Rulings taken in Task 1

- **Dependency source**: ggml from a shallow clone of the pinned commit `3cf03257`, through `-DSTRATA_GGML_DIR`,
  instead of a full-history FetchContent clone (`/data2` has 45 GB free). Cost if wrong: dependency provenance
  differs from the default configure; any ggml divergence shows up in `native_arithmetic_parity_kq256`.
- **Slot capacity**: per-selection buffers are sized for the whole chunk's selection count (not per batch), and the
  per-slot weight capacity is `max(stride, min(cap, experts_per_batch x stride))`, so a single blob larger than the
  32 MiB cap still runs alone rather than being refused. Cost if wrong: a slot is 8 x 73 MB larger than needed at
  eight 73 MB experts; the two-slot budget check still refuses an over-budget configuration.
- **Capture refusal set**: the hook exists only on the CPU pool leg, and a capture run refuses (exit 2, named)
  `--expert-slots`, `--grouped`, the A1 arm, the fp16 expert cache, `PF_PARTIAL`, `DEC_RES`, the MMQ arm, a served
  request stream, and `STRATA_KOLIBRI_POOL_ALL` set to anything but 1. Cost if wrong: a configuration that would
  have been instructive to capture has to be captured through the pool instead.
- **Replay repeats are record-major**, and blobs stay in one contiguous pageable table. Cost if wrong: repeats
  after the first see their working set from RAM rather than re-read from the store, which flatters the replay by
  less than the 9-13 % above.

### Not claimed

- Nothing here says the GPU arm is faster. Task 1 built the control; the gate is Task 4.
- The replay's absolute milliseconds are not the engine's pool time; they are 9-13 % above it, and the reason is
  still open.

## The transfer floor, measured before any kernel work

Before Task 2 spends days on kernels, the one number the whole design rests on was measured directly:
pinned host -> device on this card, with the real blob size (2,570,400 B), 117 copies per layer-chunk and one large
copy for comparison (`/data4/kolibri-gpu-streaming/h2d-probe.txt`, probe `h2d_probe.cu`, not part of the build):

| pattern | rate |
|---|---:|
| 117 x 2,570,400 B (the real layer-chunk pattern) | **9.61-9.64 GB/s** (31.2 ms per 301 MB) |
| one 281 MB copy | 9.77-9.81 GB/s |

That confirms the design's 9.56 GB/s on today's driver (580.196.07) to within 1 %, and it is stable to ±0.05 %.
The two measured expert-leg byte totals from the engine's own counters are 28.12 GB (ids511) and 756.85 GB
(ids8191). Dividing:

| distribution | CPU pool, measured | H2D floor | floor / CPU | D2H for the CPU bridge (measured shapes) | floor with both directions |
|---|---:|---:|---:|---:|---:|
| ids511 | 6,060-6,155 ms | 2,917 ms | **0.47-0.48** | 2.3 GB = 236 ms | 3.15 s = **0.51-0.52** |
| ids8191 | 85,373 ms | 78,510 ms | **0.92** | 36.4 GB = 3,780 ms | 82.3 s = **0.96** |

The CPU bridge is gate/up out (1,533 x 1,024 x 4 B), hidden back (1,533 x 584 B) and down out (1,533 x 2,560 x 4 B)
per layer-chunk, i.e. 22.75 MB x 100 or x 1,600 layer-chunks.

**What this means, stated plainly.** For the short prompt the transfer floor is half the pool's time, so the
design's premise holds and a 10 % gate is reachable. For the long prompt the bytes alone are 92-96 % of the CPU
pool's time: no amount of GPU compute, overlap or tuning can bring that distribution under the plan's 0.90 gate,
because the gate's denominator is the expert leg itself. The reason is not the card: the CPU pool already streams
756.85 GB in 85.4 s = **8.9 GB/s**, which is 92 % of this box's measured 9.64 GB/s PCIe rate. Re-reading an expert
from local RAM and copying it over PCIe cost almost the same at this size.

The lever that does change the long-prompt arithmetic is **fewer bytes**, not a faster copy: the same expert is
re-loaded in most of the 32 chunks (756.85 GB = 32 x 50 x ~197 x 2.57 MB). A bigger prompt chunk or VRAM residency
of the original quantized blobs would cut the total; both are engine changes outside this plan's scope. Recorded
here as the finding that decides the campaign, not as a change made.

## Ruling: both distributions are measured, the gate is read per distribution

Taken after the transfer floor above, by the operator in this session: **run the full campaign on the short and the
long capture**, and report the gate per distribution. The reason the long distribution is still worth building: the
bound above is a property of *this* card's PCIe, not of the design. The card is PCIe 3.0 x16 and measures
9.61-9.81 GB/s; the long prompt needs, at 756.85 GB up and 36.4 GB down inside the 0.90 x 85.373 s budget,

| what would pass the gate on the long distribution | rate needed | this card | a PCIe 4.0 x16 host |
|---|---:|---:|---:|
| both directions counted against 0.90 x CPU | 10.33 GB/s | 9.64 GB/s | ~22 GB/s, i.e. 0.42 |

so the long arm is a hardware-independent test of the pipeline, and its verdict is recorded as "bound by this
card's PCIe" with the rate it would need, not as a refutation of the design. Both arms are built and measured;
neither the plan's gate nor this ruling is relaxed.

## Task 2 — the tiled grouped products, gated by the CPU pool itself

`src/kolibri/expert_stream_kernels.cuh` holds `q4k_tile` and `q6k_tile`. A CTA is 128 threads: four warps, four rows
per warp, and one row's eight integer lanes per row are the eight threads of a shuffle group. A thread keeps
`kTileTokens = 4` accumulators, one per token, so the row's weight bytes are loaded and decoded **once for four
tokens** instead of once per token. The tile is the plan's 16 rows x 4 tokens.

The arithmetic is A1's, which is the pool's: eight int32 lanes per row, the four-value groups at `64j + 4k` and
`64j + 32 + 4k` scaled by `scales[2j]` and `scales[2j+1]`, one FMA per block, `dmin = -y.d * xdmin` for the minimum
term, the final tree `((a0+a4)+(a2+a6)) + ((a1+a5)+(a3+a7))` and `((m0+m2)+(m1+m3))`. The one freedom taken is the
integer part: four of those four-value products go through `__dp4a` in one instruction. That is legal because the
integer sum is exact in int32 — the order inside a lane cannot change the result — and it is what makes sharing the
decoded weight bytes across four tokens possible at all. The float sequence per token is untouched, so each token's
bits are the single-token bits.

### What the test proves

`tests/kolibri/expert_stream_kernel_test.cu`, 94 cases, all bitwise (`--selftest`; `--tails-only` runs 72 tail cases
for the sanitizers):

| what | against |
|---|---|
| every `nb` in {1,2,10,32} at every row tail (1,15,16,17,512,2560), 4 tokens | `kq256_rows` (Q4_K), ggml-cpu `vec_dot` (Q6_K) |
| every token count in {1,2,7,8,9,13,31,64,65} at 17 and 512 rows | same |
| the model's geometries: gate/up (nb 10, 512 rows), down Q4_K and Q6_K (nb 2, 2,560 rows) | same |
| largest finite fp16 scales, zero blocks, row starts at 144/210/7x144 bytes (unaligned) | same |
| the whole expert: gate/up tile, CPU SwiGLU, `native_quant_h` **byte for byte**, down tile | `native_gu_rows` / `native_down_rows` |
| tiled vs A1 on a subset | A1 **and** the CPU oracle (A1-only agreement is not a pass) |

Sensitivity of the data itself (the test must be able to see the mistakes the campaign is about): flipping `dmin`'s
sign in the weights (what a `+xdmin` kernel is equivalent to) changes the oracle's output, swapping adjacent Q6_K
scale bytes changes it, and rotating `ql` by 32 bytes changes it. All three are required to differ, and all three do.

The test is not vacuous: with `-(yd * xdmin)` turned into `(yd * xdmin)` in `q4k_tile`, 48 of 72 tail cases fail; with
the `(c & 1) * 32` half of the Q6_K `ql` index removed, 24 of 72 fail. Both mutations were reverted before commit.

Compute Sanitizer on `--tails-only`: memcheck 0 errors, racecheck 0 errors, synccheck 0 errors.

### Compiler report (recorded, as the plan asks)

| kernel | registers | spills | barriers | shared |
|---|---:|---|---:|---:|
| `q4k_tile` | 62 | 0 byte store, 0 byte load | 0 | 0 |
| `q6k_tile` | 62 | 0 | 0 | 0 |
| (A1, for comparison) | 61 / 64 | 0 | 0 | 0 |

62 registers at 128 threads keeps eight CTAs resident per SM (1,024 threads, half of sm_75's 2,048): the kernels are
byte-bound, not occupancy-bound, and 16x4 stays until parity says otherwise.

### Rulings taken in Task 2

- **No shared-memory staging.** The plan asks for cooperative CTA staging. It is not needed and would add the exact
  hazard it warns about: the only data shared inside a tile is the four tokens' weight bytes, and those are read by
  the four threads of one warp that hold the same integer lane — the same address, so one L1 transaction, no
  barrier, no mask. The activation blocks (11.7 KB for four tokens) are read by every row's warp and stay in L1.
  Cost if wrong: up to 16 duplicate activation reads per block from L1/L2 instead of shared memory; to be revisited
  only if the pipeline's kernel time is the limiter, which the transfer floor says it will not be.
- **Token pointers are kernel arguments, never a host array.** A kernel cannot dereference a host pointer array;
  the first version did and memcheck named it immediately (invalid 8-byte read of a host address). Four pointers
  fit in the kernel's constant bank, and a device pointer array would cost an upload per launch.
- **Per-group shuffle masks.** A row's eight lanes are one shuffle group, and a partial row cannot occur, so the
  group leaves and stays together; the mask is `0xff << 8*(lane>>3)`, not the whole warp. With the whole-warp mask
  the row tail is undefined behaviour — the first version's second bug.
- **`nt` is bounded to 1..1024 and split into tiles of four by the launcher**: a 64-token chunk is 16 launches, and
  the token count is a kernel argument so the tail tile needs no separate code path.

## Task 3 — the two-slot pipeline

`src/kolibri/expert_stream.hpp` (CUDA-free API) and `src/kolibri/expert_stream.cu` implement the design's lifecycle:

```
FREE -> H2D -> GU -> GU_D2H -> CPU_ACT_QUANT -> HIDDEN_H2D -> DOWN -> OUT_D2H -> FREE
```

Batch i loads and computes gate/up on slot i%2 while the host runs batch i-1's CPU SwiGLU, `native_quant_h`, its
hidden upload and its down product on the other slot. The slot batch i reuses was released by batch i-2's scatter,
which happened in the previous iteration, so no slot is ever waited on while the other could advance.

Bounded by the plan: eight groups and the weight cap per batch, 32 MiB per slot by default, a 256 MiB device budget,
and every buffer allocated at `create()`. A run allocates nothing, calls no `cudaDeviceSynchronize`, and publishes
`ey_h`-equivalent rows only after their event. Weights go straight from the caller's table (the engine's is pinned,
so that copy is real asynchronous DMA); the gathered activations and both results use the slot's own pinned staging.

`create()` sizes the slots for BOTH down formats of this model (Kolibri alternates Q4_K and Q6_K down between
layers), taking the larger of the two capacities. A later call may therefore use either; only the down type, the
weights, the ids, the activations and the output may differ from the view `create()` was given, and anything else is
refused by name.

### What the test proves

`tests/kolibri/expert_stream_pipeline_test.cu`, 76 cases (`--selftest`), all bitwise against
`native_gu_rows` + `native_quant_h` + `native_down_rows`:

| case | why |
|---|---|
| empty input | no GPU work, zeroed counters |
| 256 tokens x top_k 6, Q4_K down, then a second context on Q6_K | the real chunk |
| one context, both down formats, three calls | the layer alternation the engine will actually do |
| skewed groups: one expert takes a token's whole route, another every token's first slot | batches that hold one huge group next to tiny ones |
| one expert holding all 32 selections | a single group spanning the batch bound |
| 100 calls through the same two slots with the output poisoned (`0x40 + call` bytes) between them | slot reuse, stale data, allocation-free steady state |
| five failure cases (top_k 0, non-Q4_K gate, out-of-range id, wrong activation stride, 1 KiB budget) | refused by `std::invalid_argument`, nothing published |
| every allocation ordinal of `create()` failed in turn (64 ordinals, 18 real) | a partial allocation is released, not leaked |

Counters, as the plan requires: `weight_h2d_bytes` equals the sum of the UNIQUE touched blobs, `device_allocations`
is zero on a warm run, and `kernel_launches` is tile-scaled (one launch per four-token tile per product), never one
per selection.

Compute Sanitizer on `--tails-only`: memcheck, racecheck and synccheck all report 0 errors.

### Rulings taken in Task 3

- **The weight staging pins are allocated but unused.** The plan allows copying "from pinned slot staging for
  genuinely asynchronous DMA". The engine's expert table is already pinned, so staging every blob through another
  pinned copy would add a memcpy of ~2.5 MB per expert for no gain; the copy is taken straight from the caller's
  table. The pinned weight buffer stays accounted for by Task 1's capacity arithmetic (32 MiB per slot, ~64 MiB of
  host memory) rather than shrunk, because a caller with a pageable table would need it. Cost if wrong: 64 MiB of
  pinned host memory unused while the streamed flag is on.
- **Rows, not tokens, are the unit of a group.** A group's gathered activations are one row per SELECTION (in
  original order), not one row per distinct token: the same token routed to two experts of the same batch needs two
  products. Duplicated rows are the correct input, and they are why the scatter is per selection.
- **The pipeline test's weights are ggml-quantized float rows**, not random bytes. Random block bytes produce
  weights whose scales are unrelated to their nibbles and outputs up to 1e38: at that magnitude the first version of
  this test compared NaN against FLT_MAX and reported a difference that said nothing about the pipeline. Task 4
  replays the model's real captured weights through this same code, which is the fidelity check that matters.
