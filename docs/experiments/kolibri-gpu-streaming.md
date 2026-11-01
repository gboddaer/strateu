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
