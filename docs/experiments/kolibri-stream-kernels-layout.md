# Streamed expert kernels: seven changes tried, one kept

The streamed expert prefill put the prefill's expert products on the GPU (`docs/experiments/kolibri-gpu-streaming.md`).
This page records seven changes tried on top of it - three kernel layouts, the tile in both of its dimensions, the
pipeline's depth and its copy streams, and moving the SwiGLU and the quantization to the device - one at a time, each behind its own switch, each required to return the same bits as the CPU pool
before its speed was looked at. One is kept; the other six are negative results, recorded so nobody pays for them
again without new evidence. Section 5 gives where the streamed leg's time actually goes, which is why five of the six
could not move it.

Environment for every number here: Tesla T4 16 GB (sm_75, driver 580.196.07), CUDA 12.8, 2 x E5-2690 v4 with
`numactl --cpunodebind=1 --membind=1`, Kolibri-1-Q4_K_M, Release build, branch `kolibri-sram-activation` off
`kolibri-gpu-streaming` @ `19c8c92f`. Prefill is reported from the engine's own `PFchunk` line. The kernel bench is
`tools/kolibri/expert_stream_kbench.cu`:

```sh
cmake --build build-sram --target kolibri_expert_stream_kbench -j16
numactl --cpunodebind=1 --membind=1 ./build-sram/kolibri_expert_stream_kbench 40
```

It times every switch in one process on the same data, minimum of 40 launches per case, with no other GPU work in
flight. Each table below is one run; a second run of the same cases moves the ratios by a few percent and never
changes which side of 1.0 they are on.

## 1. Shared-memory staging of the tile's activations (`STRATA_KOLIBRI_STREAM_SMEM=1`) — not kept

**The idea.** A tile is 16 rows of one expert over 4 tokens. For one block of 256 values every row reads the *same*
292 activation bytes, against 128 bytes of weight codes, so the activations are the larger on-chip traffic and every
row re-reads them. Stage them once per CTA per block, then read shared memory: 73 ints per token, one per thread of
the 128. The Qwen3.8-Flash-Next engine treats this as a design dimension of its own (`sram_layout`, and "activations
copied once per workgroup into shared memory" in its GP100 kernel), which is where the idea comes from.

**Bit-identity.** Only the address a load reads from changes. The per-token lane sequence, the per-block FMA and the
final tree are untouched, and a thread whose row is past `rows` now stays alive (staging needs every thread at
`__syncthreads()`), skipping only its own loads and store. `es_smem_staging()` picks at launch; `sact` is one byte,
not one tile, where it is off, so both instantiations keep the same registers and occupancy.

| kernel bench, smem time / direct time | nt=1 | nt=4 | nt=8 | nt=16 | nt=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K gate/up (1,024 rows) | 1.262 | 0.900 | 1.048 | 1.083 | 1.075 |
| Q6_K down (2,560 rows) | 1.190 | 1.060 | 1.032 | 1.022 | — |
| Q4_K down (2,560 rows) | — | — | 0.916 | — | — |

One ratio below 1.0 at gate/up `nt=4` and one at the Q4_K down case; the rest are 2 to 26 % slower.

| engine, prefill ms | direct | shared-memory staging | ratio |
| --- | ---: | ---: | ---: |
| ids512 (511 tokens), 3 reps | 5,005 | 5,156 | 1.030 |
| ids8191 (8,190 tokens), 2 reps | 104,585 | 106,951 | 1.023 |
| ids11000 (10,999 tokens), 1 rep | 114,225 | 118,329 | 1.036 |

Full-logit dumps compared with `cmp` on all six histories, staged against the arm without it: byte-identical
(4,608,000 bytes each).

**Why it does not help.** The 32 threads of a warp are four rows for a block and token, and all four read the same
activation addresses, so the load is an L1 broadcast — one transaction serves the warp. The bytes were never the
cost. Staging replaces that with a copy and two barriers per block, which is the whole of the 2 to 3 %.

## 2. The gate row and the up row of one k in one thread (`STRATA_KOLIBRI_STREAM_GU=1`) — kept, on by default

**The idea.** A batch launches gate/up as one `2 * n_ff`-row product, so gate row k and up row k land in different
CTAs and each loads the token's activation bytes. Hold both rows in one thread: the same 4-byte activation load now
feeds four `__dp4a` instead of two. No weight byte moves — the blob keeps its `[gate rows | up rows]` order, and the
kernel maps a tile's rows to `kk` and `rows + kk`. This is the layout the Qwen engine's `swiglu_kernel(interleaved)`
and Strata's own Q2_0 blob (`gate/up codes [1280][640 B]`, gate at `2k`, up at `2k + 1`) are built around.

**Bit-identity.** Each of the two rows keeps its own accumulation sequence, its own per-block FMA pair and its own
final tree; the kernel test runs `es_q4k_gu_rows` and `es_q4k_rows(2 * rows)` on one blob and compares the whole
`[gate | up]` output with `memcmp`, for four token counts and both down formats, inside the 98-case suite.

| kernel bench, `gu-pair` time / whole-table time | nt=1 | nt=4 | nt=8 | nt=16 | nt=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K gate/up | 0.667 | 0.841 | 1.057 | 1.297 | 1.360 |

It wins where a group has enough tokens and loses on tiny groups: half the CTAs for the same work, and at `nt=1` there
are only 32 of them.

| engine, prefill ms | whole table | gate/up pair | ratio |
| --- | ---: | ---: | ---: |
| ids512 (511 tokens), 3 reps | 5,005 | 4,843 | 1.033 |
| ids8191 (8,190 tokens), 2 reps | 104,325 | 102,554 | 1.017 |

That is the shape the model has: about four tokens per expert at 511 tokens, about sixty-four at 8,190. The 511-token
case still gains, because the batch mixes small groups with ones that are not.

## 3. A plane-split Q6_K layout (`STRATA_KOLIBRI_STREAM_PLANES=1`) — not kept

**The idea.** A Q6_K block is 210 bytes, so block i starts 2 bytes further out than a 16-byte boundary and every
`ld4` in the down kernel is a 2-byte-aligned load. The Qwen engine keeps a Q6_K head copy whose bytes are "four
planes — ql (128 B per block), qh (64 B), scales (16 B), d (2 B)" precisely so a thread's words are aligned, and
128 + 64 + 16 + 2 = 210, so that copy is byte-neutral. Here the same packing runs on the device right after the
weights land (`q6k_pack_planes`, one thread per block, 105 16-bit words) into a second buffer per slot, and the down
product reads the planes.

**Bit-identity.** The values and the kernel's per-row arithmetic are the plain kernel's; the packing kernel and the
plane kernel are both compared against the CPU oracle in the kernel test (`nb` 1, 2 and 10, rows up to 2,560, 3 to 64
tokens), and the pipeline test's 76 cases pass with the layout on.

| kernel bench, planes time / block-interleaved time | nt=1 | nt=4 | nt=8 | nt=16 | nt=64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q6_K down (2,560 rows) | 1.031 | 1.012 | 1.030 | 1.031 | 1.020 |
| the same cases, second run | 1.076 | 1.039 | 1.023 | 1.022 | 1.016 |

| engine, prefill ms | block-interleaved | planes | ratio |
| --- | ---: | ---: | ---: |
| ids512 (511 tokens), 3 reps | 4,875 | 4,922 | 1.010 |
| ids8191 (8,190 tokens), 2 reps | 102,300 | 105,518 | 1.031 |

**Why it does not help.** The misalignment is real but free: the four `ld4`s of a block are inside one 128-byte sector
either way, so the unaligned load costs no extra transaction on this card, while the planes cost four base pointers
and a multiply per block. The Q8_0 case that motivated the Qwen copy is different: 34-byte blocks put each *byte*
run at a different 2-byte offset and their kernel reads 8-byte runs, which an unaligned pair of 16-bit loads really
does cost.

## 4. The tile itself: tokens per tile 8 and 16, rows per CTA 8 and 32 — none kept

The token tile was measured behind a temporary `STRATA_KOLIBRI_STREAM_TOKENS` switch, which is not in the tree: it
lost on the bench and was inside the noise in the engine, so it did not earn a permanent switch. The row tile is
`STRATA_KOLIBRI_STREAM_ROWS` and stays, defaulting to the shipped 16.

**The idea.** A CTA reads its rows' weight blocks once and uses them for `kTileTokens` (4) tokens, and each of its 16
rows re-reads the same activation bytes. A wider token tile reads each weight block once for 8 or 16 tokens instead
of once for 4; fewer rows per CTA re-read the weights more often per token but hand the card more CTAs, which a
1,024-row gate/up launch at eight tokens per group needs: 64 CTAs against 40 SMs that hold seven each.

**The kernel bench says no to the wider token tile.** `tok8 / tok4`, and `tok16 / tok4` in brackets:

| case | nt=1 | nt=4 | nt=8 | nt=16 | nt=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K gate/up | 1.417 | 1.301 | 1.517 | 0.974 | 1.097 |
| Q4_K gate/up pair | 1.783 | 1.329 | 1.738 | 1.257 | 0.994 |
| Q6_K down | 1.085 | 0.833 | 1.065 | 1.098 | — |

The reason is registers, and it is worth writing down because it decides the direction of any further tiling.
`ptxas` on this card: `q4k_tile<4>` 70 registers → 7 CTAs/SM (44 % occupancy), `<8>` 96 → 5 (31 %), `<16>` 173 → 2
(12 %); `q6k_tile<4>` 57 → 8 (50 %), `<8>` 128 → 4 (25 %). The kernel waits on its weight load every block, so
occupancy is what fills that wait, and eight live token accumulators cost more than the second read of a weight byte
buys. 16 threads more per row do not exist to give: the 8 integer lanes of a row are fixed by the pool's arithmetic.

**The row tile is the direction that helps the kernel a little.** `rows8 / rows16`, then `rows32 / rows16`:

| case | nt=1 | nt=4 | nt=8 | nt=16 | nt=32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K gate/up, 8 rows | 0.973 | 0.992 | 0.907 | 0.908 | 1.021 |
| Q4_K gate/up pair, 8 rows | 0.976 | — | 0.976 | 0.933 | 0.966 |
| Q6_K down, 8 rows | 0.994 | 1.005 | 0.985 | 0.988 | — |
| Q4_K gate/up, 32 rows | 1.055 | 1.023 | 1.050 | 1.049 | 1.062 |

**In the engine all of it is inside a percent**, which is the real answer:

| prefill ms, medians | 4 tokens | 8 tokens | 16 tokens | 8 rows | 16 rows | 32 rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ids512 (511 tokens), 3 reps | 4,822 | 4,830 | 4,891 | 4,868 | 4,822 | 4,972 |
| ids8191 (8,190 tokens), 2 reps | 102,639 | 102,387 | 102,388 | 102,417 | 102,639 | 102,803 |
| ids11000 (10,999 tokens), 1 rep | 109,491 | 109,941 | — | 110,175 | 109,491 | 109,678 |

Every run's full-logit dump is byte-identical (`cmp`, 4,608,000 bytes) to the arm without the switch, on all six
forced histories.

## 5. Why a percent is the ceiling here: what the streamed leg actually spends

`--counters` now prints the split the arm accumulates (`CNT expert-stream:`). On ids8191, 1,600 layer-chunks:

| | |
| --- | ---: |
| wall, streamed leg | 130.9 s (a `--counters` run; the same prefill takes 102.4 s without it) |
| weights H2D | 751.1 GB, i.e. 77.9 s at this box's measured 9.64 GB/s |
| gate/up + down results D2H | 35.3 GB |
| activations + hidden rows H2D | 8.6 GB |
| host SwiGLU + quantization | 10.0 s (7.6 % of wall) |
| launches / slot waits | 631,380 / 80,324 |

The 9.64 GB/s there is the campaign's single-large-copy figure. The bench now times the pattern the arm actually
issues - eight 2.32 MiB blobs out of pinned host memory, eight `cudaMemcpyAsync` calls on one stream - at **1.676 ms
per batch, 11.63 GB/s**, so the transfer floor is 64.6 s, not 77.9 s. That still leaves 38 s (0.92 ms per batch) of
the plain prefill that is not transfer, and about half the kernels' own work hidden behind the copies. This is why
every tile in section 4 lands within 1 %.

The pipeline, not the tile, is where the rest sits. Per batch the order is weights H2D (1.97 ms at 9.64 GB/s,
eight groups) → gate/up (1.04 ms) → gate/up D2H → host activation → hidden H2D → down (0.25 ms) → result D2H, and
`stage_gu(i+1)` is only issued after `stage_down(i-1)` has returned, i.e. after batch i-1's results are on the host.
So the next batch's transfer hides behind one down product (a quarter of a millisecond) instead of behind the
previous batch's gate/up kernels (a whole millisecond). Section 6 tests whether a third slot closes that, and the
answer is no.

## 6. Pipeline depth: three and four slots (`STRATA_KOLIBRI_STREAM_SLOTS`) — not kept

**The idea.** Section 5's arithmetic said the host gates the pipeline: `stage_gu(i+1)` is only issued after
`stage_down(i-1)` returned, so the next batch's copy cannot hide behind the previous batch's gate/up kernels. With
three slots the batch finished in the same iteration is i-2, and `stage_gu(i)`'s copy is already in flight while the
host works, so the copy should hide behind a kernel instead of behind the host. `Config::slots` and the schedule are
N-deep (`lag = slots - 1`), every depth publishes the same rows, and the pipeline test now runs its two model-geometry
cases at three slots as well (78 cases).

**It is consistently slower**, in every history and at every depth above two:

| prefill ms, medians | 2 slots (shipped) | 3 slots | 4 slots |
| --- | ---: | ---: | ---: |
| ids512 (511 tokens), 3 reps | 4,852 | 4,899 (+1.0 %) | 4,930 (+1.6 %) |
| ids8191 (8,190 tokens), 2 reps | 102,965 | 105,431 (+2.4 %) | 105,522 (+2.5 %) |
| ids11000 (10,999 tokens), 1 rep | 109,820 | 111,263 (+1.3 %) | — |

Full-logit dumps byte-identical at three slots on all six forced histories.

**Why the arithmetic in section 5 was wrong.** Two slots were already enough. `stage_gu(i)` returns as soon as it has
issued its copies and its kernel launches, so `stage_gu(i+1)`'s copy is issued about one host-stage later - and the
host stage is short - which is early enough to start while batch i's gate/up kernels are still running. The host was
not the gate; adding a slot only put a third stream's copies in flight against the same DMA engines and the same
VRAM, which is the 1 to 2.5 %.

So the residual is not depth and not the tile. It is inside one slot's stream: a batch's own copy (1.676 ms) and its
own gate/up kernels (1.30 ms) are serialized by the dependency, so a batch costs copy + the part of its kernels that
the *next* batch's copy cannot cover. Splitting the copy per expert group and alternating two weight sub-buffers
inside a slot - a group's copy on a copy stream while the previous group's kernels run, joined by an event per group
- is the change that attacks that, and it is a different design from this one.

## 7. A copy stream with a per-group event (`STRATA_KOLIBRI_STREAM_COPY_STREAM=1`) — not kept

**The idea.** On one stream a batch's kernels cannot start until the batch's last byte has landed, so the copy and the
compute of one batch never overlap with each other. Send the weight and hidden-row copies on their own stream, record
an event per expert group, and let each group's kernel wait for its own blob: group g's gate/up then starts while
group g+1 is still arriving. `Config::split_copy_stream`, the pipeline test's two model-geometry cases also run with it
(80 cases, all bit-identical).

**It is slower, and a self-inflicted regression is recorded with it.** The first version of the change also split the
hidden-row copy per group in the *unsplit* path - eight 2 KB copies instead of one - and that alone cost 3.8 % on the
baseline (102,965 → 106,940 ms on ids8191). The numbers below are from the fixed build, where the unsplit path is
byte-for-byte the previous one:

| prefill ms, medians | one stream (shipped) | split copy stream |
| --- | ---: | ---: |
| ids512 (511 tokens), 2 reps | 4,957 | 4,889 |
| ids8191 (8,190 tokens), 3 reps | 102,496 | 105,520 (+3.0 %) |
| ids11000 (10,999 tokens), 1 rep | 109,096 | 109,820 |

Bit-identical on all six forced histories. The reason is the same as section 6's: the two slots already overlap batch
i's copy with batch i-1's compute, so the only thing the per-group split adds is sixteen event handoffs per batch
(eight weight, eight hidden), and a handoff costs more than the overlap it buys.

**Two copy-rate measurements that rule out the source memory** (both in the bench, one stream, eight 2.32 MiB blobs):

| source | per batch | rate |
| --- | ---: | ---: |
| a hot 19 MiB pinned buffer | 1.652 ms | 11.81 GB/s |
| a 43 GiB pinned table, scattered experts (the engine's shape) | 1.839 ms | 10.60 GB/s |

So the engine's table shape costs 10 % of the copy rate, not the 35 % the pipeline loses: the arm's 2.60 ms per batch
against a 1.84 ms copy floor is not a memory-placement problem either.

## 8. SwiGLU + Q8_K quantization on the device — blocked: it is not bit-identical

This is the change sections 6 and 7 pointed at. Per batch the host sits inside the critical path: the gate/up floats
come back (D2H), the host computes the pool's SwiGLU expression, quantizes the hidden row with `native_quant_h`, and
sends it back (H2D). Doing both on the device would delete two of the four round trips.

`tools/kolibri/expert_stream_swiglu_probe.cu` answers whether the device can write the same bytes before anything is
built. It runs random rows of the model's own width (`n_ff = 512`, two Q8_K blocks) through the host's path
(`ff = (g / (1 + std::exp(-g))) * u`, then `q8k_quant_avx2`) and through three device variants of the same
expression, with the quantizer transcribed operation for operation (the magic-add rounding, the low-byte sign
extension, `d = 1 / iscale`, the 16 sums).

20,000 rows per distribution; the input rows are normal with the σ given, because how often a one-ulp float
difference crosses a quantizer boundary depends on the values' size:

| device variant | σ = 1 | σ = 3 | σ = 10 |
| --- | ---: | ---: | ---: |
| `double` throughout, `exp(double)` | 52.2 % of rows | 53.9 % | 20.0 % |
| `float`, `expf`, separate rounding — the host's own precision | 4.28 % | 0.13 % | **0.00 %** |
| `float`, `__expf` | 7.95 % | 0.57 % | 0.00 % |

The floats themselves differ in 12 % of values for the `expf` variant at every σ; what changes with the magnitude is
only how often that lands on the other side of a Q8_K rounding boundary. At σ = 10 not one of 10.24 million codes
moved, at σ = 1 about one row in twenty-three has a code that did. That is a rate, not a guarantee: the arm needs
zero, and a prefill of 8,190 tokens contains 2.4 million hidden rows.

The host's expression is `std::exp` on a float argument, which is `expf`, and CUDA's `expf` is not glibc's: the two
agree to within an ulp but not bit for bit, so one value in nine lands on a different float, and about one hidden row
in twenty-three then quantizes one Q8_K code differently. A different code is a different dot product; the arm's whole
claim is that it is not.

**So this is not an unchanged-arithmetic optimization and cannot be promoted under
[KOLIBRI_ACCURACY_POLICY.md](../KOLIBRI_ACCURACY_POLICY.md) without its own accuracy qualification and an explicit
decision.** It is left as a probe, not as a switch: the number above is what an approval would have to be argued
against, and it would buy the two round trips (0.92 ms of a 2.60 ms batch at 8,190 tokens).

## 9. The expf question, settled exhaustively over all 2^32 floats

Section 8 left a rate. The domain is finite, so this is a proof. `tools/kolibri/expf_bit_probe.cu` walks **every one of
the 4,294,967,296 float bit patterns** and compares four definitions of `exp` bit for bit - glibc's `expf` (what the
pool calls), `(float)exp((double)x)` on the host, and the same two on the device:

| pair | differing patterns (of 4,294,967,296) |
| --- | ---: |
| `(float)exp((double)x)` **device** vs the same **host** | **0** |
| `(float)exp((double)x)` host vs glibc `expf` | 170,648 (1 in 25,163) |
| CUDA device `expf` vs glibc `expf` | 176,674,308 (4.1 %) |

Three things follow, and they decide the option:

1. **There is a definition both sides compute identically**: the double-precision one, exactly, on every float. So the
   device *can* compute a SwiGLU the host also computes - the "mimic" route is possible.
2. **It is not glibc's.** 170,648 patterns differ by one ulp (first: x = 3.0457e-05, host double path 1.0000304,
   glibc 1.00003052). Adopting it means the pool's SwiGLU changes for one input in 25,163 - a numerical change, small
   but not zero, so it is not an unchanged-arithmetic optimization either.
3. **CUDA's own `expf` is the worst of the three** against glibc (4.1 % of floats), which is why section 8's probe saw
   one hidden row in twenty-three move.

So the honest options, in the order this ledger would recommend them:

- **Port glibc's `expf` to the device** and prove it with this same sweep: the target is "0 differing patterns", the
  probe already measures exactly that, and nothing numerical moves. This is the only route to the device-side SwiGLU
  that keeps the pool's bits.
- **Adopt `(float)exp((double)x)` on both sides** (host pool, host bridge and device), which is exact by construction
  and removes the two round trips, but changes 1 exponent value in 25,163 and therefore needs an accuracy
  qualification and an explicit decision under [KOLIBRI_ACCURACY_POLICY.md](../KOLIBRI_ACCURACY_POLICY.md).
- Keep the host bridge, as the branch does today.

**Integers do not help here.** The quantization is already integer arithmetic (the magic-add rounding, the sign
extension, `d = 1 / iscale`); the only non-integer step is the exponential itself, and no integer expression reproduces
a libm's rounding of it. What the integer idea does buy is the insight that the definition has to be *stated*, not
inherited - which is what the double-precision row above is: a statement whose two implementations agree on every float.

## What the four add up to

The streamed arm's remaining cost is not where the Qwen engine's wins are. Its wins are in VRAM-resident decode
kernels that reload the same weights every token, where an instruction saved per weight is a token time saved. Here
the weights arrive over PCIe and each is touched once per batch, so bytes and alignment barely matter, and what is
left is issue count per weight. The gate/up pairing removes a quarter of the activation loads and is worth 1.7 to
3.3 %; the two layout changes that only redistribute bytes are worth nothing or less; and the tile, in either
dimension, is inside the noise because the tile is not what the prefill waits on.
