# Streamed expert kernels: four layouts tried, one kept

The streamed expert prefill put the prefill's expert products on the GPU (`docs/experiments/kolibri-gpu-streaming.md`).
This page records four kernel-level layout changes tried on top of it, one at a time, each behind its own switch, each
required to return the same bits as the CPU pool before its speed was looked at. One is kept; the other three are
negative results, recorded so nobody pays for them again without new evidence, and the last section says where the
streamed leg's time actually goes, which is why three of the four could not move it.

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

The weight transfer alone is 76 % of the plain prefill, so at most a quarter of it is anything a kernel can win, and
half of the kernels' work is hidden behind the copies. That is why every tile in section 4 lands within 1 %.

The pipeline, not the tile, is where the rest sits. Per batch the order is weights H2D (1.97 ms at 9.64 GB/s,
eight groups) → gate/up (1.04 ms) → gate/up D2H → host activation → hidden H2D → down (0.25 ms) → result D2H, and
`stage_gu(i+1)` is only issued after `stage_down(i-1)` has returned, i.e. after batch i-1's results are on the host.
So the next batch's transfer hides behind one down product (a quarter of a millisecond) instead of behind the
previous batch's gate/up kernels (a whole millisecond). A third slot, or issuing the next batch's H2D before the
previous batch's event wait, is the change with the size of the remaining gap in it: 102.4 s against a 77.9 s floor.

## What the four add up to

The streamed arm's remaining cost is not where the Qwen engine's wins are. Its wins are in VRAM-resident decode
kernels that reload the same weights every token, where an instruction saved per weight is a token time saved. Here
the weights arrive over PCIe and each is touched once per batch, so bytes and alignment barely matter, and what is
left is issue count per weight. The gate/up pairing removes a quarter of the activation loads and is worth 1.7 to
3.3 %; the two layout changes that only redistribute bytes are worth nothing or less; and the tile, in either
dimension, is inside the noise because the tile is not what the prefill waits on.
