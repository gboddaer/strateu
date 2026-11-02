# Streamed expert kernels: eight changes tried, two kept

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

## 10. glibc's expf, ported to the device and proven over all 2^32 floats

The first route of section 9: reproduce this host's own `expf` on the device, so nothing numerical moves.

The C source alone is not enough, and the reason is worth recording. glibc's `expf` is an ifunc; on a CPU with FMA
this host resolves to the variant compiled with `-mfma`, and GCC kept two fusions that the C source does not show.
Disassembling the resolved target (`vmovd`/`vcvtss2sd`, then `vfmadd132sd` for `InvLn2N * xd + SHIFT`, then
`vfmsub132sd` for `InvLn2N * xd - k`) shows the arithmetic that actually runs:

```text
kd = fma(InvLn2N, xd, SHIFT)          r = fma(InvLn2N, xd, -k)      y = fma(z1, r2, y1)
```

A transcription of the C source - `z = InvLn2N * xd` as a separate rounded value, used for both - is off by one ulp
in roughly one pattern in 16 million (measured densely over [16, 64): exactly 1 in 16,777,216). With the two fusions
in place: `tools/kolibri/expf_bit_probe.cu`, over every one of the 4,294,967,296 float bit patterns,

| | patterns |
| --- | ---: |
| identical to this host's `expf` | **4,294,876,429** |
| differing | 90,853 |
| NaN inputs, counted apart | 16,777,214 |

Every difference is in the extreme-underflow tail: the smallest `\|x\|` that differs is 103.278931, where `exp` is at
or below the smallest denormal float (host `0x1p-149`, device `+0`), and NaN payloads are canonicalised differently by
the two ISAs. Neither can arise here - a gate/up product above 103, or a NaN, means the layer has already failed - and
the kernel test compares the quantized bytes on the model's own distribution rather than trusting that.

So the device can compute the pool's SwiGLU exactly, and the change is an unchanged-arithmetic optimization: the same
bytes, produced from `(float)exp((double)x)`-free code, with no libm difference left in it.

## 11. The SwiGLU and the Q8_K quantization on the device — kept, and on by default

Section 10 removed the obstacle: the device reproduces this host's `expf` on 4,294,876,429 of 4,294,967,296 float
patterns. This is the change that buys.

**The kernel.** `es_swiglu_quant`: one CTA per (row, block of 256 values), 256 threads. Each value goes through the
pool's expression (`__fmul_rn(__fdiv_rn(g, __fadd_rn(1.f, expf_glibc(-g))), u)`) and the pool's quantizer, transcribed
operation for operation from `q8k_quant_avx2` - the block max, the FIRST value of that magnitude as `mx`,
`iscale = -127/mx`, the magic-add rounding, the low byte sign-extended, the sixteen 16-value sums, `d = 1/iscale`, and
a zero block writes `d = 0` and zero codes and leaves the sums alone, as the reference does.

**Correctness, before any timing.** The kernel test compares the bytes against the host bridge (`std::exp`, then
`q8k_quant_avx2`) over 1,088 rows at σ = 1, 3 and 10, with a zero block every 97th row and both destinations
pre-filled with the same pattern: **0 differing bytes** (101 cases). The pipeline test runs the model-geometry cases
with the device activation as well: 82 cases, bit-identical. And the engine's six forced histories, full 4.6 MB logit
dumps compared with `cmp`: byte-identical at the new default.

**The engine**, medians of the prefill's own `PFchunk` line:

| | host bridge | device | ratio |
| --- | ---: | ---: | ---: |
| ids512 (511 tokens), 3 reps | 4,839 ms | **4,031 ms** | **1.200x** |
| ids8191 (8,190 tokens), 2 reps | 102,343 ms | **89,419 ms** | **1.145x** |
| ids11000 (10,999 tokens), 1 rep | 109,979 ms | **88,644 ms** | **1.241x** |
| ids16 (15 tokens) | 803 ms | 778 ms | 1.03x |
| ids512_de | 6,022 ms | 5,196 ms | 1.159x |
| ids512_code | 6,419 ms | 5,343 ms | 1.200x |

**Why it is that large.** `--counters` on ids8191, same run shape as section 5's:

| | host bridge (section 5) | device |
| --- | ---: | ---: |
| gate/up D2H | 10.1 GB | **0.0 GB** |
| hidden-row H2D | 1.43 GB | **0.00 GB** |
| host SwiGLU + quantization | 9,998 ms | **0 ms** |
| launches | 631,380 | 671,542 |
| the streamed leg's wall | 130,915 ms | **74,362 ms** |

The two round trips were not just bytes: the host had to *wait* for the gate/up results before it could compute the
hidden row, and that stall was inside every batch. With both steps on the card, a batch is gate/up -> SwiGLU+Q8_K ->
down on one stream, and the host only waits for the down results. The streamed leg is now 74 s of work against a 64.6 s
transfer floor, instead of 131 s.

This is the second change of the seven kept (the first was the gate/up pairing), and the only one that removed work
from the critical path rather than rearranging it.

## 12. Which rejections were the float effect, and which were not

Eight things were stopped on accuracy grounds in this repo's Kolibri work, or in the campaign before it. The ported
`expf` makes it worth asking which of them were the *libm* difference - a fixable last-bit artefact - rather than a
real change in the arithmetic.

| stopped | what was recorded | cause | verdict now |
| --- | --- | --- | --- |
| **the SwiGLU on the device** (`kolibri-perf-gap.md`, "The hazard that can still sink it — SwiGLU"; and section 8 here) | "CUDA's `expf` and glibc's `expf` are not guaranteed to agree to the last bit … the first thing the probe must measure" | **this effect**: CUDA's `expf` differs from this host's on 4.1 % of all floats | **explained and fixed**: glibc's expf ported and proven over 2^32 patterns (section 10), the change is now an unchanged-arithmetic one and is the default (section 11), worth 1.145x to 1.241x |
| **FP16 expert caches** (branch `e5f4815c`, `kolibri-t4-f16-correctness.md`) | mean KL 0.0037 to 1.073, worst \|dlogit\| up to 11.27, top-1 agreement down to 0.556 | **not** this effect: fp16 halves the weight mantissa, which moves a product by ~1e-3 relative - four orders above the ~1e-7 an ulp of a libm call can move it, and the KL floor for that comparison was 0.0012 | rejection **stands** |
| **prompt chunk above 8192 / 16384 on CUDA** (`DETAILS.md`) | "not bit-exact on CUDA, so the default stays 8192" | **not** this effect: a bigger chunk changes the *order the blocks are accumulated in*, not a transcendental | rejection **stands** |
| `STRATA_Q8_SM60`, `STRATA_BF16_TC`, the HIP Q8 path | "not bitwise the exact kernels" | different arithmetic by design (no `dp4a`, tensor cores, HIP's Q8) | unchanged; opt-in variants, not accuracy losses |

So exactly one of them was the float effect, it was the largest of them in prospect, and it is now closed. The lesson
that generalises: a "1 ulp" rejection is a *claim about a libm*, and it is testable exhaustively - `tools/kolibri/
expf_bit_probe.cu` answers it for `expf` in three minutes of GPU time, and the same shape of probe would answer it for
`logf`, `tanhf` or `powf` if one of them ever appears on a hot path.

## 13. The switches, and what each one is set to

Every measurement above is a switch. This is the state they are in, and it is the state the numbers in sections 4 to 11
describe:

| switch | default | measured |
| --- | --- | --- |
| `STRATA_KOLIBRI_EXPERT_STREAM` (the streamed arm itself) | unset = streamed | 1.10x to 1.64x over the CPU pool (section 5 of the prefill ledger) |
| `STRATA_KOLIBRI_STREAM_GU` — gate/up pairing | **on** | **1.017x / 1.033x** |
| `STRATA_KOLIBRI_STREAM_DEV_ACT` — SwiGLU + Q8_K on the device | **on** | **1.145x / 1.200x / 1.241x** |
| `STRATA_KOLIBRI_STREAM_SMEM` — shared-memory staging | off | 2 to 3 % slower |
| `STRATA_KOLIBRI_STREAM_PLANES` — plane-split Q6_K | off | 1 to 3 % slower |
| `STRATA_KOLIBRI_STREAM_ROWS` — rows per CTA | 16 | 8 and 32 both within a percent, 8 slightly better at kernel level |
| `STRATA_KOLIBRI_STREAM_SLOTS` — pipeline depth | 2 | 3 and 4 slower |
| `STRATA_KOLIBRI_STREAM_COPY_STREAM` — split copy stream | off | 3 % slower at 8k |

Verified with nothing set in the environment: ids512 prefill **4,140 ms** and ids8191 **89,528 ms**, with full-logit
dumps byte-identical to the reference; `STRATA_KOLIBRI_STREAM_DEV_ACT=0` reproduces the host bridge (4,807 ms on
ids512). The library's `Config` defaults stay explicit (`device_activation = false`, `slots = 2`), so the test suite
keeps covering the old path; the engine is what turns the measured-faster options on.

## 14. The three "not bitwise" GPU kernels: what they are, what they buy, and what can be done

Section 12 put them in one row. They are three different things, and the test that separates them from the `expf` case
is one question: **can the difference be written as a different implementation of the same function, or is the
difference the source of the speed?**

- `expf`: a different implementation of the same function. Port glibc's and the speed is free - which is what section 10
  did, and section 11 banked.
- These three: the difference *is* the speed. Reproducing the reference's arithmetic reproduces the shape that was
  slow, so there is nothing left to port.

### `STRATA_Q8_SM60` - the Q8_0 decode GEMV for Pascal GP100

`src/kernels/cuda/q8_sm60.cuh`, opt-in, GP100 only (the one Pascal chip without `__dp4a`). It reads the
`STRATA_Q8_PACKED` planes - the same int8 values, row-major - with a warp owning a row, 16-byte weight loads through
the read-only cache, the window's activations copied into shared memory once per workgroup, the 4-int dot on
`STRATA_DP4A` (`vmad` on sm_60), and a shuffle reduction with **no barrier per row**.

| PH402 SKU 200, 1050 MHz, Flash-Next IQ3_XXS, 32K, decode tok/s, mean of four prompts (`docs/OLDER_GPUS.md`) | |
| --- | ---: |
| the exact kernels (GSQ-RCO shard) | 28.3 |
| `STRATA_Q8_PACKED=1` - bitwise, the same values in aligned planes | 31.1 (+9.9 %) |
| `STRATA_Q8_SM60=1` | **38.9 (+25 % over the bitwise path, +37 % over the exact one)** |

GPU time per verify window: dense projections 19.2 -> 11.2 ms, head 4.25 -> 2.1 ms.

**Why it is not bitwise**, in the kernel's own header: "the per-32 products are added in another order than the exact
kernels', so outputs are not bitwise theirs (float rounding only; the int8 dots are exact)". A warp owns a contiguous
chunk of the row instead of the exact 8-lane, block-strided ownership, and that is what removes the `__syncthreads()`
per row. **What can be done:** the memory half of the win is bitwise and already shipped separately (`Q8_PACKED`,
+10 %); the remaining 25 % cannot be had without the reordered reduction. No libm is involved, so the `expf` technique
does not apply.

### `STRATA_BF16_TC` - BF16 projections on the FP16 tensor cores

Below sm_80 there are no BF16 tensor cores, so the BF16 projections (hyper-connection, router, indexer, PLE value
matrix) are converted to FP16 and run on the FP16 tensor cores instead of cuBLAS BF16 or an FP32 SIMT kernel
(`docs/NVIDIA_V100.md`). It runs by default below sm_75; `=1` forces it, `=0` turns it off on a V100.
`src/prefill/gemm_bf16_parity.cu` measures it per shape against `cublasGemmEx` on the same BF16 inputs and reports
agreement "within fp32-accumulation rounding" - i.e. not bitwise.

**Why it is not bitwise:** "FP16 tensor-core sums round differently". That is the hardware's 4-element dot unit and its
accumulation order, not a library call. **What can be done:** nothing bitwise. It is a change of arithmetic unit, so it
is either kept off where an exact path exists or taken to an accuracy qualification. No end-to-end percentage is
recorded for it in the docs, and inventing one is exactly what this ledger does not do; the harness times it per shape
(`--bench`).

### The HIP Q8 path - the AMD backend's MMQ

Not a switch: the AMD backend's MMQ multiplies quantized weights with **Q8 activations**, and the note attached to it
(`docs/AMD_HIP_PERFORMANCE.md`) is a *measured* distance rather than a claim: "Real quantized MMQ relative L2 error was
approximately 0.0026 against raw-FP32/dequantized-weight reference, reflecting Q8 activation arithmetic; this is not
bitwise numerical equivalence." Its gain is the backend existing at all.

**Why it is not bitwise:** two layers - the activation format (Q8 rather than the reference's) and the AMD dot's
accumulation order. **What can be done:** of the three, this is the one closest to a decision, because 0.0026 is already
the kind of number an accuracy qualification needs, and the bar is known: the FP16 expert caches were rejected at KL
0.0037 to 1.073 against a run-to-run floor of 0.0012.

### What to do about the class, in order of cost

1. **Take the bitwise-safe half.** Layout, alignment, vector width, staging, cache hints - none of it changes a value,
   and it is most of what makes a kernel fast on an older card. `Q8_PACKED` is the proof (+10 %, bitwise), and both of
   this campaign's wins (the gate/up pairing and the device-side activation) are of this kind: the arithmetic stayed
   the pool's, only where the bytes came from changed.
2. **Census the divergence before arguing about it.** For any candidate, count how many outputs differ and by how much
   - the shape of `tools/kolibri/expf_bit_probe.cu`, and for a kernel, its parity harness. "Differs in 4 % of rows by
   one ulp, only in the denormal tail" and "differs everywhere by 1e-3 relative" are different decisions, and the
   difference between them is a measurement, not an opinion.
3. **Only then the accuracy route.** A change of unit, format or reduction order is a numerical change: it needs a KL /
   worst-|dlogit| study on the six forced histories and an explicit decision under
   [KOLIBRI_ACCURACY_POLICY.md](../KOLIBRI_ACCURACY_POLICY.md). The FP16 arms show what a failing one looks like; the
   HIP path's 0.0026 shows what an arguable one looks like.

## 15. Would an expert cache pay? Measured, and a bigger find behind it

The question, after the streamed leg reached 74 s of work against a 64.6 s transfer floor: how much of that 751 GB is
the *same* experts fetched again? A chunk is 256 tokens and a layer has 384 to 512 experts, so most of one chunk's
experts come back in the next one.

`STRATA_KOLIBRI_STREAM_ROUTE_DUMP=<file>` writes one line per (chunk, layer) with the DISTINCT expert ids the arm is
about to fetch - measurement only, no arithmetic, no file when unset. On ids8191 (32 chunks x 50 layers, 315,690
fetches):

| | |
| --- | ---: |
| distinct experts per (chunk, layer) | median 201, max 299, of that layer's 384 |
| overlap with the previous chunk's set | mean **79.8 %**, median 80.3 % |
| distinct (layer, expert) pairs over the whole prefill | 16,604 of 50 x 384 = 19,200 |
| ceiling for an infinite cache | **94.7 % of the fetches** |

So the reuse is real and large. The obstacle is the *working set*, not the hit rate: one chunk's expert set across all
50 layers is 50 x 200 x 2.38 MB = **23.8 GB**, and the card has 16. A per-layer LRU below that size thrashes - at 128
entries per layer the hit rate is 0.9 %, because a layer's chunk needs ~200 entries before anything survives - and at
256 per layer (30.5 GB) it jumps to 78.8 %. Allocating uniformly across layers is therefore useless on this card.

Residency by *layer* rather than by recency works, and the arithmetic is simple: a resident expert of a layer is
touched ~12 times over the prefill (6,400 fetches per layer for 384 experts), so **each GB of resident experts saves
about 1.2 s at 8k tokens** - 8 GB of the card's spare capacity would buy ~9 to 10 s of the 89.4 s prefill (11 %), for
a real cache implementation. Worth having, not worth mistaking for a step change.

## 15b. A bigger find behind it: the prompt chunk size is worth 40 %, and it changes the answer

The same route dump says something about the chunk size. Fetches are `chunks x distinct-per-chunk`, and
distinct-per-chunk grows far slower than the token count: 201 at 256 tokens. So halving the chunk count halves most of
the traffic. Tried, by returning the full chunk for short prefixes as well (`STRATA_KOLIBRI_PF_CHUNK_FULL=1`, since
removed):

| | chunks | prefill | logits against the reference |
| --- | ---: | ---: | --- |
| ids512, chunk 256 (today) | 2 | 4,073 ms | identical |
| ids512, chunk 511 | 1 | **3,591 ms** | **115,200 of 1,152,000 differ, max \|d\| 3.37, 3 of 9 argmaxes flip** |
| ids8191, chunk 256 (today) | 32 | 89,507 ms | identical |
| ids8191, chunk 512 | 16 | **63,956 ms** | **1,152,000 of 1,152,000 differ, max \|d\| 3.37** |

1.40x, and a different computation. This is not a rounding artefact: every logit moves, by up to 3.4 in absolute
terms, and the greedy pick changes. The knob is gone rather than left as a trap. Section 17 localises it further: the
path is right at 256 tokens and wrong above it.

**What that means, and it is the most important line in this page:** the engine's own default changes its chunk size
at an 8,192-token prefix (256 below, 512 above - `pf_chunk_at`), so as shipped the *same model on the same machine*
computes a long prompt's rows with one arithmetic below that length and another above it. This ledger's whole bitwise
discipline - and the K28 note in `kolibri_engine.cu` that "a chunked causal prefill computes each token's row and
per-token softmax identically regardless of how the prompt is cut" - says that should not happen. Whichever of the two
is wrong, the difference between a 256-token and a 512-token chunk is now known to be 1.13x to 1.40x of prefill time,
so it is worth finding out which side is right before anything else is tuned on top of it.

It is not this campaign's branch to fix (the chunking lives in the pool's prefill path, not in the streamed arm), but it
belongs in the handoff: a reproducible 40 % waiting behind a correctness question.

## 16. The 8,192-token chunk switch: who put it there, and which leg differs

Section 15b left the question of which side is wrong. The history answers the first half. `git log -S "const int small
= 256"` and `git blame` on `pf_chunk_at`:

| commit | date | what it did |
| --- | --- | --- |
| `5bbe47d5` | 2026-10-21 | "S7 chunk tuning - an ADAPTIVE prefill chunk (T=256 while p0 < 8K, else 512)": pp512 4,845/4,856 ms at T=256 against 5,133/5,781 ms at T=512, 32K 307.2 s at 512 against 320.9 s at 256. Also states "oracle BITWISE, T <= n_swa-1 holds". |
| `4a81fd58d` | 2026-10-21 | K28: keys the rule on the prefix length instead of the position, and records the claim that "a chunked causal prefill computes each token's row and per-token softmax identically regardless of how the prompt is cut". |

Both are the maintainer's, and both were tuned on the **pool** - which is the point, because the two legs respond to the
chunk size in opposite ways. Measured on ids11000 (11,000 tokens), a forced cap of 256 against the default 512, all
four combinations run in one build:

| leg | 43 chunks (cap 256) | 22 chunks (cap 512) | |
| --- | ---: | ---: | --- |
| streamed arm | 122,176 ms | **88,660 ms** | **1.38x for the bigger chunk** |
| CPU pool (`STRATA_KOLIBRI_EXPERT_STREAM=0`) | 158,333 ms | 156,569 ms | 1.01x, i.e. flat |

and the dumps, compared byte for byte:

| | |
| --- | --- |
| pool, 256 vs pool, 512 | **DIFFER** |
| arm, 256 vs pool, 256 | **identical** |
| arm, 512 vs pool, 512 | **identical** |

So the divergence is **not the streamed arm's**: at any one chunk size the arm reproduces the pool exactly, which is the
contract it was built to keep. The chunk-size dependence is in the pool's own prefill path, and it predates this
branch. What this branch adds is the price tag: the arm pays 1.38x in transfer for the 256-token chunk while the pool
pays nothing, because the pool reads its experts from its own RAM and the arm reads them over PCIe once per chunk.

Two things follow, and they are for the maintainer rather than for this branch:

- as shipped, the engine's arithmetic changes at an 8,192-token prefix, so the same prompt 10 tokens either side of it
  is computed two ways;
- the streamed arm would be 1.38x faster on the chunk the pool prefers for long prompts, which is where it already
  runs (>= 8192 tokens), and 1.38x slower on the chunk the pool prefers below it.

Localising *where* the pool diverges (which layer, which kernel) is the next investigation, not a tuning step: a
chunk-layer hash comparison between the two chunk sizes would name the first layer that moves, exactly as the PFHASH
work did for the residency bug.

## 16b. What llama.cpp does in the mode those numbers come from

`--cpu-moe` (`-cmoe`) assigns the routed experts' **tensors to the CPU**, so the expert weights are read and multiplied
in host RAM - no PCIe traffic for experts at all - while the GPU does attention and the shared expert. Its MoE is
grouped per micro-batch exactly as ours is (`build_moe_ffn` gathers the tokens of each expert), with `n_ubatch = 512`
as the physical batch for prompt processing, and the same `n_swa` sliding-window constraint.

So the published 723 tok/s on a 512-token prompt is the *same design* as the streamed arm with the experts read from
DDR5 instead of over PCIe 3.0. This branch's counters say one 511-token prompt moves 27.9 GB of expert weights (11,712
fetches x 2.38 MB): 0.33 s at DDR5's ~85 GB/s, 2.6 s at this box's measured 10.6 GB/s. Our measured 4,031 ms against
their 0.71 s is a 5.7x gap on a link that differs by roughly 8x - which is the whole of it. Their decode, where the
weights must be re-read per token, is 18.0 tok/s (`--cpu-moe`) and 11.7 at 32K context; this branch's arm keeps
experts in VRAM for decode and measured ~33 tok/s at 512 context.

## 17. Localised: the prefill's chunk path is right at 256 tokens and wrong above it

Section 16 left the question open: is the chunk-size dependence a rounding artefact (a numerical change to argue about)
or something else? Measuring the boundary answers it.

With a temporary knob that returns the full chunk for a short prefix (not committed), ids512 - 511 tokens, reference
dump from the default 256-token chunk:

| forced chunk | chunks | logits against the 256-token reference |
| ---: | ---: | --- |
| 256 | 2 | **identical** |
| 320 | 2 | 1,151,999 of 1,152,000 differ, max \|d\| **8.16** |
| 384 | 2 | 1,151,996 differ, max \|d\| **11.05** |
| 448 | 2 | 1,151,999 differ, max \|d\| **11.29** |
| 511 | 1 | 1,152,000 differ, max \|d\| **9.79** |

So it is not a threshold near the sliding window (511 against n_swa 513) and it is not a drift: it switches on the
moment the chunk passes 256 tokens, and the difference is up to 11 in absolute logit terms - a different computation,
not a last-bit one. A rounding artefact from a changed GEMM or reduction order would show ~1e-6 relative; this is a
model that computes something else.

Where it starts, measured with the existing `STRATA_KOLIBRI_ATTN_DUMP` instrument (layer 0's attention output, the
first 256 tokens x 48 heads): **bit-identical** between a 256-token chunk and a 511-token one, and no row is left
unwritten. So the first 256 rows' attention is right, the divergence is after it - later rows, or later in the layer
(the routed experts, the shared FFN, or the Q8 quantizer that feeds them).

What that means for the branch, and it matters for how section 15 and 16 should be read:

- the 1.38x that the bigger chunk measured is **the wrong answer** being 1.38x faster. The 512-token path is the
  suspicious one, not the 256-token one;
- the engine's default takes the 512-token path for every prompt of 8,192 tokens or more (`pf_chunk_at`), so a
  long-prompt prefill runs the path that this measurement says computes something else - and the campaign's bitwise
  gates never compared the two sides, because each was checked against itself;
- the streamed arm is not implicated: at a fixed chunk size it reproduces the pool bit for bit (section 16), which is
  exactly what it was built to do.

**So the fix is not a chunk-rule change.** Making every prompt use 256 tokens would restore consistency at the cost of
the speed, and making every prompt use 512 would standardise on the path that disagrees with the reference. The fix is
to find what breaks past 256 tokens in the pool's chunk path: the next instrument is the same one, taken on rows
256-510 (compare a 511-token chunk's rows 256+ against the 256-token run's second chunk, which computes the same
tokens with a valid KV cache), then the shared-FFN GEMM and the Q8 quantizer at Tc = 511. That is a bug hunt in the
pool, with a named starting point - not something this branch should paper over.

## 18. The chunk-size bug, localised and closed

Section 17 showed the prefill computing a different answer above 256 tokens per chunk. This is what it was, how far the
fix got, and what is left.

**Two sources, found by instrumenting the tensors rather than the logits.** The existing `STRATA_KOLIBRI_ATTN_DUMP`
instrument, pointed first at layer 0's attention output and then (temporarily) at its q projection, compares a
256-token chunk against a 511-token one on the same tokens:

| tensor, layer 0, rows 0..255 | 256-token chunk vs 511-token chunk |
| --- | --- |
| q projection (`attn_q`) | 1,530,694 of 1,572,864 differ (97.3 %), max \|d\| 2.0e-5 |
| attention output, before the fix | 1,320,452 differ (84.0 %), max \|d\| 2.8e-4 |
| attention output, after the dense-projection fix | 663,962 differ (42.2 %), max \|d\| **1.8e-7** |

So the first source is the **dense projections**: cuBLAS (and the engine's own GEMM) chooses a kernel by shape, so a
511-row call and a 256-row call accumulate in a different order. Fixed by tiling every dense projection of the chunk
to **fixed 256-row calls** (`gemm_rows` in the layer path: q, k, v and attn_output), which makes any chunk size produce
exactly the M sequence a 256-token chunk decomposition produces. Verified: the 256-token path is still byte-identical
to the stored references, and the cross-chunk difference fell by three orders of magnitude.

The second source is **the attention kernel itself**, still Tc-dependent by one ulp (1.8e-7): its per-row accumulation
is over `attn_mask_bounds(P, ...)` blocks where `P = p0c + Tc`, so a row in the *first* chunk sees 256 key blocks in one
run and 511 in the other even though its own causal bound is the same. Its 1-ulp difference is amplified fifty layers
deep into whole-logit changes, which is why the symptom looked so much bigger than its cause.

**What is closed, and how.** Until that attention path is made chunk-invariant, the engine uses **one chunk size at
every prefix length - 256 tokens**, which is the decomposition every reference dump and every bitwise gate in this
repository was taken on. The adaptive rule (256 below 8,192 tokens, 512 above, commit `5bbe47d5`) is gone from the
rule; the two-branch version is the one-line revert above it in `pf_chunk_at` if a maintainer wants the 512-token path
back before the attention is fixed.

| | before | after |
| --- | ---: | ---: |
| ids512 (511 tokens), prefill | 4,073 ms (2 chunks of 256) | 4,115 ms (same, + the dense tiling) |
| ids8191 (8,190 tokens) | 90,372 ms (32 chunks of 256) | 90,372 ms, byte-identical to its reference |
| ids11000 (10,999 tokens) | 88,660 ms (22 chunks of **512**) | 124,281 ms (43 chunks of 256) |

**Read that third row honestly: the fix costs 40 % of long-prompt prefill speed**, because the faster path was the
divergent one. That is the price of one arithmetic instead of two, and it is a change to what a long prompt computes -
ids11000's stored dump is from the other side of the split and no longer matches. It needs the maintainer's decision,
not this branch's; what the branch can say is that the two paths differ by up to 11 in absolute logit terms, that the
first divergence is a shape-dependent GEMM which is now fixed, and that the residual is one ulp in the attention.

**The follow-up that buys the 40 % back, now with a named target:** make `k_attn_flash`'s key-block loop per-tile
(`min(P, p0c + q0 + QT)`) instead of per-chunk, re-run the 511-vs-256 comparison, and if the attention output becomes
bit-identical the 512-token chunk can come back with the arm's 1.38x on top of it.

## 19. The follow-up: the attention kernel is the last source, and it is priced

Section 18 left one ulp of chunk dependence in `k_attn_flash` and named it. This is what it costs to chase, and what the
chase found.

**The mask is not the cause.** `attn_mask_bounds` gives a row `n = min(pos + 1, P)` with `P = p0c + Tc`, so a row's
visible key count is its own position plus one at any chunk length; the block bound uses `min(q0 + 7, T - 1)`, which for
the same absolute positions is the same number whether the chunk is 256 or 511 tokens. Checked against the source:
the bounds are invariant, so the per-tile key-bound change this section set out to make has nothing to bite on.

**And it is not a race.** Two identical 511-token runs produced byte-identical attention dumps and byte-identical
logits.

**RETRACTED: "the non-flash attention is chunk-invariant".** That measurement was taken after the A/B knob had been
removed from `pf_chunk_at`, so `STRATA_KOLIBRI_PF_CHUNK=511` did nothing and *both* arms ran 256-token chunks - the
comparison was a run against itself. Re-run with the knob back (`STRATA_KOLIBRI_PF_CHUNK_FULL=1`, committed as a
measurement switch), the truth is the opposite:

| ids512, forced chunk 511 against 256 | logits |
| --- | --- |
| `STRATA_KOLIBRI_ATTN_FLASH=0` (cuBLAS attention) | **DIFFER**, and 511 is 3.45 s against 3.98 s |
| `STRATA_KOLIBRI_ATTN_FLASH=1` (flash attention) | **DIFFER**, and 511 is 3.62 s against 4.17 s |

So both attention paths are chunk-dependent, and the last source is *upstream of the kernel choice*. The KV dump
(`STRATA_KOLIBRI_KV_DUMP`) on layer 0 - a sliding layer, stride 513 - says where: the **K cache rows differ from token
64 onward** (193,532 of 262,144 bytes in rows 0..255, first difference at token 64), with identical headers and
identical rows 0..63. That is the next place to look: whatever writes K for a sliding layer above 64 tokens per chunk.

The retraction is recorded here rather than deleted because the lesson is the one this ledger keeps relearning: a
measurement whose switch is inert is not a measurement, and it read as a clean result.

**What that costs, measured so the choice can be made:**

| | flash (today's default) | non-flash |
| --- | ---: | ---: |
| ids512 prefill | 4,115 ms | 4,163 ms (+1.2 %) |
| ids8191 prefill | 90,372 ms | 91,059 ms (+0.8 %) |

**So the 40 % is recoverable, and this is the shape of the trade, not a recommendation this branch can take alone.**
Two ways to spend it:

1. **Keep flash, keep one chunk size** (what the branch ships): every prompt computes the same way, no reference dump
   changes, and long prompts pay the 40 % that the divergent 512-token path used to hide.
2. **Take the non-flash attention, then restore the 512-token chunk for long prompts**: chunk-invariance is restored by
   measurement, the streamed arm gets its 1.38x back (88.7 s at 11k instead of 124.3 s), and the price is a different
   attention kernel - which changes *every* prompt's logits by one to a few ulps and so needs re-baselining every
   reference dump and the maintainer's decision, not a silent default change.

A third way, and the cheapest if someone has a day: find the Tc dependence inside `k_attn_flash` itself (the mask and
the bounds are ruled out above, so it is in the body - the staging loops or the warp-liveness path are the places to
look), and keep flash's 1.2 % with full consistency.

**How far option 3 got, and the one input left.** Reading the body against the same absolute positions rules out the
rest of the mask arithmetic as well: `nmax` and `nn` come from `attn_mask_bounds(P, p0, ..)`, and for non-sliding layers
`P = p0c + Tc`, `first = 0`, so a row's bound is `min(pos + 1, P)` - its own position, chunk-independent; for sliding
layers `first = max(0, p0c + Tc - n_swa)` and the same holds. Every use of `T` in the kernel is `min(q0 + 7, T - 1)`
and `warp_live = q0 + warp < T`, both per absolute position. So a deterministic kernel with identical inputs cannot be
producing the two answers, and the inputs are therefore not identical: `q` was verified after the tiling and `k`/`v`
are its siblings, which leaves **`A.q16`, the fp16 conversion the kernel reads** (`k_to_f16(A.q_m, A.q16, Tc * H * D)`)
as the last unchecked input, or a body path that reads `T` indirectly. Dumping `q16` at layer 0 for both chunk sizes -
one more use of the `ATTN_DUMP` instrument, pointed at `A.q16` instead of `A.att_m` - is the next measurement, and it is
the one that decides between those two.

**The honest summary of the 40 %:** it was never free. It came from a chunk size that changed the answer, and the
answer changed because two kernels - the dense projection GEMM and the flash attention - are shape-dependent. One of
those is now fixed for good (the GEMM, bit-neutral, verified), the other is located to a single kernel and priced at
0.8-1.2 %.

## 20. Where the chunk size still enters: after the k projection, before the attention

Section 19 retracted a bad measurement and pointed at the K cache. This is the bisection, done with the knob that
actually works (`STRATA_KOLIBRI_PF_CHUNK_FULL=1`), on layer 0 where the input (the embeddings) is identical by
construction:

| tensor, layer 0, tokens 0..255, chunk 511 against chunk 256 | result |
| --- | --- |
| raw k projection (`A.k_m`, fp32, before norm/RoPE) | **IDENTICAL** |
| the K cache rows (`s.kc[0]`, after norm/RoPE, as stored) | **193,532 of 262,144 bytes differ, first at token 64** |

and layer 1, whose input is layer 0's output (already different), differs from token 0 - which is what a downstream
divergence looks like, so layer 0 is where the primary one is.

So the chunk size enters **between the k projection and the K cache**, i.e. in the per-token norm + RoPE + write
(`k_attn_prep_multi`) or in the ring gather that both attention kernels read (`k_kv_ring`). Reading
`k_attn_prep_multi` line by line, its per-token work is `pos = p0 + t`, `slot = pos % swa`, a D-wide sum, a per-element
scale and a RoPE rotate - all per token, with no reference to `Tc`. That leaves the ring gather, which is the one piece
of the path that takes `P` and `first`, i.e. `p0c + Tc`.

**What that means for the 40 %, stated plainly.** The chain is now: dense projections (fixed, section 18) -> k
projection (verified identical) -> **the norm/RoPE/ring step, which is where it still breaks** -> both attention kernels
-> logits. The fix is not another instrument, it is one of two edits, and both are small:

1. **Dump `A.k16`** (the gathered ring slice the attention actually reads) at both chunk sizes: if it differs, the fix
   is in `k_kv_ring`'s slice arithmetic (`P`, `first`); if it matches, the fix is in `k_attn_prep_multi`'s write, and
   the "first at token 64" is the clue (64 = 512/8, and the only 64s in that path are block sizes).
2. **Or make the whole per-layer block chunk-invariant by construction**: run the norm/RoPE/ring step and the attention
   in 256-row sub-tiles of the chunk, exactly as section 18 now does for the dense projections. Then any chunk size
   produces the same sequence of 256-row operations the reference was taken on - no dependence left to find - and the
   expert leg keeps the big chunk's 1.38x.

Option 2 is the one that is guaranteed to end this, because it removes the question instead of answering it. It is also
the larger edit: a loop around the prep/ring/attention block with `p0c + t0` and `m = min(256, Tc - t0)` threaded
through, and the buffer bounds checked for the sub-tile case.

## What the four add up to

The streamed arm's remaining cost is not where the Qwen engine's wins are. Its wins are in VRAM-resident decode
kernels that reload the same weights every token, where an instruction saved per weight is a token time saved. Here
the weights arrive over PCIe and each is touched once per batch, so bytes and alignment barely matter, and what is
left is issue count per weight. The gate/up pairing removes a quarter of the activation loads and is worth 1.7 to
3.3 %; the two layout changes that only redistribute bytes are worth nothing or less; and the tile, in either
dimension, is inside the noise because the tile is not what the prefill waits on.
