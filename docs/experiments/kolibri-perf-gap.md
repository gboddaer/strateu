# Kolibri vs Qwen3.8-Flash-Next on the Tesla T4 — performance gap investigation

**Status: Phase 0 in progress.** Purpose: measure and explain why the Qwen3.8-Flash-Next path is faster than the
Kolibri-1 path on this card *without a drafter*, decompose the gap, and rank the levers. Accuracy is preserved or
explicitly qualified; nothing here changes a default.

Branch `kolibri-perf-gap`, worktree `/data2/develop/perf-gap`, cut from `e5f4815c`. The FP16 experiment's branch,
ledger, artifacts and preserved `5bbe6c4e` control (`74b0e257…f418`) are untouched, as is the parked native-arm
worktree `/data2/develop/native-q4q6`.

## What the engines are, as measured

| | Kolibri-1-Q4_K_M | Qwen3.8-Flash-Next UD-Q4_K_XL |
|---|---|---|
| weights on disk | 47,454,113,472 B (1 GGUF) | 4 shards, **111.3 GB** (10.9 MB + 49.86 + 49.38 + 12.09) |
| active params (nominal) | ~2.8 B (6-of-384, 50 layers) | ~6 B |
| engine | `build-kolibri-cuda/strata-kolibri` (branch build) | `/data2/develop/Strata/engine/strata` (`sha256` recorded per run) |
| pack | none — repacks 903 GGUF tensors into 42.9 GiB pinned RAM, ~37 s | pre-converted pack `/home/gbo/Strata-data/packs/unsloth-ud-q4_k_xl` (`dense.bin` 1.49 GB, `native_experts.txt`, `index.txt`, `conversions.json`, `compat-bf16.json`), largest expert blob 3.99 MB |
| KV | f16 | `--kv int8` (+ `--kv-resident 32768` in the shipped config) |

## Findings recorded before any throughput comparison (Phase 0)

1. **This Qwen pack cannot run without a drafter.** The engine refuses with: *"… is a native (IQ) pack: it needs
   `--native` SHARD1, `--spec T (T >= 2)` and `--prefill CHUNK`."* Every shipped Qwen config in
   `/data2/develop/Strata/*.json` passes `--spec 4` with an MTP head
   (`strata-unsloth-ud-q4_k_xl.json`, `strata-unsloth-q8_0.json`, `strata-iq3_s.json`). So a **no-drafter** Qwen
   number cannot be produced from these packs; the closest measurable configuration is the minimum window
   (`--spec 2 --suffix-draft 0 --lookup-chain 0`), and prefill is drafter-independent while decode is not. Every
   decode figure for Qwen in this ledger must carry that label.
2. **The engine binary takes flags directly, not a `generate` subcommand** (`unknown argument: generate`), even though
   its own help text prints `strata generate …`.
3. **Cold start dominates on this box.** With `--expert-cache auto` and the shipped 24,576-slot expert profile the
   engine began preloading experts from the 111 GB file at **~108 MB/s** (29.4 GB in, 3.26 GB per 30 s, process in
   uninterruptible disk wait, GPU idle at 5.7 GiB held). Without `--expert-cache` it still loads ~76 GB RSS before it
   prints anything. On this hardware the flash path's *startup* is tens of GB of disk reads; its steady-state
   throughput is a separate measurement and cannot be inferred from a cold run.
4. **`--kv-resident 32768` with a small `--max-context` is inconsistent** (the shipped config pairs it with
   `--max-context 262144`); the Phase 0 runs use `--max-context 16384 --kv-resident 16384` and say so.

### Kolibri-1, same prompt and generation length (two runs, exit 0)

Command: `strata-kolibri --model …Kolibri-1-Q4_K_M.gguf --n 128 --prefill --cpu-threads 8 --tokens-file ids512.txt
--ctx 2048 --counters` (binary `0bc772bf…acbdf`, the FP16 branch's build, arm off).

```
PFchunk 256/511   5,434 ms
PFchunk 511/511  10,398 ms                      -> 511 tokens / 10.398 s = 49.1 tok/s
CNT prefill pool: chunks=100 experts/chunk=117 bytes=28.09 GB pool=8,569 ms (85.7 ms/chunk, 3.3 GB/s)
CNT prefill pool work: rows=153300 = 1,205.6 GFLOP at 141 GFLOP/s (62% of the 226 GFLOP/s pool roof)
CNT decode: pool 714.2 MB/tok (91.42 GB total) = 3,074 ms -> 24.02 ms/tok at 29.7 GB/s
CNT branches per tok: pool 50.0 gpu-expert 0.0 slots-path 0.0 grouped 0.0
```

## Phase 1 — accuracy-safe levers (gate G1): one variable at a time, same binary

No numerical change: same binary `0bc772bf…acbdf`, same model, same 512-id prompt, same environment except the two
variables named. `--counters` on every run.

| configuration | prefill (511 tok) | pool GFLOP/s (% of the 226 roof) | decode pool work | achieved GB/s |
|---|---:|---:|---:|---:|
| **8 workers, 2-node (baseline)** | **10,388-10,398 ms** (49.1 tok/s) | 141 (62 %) | **24.02 ms/tok** (714 MB) | 29.7 |
| 8 workers, single node | 9,920 ms (51.5 tok/s) | 147 (65 %) | 14.66 ms/tok | 48.7 |
| 13 workers, 2-node | 8,128 ms (62.9 tok/s) | 189 (84 %) | 24.17 ms/tok | 29.6 |
| **13 workers, single node** | **7,922-8,039 ms** (63.6-64.5 tok/s) | **190-194 (84-86 %)** | **12.93-12.96 ms/tok** | **55.1-55.3** |

**And it is bit-identical.** Baseline (8 workers, two nodes) against the tuned configuration (13 workers, one node),
same forced history, full logits:

```
PFchunk 511/511 10,388 ms   vs   7,954 ms
9 records x 128,000 values: bitwise identical = True   (max |a-b| = 0)
```

So the two levers together are:

* **prefill 49.1 -> 63.6 tok/s (+30 %)**,
* **decode expert work 24.0 -> 13.0 ms/token (1.85x)**, at 55 GB/s instead of 29.7 GB/s.

Both are scheduling and placement only — worker count and NUMA — with zero arithmetic change, so they need no
qualification at all. The prefill gain comes from using 13 workers instead of 8 (84-86 % of the pool's compute roof
instead of 62 %); the decode gain comes almost entirely from putting the workers **and the pinned expert table on the
same NUMA node** (24.0 -> 14.7 ms/tok from placement alone at 8 workers, a further 13 % from the extra workers).
On this box the current default splits the pool 5+4 across two nodes, which costs roughly half the decode expert
throughput.

## Phase 1b — the worker knee, and the win in tok/s

Worker sweep at single-node placement (same everything else):

| workers | prefill (511 tok) | pool GFLOP/s | % of the 226 roof |
|---:|---:|---:|---:|
| 8 | 9,920 ms | 147 | 65 % |
| **13** | **7,905 ms** | **194** | **86 %** |
| 16 | 7,904 ms | 195 | 86 % |
| 20 | 7,959 ms | 193 | 85 % |

**The knee is 13** — the engine's own recommendation, matching this box's 14 physical cores. More workers buy nothing.

End-to-end, measured as the difference between one 128-token run and one 1-token run with the same prefill
(55,210 − 51,971 ms over 127 tokens): **25.5 ms/token = 39.2 tok/s** decode, against the 30.5 tok/s baseline
(32.8 ms/token). So the two accuracy-safe levers together give:

| | baseline (8 workers, 2-node) | tuned (13 workers, single node) | change |
|---|---:|---:|---:|
| prefill, 511 tokens | 49.1 tok/s | **64.6 tok/s** | **+32 %** |
| decode | 30.5 tok/s | **39.2 tok/s** | **+29 %** |
| prefill pool compute | 141 GFLOP/s (62 % roof) | 194 GFLOP/s (86 % roof) | +38 % |
| decode expert work | 24.02 ms/tok | 12.96 ms/tok | **1.85x** |
| numerics | — | **bitwise identical** (9 x 128,000) | unchanged |

This is a **configuration change only** (worker count and NUMA placement). It needs no qualification; it is not a
default change and has not been made one.

## Phase 3a — can a bit-exact CUDA port actually exist? The arithmetic contract

Read from the tree, not assumed:

| leg | implementation the pool calls | file |
|---|---|---|
| gate/up, **Q4_K x Q8_K** | **Strata's own AVX2 kernel** `q4k_dot` | `src/kernels/cpu/kq_avx2.cpp:85-145` |
| down, **Q6_K x Q8_K** | **not** in Strata's kernels — `dot_rows()` dispatches only types 12 / 7 / 8, so Q6_K goes through ggml-cpu's `vec_dot` trait | `src/kernels/cpu/kq_avx2.cpp:217-226`; ggml's `ggml_vec_dot_q6_K_q8_K` |

`q4k_dot`, exactly as written (this is the contract a CUDA kernel must reproduce):

```
per 256-value block i:
  xd, xdmin = fp16->fp32 of the block's d, dmin
  d = y.d * xd ;  dmin = -y.d * xdmin
  sumi    = int32 vector: maddubs(q4l, q8l) + maddubs(q4h, q8h), madd'ed with the 6-bit scales,
            accumulated over the four 64-value groups -- all INTEGER, hence exact and order-free
  acc_m  += fma(dmin, (float)int32(prod_of_mins_and_q8_sums), acc_m)      (__m128, 4 lanes)
  acc    += fma(d,    (float)sumi,                       acc)            (__m256, 8 lanes)
after all blocks:
  result = hsum_float_8(acc) + reduce(acc_m)      -- two FIXED reduction trees
```

**Why bit-exact is structurally possible.** Everything order-sensitive is enumerable: the int32 sums per block are
exact; the only floating-point sequencing is (i) the per-lane FMA accumulation across blocks **in block order**,
(ii) the fixed 8-lane and 4-lane reduction trees, and (iii) FMA contraction (CUDA `fmaf` matches
`_mm256_fmadd_ps`; a separate multiply-add would not). None of that needs the GPU to be clever — one thread per output
row, eight fp32 lane accumulators in registers, blocks walked in order, then the same tree. The kernel would be
*deliberately* the same shape as the AVX2 code, not a generic GEMM.

**The hazard that can still sink it — SwiGLU.** Between the two products the pool evaluates
`(g / (1.f + std::exp(-g))) * u` on the CPU. CUDA's `expf` and glibc's `expf` are not guaranteed to agree to the last
bit, and a 1-ulp difference there would break bit-exactness for every element while leaving the arithmetic itself
correct. Options: reproduce the CPU value exactly (hard), or measure the divergence and decide whether the arm is
"bit-exact except an expf ulp" — which is a *numerical change* and would fall back under gate G2. **This is the first
thing the probe must measure, before any kernel work: compare CPU `std::exp` and CUDA `expf` over the full
single-precision range, count exact matches, and report.**

### Phase 3e — the ceiling is bytes, not dot products

The dp4a micro-benchmark (`probe/q4k_speed.cu`, 512 rows x 4 tokens = 10.5 MFLOP) is **overhead-dominated and I am not
using it**: CPU 0.89 ms on 1 thread, **2.22 ms on 13 threads** (thread spawn/join exceeds the work), GPU 0.94 ms. At
this size every number is measurement artefact; the earlier per-launch figure (0.22-0.28 ms) and these 0.9 ms numbers
disagree purely because of what is inside the timed region.

The decisive arithmetic needs no micro-benchmark, because the engine already measured both sides of it:

| | value | source |
|---|---|---|
| expert bytes streamed per 512-token prefill | **28.09 GB** | `CNT prefill pool` (Phase 0) |
| host->device bandwidth measured on this box | **9.6-10.5 GB/s** | the engine's own PCIe probe, every run |
| tuned prefill (13 workers, single node) | **8.0 s** | Phase 1 |
| compute inside that | 1,205.6 GFLOP at 194 GFLOP/s | `CNT prefill pool work` |

So if the expert bytes are moved by DMA instead of read by CPU workers:

```
streaming floor = 28.09 GB / 9.6 GB/s = 2.93 s
ceiling vs today  = 8.0 s / 2.93 s     = 2.7x
```

and that floor assumes the compute is free. It is not free, but on this card it is not the constraint: the T4's int8
path is an order of magnitude above the 194 GFLOP/s the CPU reaches, which is why Qwen — doing exactly this, DMA streaming
plus GPU products — measures 191 tok/s against Kolibri's 64.6.

**With resident weights instead of per-chunk streaming there is no floor at all**, which is why the parked native-residency
arm was attractive: it removes the 28 GB, and its problem was never speed, it was arithmetic.

**What this means for A, stated plainly.** The porting question (bit-exact vs ~1e-6) is second order: even a perfect dot
kernel cannot beat **2.7x** while every chunk streams 28 GB, and it cannot beat *unity* if the streaming is done badly.
The next measurement that matters is therefore a **chunk-scale GPU benchmark** — 117 experts x ~4 tokens, weights DMA'd
over PCIe, one launch per group — checked against the pool's **85.7 ms per chunk-layer**. If that lands near the 2.93 s
floor for a 100-chunk-layer prefill, A is worth building; if it lands at or above 85.7 ms, A is dead and the answer is
the CPU pool we already have.

### Phase 3d — the pool's arithmetic is AVX2-specific, and that redefines "bit-exact"

The probe now compares **three** implementations of the same dot on the same blocks and Q8_K activations
(512 rows x 4 tokens, generated-valid Q4_K blocks, ggml's own quantiser):

| comparison | match |
|---|---|
| **Strata's AVX2 `kq256_rows` vs ggml's own generic `ggml_vec_dot_q4_K_q8_K_generic`** | **1019 / 2048 (49.76 %)** |
| my CUDA port (ggml-generic semantics) vs Strata's AVX2 kernel | 1021 / 2048 (49.85 %) |
| my CUDA port vs ggml generic | 498-512 / 2048 (24-25 %) |

Two conclusions, both important:

1. **The two reference implementations do not agree with each other.** ggml's generic and Strata's AVX2 kernel differ in
   the last bits for half the values (~1e-6 relative; worst absolute difference 1.5e-3 on values of order 1e2-1e3). So
   "bit-exact" is not a property of "the arithmetic" — it is a property of **one specific kernel's summation order**,
   and the pool's is the AVX2 one.
2. **Which means the tolerance question is not binary.** A CUDA port that uses the *same* Q8_K activations and the same
   weight decoding but a different summation order lands ~**1e-6 relative** away — that is **~100x smaller than the
   ~1e-4 seed** which flipped the router in every arm that failed here (fp16 cache, Q8_1 MMQ, grouped-native). It is
   not the same risk at all — but it is still a *numerical change*, so under the accuracy policy it needs the G2
   protocol rather than being waved through as "bit-exact".

**Speed reality check from the same run:** the naive GPU kernel (one thread per row, 10 blocks serial, one launch per
token) takes **0.23 ms per 512-row token** against **0.29 ms for all four tokens on ONE CPU thread** — i.e. the naive
port is ~3x *slower* per token than a single core, and the pool has 13 of them. A real candidate needs dp4a/mma
vectorisation and one launch per group; the ~1e-6 exactness question above is independent of that.

### Phase 3c — the linked probe: infrastructure works, bit-exact parity not yet achieved

`probe/q4k_parity.cu` links the **real** pool kernels (`libstrata_kernels_cpu.a` + ggml-cpu) and puts a CUDA
emulation of `q4k_dot` beside `kq256_rows(12, …)` on generated-valid Q4_K blocks and Q8_K activations produced by
ggml's own quantiser, 512 rows x 4 tokens (one expert's gate *or* up):

```
bitwise: 0 of 2048 match (0.00 %), worst |diff| 499.4 at row 220
time: CPU kq256_rows 0.29-0.31 ms for 512 rows x 4 tokens (36 GFLOP/s, ONE thread); GPU 0.89-1.06 ms
```

**What is established:** the probe infrastructure works end to end (the real AVX2 kernel is callable and timed, the
CUDA kernel runs, comparands are aligned). **What is not:** the port is not bit-exact. Two concrete mapping errors were
found and fixed on the way (the Q4_K nibble layout: one byte per value, low nibble for values 0-31 of a 64-value group
and high for 32-63; and a heap-corrupting brace-initialised buffer in the probe itself), and the residual difference
dropped from 1251 to 499 — so the structure is closer, but the 6-bit scale-to-lane mapping and/or the `acc_m` term are
still wrong.

**The methodological lesson, recorded because it changes how the next step is done:** reverse-engineering the AVX2
kernel's `_mm256_shuffle_epi8` scale broadcast by trial and error is slow and unverifiable. The right reference is
**ggml's scalar `ggml_vec_dot_q4_K_q8_K`** (in the fetched ggml-cpu source), whose value-to-scale mapping is written
out explicitly; `q4k_dot` was authored to match it ("ggml_vec_dot_q4_K_q8_K, __AVX2__"), so implementing the scalar
semantics in CUDA and then checking it against the AVX2 kernel is a two-step verification instead of a guessing game.
That is the next step.

**Speed signals so far, with their limits:** the CPU AVX2 kernel does **36 GFLOP/s on ONE thread** for these dots;
the engine's measured pool roof is 226 GFLOP/s (the 13-worker configuration reached 194). The naive GPU kernel (one
thread per row, 10 blocks serial, per-token launch) is **3x slower** than a single CPU thread — which says nothing
about the GPU's ceiling, only that this kernel is unvectorised. A real candidate needs dp4a/mma and one launch for the
whole group; whether *that* can beat 194 GFLOP/s while streaming 281 MB per chunk-layer is the open question that
decides A, and it is not answered yet.

### Phase 3b — the expf/SwiGLU probe: A survives, but not the way it was written

`probe/exp_parity.cu` (nvcc -O3 -arch=sm_75, no fast-math), 40 million samples, gate values uniform in [-40, 5] plus a
dense linspace, `u` uniform in [-2, 2]:

```
exp(+x):                    12,121,875 of 40,000,000 differ bitwise (30.30 %), worst 2 ulp at x = -31.977
swiglu (g/(1+exp(-g)))*u:    8,647,247 of 40,000,000 differ bitwise (21.62 %)
```

**So a CUDA `expf` is not glibc's, and a port that computes SwiGLU on the GPU cannot be bit-exact.** That is not a
reason to abandon A, because the SwiGLU does not have to move:

**The bit-exact design that survives** — move only the two dot products, keep the elementwise stages on the CPU:

```
GPU: gate/up Q4_K x Q8_K dot          (bit-exact reproduction of q4k_dot)
D2H: gate/up outputs                  rows x 2*n_ff floats   = 1,533 x 1,024 x 4 = 6.3 MB per chunk-layer
CPU: SwiGLU  (g/(1+std::exp(-g)))*u  +  Q8_K quantisation of h   -- the existing code, unchanged
H2D: h (Q8_K)                         rows x n_ff bytes       = 1,533 x 292        = 0.45 MB
GPU: down Q6_K x Q8_K dot             (bit-exact reproduction of ggml's kernel)
D2H: down outputs                     rows x n_embd floats    = 1,533 x 2,560 x 4 = 15.7 MB
```

The intermediate traffic is ~22 MB per chunk-layer against 281 MB of expert weights, so it costs ~2 ms at the measured
9.6 GB/s PCIe — affordable against a 85.7 ms baseline. And the activation quantisation (x -> Q8_K) stays where it
already is, on the CPU, so the values the dots consume are produced by the same code as today.

The alternative — reimplementing glibc's `expf` on CUDA to the last bit — is a porting project with its own risk, and
it is not needed for a bit-exact arm.

**Revised go/no-go for A:** the Q4_K and Q6_K dots must be bit-identical on the real shapes (measurable in the next
probe), the CPU SwiGLU/quantisation round trip must stay affordable, and the whole thing must beat the measured
8.0 s prefill (63.6 tok/s) including the 28 GB of weight streaming. If the dots cannot be made bit-identical, or the
round trip eats the gain, A is reported as a measured dead end.

**Probe design (the next step, no engine change).** A standalone program in this worktree with its own build dir:
1. drive the **CPU pool kernels** (`native_fmt` / `native_quant_act` / `native_gu_rows` / `native_quant_h` /
   `native_down_rows`) at the real shapes, 13 workers single-node → the 85.7 ms/chunk-layer, 194 GFLOP/s baseline;
2. a **CUDA Q4_K x Q8_K kernel** with the contract above, bitwise-compared against `q4k_dot` on random valid blocks
   and on real expert blobs from the model;
3. the same for **Q6_K down** against ggml's x86 kernel;
4. the **expf/SwiGLU parity** measurement;
5. timing including getting the bytes to the card (28 GB per 512-token prefill at the measured 9.6 GB/s PCIe).
Go/no-go: bit-identical on (2) and (3), SwiGLU parity understood, and faster than 194 GFLOP/s *including* data
movement. Anything less and A is reported as a measured dead end.


## Phase 4 — the ranked plan, with what each lever costs and what it does to accuracy

Baseline for comparison after the free win: **Kolibri tuned 64.6 tok/s prefill / 39.2 tok/s decode** against
**Qwen 191 tok/s prefill / 7.65 tok/s decode** on the same card, same prompt, drafter contributing nothing.

| # | lever | measured value | accuracy | cost / gate |
|---|---|---|---|---|
| 1 | **13 workers + single-node NUMA** | **prefill +32 %, decode +29 %, expert work 1.85x** | **none — bitwise identical** | **done, config-only** |
| 2 | **GPU compute for the prefill expert products** (the only lever that can close the remaining **3.0x** prefill gap) | the decomposition says prefill is **compute-bound**: 1,206 GFLOP at 194 GFLOP/s = 86 % of this pool's roof, while Qwen does the same work on the GPU at 19.8 GB/s and reaches 191 tok/s. The card is idle during Kolibri's prefill. | **changed arithmetic** — every GPU expert path measured here flips the router (fp16 0.400/11.08, Q8_1 0.266/9.51, grouped-native 0.370/11.64; 97/100 chunk-layers diverge from layer 2) | largest; needs G2 approval *and* a per-row-count design that avoids the 3.7-rows-per-expert shape where earlier GPU attempts lost |
| 3 | **KV int8 (or k8v4)** | at 512 tokens the KV is a few MB, so ~0 for this prompt; real value at 8k-110k context | changed arithmetic (KV precision) → G2 | moderate; well-understood |
| 4 | **Q4_K down instead of Q6_K** (down ≈ 20 % of expert bytes; Q6_K is ~2.05x Q4_K per value) | ~20 % fewer expert bytes → helps the *memory* side; **prefill here is compute-bound, so little at 512 tokens**, more at long context | changed arithmetic → G2 | moderate |
| 5 | **Load/repack** (37 s to repack 903 tensors into 42.9 GiB pinned) | startup only; Qwen is worse (~11 min of disk reads) | none if the repacked table is cached bit-identically | small, accuracy-safe |
| 6 | *(not a lever)* **copy Qwen's decode** | Qwen's decode is **7.65 tok/s** against Kolibri's 39.2 — PLE reads block 6.06 s of its 16.7 s | — | do not copy |

**Where the gap actually is, in one sentence:** after the free fix Kolibri's prefill is 3.0x behind Qwen and its decode
is 5.1x ahead, and the prefill gap is not bytes — it is that Kolibri multiplies prompt experts on 13 CPU cores at 86 %
of that pool's compute roof while the T4 sits idle, whereas Qwen hands the same products to the GPU.


## Phase 0b — where the milliseconds actually are

| leg | Kolibri-1 | Qwen3.8-Flash-Next | what it means |
|---|---|---|---|
| prefill, 511 tokens | **10,398 ms** (49.1 tok/s) | **2,670 ms** (191.4 tok/s) | Qwen **3.9x** faster |
| prefill bottleneck | **compute**: 1,205.6 GFLOP at **141 GFLOP/s** on 8 CPU workers, 62 % of the pool's 226 GFLOP/s roof | **DMA + GPU products**: 19.8 GB/s over the row phases, CPU share 0.70 | Kolibri's prefill is **CPU-bound**; the T4's tensor cores are idle |
| expert bytes for the prefill | 28.09 GB (3.3 GB/s) | ~22.8 GB (9.6-19.8 GB/s) | same order of bytes, 3-6x the rate |
| decode per token | **24.0 ms** pool work (714 MB at **29.7 GB/s**) → ~30.5 tok/s | **130.7 ms**/token → 7.65 tok/s | Kolibri **4x** faster in decode |
| decode bottleneck (Qwen) | — | PLE table **6,063 ms blocked** of the 16.7 s, pool drain 91.5 ms/round, cache hits 63.1 % | Qwen's decode pays SSD reads and a slower pool drain it does not overlap |
| load | 37 s repack of 903 tensors into 42.9 GiB pinned | ~77 GB read at ~110 MB/s (≈11 min) + 48 s projections + PLE map | Kolibri loads far faster |
| drafter | none | `--spec 2`, **0 of 127 drafts accepted** | the comparison *is* drafter-free in effect |
| KV | f16 | int8 (+16 k resident) | Qwen's KV is half the bytes |

**Three conclusions, all measured:**

1. **The user's premise is half right, and the half that is right is the important one.** Flash is not faster overall — it
   loses decode by 4x on this card — but it wins prefill by **3.9x**, and it does so *on a model 2.3x larger*. The
   mechanism is not a better expert kernel: it is that **Qwen's prompt experts run on the GPU while Kolibri's run on 8
   CPU workers at 141 GFLOP/s**, which the engine itself reports as 62 % of the pool's compute roof. Kolibri's prefill
   is compute-bound, not byte-bound — the 3.3 GB/s is what 141 GFLOP/s *costs* over 28 GB of Q4_K/Q6_K rows.
2. **Kolibri's decode is already excellent and Qwen's is not.** 24 ms/token of pool work at 29.7 GB/s, versus 130 ms
   with 6 s of that blocked on SSD PLE reads. Any plan that "makes Kolibri more like flash" must not copy the decode
   side.
3. **The 9x asymmetry inside Kolibri is the real anomaly.** Decode moves 714 MB/token at **29.7 GB/s**; prefill moves
   28 GB at **3.3 GB/s**. Same kernels, same workers, same RAM. The difference is *row count per expert* (1 vs ~3.7)
   and the batch shape the pool runs — which is exactly the shape where the earlier GPU experiments looked bad. That
   asymmetry, not the model size, is what a plan should attack.



### Qwen3.8-Flash-Next — full run with the residency table (prefill + decode)

Command: `engine/strata --pack …/unsloth-ud-q4_k_xl --native …00001-of-00004.gguf --expert-profile
…/data/expert-profile.bin --expert-cache 1024 --tokens-file ids512.txt --max-new 128 --max-context 16384
--kv int8 --kv-resident 16384 --prefill 512 --spec 2 --suffix-draft 0 --lookup-chain 0 --pool-workers 13 --stats`
(load: ~77 GB read at ~110 MB/s before the first line; see finding 3.)

```
prefill  511 tokens in 2,670.0 ms -> 191.38 tok/s   (time to first token 3,923.5 ms)
decode   128 tokens in 16,724.6 ms ->   7.65 tok/s  (130.66 ms/token wall clock)
PLE      0.6 ms in this run; PLE io during decode: 12,832 rows, 26.3% row-cache hits, 2,880 SSD reads
         (12.1 MB), p50 4,597 us, p99 571,285 us, **blocked 6,063 ms total**
spec     "128 rounds of 2, drafts accepted 0 of 127 (0.000), 1.00 tokens per round"
experts  prefill: 5,709 streamed (all by DMA), 747 resident, 6,526 on the CPU (share 0.70)
         decode: 49.433 ms/token of CPU pool over 48 layers; 1.97 distinct experts/layer over PCIe
         expert-cache hits 69,592 of 110,213 = 63.1 % (cache 1,024 slots, 100 % full)
pool     gate/up 34.171 + quantize 0.428 + down 22.750 = 19.8 GB/s over the row phases
```

**The drafter contributed nothing** — 0 of 127 drafts accepted. So this *is* the no-drafter number the
investigation wanted, by accident of the small verify window and this pack's MTP head.

### Side-by-side on the same card and prompt (both measured this session)

| | Kolibri-1 (pool, 8 workers) | Qwen3.8-Flash-Next (int8 KV, 1,024-slot cache) | winner |
|---|---:|---:|---|
| prefill, 511-512 tokens | 58-60 tok/s (8.46-8.87 s) | **191.4 tok/s** (2.67 s) | **Qwen 3.2x** |
| decode, greedy/1 token per round | **~30.5 tok/s** | 7.65 tok/s (130.7 ms/token) | **Kolibri 4.0x** |
| expert bytes streamed for the prefill | 28.09 GB | ~5,709 x 3.99 MB ≈ 22.8 GB | similar |
| achieved streaming rate | **3.2-3.4 GB/s** (CPU pool) | **9.6-19.8 GB/s** (PCIe DMA + GPU) | **Qwen 3-6x** |
| PLE table | none | 6.06 s of decode blocked on SSD reads | Kolibri |

**This is the finding of Phase 0, and it is not the story we assumed.** Flash is not uniformly faster: it wins
**prefill by 3.2x and loses decode by 4x**. And the prefill win is not a better kernel — it is **the same bytes moved
3-6x faster**, because Strata streams experts by PCIe DMA to the GPU (the engine's own probe: 9.6 GB/s host→device)
while Kolibri's pool reads them with AVX-2 CPU workers at 3.2-3.4 GB/s. Qwen's decode is slow for reasons Kolibri does
not have: 60 % of its decode wall time is PLE table reads from the SSD (p99 0.57 s per read) and its pool drain is
91.5 ms/token.



### Qwen3.8-Flash-Next, prefill (drafter-independent), 511 tokens, no residency

Command: `engine/strata --pack …/unsloth-ud-q4_k_xl --native …00001-of-00004.gguf --tokens-file ids512.txt
--max-new 128 --max-context 16384 --kv int8 --kv-resident 16384 --prefill 512 --spec 2 --suffix-draft 0
--lookup-chain 0 --pool-workers 13 --stats` (no `--expert-cache`; the `--spec` requirement was satisfied only up to
the loader's own check, which then refused — see below).

```
prefill 511 tokens in 1 chunks, 7762.4 ms (65.8 tok/s); experts streamed 6253 (6253 by DMA, host 36.1 ms),
resident 0, on the CPU 6658 (share 0.69); PLE 4682.9 ms
```

| component | ms | share of prefill |
|---|---:|---:|
| prefill total | 7,762 | 100 % |
| **PLE table reads** | **4,683** | **60 %** |
| everything else (attention + experts + dense) | 3,079 | 40 % |

Two things follow immediately. First, **the flash path's prefill is not 3x faster than Kolibri's; it is ~66 tok/s
against Kolibri's ~58-60 tok/s** (8 pool workers, measured this session: 28.09 GB streamed in 8.46-8.87 s for 512
tokens; the 24-worker documentation figure is 51.6-53.6 tok/s). Second, **60 % of the flash prefill is PLE row reads
from the SSD**, a component Kolibri does not have at all — i.e. on this box the flash path is paying a large
SSD-table cost and is *still* marginally ahead, which means its compute+expert leg is substantially cheaper than
Kolibri's: ~3.08 s for 511 tokens ≈ 166 tok/s equivalent.

`--spec` then refused: *"--spec needs the device residency table (--expert-profile, --expert-cache and the token
graph)"*, so decode cannot be measured on this pack without the expert cache (and therefore without the ~76 GB load
and the cache preload). That is the next run.



*(to be filled as the runs complete; nothing is asserted above beyond what is written here)*
