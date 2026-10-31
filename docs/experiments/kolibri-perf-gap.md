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


### Phase 1d — shipped: the worker count now follows the GPU's NUMA node

`setup.py` used to write `--cpu-threads min(24, os.cpu_count())`, i.e. every logical CPU on every node. It now calls
`expert_worker_count()`, which reads the GPU's PCI device from `nvidia-smi`, finds its NUMA node, and counts that node's
**physical** cores (SMT siblings counted once), capped at 24, falling back to the old rule whenever the topology cannot be
read (no `nvidia-smi`, one node, `numa_node` = -1, unreadable sysfs). Six tests in
`tests/terminal/test_expert_workercount.py` pin both the rule and every fallback; the existing
`tools/test_setup_choices.py` and `tools/test_setup_amd.py` still pass.

On this box the rule returns **14** (the node's physical cores). Measured, single node, 511-token prefill:

| workers | prefill | pool GFLOP/s |
|---:|---:|---:|
| 13 (what I had swept) | 8,056 ms | 190 |
| **14 (what the rule picks)** | **7,773 ms** | **199** |

So the shipped value is at least as good as the one I had measured, and the gain is the prefill half of the free win:
**+25-30 % prefill** against the 10,398 ms the old rule produced here (8 workers, two nodes).

**Still open, and not shippable by a flag: the memory binding.** Phase 1c measured that the engine's own
`--pool-cores`/`STRATA_NUMA_POOL` knobs reproduce *none* of the decode half (23.2 ms/token against 12.9 with
`numactl`), because it is `--membind` — the pinned 42.9 GiB expert table's placement — that matters. Delivering it needs
either a launcher prefix (`numactl --cpunodebind=N --membind=N`, with a fallback when `numactl` is absent) or an
engine-side `mbind`/`set_mempolicy`, which is a change in the pool's allocation path. Both keep the arithmetic
identical, so neither needs qualification; neither is in this change.

### Phase 1c — what the free win actually requires (for the product change)

Two attempts to reproduce the `numactl` win **without** `numactl`, so that `setup.py` could emit it as flags:

| configuration | prefill (511 tok) | decode expert work | achieved GB/s |
|---|---:|---:|---:|
| 13 workers, 2 nodes (baseline) | 10,398 ms | 24.02 ms/tok | 29.7 |
| `numactl --cpunodebind=1 --membind=1`, 13 workers | **7,922-8,039 ms** | **12.93-12.96 ms/tok** | **55.1-55.3** |
| `STRATA_NUMA_POOL=0 --pool-cores <node 1's 13 physical cores> --cpu-threads 13` | **10,680 ms** | **23.16 ms/tok** | **30.8** |

**The lever is memory locality, not core pinning.** Pinning the workers to one node's physical cores — through the engine's
own `--pool-cores` — reproduces none of the gain (10,680 ms and 23.2 ms/token, i.e. the baseline). Only binding the
**memory** as well (`--membind`) recovers it (7.9 s and 12.9 ms/token). The pinned expert table is 42.9 GiB; when it is
allocated on the far node, half the workers read it across the interconnect at roughly half the bandwidth.

**Consequence for the product change:** the free win cannot be shipped as engine flags alone. Either

* **the launcher wraps the engine in `numactl --cpunodebind=N --membind=N`** when the machine has more than one NUMA node,
  `numactl` is installed, and the GPU sits on node N; `--cpu-threads` then follows that node's physical core count, or
* **the engine binds memory itself** (`set_mempolicy`/`mbind` on the pinned expert table and its workers), which is a code
  change in the pool's allocation path and works for every launcher including the app.

Both keep the arithmetic identical, so neither needs qualification — the bit-identity was already proven for the
`numactl` configuration (9 x 128,000 logits, max |a-b| = 0).

## A1 one-block oracle — root cause isolated and probe repaired

The operator's latest direction is **one-block oracle for A1, then Phase 2a**. Phase 2b stays parked;
this does not authorise an in-engine A1 rollout or change a default.

`probe/q4k_oneblock.cpp` instruments the **actual** `kq_avx2.cpp` FMA boundaries (intrinsic wrappers,
not a rewritten AVX2 implementation). Across 4,096 seeded blocks, all eight `sumi` lanes and four
minimum-product lanes agree with the scalar mapping. The structural error was simply the missing
minus sign: CPU uses `dmin = -y.d * xdmin`, the CUDA attempt used `+y.d * xdmin`.
The corrected scalar reduction matches all 4,096 output bit patterns; the old sign matches zero.

The CUDA regression check failed before the fix (0/2,048 one-block outputs, exit 1). After the
one-line sign fix, FMA/FMA matches **2,048/2,048** outputs at each of **1, 2, 10 and 32 blocks**
(256/512/2,560/8,192 columns), worst error zero. This establishes the tested dot-product probe,
**not full engine logits, all possible inputs, throughput, or a completed GPU port**.
The generic comparison also had a harness bug: it compared only the last token's GPU output
against all tokens. It now compares the corresponding token, and exactness failure returns nonzero.

Reproduce with `bash probe/run_q4k_oracle.sh`. Raw red/green outputs, host lane dump, binaries,
build log and hashes: `/data4/kolibri-perf0/a1-oracle/`. CUDA sm_75 and host AVX2 tested;
HIP/SYCL not tested (toolchains unavailable). No engine arithmetic changed in this step.

**Correction to Phase 3f:** `stream_floor.cu` queues copies and consuming kernels on a *single stream*;
they are serial, not overlapped. The 2.937/3.239 s timings are real copy/copy-plus-read measurements,
but do not prove a compute-overlapped or in-engine floor. A real expert-compute streaming pipeline
still needs its own timing. Earlier wording claiming overlap/reachability was too strong.

## Phase 2 — approved preregistration (Phase 2a next; Phase 2b parked)

The protocol below was written **before** any measurement of either lever, as the goal's G2 gate requires.
Later execution is recorded separately; the corpus and limits have not been changed.

### What is proposed

| arm | change | where the value is | why it is a numerical change |
|---|---|---|---|
| **2a — KV int8** (and optionally the `k8v4` hybrid) | the Kolibri path's KV cache goes from f16 to int8 codes + an f16 scale per 64 values | **original expectation, not a result**: smaller storage at long context. The tested staged/offloaded arm fails fidelity; no speed test is justified. Encoded bytes decrease 48.44%, not all KV traffic | KV values are quantised, so every attention score changes |
| **2b — Q4_K down (parked)** | Q6_K down weights (210 B / 256 values) become Q4_K (144 B); half the layers already use Q4_K down | **7.10 % fewer expert bytes/token** for this actual mixed-format pack; the original ~20 % estimate was wrong (see audit below) | different weight quantisation: the down result changes at the source |

### Corpus, arms and reference (fixed now, not after seeing numbers)

* **Corpus**: the same four families the FP16 experiment used, so the arms are comparable — 16- and 512-token prompts from
  `tools/kolibri/oracle_ids512.txt`; the 8,191-id KV-wrap fixture; the ~11,000-id attention-transition fixture; and the
  two extra ~512-token prompts (German, code-shaped) with their recorded ids. Nine forced decode steps each, same ids for
  every arm. For 2a the 8,191 and ~11,000 cases are the ones that matter and both are in the set.
* **Arms**: control (f16 KV, Q6_K down, current build), 2a alone, 2b alone, and both together. Every arm run twice.
* **Reference**: the pinned llama.cpp dumper (`/home/gbo/llama.cpp` `a30fb4924`, `--ngl 99 --cpu-moe 999`, f16 KV), whose
  arrangement is recorded with every dump. `strata-kolibriref` remains a diagnostic only.
* **Control calibration first, as in FP16 Task 6**: control-vs-reference is measured and reported before any candidate
  comparison. **The FP16 result must be carried into this text honestly: the control itself failed these limits against
  that reference on all six families (mean KL 0.004-0.145, worst |dlogit| up to 5.5).** So the *same* rule applies: if the
  control fails, the record reads "absolute limits not decidable against this reference", the absolute verdict is void,
  and the arms are judged on **same-engine** numbers (candidate vs control, identical inputs).

### Limits (proposed; the operator sets or replaces them)

Same-engine, per arm, per prompt, per position: **mean KL <= 1e-3, p99 <= 5e-3, worst |dlogit| <= 0.05, top-1 agreement
100 %, and no rank-1 flip where the reference margin exceeds 0.05.** Justification: identical to the FP16 proposal, kept
identical deliberately so the two reports are comparable — and because 2a and 2b each change one thing only (KV precision,
down weights), not the expert arithmetic wholesale.

### Metrics reported

KL (mean/median/p95/p99/worst) and `|dlogit|` percentiles per arm and position, top-1 agreement, reference-winner rank and
margin, **KV bytes/token and expert bytes/token measured**, prefill and decode tok/s separately, VRAM high-water. For 2a a
context sweep (512, 8,191, ~11,000) because that is where its value is; for 2b the byte accounting per token.

### State checks (mandatory, independent of KL)

Two repeats per arm plus first use, replay, prefix reuse, reset, cancellation and a clean request after cancellation, for
each arm. Non-finite values, stale outputs or unexplained same-arm variability reject regardless of the KL numbers.

### Operator decisions, recorded before any result (2026-10-30)

| # | decision | what it authorises |
|---|---|---|
| **2a+2b** | **(i) approve both as written**, limits as proposed | KV int8 (`k8v4` optional) and Q4_K down may be implemented and measured; the corpus, reference, control-calibration-first rule and limits in this section are the approved protocol; the FP16 limits were kept identical on purpose and are not re-derived after results |
| **A** | **benchmark first**, then **A1** if the floor is reachable | the chunk-scale streaming benchmark runs before any port work; if it shows the 2.93 s streaming floor is reachable, the port is built to A1 (emulate the AVX2 summation order exactly — bit-identical, no numerical qualification). If the floor is not reachable, A stops |
| **memory binding** | **launcher prefix** (`setup.py` writes the node, `serve/server.py` applies `numactl --cpunodebind/--membind`) | this is the decode half of the free win (24.02 → 12.93 ms/token measured). It must fall back cleanly when `numactl` is absent or the machine has one node, and it changes no arithmetic |

At this approval checkpoint no implementation or result existed against these approvals: they were recorded first,
as the accuracy policy requires.

### Phase 2a execution notes (before candidate end-to-end results)

Ruling: exercise **int8 only** first (not optional k8v4). The durable format is 64 signed codes
in [-127,127] plus an FP16 scale (66 bytes versus 128). Scale is max(abs(f16 inputs))/127,
rounded to FP16; codes use round-to-nearest-even and the *rounded* scale. A nonzero block
whose scale underflows uses the smallest positive FP16 subnormal; zero blocks have zero codes
and scale. Decode/attention consumes FP16-rounded reconstructions so that the existing attention
arithmetic is otherwise unchanged. This is an experimental, default-off numerical change.

Ruling: first implement a **staged compressed cache** behind `STRATA_KOLIBRI_KV_INT8=1`.
One shared FP16 K/V workspace restores a layer before its attention preparation, then only new
rows are packed and rounded back into that workspace. This reduces encoded durable-cache bytes
but adds unpack traffic and staging allocation. It does not establish a speed case for fused/direct
int8 attention, nor halve the FP16 attention traffic.

**Placement correction:** the initial claim that this preserved control placement was wrong.
Allocator inspection is **static analysis, not a runtime pointer-attribute audit**: `halloc()` in
`src/kolibri/kolibri_engine.cu` calls `cudaMalloc`, despite the nearby legacy comment about host KV.
The control therefore requests GPU KV storage. This prototype explicitly requests **host** packed
storage when `KV_DEV` is unset, plus a GPU FP16 workspace. It is an offload/compression prototype,
not a same-placement dtype comparison. The earlier predicted long-context speed benefit based
on a host-resident control is withdrawn. No speed result is claimed.

Existing mirrors are retained; pipelining, graphs, grouped decode and prefix fingerprints are
refused as recorded below. The f16 gate-off arithmetic remains unchanged.

The fixed six histories, nine records, two runs, independent reference, calibration-first rule,
limits and state checks above remain unchanged. Preserve pinned old references read-only; collect
new same-day f16 controls in this worktree/build. If numerical limits fail, no promotion or task-quality
pass is inferred; task-quality qualification still needs its own approved statistical design.

### Phase 2a build and validation checkpoint

Separate build: `/data2/develop/perf-gap/build-perf-cuda`, Release, CUDA sm_75,
`STRATA_NATIVE_EXPERTS=ON`, `STRATA_Q6K_EXPERTS=ON`, `STRATA_MMQ_KQUANTS=ON`.
Build/configuration logs and all new runs: `/data4/kolibri-perf0/phase2a/`.
No files in the FP16 evidence tree are written. The network server is not started;
mandatory state checks use short-lived stdio engine processes only.

* The CUDA format test was red with empty kernels (959 mismatches), then green with
  the implementation. The registered `kolibri_kv_int8_test` passes on the T4.
  It checks wrap, partial per-head restore, zeros, underflow scale, repeat unpack,
  and pack/roundtrip agreement. CUDA engine build passes; HIP/SYCL not run (unavailable).
* Focused Python checks: **72 passed**, one plugin deprecation warning (matrix environment,
  comparison gates, KL/layer/qualification tools, worker-count/launcher rules).
* Bare repository-wide `python -m pytest`: **collection failed, 33 errors**. Missing
  `regex`, `safetensors`, `textual`, and `libtqenc.so`, plus standalone script collection
  failures. Full names and traces are preserved in `phase2a/pytest.log`; this is **not**
  a passing repository-wide suite. No claim of a green full suite is made.
* `phase2a_run.py` collects two same-day controls for every history before candidates.
  It refuses candidate measurement if gate-off logits differ from the preserved control,
  if repetition differs, or calibration is incomplete. Cached independent-reference
  dumps are read-only, with their hashes and original commands in the manifest.
* State checks add a genuine warm-prefix/cold comparison (including generated history),
  require a positive reported reuse count, and check cancellation's actual reason.
  The older state helper's simple extension alone did not prove that reuse occurred.

The planned dedicated timing sweep was **cancelled before launch** after numerical rejection
and the operator's accuracy-first feedback. The original queue incorrectly treated successful
experiment completion as acceptance, even though the numerical results rejected the arm.
`phase2a_speed.py` now refuses rejected/incomplete numerical or state evidence **before** reading
an executable, sampling hardware or starting an engine. Regression tests were red before this
gate and green after it. An actual invocation against these results refuses before even opening
a deliberately nonexistent binary (`speed-gate-confirmation.log`).

No dedicated prefill/decode speedup is reported. Incidental correctness-run timings remain raw
observations, not a reason to continue an inaccurate optimization. This minimum numerical/state
gate does not waive task-quality qualification or separate approval before promotion.

### Phase 2a failed preflights and recovery — preserve the existing mirror path

The first matrix in `phase2a/` finished all twelve f16 controls. Every family repeated bitwise
and matched the preserved control; all six independent-reference calibrations failed the absolute
limits, so the absolute verdict is void under the preregistered rule.

The first int8 run exited **2 before generation**: the initial compatibility check rejected K/V
mirrors, which are **on by default**. That was an implementation restriction, not an int8 fidelity
result. Its logs/manifest remain in `phase2a/failed-mirror-preflight/`.

An attempted mirror-off bridge was then **rejected**, not substituted for the control. On the
16-id history, 1,151,993 / 1,152,000 logit bit patterns differed; mean KL **0.001246524**, worst
absolute difference **1.050734**, top-1 9/9. The source comment claiming mirror placement is
bit-identical is not sufficient proof for this build/configuration. Details are retained in
`phase2a/mirror-off-diagnostic.json`. No int8 numerical comparison was made against that bridge.

**Superseding ruling:** keep mirrors **explicitly on for both arms**. The packed arm restores and
quantizes the current layer's FP16 workspace *before* the existing full/incremental mirror copy;
that copy and its attention arithmetic remain unchanged. The overbroad mirror refusal was removed.
Graphs, pipelining, grouped decode and cross-layer prefix fingerprints remain refused (the latter
assumes persistent per-layer FP16 buffers and would otherwise inspect the wrong staged layer).
Neither the quantization rule nor any acceptance limit changed.

A new build and entirely fresh paired matrix use `phase2a-mirrors/`. The first build is frozen
with its original hash in `phase2a/frozen-first-build-strata-kolibri`; old results are not passed
off as controls for the changed binary. Startup with int8 plus existing mirrors succeeded:
**exit 0, `READY 2048 stop`, zero generation requests**. The initial smoke wrapper incorrectly
required a bare `READY` line; its failed derived check and raw output remain recorded, alongside
the corrected prefix-based validation (`startup-validation.json`). No extra candidate generation
was used to bypass calibration.

Recorded environments now redact inherited credentials and PI session fields without modifying
the actual child environment. Existing new-experiment manifests were sanitized too; this touched
no FP16 evidence or other protected artifacts.

### Phase 2a final result — reject the staged int8 arm; no speed sweep

The mirror-preserving build finished twelve fresh f16 controls and twelve int8 runs. Every history
has nine forced records of 128,000 logits. Both arms repeat bitwise on every history. Every f16
control matches the preserved control. All six independent-reference calibrations fail the absolute
limits, so that absolute verdict is **void**; the agreed same-engine limits still decide the arm.

| history | mean KL | p99 KL | worst absolute logit difference | top-1 |
|---|---:|---:|---:|---:|
| 16 | 0.00264595 | 0.0127555 | 0.708475 | 9/9 |
| 512 | 0.443848 | 3.63066 | 12.6105 | 9/9 |
| German 512 | 0.0392575 | 0.0796853 | 2.02248 | 8/9 |
| Code 512 | 0.0704882 | 0.137821 | 4.18967 | 7/9 |
| 8,191 | 0.0573471 | 0.158806 | 3.08119 | 7/9 |
| 11,000 | 0.00907386 | 0.0255144 | 1.55351 | 8/9 |
| **limits** | **<= 0.001** | **<= 0.005** | **<= 0.05** | **9/9** |

**All six histories reject.** Matching the English top-1 tokens does not rescue the distributions.
Source: `phase2a-mirrors/candidate-metrics.json` and the raw dumps; limits were not changed.

**Passed:** CUDA format unit test; forced-history repeatability; gate-off control bit identity;
token-level first-use/replay/after-cancel/after-reset checks for both arms; actual cancellation
reason `cancel`; warm-prefix/cold token agreement with **256 reused ids** for both arms.
These are token-level state checks, not full-logit proofs after every state transition.

**Failed:** numerical fidelity on every history. The result rejects this implementation, not every
possible int8 quantization scheme. No numerical fix, default promotion or task-quality pass is inferred.

**Inconclusive/void:** absolute fidelity versus the independent reference, because the control fails.

**Not run:** dedicated speed sweep (cancelled after the operator questioned timing a rejected arm),
paired task-quality qualification, optional k8v4, and Phase 2b. HIP/SYCL were unavailable. The broad
Python suite could not collect (33 errors); focused checks pass (72).

### Phase 2a storage and sampled VRAM — not a speed claim

The wire format is **66 bytes / 64 values**, against 128 bytes for f16: 48.44% fewer encoded bytes.
The table gives allocated durable K+V capacity, not active-token traffic. Control capacity is derived
from the same geometry and its f16 allocator; packed capacity/staging are printed by the engine.
Existing FP16 transpose mirrors are additional and unchanged in both arms.

| prompt / context capacity | f16 K+V capacity, GPU in source (MiB) | int8 K+V capacity, host (MiB) | int8 GPU workspace (MiB) | sampled total GPU peak, f16 / int8 (MiB) |
|---|---:|---:|---:|---:|
| 512 / 2,048 | 80.078 | 41.290 | 4 | 2,953 / 2,863 |
| 8,191 / 12,288 | 280.078 | 144.415 | 24 | 3,175 / 2,905 |
| 11,000 / 14,336 | 320.078 | 165.040 | 28 | 3,219 / 2,913 |

Peaks are the maximum of two runs, sampled every 0.5 s, not an allocation-level high-water mark.
Lower VRAM here mixes offload, encoding and workspace differences; it is **not** a pure quantization
benefit or a speedup. The accuracy failure makes further speed work on this arm unnecessary.
The GPU is idle, the network server stayed stopped, and no default was changed.

### A1 evidence — the multi-block oracle (branch `kolibri-a1-prefill`, from `7720e7e4`)

`probe/q4k_multiblock.cpp` instruments the **real** `q4k_dot` at its FMA boundaries and, for every block, compares
the eight integer lanes, the four minimum-correction lanes, both scale factors and the final result **bit for bit**
against a scalar re-derivation. It is host-only, so it runs while the GPU is busy, and a mapping change costs a
seconds-long host build instead of a CUDA rebuild.

| blocks per dot | trials | outcome |
|---|---:|---|
| 1 | 2,048 | 100 % bit-identical |
| 2 | 2,048 | 100 % bit-identical |
| 10 | 2,048 | 100 % bit-identical |
| 32 | 2,048 | 100 % bit-identical |

Discrimination is checked too, so a "100 %" cannot be vacuous: the **positive** minimum correction — the original
defect — is detected in **64/64** runs, and the correct sign passes **64/64**. Raw output: `a1-oracle/multiblock.txt`.

Reproduce: `bash probe/run_q4k_oracle.sh` (the CUDA half is skipped with `SKIP_CUDA=1`). The same script still runs
`q4k_oneblock` (4,096/4,096) and, with the GPU free, `q4k_parity` at nb = 1, 2, 10 and 32.

**What this does and does not establish.** The summation order is now fully pinned for the Q4_K x Q8_K dot, including
the sign convention that cost the earlier attempt. It does **not** establish anything about the engine: no kernel in
`src/kolibri` uses this order yet, so there is still no engine-level bitwise result and no speed measurement.

### A1 kernels — both expert products are bit-identical on the GPU

`src/kolibri/kq_a1.cuh` holds the two kernels; `probe/q4k_parity.cu` and `probe/q6k_parity.cu` test them against the
functions the CPU pool actually calls. One warp covers four rows and eight lanes own the eight integer lanes of the
dot, so every lane accumulates in the scalar order and no cross-lane reduction reorders anything.

| product | reference | nb = 1, 2, 10, 32 | worst difference |
|---|---|---|---|
| Q4_K gate/up | `kq256_rows` -> Strata's AVX2 `q4k_dot` | **2,048/2,048 each** | **0** |
| Q6_K down | `ggml_vec_dot_q6_K_q8_K` (the pool's trait) | **2,048/2,048 each** | **0** |

Shape guards are exercised too: null pointers, `nb` of zero or beyond 1,024, and out-of-range rows all return a
refusal, and a valid geometry returns none.

Three defects were found and fixed on the way, each invisible without a bit-level comparison:

1. **The scale of a 32-value Q6_K chunk is a pair, not one value.** `get_scale_shuffle` expands an index into the
   sub-block pair `(2n, 2n+1)`, so the first 16 values use `scales[8j+2c]` and the second 16 `scales[8j+2c+1]`.
   Using one scale for all 32 gave 0 % agreement and errors in the thousands.
2. **Each 128-value group consumes 64 `ql` bytes**, not 32; the second group's base was wrong.
3. **The Q6_K reference needs `ggml_cpu_init()` first**, otherwise the type traits are uninitialised and every
   reference value comes back 0 — which reads as a total mismatch and sends you hunting in the wrong place.

Raw output: `a1-oracle/multiblock.txt`, `a1-oracle/q6k-nb*.txt`. Reproduce: `bash probe/run_q4k_oracle.sh`.

**Scope of the claim.** This is bit-exactness of the two dot products against the functions the pool calls, measured
on random blocks with signed scales, both signs of `dmin`, and all six-bit encodings. Nothing in the engine uses them
yet: no engine-level bitwise result exists and no speed has been measured.

### A1 status — kernels proven, engine wiring not started

Done and committed on `kolibri-a1-prefill` (`33b79257`, `de29e3d6`):

| piece | state |
|---|---|
| Multi-block oracle vs the real AVX2 kernel | 100 % bit-identical at nb = 1, 2, 10, 32; sign mutant caught 64/64 |
| Q4_K gate/up kernel (`src/kolibri/kq_a1.cuh`) | 2,048/2,048 at nb = 1, 2, 10, 32, worst difference 0 |
| Q6_K down kernel | 2,048/2,048 against `ggml_vec_dot_q6_K_q8_K`, worst difference 0 |
| Shape guards | refuse null, `nb` 0 or > 1,024, out-of-range rows |

Not started: `a1-engine`. This is the honest boundary, and the reason is size, not doubt about the arithmetic.

Wiring it is a **new expert leg**, not a flag. The chunk's activations must be quantised to Q8_K, the touched expert
blobs staged, the gate/up dots run on the GPU, the results returned to the CPU for SwiGLU — which cannot move to the
GPU, because CUDA's `expf` differs from glibc's in 30 % of sampled values (Phase 3g) — and then the down dot run
again on the GPU. Each stage has to produce byte-identical `ey` rows to `pool_experts`, in the same selection order,
for 100 chunk-layers per prefill.

It also needs the card to itself: `a1-gates` and `a1-timing` cannot run while the operator's 160K server holds
14.9 GiB of the T4's 16 GiB. The kernels themselves were tested alongside it because they are tiny.

So the remaining work is the 1-3 days estimated earlier for the arm, plus the two later tasks, and it should start
with a quiet GPU.

### A1 engine wiring — the hook, and why it is a three-stage leg

The prefill expert call site is `pool_experts` / the per-shard `native_*` group. Reading
`include/strata/kernels/cpu/native_expert.hpp` settles the shape:

```
native_gu_rows(f, blob, act, nt, ff, r0, r1);   // ff[t][r] = silu(gate_r . a[t]) * (up_r . a[t])   <- SwiGLU INSIDE
native_quant_h(f, ff[k], hqp);                  // hidden -> the down activation (Q8_K / Q8_0)
native_down_rows(f, blob, hqs, nt, out, r0, r1); // out[t][r] = down_r . hq[t]
```

So the GPU leg is **three stages, with a CPU round trip in the middle**, not a kernel swap:

1. GPU: `gate_r . a` and `up_r . a` for the FF rows (the A1 kernel, already bit-exact).
2. CPU: `silu(g) * u` — this must stay on the CPU, because CUDA's `expf` differs from glibc's in 30 % of sampled
   values (Phase 3g), so moving it would change the arithmetic; then `native_quant_h`, unchanged.
3. GPU: `down_r . hq` for the N rows (the Q6_K/Q4_K kernel, already bit-exact).

That also fixes the staging requirement: per layer-chunk the GPU needs the touched expert blobs (already staged for
the grouped path) and the Q8_K activations, and the host needs FF floats per selection between stages 1 and 2 — for a
512-token chunk that is roughly 1,533 selections x FF floats, which is small, but it is a real round trip and belongs
in the timing.

The GPU was freed for this work (server stopped: 0 MiB used, port 8095 closed); the default path still references
none of these kernels, so it remains byte-identical by construction.

### A1 arm — wired, bit-identical, and 2x SLOWER (negative speed result)

The arm exists: `STRATA_KOLIBRI_EXPERT_GPU=1` sends the layer-chunk's expert products to the GPU through the same
three stages, and an unsupported layer is **refused with exit 2**, never handed back to the pool silently. Unset,
none of it runs.

**Correctness — passed.** Full logits, nine forced records of 128,000 values, compared byte for byte:

| history | arm off | arm on | preserved `5bbe6c4e` |
|---|---|---|---|
| 16 | `6d1f4fd7a3475209` | `6d1f4fd7a3475209` | identical |
| 512 | `8628012b934ed4e4` | `8628012b934ed4e4` | `8628012b934ed4e4` |
| de512 | `cf530c2c47539ba3` | `cf530c2c47539ba3` | identical |
| code512 | `a7e79144d452209d` | `a7e79144d452209d` | identical |
| 8191, 11000 | running | running | — |

The arm-off path is unchanged: its dump equals the preserved control's, so the new code cannot have moved the
default path.

An intermediate check also passed on real data before the arm was wired: at layer 2 the hidden rows were 512/512
and the down row 2560/2560 bit-identical to what `native_gu_rows`/`native_down_rows` produced.

**Speed — failed.** 511 tokens of prefill:

| arm | prefill | rate |
|---|---|---|
| off (CPU pool) | 7.989 s | 64.0 tok/s |
| on (GPU expert products) | **16.164 s** | **31.6 tok/s** |

2.02x slower. The cause is the shape, not the arithmetic: this arm runs one selection at a time (3 launches, 2
synchronisations and ~16 KB of traffic each, roughly 1,533 times per layer-chunk). The per-block kernel is a
per-lane integer loop, correct but narrow — it reaches ~68 million dots/s in the microbenchmark, while a
chunk-layer needs ~5.5 million dots on top of that traffic. Batching the selections into one launch per stage
would remove most of the synchronisation, but a *competitive* version needs a tiled gemm-shaped kernel, which is a
separate project (the parked native track measured 1.2-1.7x over the FP16 GEMM for one such shape at 1-4 rows per
expert, and lost 1.23x at 32).

**Verdict:** A1 is arithmetically solved and performance-rejected for now. It stays off; nothing is promoted. This
is the negative result the plan said would end the arm if it did not beat 8.0 s per chunk.

### A1 gates — all six histories bit-identical, state checks pass

With `STRATA_KOLIBRI_EXPERT_GPU=1`, nine forced records of 128,000 logits per history, compared byte for byte with
the arm off (and, where it exists, with the preserved control):

| history | arm on | control | |
|---|---|---|---|
| 16 | `6d1f4fd7a3475209` | identical | pass |
| 512 | `8628012b934ed4e4` | `8628012b934ed4e4` (preserved `5bbe6c4e`) | pass |
| de512 | `cf530c2c47539ba3` | identical | pass |
| code512 | `a7e79144d452209d` | identical | pass |
| 8191 | `59d1b413d88e709f` | identical | pass |
| 11000 | `f046946aed454eba` | identical | pass |

State checks with the arm on: first use, replay (`equals_first_use`), a 576-id reuse request, cancellation (reason
`cancel`), a clean request after cancellation and after a reset — **PASS** (`a1-oracle/state_a1.json`).

Prefill cost of the arm at long context: 8,190 ids took 288.5 s against 115.8 s arm-off (2.49x), 10,999 ids 388 s
against 155.4 s (2.5x). Consistent with the 2.02x measured at 511 ids.

### The 160K server's prefill rate — its configuration is NOT the cause

Same 10,999-id prompt, same binary, same 14 workers, one run each:

| configuration | prefill | rate |
|---|---|---|
| context 14,336, expert cache off | 155.4 s | **70.8 tok/s** |
| context 163,840, `--expert-slots auto` (73 slots/layer, 8,465 MiB) + NUMA binding | 157.5 s | **69.8 tok/s** |

1.4 % apart. So neither the large context nor the expert cache explains the **56 tok/s** observed when the same
prompt went through the HTTP server as its first request (11,149 tokens in 199.5 s). The difference must lie in the
server path — first-request warm-up (CUDA module load, cuBLAS handle, allocator growth) or the server's own prompt
handling — which this campaign did **not** measure. Recorded as an open, named gap rather than a guess.

### What would make me stop early

If 2a shows no gain at 8k-110k, or 2b's byte saving does not move prefill or decode measurably, the lever is dropped as
not worth a numerical change — recorded as a negative result, with no qualification run.

### Read-only byte audit while Phase 2a runs — correction, not a Phase 2b restart

`probe/expert_bytes.cpp` reads the actual GGUF metadata/tensor geometry through ggml without
loading tensor data or using the GPU. Output: `phase2a/expert-bytes.txt`.

* Q4_K: **144 B / 256 values**; Q6_K: **210 B / 256 values**. Ratio **1.45833**, not 2.05.
* Gate/up: 50 Q4_K tensors each, input/output dimensions 2,560 x 512.
* Down: **25 Q4_K and 25 Q6_K** tensors, dimensions 512 x 2,560. Half is already Q4_K.
* The pack metadata says six active experts. Exact routed weight traffic is **714,240,000 B/token**,
  agreeing with the measured **decode** counter of 714.2 MB/token. All down weights contribute 271,872,000 B/token.
* Replacing only the remaining Q6_K down weights would save **50,688,000 B/token = 7.096774 %**,
  not ~20 %. This is **decode** format geometry, not a measured speedup or a numerical result;
  prefill savings also depend on the layer-specific expert touch counts.

This corrects the old estimate in the proposal and ranking; it changes no numerical limits or arms.
The parked Phase 2b task contract also repeats the old incorrect geometry and must be corrected
with the operator before any later execution. Phase 2b remains unimplemented and unrun.

## Phase 4 — the ranked plan, with what each lever costs and what it does to accuracy

Baseline for comparison after the free win: **Kolibri tuned 64.6 tok/s prefill / 39.2 tok/s decode** against
**Qwen 191 tok/s prefill / 7.65 tok/s decode** on the same card, same prompt, drafter contributing nothing.

| # | lever | measured value | accuracy | cost / gate |
|---|---|---|---|---|
| 1 | **13 workers + single-node NUMA** | **prefill +32 %, decode +29 %, expert work 1.85x** | **none — bitwise identical** | **done, config-only** |
| 2 | **GPU compute for the prefill expert products** (the only lever that can close the remaining **3.0x** prefill gap) | the decomposition says prefill is **compute-bound**: 1,206 GFLOP at 194 GFLOP/s = 86 % of this pool's roof, while Qwen does the same work on the GPU at 19.8 GB/s and reaches 191 tok/s. The card is idle during Kolibri's prefill. | **changed arithmetic** — every GPU expert path measured here flips the router (fp16 0.400/11.08, Q8_1 0.266/9.51, grouped-native 0.370/11.64; 97/100 chunk-layers diverge from layer 2) | largest engineering effort; A1 needs full-engine bitwise/state gates and a real pipelined timing case. Any arithmetic change needs new G2 approval; the small rows/expert shape still matters |
| 3 | **Staged KV int8 — rejected** | Encoded bytes decrease, but all six fidelity cases fail. Dedicated speed testing cancelled; memory reduction also includes offload | G2 **fails**; keep f16. Optional k8v4 not run | Do not spend more speed work on this rejected arm |
| 4 | **Q4_K down instead of Q6_K (parked)** | **7.10 % fewer expert bytes** in this mixed pack (50.688 MB of 714.24 MB/token); no speed measured. Q6_K is **1.458x**, not 2.05x, Q4_K per value | changed weight quantisation → G2; no result collected | moderate; smaller byte lever than originally stated |
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
