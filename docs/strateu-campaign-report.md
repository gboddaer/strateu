# strateu — campaign report: what was done, what was measured, what remains

*Report date: 2026-10-23. Covers the Kolibri-1 engine campaign on one 16 GB Tesla T4 (sm_75).
Every number here was measured on that box; where a measurement needs two runs or a stated
machine load, it says so.*

---

## 1. Executive summary

The engine runs **Kolibri-1 (Q4_K_M)** on a **16 GB** consumer-class GPU with all 42.90 GiB of expert
weights pinned in host RAM, a bounded VRAM residency cache, and **cache misses computed in place by a CPU
pool over the pinned table** — the design goal of the fork. Prefill is a staged, pool-carried pipeline.

**One of four speed floors is met.  Accuracy is intact, and the shortfall is quantified rather than glossed:**

| floor | target | measured | verdict |
|---|---|---|---|
| prefill, 512-token prompt | 200 tok/s | **157.0 / 160.3** (wall clock) | not met |
| prefill after 32K context | 150 tok/s | **154.0 / 153.7** (wall clock) | **met** |
| decode, short context | 30 tok/s | **24.5 / 24.4** (wall clock) | **not met** |
| decode after 32K context | 20 tok/s | **12.3 / 12.2** (wall clock) | **not met** |

> **Correction (post-audit).**  An earlier version of this table reported decode as **48.0/48.3 tok/s** and claimed the
> 45-60 stretch.  Those figures were the reciprocal of the *pool's own* per-token drain time
> (`714.2 MB/token / 34.5 GB/s = 20.70 ms`), not a wall-clock token measurement, and they were wrong.  The wall-clock
> numbers above come from `tools/kolibri/bench.py` on the shipped config (loads 5.4-5.6 and 7.5-14.1); an independent
> audit's runs (29.5-30.2 short, 13.4/13.5 after 32K, single-socket `numactl`, load 4.8-11.7) bracket them, so the binding
> spread is real but neither binding reaches either decode floor.  Section 3.2 and the B0/C0 conclusions that rested on the
> same confusion are corrected below.
| KL vs llama.cpp (answers) | < 0.02 | **0.00132**, top-1 0.982 | **met**, 0.74× of llama.cpp's own CPU-vs-GPU floor |
| oracle (plain path vs reference build) | bitwise | **BITWISE** | **met**, re-anchored after every numerics change |
| prefill determinism | two runs identical | **bitwise identical**; `racecheck` 0 hazards | **met** |

**The one real defect found in shipped code was a missing barrier** in a reduction helper
(`block_max`), which made the prefill **non-deterministic run to run** — two runs of the same binary
disagreed by ~35 in the residual stream while generated tokens happened to match. It was found with
`compute-sanitizer --tool racecheck`, fixed with one `__syncthreads()`, and is now gated.

**Two large wins came from policy, not from kernels:** defaulting the expert-pool path on for *every*
layer and length (`POOL_ALL=1`) took the 512-token prefill from **38.9 → 150 tok/s (3.9×)**, because the
previous default silently routed short prompts to a GPU grouped-expert leg costing ~154 ms/layer against
the pool's 24.5 ms.

**What is not met is three of the four floors:** short-prompt prefill (157-160 vs 200), decode short
(24.5 vs 30) and decode after 32K (12.2 vs 20).  The two decode floors were briefly *reported* as met by mistake - the
48 tok/s figure was the reciprocal of the pool's per-token drain time rather than a wall-clock token measurement, and an
independent audit caught it (§3.2 and the correction note in §1).  The prefill floors are as measured: after-32K met at
154, short short of 200.  For the short-prompt prefill, five cheap levers were measured dead (§7); the remaining ~1.25x
needs kernel work (§8, Option A) that this box cannot even *locate* with its profilers, because both perturb the workload
they measure (§6.6).  The decode shortfall is a different story: with the pool at roughly half a token, the rest is the GPU
legs plus handoffs, serialised by the same residual dependency that makes the overlap neutral - so decode needs either that
dependency broken (speculation or a different placement) or fewer bytes per token, not another knob.

---

## 2. The system as it stands

### 2.1 Design

* **Weights:** 50 layers × 384 experts, Q4_K_M, 44 GB on disk. All 19,200 experts are pinned in host
  RAM (42.90 GiB) in a repacked blob layout (`[gate|up|down]` contiguous per expert).
* **VRAM:** dense weights are a 1.62 GiB fp16 copy; a bounded **expert residency cache** holds
  `EXPERT_F16=24` slots per layer (~2.15 GiB). Everything not resident is computed **in place by the
  CPU pool** reading the pinned table — no PCIe round trip for a miss.
* **Prefill:** chunked (default 512 tokens/chunk), carried by the pool as a staged pipeline
  (`PF_STAGE=1`). A two-chunk wavefront (`PF_PIPE`) exists but is **neutral** and off by default (§7).
* **Attention:** 4:1 hybrid — 513-token sliding-window rings on 40 layers, full attention on 10; RoPE on
  sliding layers only (full layers are NoPE). Default path is a tiled cuBLAS implementation
  (`ATTN_MB=512`), with SIMT and fused-flash variants switchable.
* **Routing:** top-6 sigmoid with correction bias, tiled router (`ROUTER_TILE=1`, `TK=3`, `TJ=16`).

### 2.2 Where things live

| area | path | size |
|---|---|---|
| engine (CUDA, Kolibri-1 path) | `src/kolibri/kolibri_engine.cu` | ~4,030 lines |
| dense GEMM + dequant scratch | `src/prefill/gemm.cu` | ~980 |
| CPU expert pool | `src/kernels/cpu/pool.cpp` + `include/strata/kernels/cpu/pool.hpp` | ~910 + header |
| CUDA kernels (routed experts, router, dequant) | `src/kernels/cuda/iq_kernels.cu`, `dequant_bf16.cu`, `native_mmq.cu` | ~3,800 + |
| launcher | `run-kolibri1.sh` | — |
| harnesses | `tools/kolibri/` | 22 tools |

The CPU pool is **plain C++** with no CUDA dependency — it is the portable half of the design and the
natural first bring-up target on any new accelerator (§9).

---

## 3. Measured results in detail

All wall-clock numbers are two runs unless stated; where machine load matters it is given, because this
box's pool bandwidth varies 20.5–34.5 GB/s with load and that moves decode by 1.7×.

### 3.1 Prefill

| context | time | tok/s | notes |
|---|---|---|---|
| 512 tokens | 3,412 / 3,356 ms | 150.1 / 152.6 | cold process, shipping config |
| 8,192 | 53,374 / 53,354 ms | 153.5 / 153.5 | |
| 32,767 | 212,850 / 211,542 ms | 153.9 / 154.9 | floor 150 |
| 101,414 | 1,049,326 ms | 96.6 | 100K prompt, peak VRAM 15,241 MiB |

The pool leg is **1.2 s of the 3.4 s** short-prompt wall (measured un-instrumented: 19 GB read at
15.6 GB/s). At 8K the pool is ~60% of the wall.

### 3.2 Decode (wall clock, not counters)

**A rate is only ever quoted here from a wall-clock token run.**  `bench.py`, shipping config, load stated:

| case | run 1 | run 2 | load |
|---|---|---|---|
| tg 128, short context | **24.5 tok/s** (5,228 ms) | **24.4** (5,255 ms) | 5.4 / 5.6 |
| tg 128, after 32K | **12.3 tok/s** (10,420 ms) | **12.2** (10,488 ms) | 7.5 / 14.1 |

Both are **below** their floors (30 and 20).  The pool's own drain time is ~20.7 ms of a ~41 ms token, i.e. **about half**,
so the other half is the GPU legs plus the copy handoffs - serialised by the same per-layer residual dependency that makes the
overlap neutral (section 7).  That is why decode cannot reach the reciprocal of the drain time: the token is the *sum*.

Earlier component measurements, now labelled as components rather than rates:

| context | pool drain | pool bandwidth | note |
|---|---|---|---|
| 512-prompt | 20.7-20.8 ms | 34.3-34.5 GB/s | roughly half of a decode token |
| 512-prompt | 33.7 ms | ~20.9 GB/s | load 14.6 |
| 32K | 20.7 ms | 34.5 GB/s | roughly half of a decode token |
| 32K | 34.6 ms | 20.7 GB/s | load 12.1 |

### 3.3 Serve path (the shape that matters in practice)

Three identical 415-token chat requests through `run-kolibri1.sh` (ctx 8192):

| request | prompt_ms | tok/s |
|---|---|---|
| 1 (cold) | 2,756 | **150.6** |
| 2 (warm) | 2,356 | **176.2** |
| 3 | 2,430 | 170.8 |

`cache_n = 0` on all three: this is **not** KV reuse. It is one-time process state (expert-cache fill,
cuBLAS handles, pinned warm-up) amortising — ~17% of the cold cost. So the *steady-state* short-prompt
figure is **~176 tok/s**, 12% short of the 200 target rather than 33%.

---

## 4. Accuracy and determinism

* **KL vs llama.cpp** (7-prompt parity set, 136 steps / 109 answer steps): answers **0.00132**
  (median 0.000155, p99 0.0194, top-1 0.9817); all steps 0.00238 (top-1 0.9779). For reference,
  llama.cpp's own CPU backend vs its GPU backend scores 0.00178 / 0.00250 on the same set — i.e. this
  engine is **closer to llama.cpp-GPU than llama.cpp-CPU is** (ratio 0.74 on answers).
* **Oracle:** the plain per-token path is **bitwise identical** to the reference build
  (`strata-kolibriref`) on the gate fixture, re-anchored after each deliberate numerics change.
* **Determinism (fixed this campaign):** the prefill used to be non-deterministic. `racecheck` pinned
  50 hazards to `k_attn_softmax16`; root cause was `block_max` returning `sh[0]` **without a trailing
  `__syncthreads()`**, so the next helper could overwrite the shared array while another thread was still
  reading it. One barrier fixed it: `racecheck` now reports **0 hazards**, and two runs of the shipped
  configuration are **bitwise identical** (0.0), including 128 decode steps at 32K. The barrier is free
  (32K prefill unchanged within noise).
* **100K needle (five fixtures, distractor-bearing haystacks):** **recall 4/5, exact 3/5.**

| fixture | tokens | needle | result |
|---|---|---|---|
| `needle_32k90` | 33,364 | `velvet-willow-107` | exact |
| `needle_64k90` | 65,876 | `velvet-willow-107` | exact |
| `needle_anchor` | 101,415 | `velvet-willow-107` | exact |
| `needle_d90b` | 98,591 | `quartz-willow-884` | `quart-willow-884` — one letter short (llama.cpp on the same fixture: `quartz-willow-107`, **wrong number**) |
| `needle_d90_seed777` | 98,591 | `glacier-velvet-556` | answers a different instruction in the haystack (llama.cpp: correct) |

Scored by `tools/kolibri/needle_audit.py` against each fixture's own framing sentence
("The secret code word for this text is: X"), on the **shipped** path. Note the deviations are recorded,
not hidden: one is a one-letter miss where the reference is also wrong, the other is a genuine miss.

---

## 5. What was fixed or added this campaign

| change | effect | evidence |
|---|---|---|
| `block_max` barrier | prefill determinism: 3.5e1 divergence → **0.0** | `racecheck` 50 → 0 hazards; two-run bitwise |
| `POOL_ALL` default 2 → 1 | short-prompt prefill **38.9 → 150 tok/s (3.9×)**, neutral at 8K/32K | two runs each, ids unchanged, oracle BITWISE |
| async pool-input copy + per-chunk host double buffer (`A2a`/`A2b`) | correct scoping, speed-neutral; prerequisite for future prefetch | gated; recorded as neutral, not as a win |
| zero-size dequant guard | a zero-element conversion is now a no-op instead of a launch failure | code |
| engine layer dump inside the chunked prefill | 100K layer comparisons became practical (1 + n_gen rows instead of a per-token prefill) | 1,646-row dump validated against the reference at 512 tokens |
| reference tool: `--kv-type`, `STRATA_DUMP_BATCH`, bounded dump | made llama.cpp's own stability measurable at 100K | see §6.5 |
| `tools/kolibri/needle_audit.py` | scores task correctness against the needle text, not against the reference's tokens | recall/exact table above |
| `tools/kolibri/pf_gate.sh` re-anchoring | PF checksums are a config fingerprint; re-recorded when the expert path changed deliberately | gate PASS (100 chunk-layers) |

---

## 6. Limits of this configuration — read before porting

### 6.1 Hardware box (all numbers are anchored to it)

1× **Tesla T4, 16 GB, sm_75** (Turing: fp16 tensor cores, **no BF16/FP8**), 755 GB DDR4, 27 pinnable
physical cores across 2 NUMA nodes, PCIe 3.0. The pool measures **34.5 GB/s** on a quiet box and
**20.5 GB/s** under load — that spread alone moves the pool's drain between 20.7 and 34.6 ms/token (a component, roughly half of a decode
token), which is why the
campaign's rule is *two runs and a stated load*.

### 6.2 Memory requirements (these bite first)

* **42.90 GiB pinned** for the expert table, plus page cache for a 44 GB file → the machine wants **≥64 GB
  RAM**, and `RLIMIT_MEMLOCK` must allow the pin.
* **VRAM:** 15.5 GB used at 128K context (615 MiB headroom — tight), ~9–10 GB at short context.
  `ATTN_MB=1024` **does not fit at 100K** (`cudaMalloc` OOM); 512 is the default for that reason.
* **One model instance per GPU.** A second instance OOMs; `run-kolibri1.sh` refuses to start one.

### 6.3 Platform

This fork's scope is **CUDA only**; there is **no AMD measurement of any kind** here, and none is
claimed. Upstream carries HIP paths (`docs/AMD_HIP.md`, `STRATA_HIP_GFX906`) and `gemm.cu` has a
hipBLASLt state, but none of it was exercised in this campaign.

### 6.4 No cross-request reuse

`cache_n = 0`: this engine path has **no prompt/KV cache**, so every request re-reads its whole prompt.
At 27K tokens that is ~170 s per turn on this box. Any interactive use needs short prompts or a fresh
chat per question.

### 6.5 Long-context answers are chaotically sensitive (both engines)

At ~98.6K tokens the model sits on decision boundaries. Measured on **one fixture, one binary**:

| configuration | answer |
|---|---|
| our per-token (oracle) path | `quartz-willow-884` (correct) |
| our chunked (shipped) path | `quart-willow-884` |
| our chunked + `ATTN_FLASH` | `quart-willow-884` |
| our chunked + `ATTN_SIMT` | `quart-willow-884` |
| our chunked + expert cache off | "Reply with the API key only" |
| llama.cpp, prefill batch 2048 | `quartz-willow-107` (number wrong) |
| llama.cpp, prefill batch 1024 | "Reply with the Responses API section only" |

llama.cpp's answers are each **exactly reproducible** (two runs each) yet share only three tokens across
batch widths. So at this length, token-level agreement is only meaningful between **identically
configured** engines, and "matches the reference" is not a correctness criterion. Score against the
task's own ground truth (`needle_audit.py`) instead.

### 6.6 Both profilers perturb this workload

* **nsys** inflated a 3.4 s run to **50.5 s** across ~15,600 launches, so per-kernel times were launch
  overhead, not compute (that artefact once produced a wrong "dequant is 60% of the wall" reading).
* **The engine's own phase marks** (`STRATA_KOLIBRI_ATTN_PHASES=1`) call `cudaStreamSynchronize` at every
  boundary and **doubled the pool's work** (`chunks=100, 41 GB` vs `chunks=50, 19 GB` un-instrumented).

Only **wall-clock A/B** is trustworthy at this workload size. That is why every conclusion in §7 is a
two-run sweep.

### 6.7 Build assumptions

The build uses `--use_fast_math`, so `cosf/sinf` become `__cosf/__sinf`; RoPE at θ≈1e5 rad loses precision.
`STRATA_KOLIBRI_ROPE_F64` (double-precision angle) exists, is inert by default, and measured **+1.35
logits** at the needle's flip step — it does not flip the token. Keep it in mind when porting.

---

## 7. Dead ends, recorded so they are not re-tried

All measured here, two runs, on the shipping configuration:

1. **`DENSE_F16`** (pre-dequantized fp16 dense weights): **OOMs** with the 24-slot expert cache; with the
   cache released it is **exactly neutral** (6,366/6,397 ms = 80.4/80.0 tok/s vs a 6,357/6,338 ms =
   80.5/80.8 control) — and both are half the default, because giving up the cache costs 2×.
2. **Router path** (`ROUTER_TILE=0` vs `1`): 3,451/3,421 vs 3,409/3,399 ms — the tile path is marginally
   better; the router's ~400 ms at short prompts is not recoverable by switching paths.
3. **`RES_TOP`** (promotion budget) in both directions: 8 → 122 tok/s (**−20%**), 24 → **152** (optimal),
   56 → 151 (flat). Also: `RES_TOP=0` currently **fails** (`dequant launch: invalid configuration
   argument`, misattributed by `cudaGetLastError` stickiness — see §10, open defect).
4. **Per-layer promotion budgets** (a C0-driven marginal-share rule, implemented and swept because the
   per-layer coverage spread is 24%–98%): 150.4/149.2 vs stock 147.6/146.8 — **null**, reverted rather
   than kept.
5. **Overlap re-scoping**: neutral. And the `max(pool, GPU)` ceiling is **unreachable**, because layer
   L's residual depends on L's pool result, so L+1's GPU legs cannot be enqueued before it. The wall
   decomposes into pool 32.5 s + GPU 12.5 s + copies 4.7 s + host 4 s = the 53.7 s measured at 8K —
   the *sum*, i.e. no overlap, by data dependence rather than by a missing enqueue.
6. **`ATTN_MB=1024` at 100K**: does not fit (`cudaMalloc` OOM).

---

## 8. What remains

### 8.1 Option A — the kernel project (the only route that can reach 200)

**Why.** The short-prompt wall is 3.4 s, of which the pool is 1.2 s (already at **103% of its arithmetic
roof**) and the rest includes an expert-cache fill that expands hot experts from **4.5-bit storage into
16-bit fp16** — 1,004 promotions writing ~7.9 GB over PCIe at short prompts. Keeping residency in
**native Q4_K blobs** would cut those bytes **3.5×** (~7.9 GB → 2.2 GB), remove the expansion launches,
and free **~7 GB of VRAM** (which also de-risks 128K context and makes `ATTN_MB=1024` plausible). The
serving side then needs a **Q4_K GEMM that reads blobs from VRAM**; the tree's only grouped kernel does
read blobs but runs **100× off** its bandwidth floor (154 ms/layer vs 1.3 ms at 320 GB/s), so it must be
written or rebuilt, not merely selected.

**Plan (order matters).**

1. **Write the parity test first.** A kernel-level test in the tree's existing `*_parity` convention that
   runs the new Q4_K-from-VRAM GEMM and the current `dequant_f16` + fp16 GEMM on identical weights,
   rows and activations, and compares element-wise (bitwise, or KL-bounded if the accumulation order
   differs). Without this, a numerical kernel is unsafe to iterate on.
2. **Implement the kernel.** Group rows by expert (the pool already computes that grouping), read
   `Q4_K` blocks straight from VRAM slots, fp32 accumulate, tile shapes tuned for the target
   architecture. Keep the existing path as the A/B arm behind an env knob.
3. **Switch residency to blobs (`C1`).** Slot = raw blob (`b.stride` bytes) instead of
   `f16_stride = 3·FF·N` halves; a promotion becomes a plain H2D blob copy; serving gathers the picked
   rows and calls the new kernel.
4. **Gates (all of them, in this order):** cache-off arm PF_HASH + oracle **BITWISE** (the new path must
   not touch the miss path at all); cache-on KL **≤ 0.00132** (the current baseline — in-kernel dequant
   should be *closer* to llama.cpp, since it removes the fp16 re-rounding); two-run identical ids;
   `needle_audit.py` ≥ current (recall ≥ 4/5, exact ≥ 3/5); short-prompt prefill two runs, target ≥ 150
   with **200** as the goal; and no regression at 8K / 32K.
5. **Record** the fill-bytes drop, the freed VRAM, and the achieved tok/s, and re-anchor the floor table.

**Estimate: 2½–4 focused days.** Kernel ~1–2 days, residency switch ~1 day, gate set ~½ day.
**Risk:** medium-high (numerical; the gates above are the safety net). **Odds of landing 200:** ~60%.
Arithmetic best case is ~1.9 s per short prefill (~270 tok/s); realistic 185–200 after what does not
shrink.

### 8.2 Option C — the cheap partial (≈1 day, low risk, does **not** reach 200)

1. **Batch the fill's conversions**: gate and up are contiguous in the repacked blob, so one launch can
   cover both (`rows = 2·FF`) with `down` separate — three launches per promoted expert become two.
2. **Do the MoE weighted sum on the CPU**: the pool already holds the K=6 selected rows per token
   (7.9 MFLOP per chunk), so sum them there and copy up only the summed result
   (`Tc·N·4` = 5.2 MB) instead of all selections (`NKc·N·4` = 31.5 MB per chunk-layer) — a **6×** cut of
   that copy.
3. Same gate set as Option A steps 4–5.

**Expected +10–20%** short prefill (~165–180 tok/s). Worth doing while in the code, not worth doing for
the floor.

---

## 9. Starting over on a different machine — AMD R9700 AI Pro first

> Everything in this section is *planning*, not measurement. No AMD code was run in this campaign.

### 9.1 What is portable as-is (bring this up first)

* **The CPU expert pool** — `src/kernels/cpu/pool.cpp` + `pool.hpp`: plain C++, no CUDA. It implements
  the whole "misses computed in place" design and works on any x86-64 machine.
* **Model loading and geometry** — GGUF reader, repacked blob layout, tokenizer.
* **Every harness** — `pf_gate.sh` (build/ids/PF_HASH/oracle), `kl.py`, `parity.py`,
  `compare_dumps.py`, `needle_audit.py`, `bench.py`, `speed_config.sh`, `run-kolibri1.sh`.
* **The whole env-knob surface** (§11) — it is the experiment matrix, independent of the backend.

A sane first milestone on the new box: **the pool path + any working dense/attention kernels, passing
oracle and KL**, before touching expert kernels. That is a complete, correct engine (its speed is the
prefill floor without GPU expert help).

### 9.2 What is CUDA-specific and must be ported or replaced

| piece | path | notes |
|---|---|---|
| engine | `src/kolibri/kolibri_engine.cu` | ~4,030 lines; `__half`, streams, events, `<<<>>>`, `cudaMemcpyAsync`, pinned `cudaHostAlloc`, per-chunk events |
| attention (tiled / SIMT / flash) | same file | the tiled path is `cublasGemmStridedBatchedEx` + a custom `k_attn_softmax16`; flash is a fused kernel |
| dense GEMM + dequant scratch | `src/prefill/gemm.cu` | uses cuBLAS with a fixed workspace, `cublasSetAtomicsMode`, and a `hipblaslt_state_` hook upstream already provides |
| expert kernels | `src/kernels/cuda/iq_kernels.cu` (3,824 lines), `native_mmq.cu`, `dequant_bf16.cu` | the grouped Q4_K path, mmvq (M=1) and mmq (M>1) shapes; **dp4a/int8 and MMA shapes are architecture-specific** |
| tooling | — | `compute-sanitizer` (for the racecheck class of bug), `nsys` → `rocprof`/`omniperf` equivalents |

Check upstream `docs/AMD_HIP.md` and the `STRATA_HIP_GFX906` CMake paths first; they exist but are
untested here.

### 9.3 Verdicts from this box that must be **re-derived**, not inherited

The plan carried an explicit rule about this, and it applies to the whole table:

1. **"The CPU pool beats the GPU expert path."** True here (22.7 vs 16.1 tok/s in decode) because the pool
   gets 34.5 GB/s and the card has 15.5 GB of VRAM. With **32 GB** and different memory bandwidth the
   sign can flip — re-run the comparison before committing to the pool-only design.
2. **"Decode residency serves 0 of 0 picks."** Also a consequence of 15.5 GB. On 32 GB the residency
   cache can hold far more, and the fill (which costs ~17% of a cold short run here) amortises over a
   much larger cache.
3. **All four floors** are *same-box-anchored* by design: re-measure llama.cpp on the new machine and
   re-anchor. Do not carry 150/200/30/20 over as targets.
4. **`ATTN_MB` and context limits** — 1024 failed at 100K here purely for VRAM; on 32 GB it should be
   tried again, together with 128K context.
5. **`DENSE_F16`** — OOM here, exactly neutral when it fit. On 32 GB it fits *alongside* the expert cache;
   re-measure rather than citing the null.
6. **fast-math RoPE** — the `ROPE_F64` lever (+1.35 logits here) may behave differently with a different
   compiler/`--use_fast_math` policy; it is inert by default, so nothing depends on it.

### 9.4 A first-week checklist on the new box

1. Build upstream Strata **and** this fork; confirm the GGUF loads and geometry asserts match
   (50 layers × 384 experts, 2560 embd, top-6, 513-token sliding window on 40 layers).
2. Run the **pool path** with whatever dense kernels work; get `oracle.sh` to **BITWISE** on
   `tools/kolibri/oracle_ids512.txt` and `pf_gate.sh` to PASS (it needs `KOLIBRI_CM`).
3. Build a **llama.cpp reference on the same box** and run `parity.py` + `kl.py` — that establishes the
   local floor and the local fixture set.
4. Add a **determinism check** from day one: two-run identical ids + a race detector (`compute-sanitizer`
   on ROCm, or `rocprof`'s race tooling). This campaign lost days to a single missing barrier; the
   cheapest insurance is running the detector on the first kernel that lands.
5. Then port kernels in dependency order: dequant/dense → attention (softmax, PV) → router → grouped
   experts. Gate each on the same four things (ids, PF_HASH, oracle, KL) before moving on.
6. Re-derive §9.3's verdicts, then re-anchor the floor table.

### 9.5 Fixtures you must regenerate (they live in `/tmp` here, not in git)

The harnesses take explicit paths, but the campaign's fixtures were generated on this box:

* `bench_ids512.txt`, `bench_ids32k.txt`, `ids8k.txt` — token id lists from any text (tokenize with the
  model's tokenizer, e.g. via `strata_tokenizer` or `needle_bench.py`'s helper).
* the **parity** set — regenerate with `tools/kolibri/parity.py` from the GGUF and a llama.cpp build.
* the **needle** fixtures — regenerate with `needle_bench.py`'s haystack/needle construction (the driver
  used in this campaign, `/tmp/needle_drive.py`, is 90 lines and builds the prompt exactly as
  `needle_bench.py` does; **it was not committed** — regenerate rather than copy paths).
* the **oracle** fixture *is* in-tree: `tools/kolibri/oracle_ids512.txt`.

### 9.6 Executed on the R9700 (2026-10-24) — the re-derivation, with its numbers

This section was a checklist; it is now done, and `docs/AMD_R9700_PORT.md` is the ledger behind every
line here. Box: host `rizo`, Radeon AI PRO R9700 (gfx1201, 32 GB), ROCm 7.2.2, Ryzen 7 5700G
(8 physical cores, one NUMA node), 93 GiB RAM, 90-91 GiB free per run, GPU idle at start,
`--cpu-threads 8`, shipped config, two runs per arm.

| | this box (16 GB T4) | **R9700 (32 GB)** |
|---|---|---|
| prefill, 512-token prompt | 157.0 / 160.3 tok/s | **87.8 / 85.4 tok/s** (26.3 / 26.2 end to end; prefill pool on, 8 threads) |
| prefill, 99,299-token prompt | - | **102.3 / 100.8** (`ATTN_MB=512`), **105.2 / 105.8** (`1024`) |
| prefill, 127,811-token prompt | - | **94.1 / 94.2** (`512`), **100.4 / 100.8** (`1024`) |
| decode, short context | 24.5 / 24.4 tok/s | **15.8 / 15.8 tok/s** |
| decode, long context | 12.3 / 12.2 (after 32K) | **6.7 / 5.5** (at 100K) |
| KL vs llama.cpp, answers | 0.00132 = 0.74x of that floor | **0.00182**, top-1 0.9884 = **0.97x of this box's floor** (0.00187) |
| 100K needle | recall 4/5, exact 3/5 | **recall 5/5, 0 not exact** |
| ctest | - | **60 / 62**, both failures box-scoped |

Against §9.3's list of verdicts that had to be re-derived rather than inherited:

1. **"The CPU pool beats the GPU expert path" - the sign did flip.** `POOL_ALL=0` measured 29.1 / 29.0 s
   against the shipped pool-on 32.3 / 31.1 s, and the counters say why the original reasoning broke:
   every expert layer already runs through the pool here (`pool 50.0` of `50.0` layers/token,
   `gpu-expert`, `slots-path` and `grouped` all zero), the host streams 32.4 GB/s, and `VRAM free=20.56
   GiB of 31.86 GiB after dense upload` leaves no room for a GPU-resident placement of a 42.90 GiB
   table. There is no placement to choose, so the pool-only design is not a win to defend here - it is
   the only shape, and turning it off for prefill is slightly faster.
2. **"Decode residency serves 0 of 0 picks" - true by default, and turning it on is a large loss.**
   `DEC_RES` is opt-in and default-off, so 0 of 0 is the shipped path, not a failure. With it on, the
   cache serves **23513 of 38400 picks (61.2%)** and streamed bytes fall from 714.2 to 276.7 MB/tok -
   and the run takes **360.2 / 363.3 s against 32.3 / 31.1 s**. Residency engages well; what it replaces
   streaming with (a per-layer fp16 gemm at T=1) costs far more than the bytes it saves on RDNA4. The
   fill-amortisation argument in §9.3 assumed the saved streaming was expensive. It is not, on this box.
3. **The four floors are not carried over.** The AMD table above is the anchor; 150/200/30/20 are not
   AMD targets. The reference for KL is a llama.cpp built on this box (`/fast/llama-server-kolibri/
   llama.cpp` at `f5027fb21`, `build_rocm_rizo`), and the measured 0.00182 sits at 0.97x of that
   build's own CPU-vs-GPU floor - so it is at llama.cpp's noise, though it does not clear the 0.00132
   this box recorded against a different llama.cpp revision.
4. **`ATTN_MB=1024` is a win here, not an OOM.** It is faster at both long contexts (105.5 vs 101.6 at
   100K, 100.6 vs 94.2 at 128K), and `DENSE_F16=1` alongside the 24-slot expert cache runs clean at
   100K (942.7 / 958.0 s) where this box OOMed.

§10's open defect reproduces on HIP: `RES_TOP=0` dies with `dequant launch: invalid configuration
argument` on gfx1201 too, with no `gen:` line emitted - so its 15.3 s wall is time-to-failure and must
never be read as a fast arm in any table. `ROPE_F64=1` costs nothing measurable here (39.6 / 39.5 s
against a 39.3 / 39.3 s baseline), and `RES_TOP=24` against 56 is identical
(39.3 / 39.3 s) once residency is off.

---

## 10. Open defect (recorded, not fixed)

`RES_TOP=0` exits 1 with `dequant launch: invalid configuration argument`. A zero-size guard was added
in `dequant_bf16.cu`, which did **not** cure it: `cudaGetLastError()` is sticky, so the message is almost
certainly a **zero-block launch elsewhere** (the promotion or slots path) being reported at the next
dequant checkpoint. Off the shipped path (`RES_TOP=56`), harmless in practice, and left with the evidence
rather than a speculative patch.

---

## 11. Environment knobs (the experiment matrix)

**Shipped defaults** in `run-kolibri1.sh`: `PF_STAGE=1`, `EXPERT_F16=24`, `ROUTER_TILE=1`,
`ROUTER_TK=3`, `ROUTER_TJ=16`, `ATTN_MB=512`, `RES_TOP=56`, `GGML_CUDA_DISABLE_GRAPHS=1`.

| group | knobs |
|---|---|
| prefill scheduling | `PF_STAGE`, `PF_PIPE`, `PF_CHUNK`, `PF_CHUNK_KEY`, `PF_FENCE`, `PF_LANES`, `PF_HASH` |
| expert routing path | `POOL_ALL` (0/1/2; **default 1**), `EXPERT_F16`, `RES_TOP`, `DEC_RES`, `DEC_XEXP`, `LPT`, `EXPERT_HIST` |
| attention | `ATTN_MB`, `ATTN_KT`, `ATTN_KG`, `ATTN_SIMT`, `ATTN_FLASH`, `ATTN_MMA`, `ATTN_COALESCED`, `ATTN_PV4`, `ATTN_2PASS_MAX` (CLI), `ATTN_PHASES`, `ATTN_DUMP`, `ATTN_DUMP_LAYER` |
| decode | `DEC_F16`, `DEC_GEMV_V4`, `DEC_PHASES`, `KV_DEV` |
| numerics | `ROPE_F64`, `PV_SCALE`, `GEMM_DETERMINISTIC` |
| dense | `DENSE_F16`, `ROUTER_GEMM` |
| misc | `POISON_ZERO_SLOT` |

**Known-good / known-dead on this box:** `POOL_ALL=1` (big win), `EXPERT_F16=24` + `RES_TOP=56`
(optimal), `ATTN_MB=512` (1024 OOMs at 100K), `PF_PIPE` (neutral), `DENSE_F16` (dead), `PV_SCALE=4096`
(inert — the fp16 probability store is not the divergence source), `STRATA_GEMM_DETERMINISTIC` (does
**not** fix the prefill; the bug was a barrier — see §4).

---

## 12. Provenance

* Worktree commits of record (the published history carries the same content under rewritten hashes,
  because generated artifacts and the private plan-of-record docs were stripped before publishing):
  the determinism fix `5dfa381a`; the `POOL_ALL` default `5ae16361`; the race discovery `f71c65df` and
  the determinism pricing `3fd82e74`; the needle re-scoping `5f25c531`; the C0 coverage curve
  `44791328` (+ its pricing correction `1d47d737`); the B0 gemv table `faaa040a`; the A2b timeline
  `2e882b5f`; the A2a completion `b18c02f4`; the overlap-ceiling correction `ad9cb6f3`; the floor table
  `93e49743`; the instrument caveat `70a24da3`; the dead-lever record `86da7f9e`; the serve-path
  measurement `9f892bc8`.
* The full engineering ledger — every measurement, every rejected hypothesis, every instrumentation
  caveat — is `docs/superpowers/plans/*.md` in the **private** worktree. It is deliberately **not**
  published (it contains internal hostnames and machine paths). If you want it on the new machine,
  copy it out-of-band; do not add it to the public history without sanitising it.
* Licence: MIT, upstream copyright retained (`LICENSE`), attribution in `NOTICE`.

---

## 13. Post-audit corrections and gate accounting

An independent audit rejected the first completion claim. Three of its findings are recorded here as corrections, because
they change what this report is allowed to assert.

### 13.1 The decode method error (fixed in §1 and §3.2)

The decode rates first published here (48.0/48.3 tok/s, "stretch 45-60 reached") were the reciprocal of the **pool's own
per-token drain time** - `714.2 MB/token / 34.5 GB/s = 20.70 ms` - not a wall-clock token measurement.  Wall-clock
measurement with `bench.py` gives **24.5 / 24.4** (short) and **12.3 / 12.2** (after 32K).  Consequences recorded:

* B0's conclusion that "the pool is 99% of the decode wall" is wrong: the drain is roughly **half** a token, and the other
  half is the GPU legs plus the copy handoffs, serialised by the per-layer residual dependency.
* C0/C1's decode *throughput* pricing (absorbed rows -> decode seconds) does not follow; the VRAM-headroom argument does.
* The standing rule now in force: a rate is quoted only from a wall-clock token run (`bench.py`, or the server's own
  `timings`), with the invocation recorded; counter-derived `ms/tok` is a component and is labelled as one.

### 13.2 S8a's own gate: measured, and NOT met

S8a (prefill pipelining) carried its own acceptance gate, distinct from the 150 tok/s floor: **32K wall <= 205 s
(>= 160 tok/s)**.  Measured on the shipped binary, wall clock, two runs: **212,765 ms and 213,263 ms = 154.0 / 153.7 tok/s**.
So the S8a gate is **not met**, while the 150 tok/s floor is met.  Both numbers are true; they are different targets, and
conflating them was a reporting error in the earlier claim.  The stretch targets (300 tok/s at 32K, 400 at short) are unmet
and unreachable by scheduling for the reason recorded in section 7.5 (the residual dependency forbids the overlap).

### 13.3 `:2244`: what the record actually says

An earlier caveat of mine said the illegal-memory-access crash was "never independently root-caused".  That understated the
record: the private ledger carries a **named cause with a verified reproduction and guards** for it, and the blocking ids
copy that crashed was subsequently replaced by A2a's asynchronous publish.  The accurate statement is: diagnosed, reproduced,
guarded - and the crashing path retired.  No such fault has occurred in any run since, including a 101,414-token prefill,
pool-all arms and 128-step decodes.

### 13.4 What the audit changed, in one line

The accuracy instrument (oracle BITWISE), the determinism fix, the prefill path and the after-32K prefill floor all hold up;
the decode floors do not, and the documents now say so.
