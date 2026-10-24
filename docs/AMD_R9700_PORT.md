# AMD R9700 AI Pro port log (Kolibri-1, gfx1201)

Running record of porting the Kolibri-1 engine path (`src/kolibri`) from CUDA to AMD HIP.
Every number here is measured on the box below and says so; nothing is inherited from the
Tesla-T4 campaign (`docs/strateu-campaign-report.md` §9.3 says which verdicts must be re-derived).
Branch `feat/amd_r9700`.

## The box

| | |
|---|---|
| CPU | AMD Ryzen 7 5700G, 8 physical cores / 16 threads, 1 NUMA node |
| RAM | 93 GiB total, ~91 GiB available at idle |
| GPU (used) | AMD Radeon AI PRO R9700, gfx1201, 31.9 GiB VRAM, PCI 03:00.0 |
| GPU (excluded) | Radeon PRO W7800 48GB (gfx1100) and a Cezanne iGPU (gfx90c) are both visible to the runtime and must not be used - `run-kolibri1.sh` resolves the card by name and refuses another one |
| Software | ROCm 7.2.2 (`/opt/rocm`), hipcc 7.2.53211, hipBLAS 1.2.70202, hipBLASLt present, cmake 4.4.3 (from `requirements.txt`), ninja 1.12.1, kernel amdgpu |
| Model | `/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf` (44 GiB, Q4_K_M) |
| Limits | `ulimit -l` = 8 MiB (soft and hard), no sudo used anywhere in this port |

## Build

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201 -DCMAKE_PREFIX_PATH=/opt/rocm
```

`Strata: HIP enabled, arch gfx1201`; hipBLASLt found, so solution-table dispatch is available.
`ninja` builds the whole tree with 0 errors after five classes of fix (commit `f901ca06`):
`attn_mask_bounds` needed `__host__ __device__` under `__HIPCC__` (clang's HIP front end does not
give a host function implicit device linkage the way nvcc does), the `attn_pv4_on` /
`attn_coalesced_on` device guards keyed on `__CUDA_ARCH__` where HIP's device pass defines
`__HIP_DEVICE_COMPILE__`, and the compat shim was missing `cublasSetAtomicsMode`,
`CUBLAS_ATOMICS_NOT_ALLOWED`, `cublasGemmBatchedEx` (hipBLAS takes `const void * const *` where
cuBLAS takes `const void **`) and `cudaMallocManaged` / `cudaMemAttachGlobal` /
`cudaStreamCaptureModeRelaxed`.

Deferred, recorded: the `mma.sync.aligned.m16n8k8` flash-attention path is NVIDIA PTX, so it is
compiled for CUDA only and `STRATA_KOLIBRI_ATTN_MMA` reads as off on HIP. It is opt-in and off by
default, so the AMD default path loses nothing.

## ctest on the HIP build

**60 of 62 pass** (62.5 s). Both failures are box-scoped, not port defects:

| test | why it fails here |
|---|---|
| `ple_parity` | wants `bench/micro/../../Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf`, which is not on this box. Not a Kolibri test. |
| `platform_memory_test` | asserts POSIX `mlock()` works; `ulimit -l` is 8 MiB here, so `lock_resident` reports `locked=0 MiB`. That path serves `src/kernels/ngram.cpp`, not Kolibri's expert table. |

The CUDA campaign reported 122 tests; this HIP build registers 62, so the comparison is not
like-for-like and is recorded rather than scored.

## Loading Kolibri-1 Q4_K_M on the R9700

```
strata-kolibri: .../Kolibri-1-Q4_K_M.gguf: 903 tensors (layout checked); experts 50 x 384 blobs
repacked, 42.90 GiB pinned; dense 1.62 GiB on the GPU; native expert kernels for 25 / 50 layers;
189.8 s
```

- Geometry matches the campaign's asserts: 50 layers x 384 experts, layout checked.
- **42.90 GiB of pinned host table allocated with no root and `ulimit -l` = 8 MiB.** A standalone
  probe confirms it: `hipHostMalloc(43 GiB)` returns success at that limit on ROCm 7.2.2. The T4
  report's "RLIMIT_MEMLOCK must allow the pin" is a CUDA-era requirement that does not carry over.
  POSIX `mlock()` is a different call and does still fail at 8 MiB (that is `platform_memory_test`).
- "native expert kernels for 25 / 50 layers" is the same split the CUDA record describes - the
  Q4_K_M file's Q6_K-down layers are outside the GPU grouped kernel and go through the CPU pool
  (`kolibri_engine.cu:212`). Not a port gap.
- Cold load 189.8 s (the 44 GiB file is read from disk); the second run, with the file in page
  cache, finished the whole oracle command in 63.5 s wall clock. Peak RSS 91.7 GB on a 93 GiB box -
  the pinned table and the file's page cache together nearly fill RAM, so this box has no room for
  a second instance or a second large model.

## Race and memory checking on ROCm 7.2.2 (a platform gap, recorded)

The CUDA campaign's determinism insurance was `compute-sanitizer --tool racecheck`. **ROCm 7.2.2 has
no equivalent for RDNA4**, measured rather than assumed:

| what | state on this box |
|---|---|
| `hip-memcheck` | gone from ROCm after 4.x; not packaged here |
| device ASan (`-fsanitize=address`) | refused: "not currently supported" for gfx1201, needs an `xnack+` arch |
| `rocm-validation-layer` | not in this box's apt index |
| `rocm-debug-agent` 2.1.0, `rocgdb` | installed - GPU faults produce an inspectable core dump, which is fault recovery, not race detection |
| `rocprof`, `rocprof-compute`, `rocprofiler-sdk` 1.1.0 | installed - kernel timing and counters, no race or bounds checking |
| `omnitrace`, `omniperf` | available in apt, not installed (needs sudo); profiling, not checking |
| `valgrind` 3.24 | installed - host-side, which is where the CPU expert pool's races would be |

So the gates actually run on this box are: two-run identical ids, two-run identical `PF_HASH`
lines, a serialized-launch arm (`HIP_LAUNCH_BLOCKING=1`) to catch work that only looks ordered
because launches overlap, and `rocgdb` + `rocm-debug-agent` if a kernel ever faults. For the CPU
pool (`src/kernels/cpu/pool.cpp`, plain C++ threads) the host tools apply directly. This is weaker
than `racecheck`, and it is stated as such rather than quietly substituted.

## Harness artefacts that had to be re-pointed at this box

The CUDA campaign left its harnesses pointing at its own machine. Each of these is a silent failure,
not a loud one, so they are recorded:

| harness | default on the CUDA box | what this box needs |
|---|---|---|
| `pf_gate.sh` `BIN` | `/tmp/kolibri_bench_node1.sh` | `KOLIBRI_BIN=$PWD/build/strata-kolibri`; without it the gate reports "no gen: line (the prefill did not run)" while the engine is fine |
| `pf_gate.sh` `M`, `oracle.sh` `MODEL` | `/data4/models/...` | `KOLIBRI_MODEL=/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf` |
| `pf_gate.sh` expectations | `/tmp/pf_gate_expect_*.txt` | not in git; recorded on this box with `--record`, which is what the script is designed for |
| `/tmp/ids64.txt` | a CUDA-box scratch file | regenerated from `tools/kolibri/oracle_ids512.txt`, which is **comma-separated**, so `tr ',' '\n'` - splitting it on spaces yields zero ids and the gate dies before its first run |
| `pf_gate.sh --cpu-threads 24` | 27 pinnable cores | hardcoded 24; the engine caps it itself ("8 lane(s) = the host + 7 thread(s), 7 pinnable physical core(s), pinned (capped from --cpu-threads)"), so it is oversubscribed-but-correct here |

### Reference llama.cpp: what it took, and the current blocker

`~/develop/llama.cpp` carries **part** of the Kolibri patch: `src/models/kolibri1.cpp` and
`LLM_ARCH_KOLIBRI1` exist, but the `src/llama-vocab.cpp` half does not, so the reference refused the
file with `error loading model vocabulary: unknown pre-tokenizer type: 'kolibri1'`. The whole-patch
apply fails only because of `conversion/base.py` (drifted) and `src/models/kolibri1.cpp` (already
present); applied per file, `src/llama-vocab.cpp`, `src/llama-arch.cpp`, `src/llama-model.cpp` and
`src/models/models.h` each apply cleanly. That is what is being applied and rebuilt now into
`build_rocm_strata` (log `/tmp/llm_patch.log`); `tools/kolibri/kolibri_llama_dump.cpp` is then
rebuilt against it into `build/tools/kolibri_llama_dump` per the header comment in that file.
`parity.py` takes `--ref build/strata-kolibriref --dump build/tools/kolibri_llama_dump --model GGUF
--out DIR --n 64 --ngl 99 --cpu-moe 50`; its first attempt failed with "no gen" style rc 1 for
exactly this vocabulary reason (`/tmp/parity-amd/p00.llama.log`).

Resolved: the reference needs exactly **two** files applied over that checkout, not the whole patch -
`src/llama-vocab.cpp` (the `kolibri1` pre-tokenizer) and `src/llama-hparams.h` (the
`LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID_LOGIT_ADD` enum). Applying `src/models/models.h` or
`src/llama-model.cpp` on top duplicates what the checkout already has (redefinition of
`llama_model_kolibri1`, duplicate case values), and `git apply --check` passes those hunks because it
checks context, not meaning. `build_rocm_strata` builds clean with just the two.

### BLOCKER: no llama.cpp on this box can run Kolibri 1, so the KL gate has no reference

`kl.py`'s reference P is llama.cpp. Getting one to run Kolibri on `rizo` has failed in a chain of
ways, all one root cause - `~/develop/llama.cpp` is at upstream `0a1ef309c`, which is **newer than
the Kolibri patch** and has drifted under it:

| attempt | result |
|---|---|
| as found | `unknown pre-tokenizer type: 'kolibri1'` (the `llama-vocab.cpp` half was missing) |
| whole patch | `kolibri1.cpp` already exists, `conversion/base.py` hunk fails |
| + `models.h`, `llama-model.cpp` | redefinition of `llama_model_kolibri1`, duplicate case values |
| + `llama-graph.cpp` only | `llama-graph.cpp:2265: GGML_ASSERT(exp_probs_b != nullptr && "SIGMOID_LOGIT_ADD gating requires exp_probs_b")` |

The last one is the real wall: the newer graph code never receives the router bias for this arch,
which is what the patch's conversion/`tensor_mapping` half and an older graph API together provided.
The patch's base commit `c222b50c1d4f...` is **not in this clone** (it is the patch author's fork), so
the intended base revision cannot be checked out here. Current state of the user's tree: `llama-vocab.cpp`,
`llama-hparams.h`, `llama-graph.cpp` modified by hand (revert with `git checkout --` on those three);
`build_rocm_strata` builds clean but aborts at context init for Kolibri.

What would unblock it, in order of preference: (1) the llama.cpp revision the CUDA campaign actually
used for its recorded KL - it may live on the campaign box rather than here; (2) an upstream
llama.cpp revision from around 2026-10-03 (the patch date) to build the full patch against; (3) a
Kolibri-capable llama.cpp build supplied directly.

## §9.3 decode-dominated verdicts, with the engine's own counters

Shape: 512-token prompt, `--n 128` decode, ctx 8192, `--cpu-threads 8`, box 91 GiB free. Two clean
runs per arm for wall clock; a third run per arm with `--counters` for attribution, because the
engine's own comment at `kolibri_engine.cu:362` says counters add a 12 KB D2H and must never run in
a timed run. Counters are the `--counters` flag, not an env var - the first sweep's `cnt:` grep
matched nothing because there is no `cnt:` prefix; the lines are `CNT ...`.

| arm | wall run1 / run2 | counters |
|---|---|---|
| shipped config, pool on (default `POOL_ALL=1`) | 32.3 s / 31.1 s | `pool workers=8, streamed 714.2 MB/tok (91.42 GB), 32.4 GB/s`; `branches per tok: pool 50.0 gpu-expert 0.0 slots-path 0.0 grouped 0.0`; `decode residency: 0 of 0` |
| `POOL_ALL=0` | 29.1 s / 29.0 s | same 714.2 MB/tok and 32.5 GB/s in decode; the difference shows in prefill (`pool=462 ms` vs `970 ms`) |
| `DEC_RES=1` (fp16 residency on) | **360.2 s / 363.3 s** | `decode residency: 23513 of 38400 picks served from the fp16 cache (61.2%)`, streamed down to 276.7 MB/tok - and **11x slower** |
| `DEC_RES=1 EXPERT_HIST=1` | 362.1 s / 363.7 s | same 61.2%; the history seed changes nothing here |

**Residency verdict.** On the T4, `DEC_RES` was the knob that kept hot experts VRAM-resident so the
pool streamed less. On this box it does exactly that - 61.2% of decode picks served from the fp16
cache, streamed bytes down from 714.2 to 276.7 MB/tok - and the run gets **11x slower** (360 s
against 31 s). The streaming it saves is cheap here (32.4 GB/s from a host that is not competing with
a 5.7 GB/s PCIe link); what it replaces it with is a per-layer fp16 gemm at T=1, and on RDNA4 that
path costs far more than the bytes it avoids. So the AMD answer to §9.3's residency question is *off*,
and the reason is not that residency fails to engage - it engages well - it is that the GPU-side
alternative is the wrong trade on this card.

**Placement verdict.** Every expert layer on AMD goes through the pool (`pool 50.0` of `50.0`
layers/token; `gpu-expert`, `slots-path` and `grouped` all zero), and turning the pool off for
prefill is slightly *faster* rather than the large loss it was on the T4. `VRAM free=20.56 GiB of
31.86 GiB after dense upload` says why there is no room to change: the 42.90 GiB expert table cannot
be resident, so there is no GPU-resident placement to choose.

## The KL gate (task 5) - measured, and it does not clear the inherited number

Reference: `/fast/llama-server-kolibri/llama.cpp` at `f5027fb21` (the user's Kolibri-capable tree,
patch applied in the working tree), built fresh here as `build_rocm_rizo` with
`-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`; `tools/kolibri/kolibri_llama_dump.cpp` built
against it. It loads the Q4_K_M file and generates, which the drifted `~/develop/llama.cpp` checkout
could not.

`parity.py` (8 prompts, `--n 64 --ngl 99 --cpu-moe 50`): **5 of 8 prompts identical greedy answers**, worst layer rel err 22.20, median layer rel err ~1.15.

`kl.py` teacher-forced, 520 steps / 344 answer steps over the same 8 prompts:

| leg | mean KL | median | p99 | top-1 |
|---|---|---|---|---|
| llama.cpp CPU vs llama.cpp GPU (the noise floor), answers | 0.00187 | 0.000261 | 0.0172 | 0.9884 |
| **strata vs llama.cpp GPU, answers** | **0.00182** | 0.000086 | 0.0253 | 0.9884 |
| llama.cpp CPU floor, all steps | 0.00197 | 0.000120 | 0.0254 | 0.9923 |
| strata, all steps | 0.00224 | 0.000073 | 0.0451 | 0.9923 |

**Read it straight: the gate as written is `<= 0.00132` and the AMD path measures 0.00182 on the
answer steps, so it does not clear that number.** What it does clear is the criterion that number was
originally standing in for - the campaign reported its 0.00132 as *0.74x of llama.cpp's own
CPU-vs-GPU floor*, and here strata sits at **0.97x of this box's floor** (0.00182 against 0.00187),
with identical top-1 agreement to the floor at 0.9884. Strata is at llama.cpp's own noise, not above
it. The absolute numbers are not comparable across the two boxes anyway: KL is measured *against a
reference*, and this is a different llama.cpp revision from the one the Tesla-T4 campaign recorded
against.

Per-prompt answer KL (strata, floor in brackets): p00 0.00280 [0.00431], p01 0.00198 [0.00179],
p02 0.00208 [0.00208], p03 0.00044 [0.00069], p04 0.00045 [0.00035], p05 0.00100 [0.00116],
**p06 0.02561 [0.00452]**, p07 0.00521 [0.00118]. p06 is the outlier - 4 answer steps at 5.7x its
floor - and it is also one of the three prompts whose greedy answer diverged in parity, so the two
observations are the same observation.

## The 100K needle audit (task 5, second half) - recall 5/5

The campaign's driver was never committed (`/tmp/needle_drive.py`, "regenerate rather than copy
paths"), so the fixtures are rebuilt by `tools/kolibri/make_needle_fixtures.py`, which writes
`/tmp/needle_<name>/{90_prompt.txt,90.ids}` - the two files `needle_audit.py` looks for. Five
fixtures of ~99.3K tokens, needle at 5/25/50/75/90% depth, each needle word asserted to occur exactly
once in its prompt, framing sentence in the form the audit's regex matches.

`needle_audit.py --run` on the shipped path, box load 91 GiB free:

| fixture | tokens | needle | result |
|---|---|---|---|
| `needle_a05` | 99,298 | `quartz-willow-884` | PASS |
| `needle_b25` | 99,298 | `cobalt-orchard-311` | PASS |
| `needle_c50` | 99,299 | `amber-glacier-742` | PASS |
| `needle_d75` | 99,298 | `velvet-cedar-209` | PASS |
| `needle_e90` | 99,298 | `quartz-willow-107` | PASS |

`5 answer(s) audited, 0 not exact`. The gate is recall >= 4/5 and exact >= 3/5; this is 5/5 on both
as the harness scores them (`needle in text` over the whole decoded generation - the line it prints
is only the answer's last line, which is why a PASS can display as `'Answer:'`). Worth noting for
anyone reading the table: `needle_e90` uses `quartz-willow-107`, the word the campaign recorded as
*llama.cpp's own wrong answer* on a different fixture - here it is the needle, and it was found.

## Wall-clock rates on the AMD path (two runs each, box 90-91 GiB free, GPU idle at start)

The engine prints no rate, so these are differences between runs that differ only in decode length,
with the engine's own load line (13.4 s warm) subtracted for prefill. Shape: 512-token prompt
(`oracle_ids512.txt`), ctx 8192, `--cpu-threads 8`, shipped config.

| measurement | run1 | run2 |
|---|---|---|
| prefill, 512 tokens, load excluded | 43.4 tok/s (11.8 s) | 51.7 tok/s (9.9 s) |
| prefill, 512 tokens, end to end (load included) | 20.3 tok/s | 22.0 tok/s |
| decode, tokens 130-257 (the steady window) | **15.8 tok/s** (8.1 s) | **15.8 tok/s** (8.1 s) |
| decode, tokens 2-129 (first window after prefill) | 21.3 tok/s | 16.2 tok/s |

Long context, same engine, 8 decode tokens, load subtracted:

| prompt | `ATTN_MB=512` | `ATTN_MB=1024` |
|---|---|---|
| 99,299 tokens (ctx 105024) | 102.3 / 100.8 tok/s | 105.2 / 105.8 tok/s |
| 127,811 tokens (ctx 131072) | 94.1 / 94.2 tok/s | 100.4 / 100.8 tok/s |

Two things follow. The **marginal prefill rate is ~105 tok/s**, and the short-prompt figure is low
because roughly 6 s of fixed cost is spread over only 512 tokens - at 100K that same engine runs
2.4x faster per token. And **`ATTN_MB=1024` is faster than 512 at both long contexts** (105.5 vs
101.6 at 100K, 100.6 vs 94.2 at 128K), so on this card the larger attention slab is a win, not just
a fit; the T4's OOM at 100K does not reproduce here, at 32 GB, with the expert table in host RAM.

Against the CUDA record (`strateu-campaign-report.md`: prefill 157.0/160.3 tok/s at 512 tokens,
decode 24.4/24.5 tok/s short context) the AMD path is about 3x slower on short prefill and about
1.5x slower on decode. That comparison needs a caveat: the campaign's table says "wall clock" without
saying whether its load time was subtracted, and the AMD numbers here are quoted both ways for that
reason. Decode at long context is not measured yet - each such run is ~16 min - so the CUDA row
"tg 128 after 32K, 12.3 tok/s" has no AMD counterpart yet.

## §9.3 verdicts, remaining arms (two runs each, same box and load as above)

Short shape is `n=257` on the 512-token prompt, so arms compare against the baseline's 39.3 / 39.3 s.

| arm | wall run1 / run2 | verdict |
|---|---|---|
| baseline (`RES_TOP=56`) | 39.3 s / 39.3 s | - |
| `ROPE_F64=1` | 39.6 s / 39.5 s | no measurable cost at this shape (prefill 51.5 vs 51.7 tok/s). RoPE in fp64 is free here, so there is no accuracy-vs-speed trade to argue on this card - it can stay on if it is more accurate |
| `RES_TOP=24` (vs 56) | 39.3 s / 39.3 s | identical to within 0.01 s - the prefill promotion ceiling makes no difference once residency is off, which is consistent with the `DEC_RES` result above |
| `DENSE_F16=1` | 39.8 s / 39.9 s | +1.4% over baseline, within run-to-run spread |
| `RES_TOP=0` | **fails** | `dequant launch: invalid configuration argument` - a zero-sized launch, reproduced on a second run. Not a configuration: the engine has no guard for a zero promotion ceiling. Its 15.3 s wall is time-to-failure, and it emits no `gen:` line, so anyone reading a timing table must not treat it as a fast arm |

Long context, 99,299-token fixture, ctx 105024:

| measurement | run1 | run2 |
|---|---|---|
| decode at 100K (`ATTN_MB=1024`), 128 tokens | **6.7 tok/s** (18.9 s) | - |
| `DENSE_F16=1` alongside the 24-slot expert cache | 942.7 s, rc 0, 8 tokens | 958.0 s, rc 0, 8 tokens |

Decode falls from 15.8 tok/s at short context to 6.7 tok/s at 100K - attention over 99K tokens more
than doubles the per-token cost, and that is the number the CUDA record's "tg 128 after 32K, 12.3
tok/s" has been waiting for an AMD counterpart of. `DENSE_F16` next to the expert cache at 100K runs
cleanly at 942.7 / 958.0 s, against the T4 record where that combination OOMed: at 32 GB with the
expert table in host RAM there is room the T4 did not have.

## Gate re-confirmation, geometry evidence, and the detector result

Re-run against the current binary (`build/strata-kolibri`, gfx1201), box `rizo`, 90 GiB free:

```
0. build       PASS
  1. ids smoke   PASS  (gen: 3515 310 262 14292)
  2. PF_HASH     PASS  (100 chunk-layers)
  3. oracle      PASS  (BITWISE PASS: gen: 286 3515 310 262 14292 1444 ...)
GATE PASS
```

`tools/kolibri/oracle.sh check` on its own: `BITWISE PASS: gen: 286 3515 310 262 14292 1343 1444 46
17156 32 51 48 58 4037 262 1890` - the AMD oracle, sixteen ids, reproduced from the committed
`oracle_ids512.txt`.

### Geometry, read from the model file rather than asserted from the load line

The load line says "layout checked", which is a claim; these are the values the shared validator
(`core::geometry_from_gguf`, which enforces the four-sliding-to-one-full pattern and a positive
window) actually matched on this file - GGUF v3, 903 tensors, 46 kv pairs, architecture and
pre-tokenizer both `kolibri1`:

| key | value | the criterion it satisfies |
|---|---|---|
| `kolibri1.embedding_length` | 2560 | 2560 embd |
| `kolibri1.expert_count` / `expert_used_count` | 384 / 6 | 384 experts, top-6 |
| `kolibri1.attention.sliding_window` | 513 | the 513-cell window |
| `kolibri1.attention.sliding_window_pattern` | 50 entries, 40 set and 10 clear | window on 40 of 50 layers |

Together with the engine's own load report (`experts 50 x 384 blobs repacked, 42.90 GiB pinned`),
every geometry the campaign asserted is now evidenced with a number rather than with the word
"checked".

### The detector criterion, and what this stack will and will not do

The criterion asks for a named ROCm race/memory detector run per ported kernel. There is no
`racecheck` equivalent for gfx1201 on ROCm 7.2.2 (the table earlier in this log covers why: no
`compute-sanitizer`, `hip-memcheck` removed after 4.x, device ASan refused for want of `xnack+`,
validation layer not packaged). The one memory-side detector this box does have is
`rocm-debug-agent` 2.1.0, and it was run properly rather than assumed:

- `HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2` with `ROCM_DEBUG_AGENT_OPTIONS=--all`, full
  prefill + 16-token decode: rc 0, 16 tokens, **zero** wavefront / memory-violation / illegal-instruction
  / queue-error lines.
- To prove that silence meant "clean" rather than "not loaded", the library was checked in the live
  process: 5 mappings of `librocm-debug-agent.so.2.1.0`, `HSA_TOOLS_LIB` present in its environment.
  A `SIGQUIT` mid-prefill then produced **no wavefront dump at all**.

So the honest statement is: the agent loads but does not deliver wavefront state on gfx1201 under
ROCm 7.2.2, which makes it a null result rather than a passing detector. What is actually covered is
weaker than `racecheck` and is stated as such - no memory exception is raised by any ported kernel
across prefill, decode, the pool path and the long-context runs, and determinism rests on the
two-run bitwise identity of generated ids plus `PF_HASH` over 100 chunk-layers, which is a
reproducibility guarantee, not a data-race proof. Anyone who needs a race proof on this path needs a
detector that does not exist for this architecture yet, and should not read the two-run identity as one.

## The AMD gates are now reproducible from the tree, and one KL lever that did not exist

Everything the AMD gates compare against used to live in `/tmp`, which means a reboot silently
un-reproducible them. They are committed now, named for the backend so the CUDA references next to
them are untouched:

| file | what it pins |
|---|---|
| `tools/kolibri/oracle_gfx1201.txt` | the 16 generated ids the plain path must reproduce bitwise |
| `tools/kolibri/pf_gate_expect_smoke_gfx1201.txt` | the 4-token ids-smoke expectation |
| `tools/kolibri/pf_gate_expect_hash_gfx1201.txt` | the 100 chunk-layer `PFHASH` lines |

With those in the tree, the whole AMD gate set reproduces from a clean checkout with four exports and
two commands (verified: `oracle.sh check` -> `BITWISE PASS`, `pf_gate.sh` -> `GATE PASS` against the
committed files, not the `/tmp` ones):

```bash
export KOLIBRI_MODEL=/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf
export KOLIBRI_BIN=$PWD/build/strata-kolibri KOLIBRI_CM=$PWD/.venv/bin/cmake HIP_VISIBLE_DEVICES=0
tools/kolibri/oracle.sh check tools/kolibri/oracle_gfx1201.txt
tr ',' '\n' < tools/kolibri/oracle_ids512.txt | head -64 > /tmp/ids64.txt   # pf_gate reads this path
bash tools/kolibri/pf_gate.sh
```

That `tr | head -64` line is not a guess about the fixture: `/tmp/ids64.txt` was compared byte-for-byte
against it and is identical, so the 64-id input the ids-smoke leg uses is derived from the committed
512-id file rather than being a third unreproducible artefact.

### `--act f32` does not move KL, so the bound was not reachable that way

`kl.py` exposes `--ref-args`, and `strata-kolibriref` defaults to `g_act_q8 = true` (q8-quantized
activations), so f32 activations looked like a legitimate way to close the remaining KL gap. It is
inert on this path: re-running the whole KL set with `--ref-args "--act f32"` reproduced **every**
per-prompt number exactly - all eight prompts, both the all-steps and answers-only means, medians and
p99s, to four decimals. The reason is in `kolibriref.cu:274` (`if (quant && g_act_q8)`): the flag only
matters where the path quantizes at all, and the KL path does not. Worth recording so nobody spends an
afternoon on it; note also that `kl.py` unlinks the strata logits after scoring (line 101), so
comparing the two runs' `.strata.full` files proves nothing - the per-prompt KL lines are the evidence.

So the KL result stands as measured: **0.00182 on the answer steps against a bound of 0.00132**, at
0.97x of this reference's own CPU-vs-GPU floor of 0.00187. That last number is the awkward one - the
bound sits *below* the noise floor of the reference it is measured against on this box, which is
recorded as a deviation rather than argued away.

### The plain-kernel reference path gets to 0.84x of the floor; neither path reaches 0.00132

`strata-kolibriref` also takes `--kernels plain` (the straightforward reference kernels rather than
Strata's fused ones), which is the configuration `oracle.sh` gates on. Re-measured against the same
cached llama.cpp logits:

| reference configuration | answers mean KL | median | p99 | top-1 | ratio to floor |
|---|---|---|---|---|---|
| llama.cpp CPU vs llama.cpp GPU (the floor) | 0.00187 | 0.000261 | 0.0172 | 0.9884 | 1.00 |
| `--kernels strata` (shipped fast path) | 0.00182 | 0.000086 | 0.0253 | 0.9884 | 0.97 |
| **`--kernels plain`** | **0.00158** | 0.000086 | 0.0264 | 0.9942 | **0.84** |

So the AMD path can be brought to 0.84x of its own reference's noise floor - closer to llama.cpp-GPU
than llama.cpp-CPU is, by 16% - and the bound of 0.00132 is still not reached on either configuration.
The reason is structural rather than a defect to chase: **the bound sits below the floor of the
instrument it is measured against on this box** (0.00187 here). On the Tesla-T4 box the same floor was
0.00178 and Strata scored 0.00132, i.e. 0.74x of that floor - the absolute number was only ever
achievable because that box's CPU-vs-GPU divergence happened to be smaller. The floor is a property of
the box (this CPU against this GPU, their respective rounding), so an absolute KL bound silently
imports the old box's numerics, which is the exact thing §9.3 forbids inheriting.

A second incomparability, recorded so nobody over-reads the table: the campaign's 0.00132 came from a
7-prompt set with 109 answer steps; this box's parity fixture is 8 prompts with 344 answer steps.
Different step populations, different means.

The honest summary line for the AMD path is therefore **0.00182 shipped / 0.00158 plain, at 0.97x and
0.84x of this box's own floor**, with the criterion's absolute bound of 0.00132 unmet and the reason
given above - not a claim that the bound was met.

## A memory detector that actually applies: valgrind over the ported host-side paths

The GPU kernels have no `racecheck` equivalent on this architecture (above), but the CPU expert pool,
the expert gather and the dequant the pool feeds are **host** code, and for host code `valgrind` is a
real memory detector rather than a substitute. Run per ported path, `--error-exitcode=99
--leak-check=summary --errors-for-leak-kinds=definite --error-limit=no`, box `rizo`, R9700 as the
device (`HIP_VISIBLE_DEVICES=0`):

| test | result | valgrind error lines |
|---|---|---|
| `native_expert_parity --bf16-embd` | `0 failures`; "97 gathered rows, 0 differ; dequant of 156160 values, 0 differ in any bit" | **0** |
| `kolibri_router_parity` | `kolibri_router_parity OK` | **0** |
| `s2_expert_grouped_parity` | `PASS` | **0** |
| `expert_profile_version_test` | `OK` | **0** |

Note on the first row: run bare, `native_expert_parity` exits 2 after printing a usage line - it wants
`--bf16-embd`. That is an argument, not a failure, and it is worth writing down because "rc=2" in a
sweep log reads like a defect. With the flag it runs clean.

So the detector picture on the AMD path, stated precisely: **host-side ported code is valgrind-clean**
(definite leaks and invalid accesses, four paths, zero errors); **GPU-side ported code raises no memory
exception** under `rocm-debug-agent` across prefill, decode, pool and long-context runs, though that
agent produces no wavefront output on gfx1201 and so cannot be claimed as a passing race check; and
**no data-race detector exists for this architecture**, which is why the two-run bitwise identity of
generated ids plus `PF_HASH` over 100 chunk-layers is offered as reproducibility evidence and not
dressed up as a race proof.

### A fixture-shape hypothesis, tested and refuted

The n=64 parity fixture never emits `<|im_end|>` inside 64 tokens, so `kl.py` counts **every** step as
an answer step (`p00: 65 steps (65 answer)`) - including the late, most-divergent ones - while the
campaign's population was 109 answer steps over 7 prompts (~16 each). That looked like the reason the
AMD number sat above the bound, so it was tested: parity re-run at `--n 24` and KL recomputed on it.

It made things **worse**: 155 answer steps, mean KL **0.00199** at **1.12x** a floor of 0.00178
(the shipped-kernel n=64 figure was 0.00182 at 0.97x). The hypothesis is dead - truncating the
generation does not buy accuracy, and the floor moves with the population too. Recorded because a
plausible-sounding explanation that fails a test is worth more than an untested one, and because it
means the 0.00132 bound survived three separate legitimate attempts to reach it:

| attempt | answer-step KL | verdict |
|---|---|---|
| shipped fused kernels, committed fixture | 0.00182 (0.97x floor) | baseline |
| `--act f32` | 0.00182 | inert - `kolibriref.cu:274` only consults `g_act_q8` where the path quantizes |
| `--kernels plain` | **0.00158 (0.84x floor)** | real improvement, still above the bound |
| terminating-answer fixture (`--n 24`) | 0.00199 (1.12x floor) | hypothesis refuted |

The standing reading is unchanged and is a measurement, not an excuse: the bound sits below the floor
of the instrument it is quoted against on this box (0.00178-0.00187 depending on population), so it
cannot be met here by any engine that is not llama.cpp itself, and meeting it on the T4 was a property
of that box's smaller CPU-vs-GPU divergence.

| `pinned_shared_test` | `OK` - the pinned-allocation path the whole design rests on | **0** |
| `file_expert_source_test` | `PASS` | **0** |

## Where the port stands (2026-10-24)

Done and committed on `feat/amd_r9700`: launcher (`251a079c`), HIP build + load + ctest (`f901ca06`,
`f53f87a6`), `pf_gate` PASS and this log (`f86ff892`).

Reproducing the gates on this box needs four env exports, because the harnesses default to the
CUDA box: `KOLIBRI_MODEL=/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`,
`KOLIBRI_BIN=$PWD/build/strata-kolibri`, `KOLIBRI_CM=$PWD/.venv/bin/cmake`, `HIP_VISIBLE_DEVICES=0`.
The AMD oracle is recorded at `/tmp/kolibri_oracle.txt`, the pf_gate expectations at
`/tmp/pf_gate_expect_{smoke,hash}.txt`, and the 64-id fixture at `/tmp/ids64.txt` (from
`oracle_ids512.txt` split on commas). None of the three is in git yet - committing an AMD oracle
file alongside `oracle_post_s2a.txt` is a decision for the maintainer, not a silent change.

### In progress: the llama.cpp reference (task 5)

The KL gate needs a llama.cpp that understands Kolibri 1, and neither candidate had one:

- `~/develop/llama.cpp` **does** carry Kolibri support (`src/models/kolibri1.cpp`), and the
  `kolibri1-llama.cpp.patch` beside the Q8_0 files does **not** apply to it - it is already applied
  and has drifted (`conversion/base.py` hunk fails, `kolibri1.cpp` already exists). Do not re-apply it.
- Its `build_rocm/` is from Oct 12 and has no kolibri object, and no dump tool.
- The tree Strata fetches (`build/_deps/strata_llamacpp-src`) is upstream llama.cpp with **no**
  Kolibri support, so it cannot be the reference.

So a fresh ROCm build of the patched checkout is running into `~/develop/llama.cpp/build_rocm_strata`
(leaving the user's `build_rocm` untouched), logging to `/tmp/llm_ref.log`. Next steps after it
finishes: build `tools/kolibri/kolibri_llama_dump.cpp` against it (that is parity.py's `--dump`, with
`--ref` the llama.cpp binary), then `parity.py` and `kl.py` for the KL number, then the needle
fixtures for `needle_audit.py`. `parity.py` and `kl.py` both take `--ref BIN --dump BIN --model GGUF
--out DIR` and pass `STRATA_DUMP_CTX=8192` and `GGML_CUDA_DISABLE_GRAPHS=1` themselves.

## Gates on the AMD path


`pf_gate.sh` was recorded on this box (`--record`) and then run twice:

```
  0. build       PASS
  1. ids smoke   PASS  (gen: 3515 310 262 14292)
  2. PF_HASH     PASS  (100 chunk-layers)
  3. oracle      PASS  (BITWISE PASS: gen: 286 3515 310 262 14292 1343 1444 46 17156...)
GATE PASS
```

Identical on both runs, including the 100 `PFHASH` chunk-layer checksums - so the prefill pool and
the flash attention path are reproducible on gfx1201, which is the property `racecheck` was standing
in for on the CUDA box. The plain-path oracle is bitwise against the oracle recorded here.

## Determinism and the oracle


Two consecutive runs of the pinned plain-path command produced **identical ids**:

```
gen: 286 3515 310 262 14292 1343 1444 46 17156 32 51 48 58 4037 262 1890
```

Against the CUDA box's recorded oracle (`tools/kolibri/oracle_post_s2a.txt`,
`... 32 50 55 58 4037 262 7771`): **14 of 16 ids match, first divergence at the 11th generated
token.** That is the expected cross-backend split - `docs/AMD_HIP.md` states plainly that HIP does
not claim bit-identical answers against CUDA, and hipBLAS and cuBLAS do not share an accumulation
order. The usable gates on this box are therefore: two-run identity (met), an AMD-recorded oracle
as the standing reference, and KL against a llama.cpp built on this box.
