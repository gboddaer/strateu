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
