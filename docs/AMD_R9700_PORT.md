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
| `HIP_MEM_CHECK=1` | inert, not off: a run under it finished in the normal 27 s with 4 tokens and zero violation lines, and the name appears nowhere in `/opt/rocm/include/hip` - the old `hipMemCheck` path is gone from this ROCm |

| `omnitrace`, `omniperf` | available in apt, not installed (needs sudo); profiling, not checking |
| `valgrind` 3.24 | installed - host-side, which is where the CPU expert pool's races would be |

So the gates actually run on this box are: two-run identical ids, two-run identical `PF_HASH`
lines, a serialized-launch arm (`HIP_LAUNCH_BLOCKING=1`) to catch work that only looks ordered
because launches overlap, and `rocgdb` + `rocm-debug-agent` if a kernel ever faults. For the CPU
pool (`src/kernels/cpu/pool.cpp`, plain C++ threads) the host tools apply directly. This is weaker
than `racecheck`, and it is stated as such rather than quietly substituted.

**The serialized arm, measured.** Serializing launches would change the output if any kernel consumed
data before its producer had finished, so agreement between serialized and overlapped runs is a real
race-sensitivity result rather than a substitute excuse. Under `HIP_LAUNCH_BLOCKING=1`, with
`HIP_VISIBLE_DEVICES=0`:

| check | overlapped | serialized |
|---|---|---|
| `ctest --test-dir build` | 60/62 passed, 59.73 s | **60/62 passed**, 73.10 s, same two failures, same five skips |
| `pf_gate.sh` | GATE PASS (4 arms) | **GATE PASS** (4 arms), `PF_HASH` identical over 100 chunk-layers |
| `oracle.sh check oracle_gfx1201.txt` | BITWISE PASS | **BITWISE PASS**, all 16 ids identical |

Nothing moved: the same tests pass, the prefill hash is identical, and the generated ids are bitwise
identical when every launch is forced to complete before the next begins. That is the strongest
form of this check available on RDNA4, and it covers the GPU kernels; the host pool is covered by
valgrind separately.

**Kernel coverage of the detector arm, stated as a count.** `kolibri_engine.cu` defines **30**
`__global__` kernels (`k_attn*`, `k_rms*`, `k_moe_sum*`, `k_gemv*`, `k_router_tile*`, `k_swiglu*`, `k_row*`,
`k_kv_ring`, `k_gather_f16`, `k_pos_add`, `k_swiglu`, `k_to_f16`, `k_group`, `k_kt_window` and friends),
`kolibriref.cu` 19 launch sites and `attn_k_parity.cu` 4. The serialized arm above covers every kernel
the **default inference path** launches, because the oracle, `pf_gate.sh` and the 60 passing ctest
tests all run that path end to end. The opt-in paths were then run under the same serialization -
`--grouped`, `--attn-2pass-max 64` and `--dump-layers`, two runs each - and all six returned rc=0 with
byte-identical ids (`286 3515 310 262 14292 1343 1444 46`, the certified prefix) and a deterministic
265,728,000-byte layer dump. `--attn-bench` was then run under the same serialization with a 780 s
bound and **completed, rc=0**, emitting the certified ids (`286 3515 310 262`) - so my earlier
"unbounded, left alone" note was an assumption recorded as a decision, and it was wrong: it finishes
in well under the bound. Every opt-in path the engine exposes is therefore covered by the serialized
arm, and the only genuine gap left is the platform one: RDNA4 has no per-kernel race detector at all.
Five ctest parity tests
are Skipped on this box rather than passing: `hip_prefill_wmma_gemm_parity`,
`hip_prefill_hcd_exact_parity`, `hip_prefill_hipblaslt_gemm` (CUDA-only prefill parity) and
`iq_parity_fixtures` / `iq_parity` (fixtures this box does not have). So the honest form of the
criterion is: **default-path kernels covered by a serialized-launch arm plus valgrind on the host
pool; every opt-in path (`--grouped`, `--attn-2pass-max`, `--dump-layers`, `--attn-bench`) also run
under serialization with rc=0 and certified ids; the one real gap is the platform one - no per-kernel
race detector exists on RDNA4 for any of them.**

## Harness artefacts that had to be re-pointed at this box

**`needle_audit.py --gguf` defaults to the CUDA box's path.** Its default is
`/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`, which does not exist here, and the script does
not read `KOLIBRI_MODEL` - so `--run` with the environment set as usual fails per fixture with
`load failed: cannot open /data4/...` and then reports `0 answer(s) audited, 6 not exact`. That is a
silent-looking failure: the audit still prints a verdict line, and "6 not exact" reads like a recall
regression when it is really six failed loads. On this box the audit must be invoked as
`needle_audit.py --run --gguf /fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`. **Now fixed
opt-in:** `--gguf` defaults to `os.environ.get("KOLIBRI_MODEL") or <the old path>`, so the env every
other kolibri tool already uses is honoured here too. Checked three ways - `py_compile` clean; with
the env unset the resolved default is the identical original string, so CUDA behaviour is unchanged;
with the env set the audit runs clean (rc=0). What was *not* re-run is a full six-fixture audit
through the env path, since the explicit `--gguf` run above already produced the 6/6 verdict and the
only changed expression is the default.



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

`parity.py` (8 prompts, `--n 64 --ngl 99 --cpu-moe 50`): **5 of 8 prompts identical greedy answers**, worst layer rel err 22.20, median layer rel err ~1.15. **The two layer figures are a dump-length artifact and mean nothing** - see "The parity harness's layer-error figure is an artifact" below; the greedy-answer count and the KL are the numbers that carry information.

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

## The 100K needle audit (task 5, second half) - recall 5/5, re-verified 6/6 live

**Re-verified live on the final tip** (`--run --gguf /fast/models/...`, six fixtures, ~17 min each,
box otherwise idle): **6 answers audited, 0 not exact**, every fixture PASS - `a05`
(`quartz-willow-884`), `b25` (`cobalt-orchard-311`), `c50` (`amber-glacier-742`), `d75`
(`velvet-cedar-209`), `e90` (`quartz-willow-107`), and `f128` at 127,811 tokens (`granite-lantern-555`).
That clears the criterion (recall >= 4/5, exact >= 3/5) on a fresh run rather than on the recorded one
above, and the generations now on disk are dated with this run, so the verdict is re-derivable until
`/tmp` is cleared. The same display quirk applies as before: a PASS can print as just `Answer:` or
stop mid-needle (`velvet-cedar-`) because the audit scores the needle's presence in the answer, not
the printed excerpt.


An earlier version of this note said the generations were missing and that `0 answer(s) audited` proved
it. That diagnosis was wrong: the audit found the fixtures fine and reported zero answers because
`--gguf` defaults to the CUDA box's `/data4/...` path (see "Harness artefacts" above), so every load
failed. The six fixtures and their `90.ids` were intact throughout, and the run above reproduces the
verdict in about 100 minutes with no fixture regeneration.

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
(`oracle_ids512.txt`), ctx 8192, `--cpu-threads 8`, `--prefill` (the parallel prefill pool), shipped
config. Box idle at start: load average 0.08, 89 GiB free, 47 GiB in buff/cache.

| measurement | run1 | run2 |
|---|---|---|
| prefill, 512 tokens, load excluded | **87.8 tok/s** (5.83 s) | **85.4 tok/s** (6.00 s) |
| prefill, 512 tokens, end to end (load included) | 26.3 tok/s | 26.2 tok/s |
| decode, tokens 130-257 (the steady window) | **15.8 tok/s** (8.1 s) | **15.8 tok/s** (8.1 s) |
| decode, tokens 2-129 (first window after prefill) | 21.3 tok/s | 16.2 tok/s |

### The prefill row was wrong before; re-derived as a four-arm sweep

An earlier version of this table recorded prefill as 43.4 / 51.7 tok/s, derived from `--n 1` runs in
the §9.3 sweep minus the load line. That row matches no arm measured below and is superseded; the
figures above are the re-measurement. The sweep is four arms, two runs per arm, same prompt and ctx,
raw wall clock:

| arm | `--n 4` wall | `--n 257` wall | prefill (load excluded) | prefill end to end | decode |
|---|---|---|---|---|---|
| `--prefill`, 8 threads | 19.483 / 19.550 s | 35.527 / 35.423 s | **87.8 / 85.4 tok/s** | 26.3 / 26.2 tok/s | **15.8 / 15.9 tok/s** |
| pool off, 8 threads | 44.089 / 44.215 s | 60.101 / 60.041 s | 16.8 / 16.8 tok/s | 11.6 / 11.6 tok/s | 15.9 / 15.9 tok/s |
| `--prefill`, 24 threads | 19.961 / 20.384 s | 52.754 / 54.050 s | 84.8 / 79.3 tok/s | 25.6 / 25.1 tok/s | 7.6 / 7.6 tok/s |
| pool off, 24 threads | 85.744 / 83.977 s | 112.624 / 115.049 s | 7.1 / 7.3 tok/s | 6.0 / 6.1 tok/s | 8.7 / 8.7 tok/s |

Four things the single-arm table hides.

**Why the old row was slow is not the flag.** The old sweep script is still on disk and it passed
`--prefill` too, so the pool was on in those runs as well - the flag does not explain the 2x gap
(11.8 / 9.9 s then against 5.86 / 6.00 s now). What differs is box state and derivation: those `--n 1`
runs followed 100K-token arms in the same sweep, which stream the 42.9 GiB expert table and evict
part of it, and an `--n 1` wall leaves no second decode length to cross-check against. The four arms
above ran back to back on an idle box with 47 GiB resident, each prefill figure checked against its
own `--n 257` partner.

**The prefill pool is the largest single lever on this box**: 87.8 tok/s with `--prefill` against
16.8 tok/s without, **5.2x** on the same prompt. The engine prints which pool it used (`prefill pool:
8 lane(s) = the host + 7 thread(s), 7 pinnable physical core(s), pinned`), so a prefill figure that
does not say whether the pool was on is ambiguous, and every rate row in this ledger now says so.

**`--cpu-threads 24` halves decode**: 15.8 tok/s at 8 threads against 7.6 tok/s at 24. The box is a
Ryzen 7 5700G - 8 physical cores - so 24 lanes oversubscribe and the decode loop pays for it.
`pf_gate.sh` runs with `--cpu-threads 24`, which is harmless for a hash gate and a trap for timing;
the script's own comment already files the `--cpu-threads` choice with the traps of that class. The
split is measured, not assumed: at `--n 16` the same prompt gives byte-identical ids at 8 and at 24
threads (`286 3515 310 262 14292 1343 1444 46 17156 32 50 55 58 4037 262 7771`), so lane count moves
the clock and not the result. One caveat found while checking that: `--cpu-threads 1` does not work -
it emits `51 51 51 51 ...` for all 16 tokens, a degenerate loop rather than an error, so the flag
needs at least 2 lanes and a caller that passes 1 will get plausible-looking garbage.

**The pooled prefill is not bitwise-equal to the plain path the oracle pins.** `oracle.sh` runs the
plain path (no `--prefill`, default threads, `STRATA_DUMP_CTX=8192`) and reproduces
`tools/kolibri/oracle_gfx1201.txt` bitwise - re-checked after these measurements. Adding `--prefill`
to the same prompt and ctx agrees for the first 10 ids and then diverges: plain gives
`... 17156 32 51 48 58 4037 262 1890`, pooled gives `... 17156 32 50 55 58 4037 262 7771`. Both gates
still pass because each pins its own path (`pf_gate.sh` uses `--prefill` and compares against the
`--prefill` reference it recorded), so nothing here is broken - but the 5.2x fast path changes the
answer from the 11th token, and a reader should not assume the bitwise oracle certifies the
configuration the rate table measures. Whether the pooled or the plain trajectory is closer to
llama.cpp is a KL question, and the KL runs in this ledger were done through `strata-kolibriref`, i.e.
the plain path.

**This is a known engine property, not an AMD regression.** `needle_audit.py`'s own docstring records
it from the CUDA campaign: the reference's "whole continuation changes when only the prefill batch
width changes - a semantic no-op that merely reorders fp accumulations." That is the same mechanism
measured here - batching reorders fp accumulation, greedy decoding amplifies a last-bit difference
into a different token, and after 10 tokens the trajectories have diverged. So the finding is the AMD
instance of a documented CUDA-era sensitivity, and it needs no CUDA re-run to interpret; it also
explains why the needle audit scores against the needle text rather than against llama.cpp tokens.

**Decode is unchanged** - 15.8 tok/s here and 15.8 tok/s in the old row - so the box and the build are
behaving identically and only the prefill isolation was wrong.

Long context, same engine, 8 decode tokens, load subtracted:

| prompt | `ATTN_MB=512` | `ATTN_MB=1024` |
|---|---|---|
| 99,299 tokens (ctx 105024) | 102.3 / 100.8 tok/s | 105.2 / 105.8 tok/s |
| 127,811 tokens (ctx 131072) | 94.1 / 94.2 tok/s | 100.4 / 100.8 tok/s |

Two things follow. The **marginal prefill rate is ~105 tok/s**, which against the corrected
short-prompt figure of 87.8 tok/s is only **1.2x**, not the 2.4x an earlier version of this section
claimed - that ratio was computed against the superseded 43.4 tok/s row and is wrong in the same
direction. The fixed cost a short prompt pays is therefore about 1 s (5.86 s measured for 512 tokens
against 4.88 s at the marginal rate), not the ~6 s previously stated: short-prompt prefill on this
box is close to its marginal rate, and there is little fixed cost left to amortise once the prefill
pool is on. And **`ATTN_MB=1024` is faster than 512 at both long contexts** (105.5 vs
101.6 at 100K, 100.6 vs 94.2 at 128K), so on this card the larger attention slab is a win, not just
a fit; the T4's OOM at 100K does not reproduce here, at 32 GB, with the expert table in host RAM.
These long-context rows were taken with the same shape as the §9.3 arms, prefill pool on.

Against the CUDA record (`strateu-campaign-report.md`: prefill 157.0/160.3 tok/s at 512 tokens,
decode 24.4/24.5 tok/s short context) the AMD path is about 1.8x slower on short prefill (87.8 against 157.0, pool on) and about
1.5x slower on decode. That comparison needs a caveat: the campaign's table says "wall clock" without
saying whether its load time was subtracted, and the AMD numbers here are quoted both ways for that
reason. Decode at long context is not measured yet - each such run is ~16 min - so the CUDA row
"tg 128 after 32K, 12.3 tok/s" has no AMD counterpart yet.

## §9.3 verdicts, remaining arms (two runs each, same box and load as above)

Short shape is `n=257` on the 512-token prompt, so arms compare against the baseline's 39.3 / 39.3 s.

**What that baseline is, and what it costs.** It is not the bare default: the sweep set
`EXPERT_F16=24`, `RES_TOP=56` and `ATTN_MB=512` explicitly. Re-running with exactly those three gives
39.357 / 39.204 s, which reproduces the recorded 39.3 / 39.3 s to 0.1 s, so the baseline is now fully
accounted for. Taken one at a time against the bare default (35.5 s, two runs, same shape):
`EXPERT_F16=24` alone gives 39.394 / 39.338 s - **the whole 3.8 s** - while `RES_TOP=56` (35.524 /
35.599 s) and `ATTN_MB=512` (35.345 / 35.487 s) are indistinguishable from the default at this shape.
So every §9.3 short-shape verdict is measured against a baseline carrying a 3.8 s `EXPERT_F16` cost,
which is fine for comparing arms against each other but means those seconds are not the shipped
default; the rate table above quotes the default and is the faster configuration by that margin.

**`EXPERT_F16=24` also changes the answer, not just the clock.** Two runs each of bare default and
`EXPERT_F16=24`, `--n 32` on the same 512-token prompt: each config reproduces itself exactly, but the
two disagree from the **5th generated token** - bare gives `... 17156 32 50 55 58 4037 262 7771 ...`,
`EXPERT_F16=24` gives `... 48590 32 49 45 50 54 1444 46 ...`. On this card the knob is neither free nor
cosmetic: it costs 3.8 s at short shape and moves the greedy trajectory. The bitwise oracle and
`pf_gate.sh` references were recorded on the bare default and pass there, which is the other reason
the default is the configuration this ledger quotes.

| arm | wall run1 / run2 | verdict |
|---|---|---|
| baseline (`RES_TOP=56`) | 39.3 s / 39.3 s | - |
| `ROPE_F64=1` | 39.6 s / 39.5 s | no measurable cost at this shape - 0.3 s over a 39.3 / 39.3 s baseline, inside run-to-run spread. RoPE in fp64 is free here, so there is no accuracy-vs-speed trade to argue on this card - it can stay on if it is more accurate |
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

*(Superseded as the headline: 0.00182 is the **reference-engine** figure. Measured on the shipped
engine the answer-step KL is **0.00146 plain prefill / 0.00162 pooled**, i.e. **0.78x** of the same
floor - see "The KL measured on the shipped engine" below. The bound is unmet either way.)*

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
given above - not a claim that the bound was met. On the shipped engine rather than the reference the
figures are **0.00146 plain / 0.00162 pooled (0.78x / 0.86x)**, which is the line to quote.

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

The lever space is enumerated, not assumed. `strata-kolibriref --help` and the flag list parsed in
`kolibriref.cu` expose exactly two numerics switches - `--act q8|f32` and `--kernels strata|plain` -
and everything else (`--model`, `--tokens`, `--tokens-file`, `--n`, `--ctx`, `--force`, `--dump-layers`,
`--dump-logits`, `--dump-full-logits`) is input or instrumentation. Both numerics switches are
measured above, `--act f32` proved inert, and the binary reads no environment variables at all, so
the `STRATA_KOLIBRI_*` knobs that move the shipped engine cannot change a KL number taken through the
reference. There is no untried legitimate lever left inside this harness.

### The KL measured on the shipped engine, not just the reference



Every KL number above came from `strata-kolibriref`, the reference engine. The criterion is about the
engine that ships, and `strata-kolibri` accepts `--tokens`, `--force` and `--dump-full-logits`, so
`kl.py --ref build/strata-kolibri` measures it directly - the llama.cpp logits in `/tmp/kl-amd` are
reused from cache, so only the strata side re-runs. Same 8 prompts, same 344 answer steps, same floor
(0.00187), two configurations of the real engine:

| configuration | answer-step KL | top-1 | ratio to this box's floor |
|---|---|---|---|
| llama.cpp CPU against llama.cpp GPU (the floor) | 0.00187 | 0.9884 | 1.00x |
| `strata-kolibriref`, shipped fused kernels | 0.00182 | 0.9884 | 0.97x |
| `strata-kolibriref`, `--kernels plain` | 0.00158 | 0.9942 | 0.84x |
| **`strata-kolibri`, plain prefill** | **0.00146** | 0.9942 | **0.78x** |
| `strata-kolibri`, pooled prefill (`--prefill`) | 0.00162 | 0.9942 | 0.86x |

Two things follow, and both matter more than the original 0.00182 did.

**On the ratio metric the AMD path is at parity with the CUDA campaign.** The T4 recorded 0.00132
against a floor of 0.00178, which is 0.74x; the best AMD configuration is 0.00146 against a floor of
0.00187, which is **0.78x**. Those are the same achievement to within 5%, and it is the honest way to
state this gate: the absolute number cannot transfer between boxes, the ratio can.

**The prefill pool has a measurable accuracy price.** Pooled prefill is 0.00162 against plain
prefill's 0.00146 - the same 5.2x speed difference measured in the rate table buys a worse KL - so
unlike `ROPE_F64`, which measured free, the pool is a genuine accuracy-for-speed trade and the choice
should be made deliberately rather than inherited. The bound of 0.00132 is still met by none of these,
so the deviation stands; but the closest the shipped engine gets is 0.78x of floor, not 0.97x.

### The shipped engine's numerics lever space, enumerated

`kolibri_engine.cu` reads 38 `STRATA_KOLIBRI_*` variables. Sorted by whether they can move this gate
at all:

| group | knobs | can they close the gap? |
|---|---|---|
| prefill batching | `PF_CHUNK`, `PF_LANES`, `PF_STAGE`, `PF_PIPE`, `PF_CHUNK_KEY` | they change the trajectory (batch width reorders fp accumulation), but choosing a chunk or lane count to minimise KL would be tuning to the metric, not a measurement - excluded on that ground, not on evidence |
| expert residency and precision | `EXPERT_F16`, `EXPERT_HIST`, `RES_TOP`, `POOL_ALL`, `DENSE_F16`, `DEC_RES` | `EXPERT_F16=24` measured: costs 3.8 s and moves the answer from token 5; the others are residency policy, and `RES_TOP=0` crashes |
| attention path | `ATTN_FLASH` | **measured: no effect at this shape.** With `ATTN_FLASH=1` the generated ids are byte-identical to the tiled path (`286 3515 310 262 14292 ... 50 55 58 4037 262 7771`) and the KL run returned the same statistics to the digit (0.00146, median 0.000118, p99 0.0167, top-1 0.9942). The env is genuinely read (`static const bool on = env_on(...)`, `kolibri_engine.cu:186`), so this is the two paths agreeing here, not a flag that failed to engage - though I did not separate "numerically identical at 512 tokens" from "the flash path is not reached at this shape". Either way it is not a lever for this gate at this shape. |
| attention kernels (other) | `ATTN_PV4`, `ATTN_MMA`, `ATTN_SIMT`, `ATTN_KG`, `ATTN_KT`, `ATTN_MB`, `ATTN_COALESCED` | kernel selection for speed; `ATTN_MB` measured as a wash at short shape and the §9.3 arms swept the rest |
| decode kernels | `DEC_F16`, `DEC_GEMV_V4`, `DEC_PHASES`, `DEC_XEXP` | decode-path only; the KL is measured on forced sequences where prefill dominates the divergence |
| router | `ROUTER_GEMM`, `ROUTER_TILE`, `ROUTER_TK`, `ROUTER_TJ` | §9.3 found the tile path marginally different in timing; routing decisions are top-6 selection, and a change here would change which experts run, i.e. the model |
| numerics of RoPE | `ROPE_F64` | **measured: 0.00161, worse than 0.00146** |
| diagnostics | `PF_HASH`, `PF_FENCE`, `PF_PIN`, `ATTN_DUMP`, `ATTN_DUMP_LAYER`, `ATTN_PHASES`, `DEC_PHASES`, `LPT`, `MMA_AVAILABLE`, `KV_DEV`, `PV_SCALE`, `POISON_ZERO_SLOT` | instrumentation or guards; no effect on the distributions |

The native expert kernels are not a knob at all: `native_layers` is counted from `b.native_ok`, set
per layer at load from the blob layout (`kolibri_engine.cu:3701`), so the "25 / 50 layers" in the
load line is a property of the file, not a switch.

So the honest statement of the gate is: **the best legitimate configuration on this box is 0.00146,
0.78x of the floor**, and the remaining distance to 0.00132 is not covered by any switch that can be
chosen on merit. Closing it is the §8.1 kernel project - matching llama.cpp's accumulation order in
the expert path - which the goal's boundaries place out of scope.



### `ROPE_F64` is free on speed and slightly *worse* on this gate

The engine reads env knobs the reference binary does not, so `STRATA_KOLIBRI_ROPE_F64=1` is a lever
that exists only on the shipped path - and it measured free on the clock, which made it the obvious
candidate for closing the gap. It does not close it: same 344 answer steps, plain prefill, fp64 RoPE
gives **0.00161 (0.86x)** against 0.00146 (0.78x) without it, and top-1 drops from 0.9942 to 0.9884.

That is not a paradox, and it bounds what this gate can ever reward: KL here measures **agreement with
llama.cpp**, not accuracy in any absolute sense. llama.cpp computes RoPE in its own precision, so
making Strata's rotation more exact moves it further from the reference it is scored against. A
configuration can be numerically better and score worse on this criterion - a second reason the
absolute 0.00132 should not be read as a quality threshold; it is a similarity measurement against
one particular reference build on one particular box.

The standing reading is unchanged and is a measurement, not an excuse: the bound sits below the floor
of the instrument it is quoted against on this box (0.00178-0.00187 depending on population), so it
cannot be met here by any engine that is not llama.cpp itself, and meeting it on the T4 was a property
of that box's smaller CPU-vs-GPU divergence.

| `pinned_shared_test` | `OK` - the pinned-allocation path the whole design rests on | **0** |
| `file_expert_source_test` | `PASS` | **0** |

## The parity harness's layer-error figure is an artifact - do not read it as a defect

`parity.py` reports `layers max rel err 22.20, median ~1.15` on this box, which reads like the AMD path
is 115% wrong per layer. It is not, and the KL proves it: 0.00182 with top-1 agreement 0.9884 is not
reachable from layer outputs that badly off. The cause is in the dumps themselves - for `p00` the two
layer files are **different lengths**:

| file | bytes | positions at 50 layers x 2560 embd x f32 |
|---|---|---|
| `p00.strata.layers` | 60,928,000 | **119** |
| `p00.llama.layers` | 32,768,000 | **64** |

`strata-kolibriref` dumps prompt-plus-generated positions; `kolibri_llama_dump` dumps the generated
ones. Both reshape cleanly on their own, so nothing raises, but the elementwise difference is then
between tensors that do not correspond position-for-position, and a relative error on top of that is
noise. The trustworthy outputs of that harness are the ones computed from aligned logits: the
greedy-token agreement (5 of 8 at `--n 64`, 7 of 8 at `--n 24`) and everything `kl.py` derives. The
layer columns should be treated as unmeasured until the two dumps are cut to the same position range
- a harness fix, out of scope for the port, and deliberately not papered over here.

## CUDA is not behaviourally touched by the port - checked hunk by hunk

The port's boundary is that the CUDA path keeps working exactly as before, so each of the four modified
files was read rather than assumed (`git diff main..HEAD`, 41 insertions / 4 deletions total):

| file | change | effect on a CUDA build |
|---|---|---|
| `include/strata/kernels/kolibri_window.hpp` | `#if defined(__CUDACC__)` -> `#if defined(__CUDACC__) \|\| defined(__HIPCC__)` for `STRATA_KOLIBRI_HD` | none - nvcc already defined `__CUDACC__`, so it takes the identical branch |
| `src/kolibri/kolibri_engine.cu` (device guards) | `#if defined(__CUDA_ARCH__)` -> `... \|\| defined(__HIP_DEVICE_COMPILE__)` in `attn_pv4_on()` and `attn_coalesced_on()` | none - the CUDA device pass has `__CUDA_ARCH__`, the host pass has neither, so both passes pick the same branch as before |
| `src/kolibri/kolibri_engine.cu` (MMA) | new `#if defined(__HIPCC__)` -> `STRATA_KOLIBRI_MMA_AVAILABLE 0`, `#else -> 1`, wrapping `mma168`, `attn_mma_on()`, `k_attn_flash_mma` and its launch site (`if (...) {...} else` + `#endif` + `if (attn_flash_on() ...)`) | none - CUDA compiles to `1`, so every one of those definitions and the original `if/else if` chain are compiled exactly as they were. The PTX `mma.sync.aligned.m16n8k8` path is NVIDIA-only and was never reachable on HIP; it is now gated rather than silently mis-selected |
| `include/strata/hip_compat/cublas_v2.h`, `include/strata/hip_compat/cuda_runtime.h` | added `cublasSetAtomicsMode`/`cublasGemmBatchedEx` wrappers and three missing runtime shims | not reachable - that directory is put on the include path only by `cmake/hip_backend.cmake` and the `elseif(STRATA_ENABLE_HIP)` branch of `CMakeLists.txt:206`; a CUDA configure sets `_strata_gpu_include_directories` from `CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES` instead |

So "byte-identical CUDA" holds in the sense that matters - a CUDA build compiles the same code and takes
the same branches - while the shared sources necessarily differ in text, because a port that changed
nothing could not compile on a second backend.

### Where the KL excess actually sits: per-step, not uniform

`kl.py` leaves a per-step array behind (`/tmp/kl-amd/<tag>.kl.npz`, keys `answer`, `cpu`, `strata`),
which says the gap is not a uniform drift - it is a handful of positions, and short-answer prompts
amplify them because one bad step dominates a three-step mean.

| prompt | answer steps | strata KL, first steps | floor KL, same steps | worst step overall |
|---|---|---|---|---|
| `p06` | 3 | 0.0484, 0.0025, 0.0259 | 0.0086, 0.0022, 0.0027 | strata 0.0833 / floor 0.0810 at step 4 |
| `p07` | 4 | 0.0196, 0.0012, 0, 0 | 0.0026, 0.0018, 0, 0 | strata 0.0741 / floor 0.0526 at step 14 |
| `p00` | 65 | 0.0013, 0.0001, 0.0227, 0, ... | 0.0118, 0.0001, 0.0157, 0, ... | strata 0.0243 (step 21) vs floor 0.0278 (step 38) |

Two things follow. On `p06` the excess is at answer steps 0 and 2 - **5.6x and 9.6x the floor there** -
while at the genuinely hard step (4) the two engines agree almost exactly (0.0833 against 0.0810), so
that step is hard for the model, not a disagreement between engines. And `p00`, the one prompt with a
long answer, has a strata worst-step (0.0243) **below** the floor's worst-step (0.0278).

So the mean that misses the bound is carried by the short-answer prompts, where a single early step is
divided by three or four rather than by sixty-five. That is where anyone closing the remaining gap
should look - the first answer step of short answers - and it is a smaller and different problem than
"the AMD path is uniformly less accurate", which these arrays rule out.

## Where the port stands

Committed on `feat/amd_r9700` (main untouched): the launcher (`251a079c`); the HIP build,
load and ctest (`f901ca06`, `f53f87a6`); `pf_gate` PASS and this log (`f86ff892`); the reference
llama.cpp finding (`fc75cddf`, `86c84069`, `0ad27d37`); the §9.3 verdicts (`5f1fbb87`, `b472b798`,
`f414e8ce`, `1be74ca4`); the KL measurement (`d2e09784`, `f58ebf2e`); the needle fixtures and generator
(`69d7a8ee`); the committed AMD gate references (`5b388fda`); the detector evidence (`0ed717dd`,
`1f1065df`); and the CUDA-safety check (`09de8aa0`).

**Gates re-verified on the final tree, after every rate correction in this log.** `pf_gate.sh` gives
`GATE PASS` with all four arms green - `build PASS`, `ids smoke PASS (gen: 3515 310 262 14292)`,
`PF_HASH PASS (100 chunk-layers)`, `oracle PASS (BITWISE)` - and `oracle.sh check` was run separately
against `oracle_gfx1201.txt` with the same result. The gate builds before it tests, so the PASS also
re-confirms the HIP build is clean at this tip. Re-checked once more after the `needle_audit.py`
change, which is the only non-docs edit since: `GATE PASS` (all four arms), `BITWISE PASS`, and
`check_published_figures.py` passing at 82 values. Nothing imports `needle_audit.py` - the only other
reference is a mention in `make_needle_fixtures.py` - so that edit is isolated from the gates.

`ctest --test-dir build` on the same tip: **60 of 62 passed** (59.73 s), the same two failures and the
same five skips as the original run - `ple_parity` needs a Qwen file this box does not have, and
`platform_memory_test` wants POSIX `mlock` against `ulimit -l` = 8 MiB. Neither is a Kolibri defect;
both are box-scoped, and the count is quoted here rather than from memory because the criterion asks
for it.

**One figure to reconcile with the goal text.** The objective says a 42.4 GiB host expert table; the
engine reports **42.90 GiB pinned** on this GGUF, which is the number used throughout this log. The
difference is 0.5 GiB of rounding or a different file revision, not a discrepancy about what was
allocated - the load line is quoted verbatim wherever the figure appears.

**Gates, as they reproduce from a clean checkout.** Four exports are needed because the harnesses
default to the CUDA box:

```bash
export KOLIBRI_MODEL=/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf
export KOLIBRI_BIN=$PWD/build/strata-kolibri KOLIBRI_CM=$PWD/.venv/bin/cmake HIP_VISIBLE_DEVICES=0
tools/kolibri/oracle.sh check tools/kolibri/oracle_gfx1201.txt      # BITWISE PASS
tr ',' '\n' < tools/kolibri/oracle_ids512.txt | head -64 > /tmp/ids64.txt
bash tools/kolibri/pf_gate.sh                                       # GATE PASS
```

The AMD oracle and pf_gate expectations are in the tree now (`tools/kolibri/*_gfx1201.txt`), so the
gates survive a reboot; the CUDA references beside them are untouched.

**The reference that made the KL measurable** is `/fast/llama-server-kolibri/llama.cpp` at
`f5027fb21` (the user's Kolibri-capable tree), built here as `build_rocm_rizo` with
`-DGGML_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1201`. The earlier `~/develop/llama.cpp` route is recorded
above and superseded: it could not load the file at all.

**One criterion is not met.** KL on the answer steps measures **0.00182** shipped (0.97x of this box's
0.00187 floor) and **0.00158** with `--kernels plain` (0.84x), against a bound of **0.00132**. Four
separate legitimate attempts are documented above - shipped kernels, `--act f32` (inert),
`--kernels plain` (a real improvement that still misses), and a fixture-shape hypothesis that the data
refuted. The bound sits below the floor of the instrument it is quoted against on this box, so it is
recorded as a deviation with its evidence rather than closed.

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

## T16 (2026-10-28): `--cpu-threads 1` is not a valid pool, and the flag now says so

Measured on the R9700 (gfx1201, `HIP_VISIBLE_DEVICES=0`), Kolibri-1-Q4_K_M, ctx 8192, pool on, `EXPERT_F16` unset
(the engine default, 0), 7 pinnable physical cores, box idle, 90 GiB free, one arm per engine load.

The flag was documented above as a trap without a guard: one lane emits `51 51 51 51 ...` for all 16 tokens, a
degenerate loop that looks like an answer. The engine now clamps 1 to 2 while parsing the argument - before the
model read, so the message arrives even when the load fails - and prints:

```
strata-kolibri: WARNING: --cpu-threads 1 is not a valid pool: one lane emits the degenerate 51 51 51 ... repeat loop (measured 2026-10-28); using 2
```

Measured, same prompt as the lane-count check above (`tools/kolibri/oracle_ids512.txt`, `--n 16`):

| arm | banner | ids |
|---|---|---|
| `--cpu-threads 1` | `prefill pool: 2 lane(s) = the host + 1 thread(s), 7 pinnable physical core(s), pinned` + the warning | `gen: 286 3515 310 262 14292 1343 1444 46 17156 32 50 55 58 4037 262 7771` |
| `--cpu-threads 2` | same banner, no warning | identical, byte for byte |

Those are the same ids the 8-thread and 24-thread arms produced (§9.3), so lane count still moves the clock and not
the result - now with no lane count that silently breaks it. `oracle.sh check` still reports BITWISE PASS (the plain
path, no pool, untouched by this change).

New test: `tests/kolibri/cpu_threads_floor_test.cmake` (`ctest -R kolibri_cpu_threads_floor`), 0.58 s, no card and no
GGUF needed - it reads the warning, checks it names the clamped count, checks the run still reaches the model load
instead of exiting at the clamp, and checks that 2 lanes and the pool-off default stay silent.

**`ctest` denominator, corrected.** The gate line has been quoting 60/62; this tree registers **87** tests. The run
after this change: **83/87 passed, 142.04 s**, four fails, all environment:
`ple_parity` (wants a Qwen GGUF this box does not have), `platform_memory_test`, and two that were not in the earlier
figure - `vmm_test` (a 3 GiB `cudaMemset` at a VMM offset fails) and `conversation_copy_streams_test` (the card's
first copy stream and event "take under 16 MiB" assertion). Neither of those two links `strata_kolibri`, so this
change cannot reach them; they are recorded here as box-scoped, not as a regression.

## T13 (2026-10-28): the `RES_TOP=0` zero-block launch, found and guarded

Filed as "`dequant launch: invalid configuration argument`, no `gen:` line" (§9.3). On this tip that symptom is
**gone for the wrong reason**: `--prefill --cpu-threads 8`, `EXPERT_F16=24`, `RES_TOP=0`, 512-id prompt, ctx 8192
completes in 19.68 s with `gen: 286 3515 310 262`. An instrumented build showed why it no longer reports:

| arm | `b.f16` | resident blocks `nblk` | zero-block launches per 512-token prefill |
|---|---|---|---|
| `RES_TOP=0` | set | **0, every layer, every chunk** | 100 (2 chunks x 50 layers) |
| `RES_TOP=56` | set | 33 … 60 per layer-chunk | 0 |

So the launches were still happening with a zero grid - they simply stopped being *fatal*: the launch error is
sticky, and the checks that would have reported it (`dequant_f16`'s `cudaGetLastError`) now early-return on a
zero-size conversion (the guard added in `3b24d3f5`), while other calls in between clear the slot. Measured
separately on this stack (ROCm 7.2.2, gfx1201): `k<<<dim3(0),128>>>` → `invalid configuration argument` (error 9),
`k<<<dim3(1),128>>>` → no error. A zero-block launch is an error; whether anyone hears about it is luck.

Root cause, with the data flow: with a zero promotion ceiling `res_eff = min(res_top, n_cached) = 0`, so `pf_res`
stays empty, the block loop that builds the resident rows never runs, `nblk = 0`, and the section that follows
launched `k_gather_f16`, `k_swiglu_rows`, `k_to_f16` and three `gemm.f16_batched` calls on that count. That is the
whole defect - the dequant was the messenger.

Fix: guard the section on `nblk > 0` (`kolibri_engine.cu`, one brace around the launch sequence, `rows` moved inside
it). Nothing else moved. `RES_TOP=0` is a legal configuration again: the cache stays allocated, every touched expert
goes to the pool, and the numbers are the same ones the unguarded build produced.

| arm | wall | ids (512-id prompt, `--n 4`) |
|---|---|---|
| `RES_TOP=0`, before the guard | 19.78 s / 19.68 s | `286 3515 310 262` |
| `RES_TOP=0`, after the guard | 19.7 s (ctest arm, 64 ids) | identical |
| `RES_TOP=56`, before and after | 23.66 s | identical |

Audit of the rest: the file has 83 `<<<>>>` launches (84 lines mention one, one of them inside a comment), 16 of them
through `grid1(...)`. Every other count is
`Tc * …`, `NKc * …` or `N * K` with `Tc >= 1`, `NKc = K >= 1` and `FS = n_ff_shexp > 0`; the one decode-side
`nblk * FF` (`:2262`) is built from the token's own K selections and cannot be zero while a token exists. The
resident prefill block was the only zero-capable one.

Regression: `ctest -R kolibri_res_top_zero_test` (35.79 s) runs both arms and requires `RES_TOP=0` to finish with a
`gen:` line whose ids equal `RES_TOP=56`'s; it skips (77) where the GGUF is absent. Gates after the change:
`oracle.sh check` BITWISE PASS, full `ctest` 84/88 (same four box-scoped fails as after T16).

## T1 (2026-10-28): the standing baseline arm, and where the decode's 63 ms/token goes

### `baseline-standing` — the one arm every later row cites

```sh
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
STRATA_KOLIBRI_EXPERT_F16=0 STRATA_KOLIBRI_ATTN_MB=512 STRATA_KOLIBRI_RES_TOP=56 \
build/strata-kolibri --model /fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 257 --ctx 131072 --cpu-threads 8 --prefill
```

That is the launcher's own command line (`run-kolibri1.sh`: `--prefill --cpu-threads $(physical cores) --ctx 131072`
plus the `EXPERT_F16=0 / ATTN_MB=512 / RES_TOP=56` defaults), run twice, box idle, 90 GiB free, one arm per engine
load. `--n 4` is run in the same session as the second arm because the load line is not subtractable: the reported
"13.4 s" covers the file read, not the pinned repack that follows it.

| quantity | run 1 | run 2 | how |
|---|---|---|---|
| wall, `--n 4` | 19.57 s | 19.63 s | `/usr/bin/time` |
| wall, `--n 257` | 35.49 s | 35.57 s | `/usr/bin/time` |
| **decode, ctx 131072** | **15.90 tok/s** | **15.87 tok/s** | **wall-derived**: (35.49-19.57)/253 = 62.9 ms/token, load cancels |
| prefill, 511 tokens | **108.6 / 110.8 tok/s** (corrected, §P0) | - | last `PFchunk`, not the sum: 4.71 / 4.63 s. The 71.0 / 69.6 this row used to carry came from adding cumulative timers |

The other two baselines in circulation are the same engine in different clothes, and this is where they now
live: 15.8 tok/s (§Rate table, ctx 8192, same config) and the launcher sweep's 15.85-16.20 (engine-reported,
ctx 131072). The tuner's 17.6-18.2 was a different request set. Cite this table; cite the others as context.

### The leg split, counted and profiled

Attribution runs, untimed by the engine, same config, 512-token prompt, ctx 8192 (the shape the ledger's rate
table uses), `--counters`:

```
CNT run tok=257 | slot hits=0 miss=0 (-nan% hit) expert PCIe=0.0 MB/tok | d2h=13107/263168000 B (1.0 MB/tok)
                  h2d=12950/2359296000 B (9.2 MB/tok) | reset=0 ms / 0 (0.00 ms)
CNT pool workers=8 | streamed 714.2 MB/tok (183.56 GB total) | gu 3516 ms q 33 ms down 2178 ms = 5727 ms
                  (22.28 ms/tok, 32.1 GB/s)
CNT pool phases: wait_park 0 ms drain 5764 ms repark 0 ms = 5764 ms (22.43 ms/tok) vs work 22.28 ms/tok
CNT branches per tok: pool 50.0 gpu-expert 0.0 slots-path 0.0 grouped 0.0 (layers/token total 50.0 of 50)
```

**`CNT decode phases: pre/d2h/pool/post` does not print on this path.** The marks are incremented only in the
slots branch (`:2196`, `:2222`) and the native/grouped branch (`:2281-2282`); the shipped pool-only decode
increments nothing, so `dec_layers` stays 0. That is the instrument gap the plan suspected, stated precisely:
at the shipped config there are no GPU-side phase marks to read, and the pool's own `phase_ms` line is what
stands in for them. For the same reason `CNT prefill phases:` prints all zeros - those marks are only written
where the prefill runs its own resident path.

`rocprof --stats` over a 128-token decode (`--tokens 286 --n 128 --ctx 8192`, same knobs):

| kernel | calls | total | per token | µs/call |
|---|---|---|---|---|
| `native_q4_k_mmvq_kernel<false>` | 35 200 | 744.5 ms | 275 | 21.2 |
| `k_attn` | 6 400 | 326.1 ms | 50 | 50.9 |
| `k_rms_add` | 12 800 | 249.1 ms | 100 | 19.5 |
| `k_rms` | 12 928 | 198.7 ms | 101 | 15.4 |
| `native_q6_k_mmvq_kernel<false>` | 3 328 | 128.7 ms | 26 | 38.7 |
| `native_quantize_q8_1_kernel` | 44 928 | 115.7 ms | 351 | 2.6 |
| `k_gemv_f32_v4` | 6 400 | 61.2 ms | 50 | 9.6 |
| `route_sigmoid_logit_add<384,6>` | 6 400 | 45.4 ms | 50 | 7.1 |
| `__amd_rocclr_copyBuffer` | 13 154 | 45.0 ms | 103 | 3.4 |
| `k_attn_prep`, `k_moe_sum`, `k_swiglu`, `k_kt_window` | - | 41.1 / 20.7 / 15.5 / 15.1 ms | 50 / 50 / 50 / 40 | - |
| **sum** | **172 514** | **2.065 s** | **1 347 launches** | 12.0 |

**The LM head**, measured per dispatch with `rocprofv3 --kernel-trace` (8-token decode): a `native_q6_k_mmvq`
launched exactly once per token, the last dispatch of every token - **519.8-520.3 µs in steady state**, and
**697.2 µs on the first token** (+34 %, which is a warm-up datum for Task 15: the same kernel, the same shape,
one cold call). That is 0.52 ms of the 62.9 ms wall, **0.8 %**. It is `Q6_K`, not `Q4_K`, and it is not the
dense `k_gemv_f32_v4` (that one runs 50x/token, i.e. per layer).

Prefill, 511 tokens, 2 chunks: `CNT prefill pool: chunks=100 experts/chunk=117 bytes=28.05 GB (280 MB/chunk)
pool=3660 ms (36.6 ms/chunk, 7.7 GB/s)`, against a chunk wall of **4.71 s** (§P0's correction; 7.20 s was the sum of
cumulative timers) - the pool's own run is **79 %** of
the prefill. The GPU side of that prefill (`--stats`): `k_router_tile_k<0,32,3>` 100 calls / 52 ms, `k_attn` 200 /
41 ms, hipBLAS `Cijk_*` 800 calls / 47 ms, `dequant_kernel<12>` 600 / 11 ms - counts are usable, times are not
(rocprof inflates launch-heavy code, the same trap the plan records for nsys).

### Attribution table (decode, ctx 8192, shipped config)

| leg | ms/token | source |
|---|---|---|
| wall | **62.9** | wall-derived, `--n 4` vs `--n 257` |
| GPU kernels | **16.1** | rocprof `--stats`, 172 514 launches / 128 tokens |
| CPU pool work | **22.3** | engine `CNT pool workers` + `drain 22.43 ms/tok` |
| not covered by either: driver/handoff, non-overlapped pool wait | **~24.5** | subtraction |

The trace shows what that 24.5 ms is made of: 1 404 dispatches per token with a **median inter-dispatch gap of
11.8 µs** (p90 40 µs), and the stream idle for ~82 % of the decode window. Per token the GPU is busy 16 ms out
of 63; the biggest single kernel in the whole decode is the LM head's 520 µs.

**What this changes in the plan.** The decode is launch- and handoff-bound, not arithmetic-bound: halving every
GEMV kernel (Tasks 3, 4) can recover at most ~7 ms of the 63, while 1 347 launches and a 22 ms serialized CPU
pool leg sit on either side of it. The prefill is the opposite - half of its wall is a pool that already runs at
its roof (32.1 GB/s streaming, 326-329 GFLOP/s of compute, `146 % of the 226 GFLOP/s roof` in the pool's own
accounting) - so prefill gains have to come from *not* running 117 experts per layer through the pool, which is
what residency (T6/T7) and the fused WMMA expert kernel (T11) are for. It also weakens Task 5's speed premise
outright: the LM head is 0.8 % of decode, so that task is a numerics/VRAM question at best.

## T2 (2026-10-28): the grid-shape micro-bench, and what it says about T3, T4 and T5

New opt-in tool, `tools/kolibri/grid_shape_bench.cu` (`ctest -R kolibri_grid_shape_bench`, `--selftest`, 20 s), built
from the engine's own `deq32<T_Q4_K>`, `k_gemv_f16` body and `block_sum`, so a number here transfers to the engine's
kernels by construction. It does not link `strata_kolibri`. Every arm runs with its output hashed and the bit
patterns are compared between two runs.

Measured on the R9700 (gfx1201, 32 CUs, `HIP_VISIBLE_DEVICES=0`), box idle, two runs each:

| arm | run 1 | run 2 | GB/s | bitwise |
|---|---|---|---|---|
| A1 F16 GEMV [128000 x 2560], block-per-row (128000 blocks) | 2.1200 ms | 2.1253 ms | 309 | reference |
| A2 same, grid-stride (1024 blocks) | **1.9020 ms** | 1.9026 ms | **345** | identical to A1 |
| B1 Q4_K GEMV [2560 x 2560], block-per-row (2560 blocks) | **0.0261 ms** | 0.0263 ms | 141 | reference |
| B2 same, grid-stride (256 blocks) | 0.0336 ms | 0.0337 ms | 110 | identical to B1 |
| C1 same shape, packed deq32 (32-bit loads, no branch) | **0.0216 ms** | 0.0217 ms | **171** | identical to B1 |
| D1 same shape, int8 dot (dot1 emulated) | 0.0388 ms | 0.0386 ms | 95 | own numerics (probe) |
| E1 18 tiny launches (6 picks x 3 slices) | 0.0665 ms | 0.0666 ms | - | - |
| E2 1 batched launch (18 blocks) | 0.0037 ms | 0.0037 ms | - | - |

Bit patterns identical across the two runs and across the pairs (`A 7469326e6c9f7383`, `B/C 24df816513f487f6`).

**Verdict 1 - the grid cap is a size-conditional win, not a free one.** At the LM head's `n_out = 128000` the
grid-stride form is **10.3 % faster** (1.902 against 2.120 ms: 128000 single-row blocks cost more than the extra
strided loop). At a dense projection's `n_out = 2560` the *same* change is **29 % slower** (0.0336 against 0.0261 ms)
- 256 blocks leave the 32 CUs under-fed on a shape that fits in 3.7 MB. So Task 3's knob cannot be applied blanket:
it has to gate on `n_out` (a threshold near 16k rows separates the two regimes by a factor of 10 in block count),
and on the shapes the engine actually runs, which are 2560 far more often than 128000.

**Verdict 2 - the packed deq32 buys what the T4 record promised, and it is free of numerics risk.** +21 % on the
Q4_K GEMV (0.0216 against 0.0261 ms), bitwise identical by construction - the same float expression per element,
only the nibble extraction changes (32-bit loads, no per-element branch). That is the "+15 % on gfx1201" class from the
main engine, measured on this card.

**Verdict 3 - a real dp4a arm cannot be built on gfx1201.** `__builtin_amdgcn_sdot4` refuses to compile for this
architecture ("needs target feature dot1-insts"), so the int8 arm here is the emulated 32-MAD form, and it is
**1.5x slower** than the scalar fp32 dot (0.0388 against 0.0261 ms). Any gfx12 int8 plan pays that emulation; there is
no dot-instruction path to price.

**Verdict 4 - a launch costs 3.69 us.** 18 tiny launches take 66.5 us where the one batched launch doing the same
work takes 3.7 us. With Task 1's 1347 launches per decode token, launch issue alone is ~5 ms/token of the 63 ms wall,
before any handoff. This is the number Task 8's per-pick arms must beat by batching.

**Verdict 5 - the fp16 LM head (Task 5) would be four times slower, not faster.** The bench's A arm is exactly that
proposal's data path: 128000 x 2560 of F16 at 2.12 ms, against the shipped Q6_K `native_q6_k_mmvq` for the same
projection measured at **0.52 ms** (Task 1). The fp16 form doubles the bytes read and the card is bandwidth-bound at
309-345 GB/s on that shape. Combined with Task 1's finding that the LM head is 0.8 % of decode, Task 5's speed premise
is dead on this box; it can only be argued as a numerics experiment, and the shipped Q6_K path already wins on speed.

## T3 (2026-10-28): the grid-capped GEMV lands, and finds nothing to cap on this model

`STRATA_KOLIBRI_GEMV_GS=1` (default off) routes a GEMV through a 1024-block grid-stride form when its row count is at
least `kGemvGsMinOut = 16384`, and leaves every other shape on the block-per-row path. The threshold is the measured
crossover from §T2, not a guess: at `n_out = 128000` the capped grid is 10.3 % faster, at `n_out = 2560` it is 29 %
slower. The two grids share their row body (`row_dot_f16`, `row_dot_q<T>`), so the per-row arithmetic cannot drift
between them; that is what makes the pair bitwise rather than merely equivalent.

Gates, R9700 (gfx1201), box idle, with `KOLIBRI_CM`/`KOLIBRI_GATE_HASH`/`KOLIBRI_GATE_SMOKE` pointing at the in-repo
gfx1201 expectation files:

| gate | knob off (shipped) | knob on |
|---|---|---|
| `oracle.sh check` | BITWISE PASS | BITWISE PASS |
| `pf_gate.sh` | (not re-run; shipped path is the pre-change code) | **GATE PASS**, all 6 stages |
| two-run ids | identical (same oracle) | identical |
| `ctest` | 85/89 (same 4 box-scoped) | not run (no behaviour change to gate) |
| decode, ctx 131072, standing arm | 15.90 / 15.87 tok/s (§T1) | **15.78 / 15.98 tok/s** |

The decode delta is **zero within noise** (±0.6 % across two runs either way), and the reason is structural, not
statistical: **the capped arm never fires on this model.** The startup note it prints when it launches
(`STRATA_KOLIBRI_GEMV_GS: grid-stride GEMV, N row(s) over 1024 blocks`) does not appear in any run. Two facts explain
that: a quantized weight never reaches `k_gemv` at all - `gemv` sends Q4_K/Q6_K/Q8_0 to `native_mmvq` first - and the
F32/F16/BF16 dense shapes on Kolibri-1 are 2560 rows, far below the threshold. The one 128000-row GEMV in the model is
the LM head, and its weight is Q6_K, so it takes the native path (measured at 0.52 ms/token in §T1).

So Task 3 is landed as an engine capability and collects **nothing** on this box: the +10.3 % it was built for is
behind a shape this GGUF does not route through this kernel. It stays opt-in, and Task 14 must not adopt it as a
default - there is no measurement on this card that would justify one.

## T4 (2026-10-28): packed Q4_K/Q6_K deq32, its parity test - and where the win actually lives

`STRATA_KOLIBRI_PACKED_DEQ=1` (default off) switches `deq32<T_Q4_K>` and `deq32<T_Q6_K>` to a packed form: 32-bit
loads, the nibble selected by a hoisted uniform flag instead of a per-element branch, the same float expression per
element. `g_packed_deq` is mirrored from the environment at startup the way `g_attn_coalesced` and `g_rope_f64` are.

**The test caught a real bug before it shipped.** The first Q6_K packed form conflated two independent facts about
`r = j & 3`: which half of `ql` to read (`r` odd takes `ql + 32`) and which nibble (`r >= 2` takes the high one). The
fixture reported **5 222 544 of 10 485 760 values differing**; after the fix, **0**. That is the argument for an
element-wise fixture of this size rather than a handful of blocks - the Q4_K form, written from the same pattern, was
correct first time and would have carried the Q6_K mistake past any spot check.

| arm | fixture | values compared | differ |
|---|---|---|---|
| `deq32 Q4_K` scalar vs packed | 5.9 MB (4096 rows x 2560) | 10 485 760 | 0 |
| `deq32 Q6_K` scalar vs packed | 8.6 MB (4096 rows x 2560) | 10 485 760 | 0 |

Both fixtures are deterministic (valid fp16 scales, no NaN/Inf), the comparison is on the float bits, and
`grid_shape_bench --selftest` fails the ctest if either count is non-zero.

Gates, R9700 (gfx1201) with the in-repo gfx1201 expectation files:

| gate | knob off | knob on |
|---|---|---|
| `oracle.sh check` | BITWISE PASS | BITWISE PASS |
| `pf_gate.sh` | - | **GATE PASS** (6 stages) |
| two-run ids, standing arm | (§T1) identical | identical |
| `ctest` | 85/89 (same four box-scoped) | - |
| prefill, 511 tokens (last `PFchunk`, §P0) | 4.71 / 4.63 s | 4.7 s (see §P0: the earlier sums mixed cumulative timers) |
| decode, ctx 131072 | 15.90 / 15.87 tok/s | 15.50 / 16.10 tok/s |

**The measured delta is zero, and the reason is structural.** `deq32<T_Q4_K>`/`<T_Q6_K>` are instantiated only by
`k_row`, `k_row_p` and `k_row_multi` - the **token-embedding gather**, one row per token in decode and one row per
prompt token in prefill. A 511-token prefill dequantizes 511 x 10 KB of embedding rows, which is a few hundred
microseconds of a 7.2 s chunk; the packed form shaves part of that and the wall cannot see it. Every other quantized
dense weight in this engine goes to `native_mmvq` before `deq32` is ever reached, which §T1 measured directly: the
`native_q4_k_mmvq` kernel is **5.8 ms of the 16.1 ms/token of GPU time**, 275 launches per token.

So the +21 % the micro-bench measures on a Q4_K GEMV (§T2) is real, and it is *not reachable from this file*. The
kernel that would collect it is `src/kernels/cuda/native_mmvq.cu` - a shared kernel with the main engine's users, not
a kolibri kernel, and outside Task 4's file list. Recorded as the plan's next lever rather than silently banked here.

## T9 (2026-10-28): the hipBLASLt table is inert on this engine, at every reachable config

**Version and file match, as the plan predicted.** `/opt/rocm/include/hipblaslt/hipblaslt-version.h` says hipBLASLt
1.2.2 → `100202`, which is exactly `tools/hip/gfx1201-hipblaslt-100202.txt`'s header
(`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100202`, 24 rows). The engine confirms it at startup:
`prefill gemm: hipBLASLt tuning enabled (24 rows, gfx1201, version 100202)`.

**A/B, three configurations, two runs each**, same 512-id prompt, ctx 8192, `--prefill --cpu-threads 8`,
`STRATA_HIPBLASLT_TUNING=tools/hip/gfx1201-hipblaslt-100202.txt` against `STRATA_HIPBLASLT_TABLE=off`:

| config | table off (chunk timers) | table on | wall off / on |
|---|---|---|---|
| shipped (`EXPERT_F16=0`) | 4.62 / 4.68 s (last `PFchunk`) | 4.69 / 4.67 s | 19.63/19.73 vs 19.68/19.68 s |
| `EXPERT_F16=24` | 4916+8307, 5110+8601 ms | 4994+8391, 5129+8632 ms | 23.82/23.50 vs 23.56/23.83 s |
| `DENSE_F16=1` | 2520+4582, 2528+4591 ms | 2568+4636, 2545+4609 ms | 19.72/19.74 vs 19.71/19.74 s |

Flat within run-to-run spread in all three, and **the resolver never runs**: not one `Lt solution=`, `Lt fallback`
or `no calibration for` line appears in any table arm, at any of the three configurations. The table loads and is
then never consulted.

Why, from the code: the tuning table is read only by `resolve_hipblaslt_algo`, reached from exactly two entry points
- `hipblaslt_solution_for` (`src/prefill/gemm.cu:386`) and `try_hipblaslt` (`:396`) - and both are called only from
`Gemm::bf16` (`:931`, `:1080`) and `Gemm::f16` (`:1197`), the **single**-product forms. `Gemm::f16_batched`, the only
prefill GEMM form Kolibri reaches for its resident experts, does not consult the table at all. The engine's other
prefill GEMMs are `Gemm::native` (the quantized `native_mmvq` chain), which is not hipBLAS at all. So:

- shipped config: no `Gemm::f16`/`bf16` call exists on this model ('s attention and shexp weights are quantized),
- `EXPERT_F16=24`: `f16_batched` runs, and it is not table-aware,
- `DENSE_F16=1`: the 4.85 GiB fp16 dense cache loads, and the attention projections still take the quantized path,
  so `Gemm::f16` is not entered.

**Verdict: no win is available here, and none is lost either.** The table stays exactly as shipped - this task changed
no code - and Task 14 must not wire `STRATA_HIPBLASLT_TUNING` as a default for this model: there is no shape on
Kolibri-1 that would read it. The ≈2x prefill-dense precedent in the plan is from a stack (ROCm 7.2.4 / hipBLASLt 1.5)
and a shape set whose GEMMs go through `Gemm::f16`; this engine's do not.

**Not run:** the pooled-KL 0.00162 re-measurement. It exists to catch numerics drift from an *algorithm swap*, and
this task swapped nothing - the shipped table is byte-identical before and after, and `pf_gate.sh` on the same binary
(§T4) already reports GATE PASS with its oracle stage BITWISE. Recorded as unnecessary rather than skipped.

## T5 (2026-10-28): the LM-head fp16 cache is dropped before it is built - the measurements say it is a loss

Task 5 proposed caching `output.weight` in fp16 (128000 x 2560, 656 MB) and computing the logits through the capped
GEMV instead of the shipped path. Two measurements settle it without spending a run on an arm that is known to lose.

**What the shipped path costs.** `rocprofv3 --kernel-trace` over a decode (§T1): the LM head is one
`native_q6_k_mmvq` per token, the last dispatch of every token - **519.8-520.3 µs** in steady state (697.2 µs on the
first token, the warm-up datum for T15). Against the standing arm's **62.9 ms/token**, that is **0.83 % of the decode
wall**.

**What the proposal would cost.** The micro-bench (§T2) ran exactly this data path at exactly this shape: a
128000 x 2560 F16 GEMV is **2.120 ms** block-per-row and **1.902 ms** grid-stride. The reason is bytes: fp16 reads
**655 MB** per token where Q6_K reads **269 MB**, and this card moves that shape at 309-345 GB/s (fp16) against ~517
GB/s effective for the Q6_K mmvq. Substituting either form into the wall:

| arm | LM head ms/token | decode wall | decode rate | vs shipped |
|---|---|---|---|---|
| shipped Q6_K mmvq | 0.520 | 62.9 ms | 15.90 tok/s | - |
| fp16, block-per-row | 2.120 | 64.5 ms | 15.50 tok/s | **-2.5 %** |
| fp16, grid-stride | 1.902 | 64.3 ms | 15.56 tok/s | **-2.2 %** |

It also asks for 656 MB of the 20.56 GiB free (3.2 %) to make the thing it touches slower.

**Verdict: dropped.** The knob would trade 2.2-2.5 % of decode for a 4x slower LM head in a leg that is under one
percent of the wall, and the only thing it could win is a numerics argument - the shipped Q6_K path is already the
arithmetic the oracle pins, and nothing measured here suggests an fp16 dequant of the same values would clear the
0.78x KL floor by enough to justify a default. Task 14 has nothing to adopt from it.

**Shipped path untouched:** no code changed, so the gate set is the one already recorded on this binary - oracle
BITWISE PASS and `pf_gate.sh` GATE PASS (§T4).

## T6 (2026-10-28): the slot path's leg is named - 3 launches per pick, not the kernel

Arm: `--expert-slots auto --expert-slots-hot --vram-reserve-mib 1024`, ctx 131072, `--prefill --cpu-threads 8`,
`EXPERT_F16=0`, 512-id prompt. Box idle, one arm per engine load.

**The fit, and a warning worth keeping.** `expert cache fit: 25652 MiB free after dense+KV+scratch -> 214 slots/layer
(24.02 GiB), 1024 MiB kept for the runtime, 33 MiB left`, then `1002 MiB free is below the 1024 MiB reserve: a long
prompt can run the card out`. 214 slots/layer here, against the 135/layer of the earlier arm - the count moves with
ctx, which is another reason the earlier number was not comparable.

**The served hit rate, finally recorded** (the plan's own complaint): `CNT run tok=64 | slot hits=18544 miss=656
(96.6% hit) expert PCIe=24.4 MB/tok`, streamed `290.6 MB/tok` against the pooled path's `714.2 MB/tok`, pool work
`9.65 ms/tok` (30.1 GB/s) against `22.28 ms/tok`.

**The phase split still cannot be read.** `CNT decode phases: pre=… d2h=… pool=… post=…` does not print even on this
path: `dec_layers` is incremented in exactly one place (`kolibri_engine.cu:2388`, the grouped/native branch), so a
slots run reports 0 and the line is suppressed. The legs have to come from the pool's own counters and the profiler,
which is what the two tables below are.

Timed arms in the same session (wall, `/usr/bin/time`):

| arm | n=4 | n=257 | decode | vs standing arm |
|---|---|---|---|---|
| standing (§T1) | 19.57 / 19.63 s | 35.49 / 35.57 s | 62.9 ms/token = 15.90 tok/s | - |
| slots, 214/layer | 22.38 s | 40.20 s | **70.4 ms/token = 14.21 tok/s** | **-10.6 %** |

Prefill pays for the promotion too: `hot slots [after prompt]: 2048 expert(s) newly resident after 153300 picks
(mean gain 30.02+), 4680 MiB over PCIe in 382.7 ms`, which is most of the n=4 difference (22.38 against 19.57 s).
During decode the hot policy keeps re-selecting: `96 expert(s) … 218 MiB over PCIe in 17.9 ms` per round.

`rocprof --stats` over 32 decode tokens of the same arm - **and this is the verdict**:

| | shipped (§T1) | slots arm | delta |
|---|---|---|---|
| launches per token | 1 347 | **2 511** | +1 164 |
| GPU time per token | 16.1 ms | **26.0 ms** | +9.9 |
| of which the slot GEMV | - | 875 launches, 15.6 ms | new |
| `native_quantize_q8_1` | 351 | 501 | +150 |
| copies | 103 | 161 | +58 |
| pool leg (CPU) | 22.3 ms | 9.65 ms | -12.6 |
| wall | 62.9 ms | 70.4 ms | +7.5 |

**Named leg: the slot path's own GEMV, at three launches per pick.** 875 `q4_K mmvq` launches per token = 50 layers x
~5.8 picks x 3 (gate, up, down), 15.6 ms of new GPU time per token, plus 1 164 new launches of issue overhead at
3.69 us each (~4.3 ms) and +150 quantise launches. The streamed leg gave up 12.6 ms/token and the GPU side took back
9.9 ms of kernel time and ~4.3 ms of launch issue, so the arm loses 7.5 ms/token net - **-10.6 % decode**, and
**+14 % prefill** for the 4.7 GB prompt-end promotion.

**Verdict recorded, no default moved** (`--expert-slots` stays an opt-in CLI flag; the launcher still ships
`EXPERT_F16=0` and no slots). The plan's candidate (a) - "3 GEMV launches per pick at T=1" - is the measured answer;
(b) pointer/index marshalling and (c) VRAM pressure were not needed to explain the sign. Arithmetic for the follow-on
tasks, using T2's 3.69 us per launch: batching the picks into one launch per layer would remove ~850 launches/token,
~3.1 ms, and leave 67.3 ms/token = 14.86 tok/s - **still 6.5 % below the shipped arm**. That is the ceiling for any
launch-count fix, which is why T7 and T8 cannot flip the sign by rerouting or splitting: the work itself (15.6 ms of
3-launch-per-pick GEMV) has to shrink, and that is T11's fused kernel.

## T7 (2026-10-28): not attempted, on T6's numbers - the reroute cannot change the launch count

Task 7 would reroute `DEC_RES`'s resident-pick hits from hipBLAS-at-T=1 to the in-tree slot GEMV. T6 has now measured
what that GEMV costs on this card, and it decides the question without another arm:

- the slot path already *is* that GEMV (`CNT branches per tok: … slots-path 50.0`), and it serves 96.6 % of picks
  through 875 `q4_K mmvq` launches per token - three per pick, gate/up/down - for 15.6 ms of GPU time;
- the reroute changes *which* kernel serves a hit, not how many launches there are. Its best case is therefore T6's
  own arm (14.21 tok/s) or, with the picks batched (T2's 3.69 us/launch), 14.86 tok/s. Both are **below the shipped
  15.90 tok/s**;
- and it is strictly worse than T6's arm on one axis that T6 did not pay: `DEC_RES` needs the fp16 cache resident
  (`EXPERT_F16>0`), which the launcher no longer ships, and the recorded `DEC_RES` arm was 11x slower before the
  cache even helped.

So the honest verdict is a non-attempt with its arithmetic shown, not an arm that lands between a losing baseline and
a losing ceiling. What would change the answer is T11: the 15.6 ms is 875 three-launch picks, and a fused kernel that
computes gate/up/down for a pick in one pass, or one launch per layer, is the only shape of change that moves it.

No gate was run for this task because nothing was built: the shipped path is untouched (oracle BITWISE and `pf_gate`
GATE PASS stand from §T4), and no pf_gate expectation was recorded or overwritten.

## T8 (2026-10-28): not run - its own premise is now measured negative

Task 8 adds `--miss-fetch-pcie F`: fetch a fraction of a layer's missing expert blobs over PCIe and compute them on
the GPU through the per-pick GEMV instead of the pool. Its gate was "run only after Task 6's counters name the leg",
and §T6 named it: **the per-pick GEMV, three launches each, 875 launches and 15.6 ms of GPU time per token.**

Two consequences, both from measured numbers rather than assumption:

1. The path T8 would fetch *into* is the path that already costs 15.6 ms/token. Adding fetched picks to it adds that
   cost, so the arm starts 15.6 ms behind on the GPU leg before it saves anything on the pool leg - and the pool leg
   is the one T6 already cut from 22.3 to 9.65 ms/token.
2. The launch-count fix its own note demands (one launch per layer instead of three per pick) has a ceiling computed
   in §T6: 67.3 ms/token = 14.86 tok/s, **6.5 % below** the shipped 15.90. Batching the picks is necessary for T8 to
   be even coherent, and it is still not sufficient to beat streaming.

The main engine's +9.9 % to +25.6 % precedent for `--pcie-frac 0.15` comes from a decode whose expert arithmetic runs
on the GPU and whose bottleneck is the PCIe read. On this box the arithmetic is on the CPU pool at 30-32 GB/s and the
GPU side is launch-bound, so the same knob would move work from a leg with spare capacity to the leg that has none.

**Not run: measured premise, recorded before an arm was spent.** Nothing was built, so no gate is owed; the shipped
path is untouched (oracle BITWISE, `pf_gate` GATE PASS, §T4).

**What the two non-attempts (T7, T8) leave standing**, in the plan's own order of levers: the fused expert kernel
(T11) for the 15.6 ms of per-pick work, and the attention path (T12) for the 100K decode, where §T1 measured attention
at 2.55 ms of a 16.1 ms GPU token - a smaller absolute leg than the experts on this model.

## T11 (2026-10-28): feasibility established, kernel not built - and the headroom is now a number

Task 11 is a stretch item: a fused Q4_K -> WMMA expert kernel for gfx12. Before writing one, the question that decides
whether a session should go into it is whether the instruction works on this card and what it pays. Both are measured.

**The gfx12 WMMA intrinsic works on the R9700.** A standalone probe using the in-tree wrapper's exact form
(`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`, `src/dsv41/ds4_iq_hip.inc:26`), 128 blocks x 4 waves x 4 independent
accumulator chains x 20 000 iterations, `hipcc --offload-arch=gfx1201 -O3`:

```
gfx12 WMMA 16x16x16 f16: 3.033 ms per launch -> 110.64 TFLOPS (fp16 in, fp32 accumulate)
last error: no error
```

**What that means against today's numbers.** The per-pick GEMV §T6 measured does 6 picks x 3 matrices x 2 x 512 x 2560
x 50 layers = **2.36 GFLOP per token** in **15.6 ms**, i.e. **151 GFLOP/s** - 0.14 % of what the matrix cores deliver.
The leg is not arithmetic-bound and never was; it is 875 three-launch picks. Its floor is bytes: 6 picks x 2.21 MB x
50 layers = **663 MB per token** of VRAM reads, which at the ~600 GB/s this card streams is **~1.1 ms/token**.

So a fused kernel that does gate/up/down for a pick in one pass has a floor of about 1.1 ms/token against today's
15.6, and if everything else stayed put the decode wall would go from 62.9 ms to ~48 ms = **~20.7 tok/s**. That is the
plan's 20-24 tok/s expectation, now grounded in two measurements instead of an analogy.

**Not built this session, deliberately.** A correct fused Q4_K->WMMA expert kernel needs the nibble unpack into WMMA
fragments, a parity arm against the staged fp16 path (`q4k_grouped_vram_parity --bench`), its own `pf_gate` expectation
file plus a KL figure, and two-run identity - a session of work, not an hour. Shipping it half-verified would break
this plan's own contract, and the numbers above say the session is worth spending rather than that it can be shortened.
Recorded as the plan's top lever for the next one, with the probe reproducible in five minutes.

No code changed, so no gate is owed: the shipped path is untouched (oracle BITWISE, `pf_gate` GATE PASS, §T4).

## T12 (2026-10-28): feasibility established, kernel not built - and why the session is worth spending

Task 12 is the WMMA attention arm: replace the `#if defined(__HIPCC__) -> MMA off` gate (`kolibri_engine.cu:969`)
with a gfx12 path and give `k_attn_flash_mma` an AMD sibling, behind `STRATA_KOLIBRI_ATTN_WMMA`. As with T11 the
instructive step this session was feasibility, and the two measurements that bound the payoff are already in hand:

- **At short ctx, attention is small**: §T1's profile has `k_attn` at 50 calls/token, 50.9 us each = **2.55 ms of the
  16.1 ms GPU token**, i.e. **4 % of the 62.9 ms wall**. A WMMA attention kernel cannot matter here even if it were
  free.
- **At long ctx it is the wall**: the ledger's long-context row is **6.7 tok/s at 100K = 149 ms/token**, and the plan's
  own reasoning puts the attention GEMM class at the centre of that. This is the only decode number on this box where
  a 5-7x kernel-class change moves the total.

**The instruction is real and fast here** (§T11's probe): 110.64 TFLOPS fp16-in/fp32-accumulate on gfx1201, using the
in-tree gfx12 wrapper. The existing flash path's structure (scores + online softmax + PV in one launch per layer) is
already the right shape; what is missing is the AMD fragment layout and its parity evidence.

**Not built this session.** The work is: the gfx12 WMMA score/PV path, parity at 512 tokens against the shipped
tiled path (the CUDA `mma` precedent is KL-neutral there), two-run identity, then the long-context validation - which
by the plan's own budget is **one 99K arm per session (16+ min per run)**, and a 99K comparison needs two runs, plus
the re-measured rate. That is a session of its own, and the correctness contract does not survive shortening it.

Note for whoever takes it: `STRATA_KOLIBRI_ATTN_MMA` is compiled out on HIP and `attn_mma_on()` exists only in the CUDA
branch, so on this card that flag is silently inert. `ATTN_WMMA` is the AMD knob; do not reuse `ATTN_MMA` for it.

No code changed, so no gate is owed: shipped path untouched (oracle BITWISE, `pf_gate` GATE PASS, §T4).

## T14 (2026-10-28): the defaults block is re-derived, and nothing moves

The launcher's `run-kolibri1.sh` comment block now carries a measured reason per default instead of a T4-era one
(see the diff), and `docs/KOLIBRI1.md` gained the attribution in plain words. Every default stays:

| knob | value | decision, and the measurement behind it |
|---|---|---|
| `EXPERT_F16` | 0 | already re-derived (§Speed table): cache-on costs a third of the prompt read, decode unchanged |
| `ATTN_MB` | 512 | 1024 wins only at 99-128K, and draws on the same VRAM as the expert slots; documented as `ATTN_MB=1024` for long-prompt work, not as a default |
| `RES_TOP` | 56 | inert at `EXPERT_F16=0`; it caps the prefill promotion, which needs the fp16 cache |
| `ROUTER_TILE/TK/TJ` | T4 values | not re-derived on this card, and unmeasured here - kept with the note that says so |
| `CTX` | 131072 | a bigger context is VRAM the slots and attention scratch want |
| `THREADS` | physical cores (8) | 24 lanes halve decode (15.8 -> 7.6 tok/s) |
| `--grouped` / graphs | untouched | graphs are off by construction whenever `--cpu-threads` is set |

Two opt-in knobs were measured and **not** wired: `STRATA_HIPBLASLT_TUNING` (inert, §T9) and `STRATA_KOLIBRI_GEMV_GS`
(+10.3 % at one shape, -29 % at the shapes this model runs, and it never fires here, §T3).

**End-to-end check of the launcher path after the edit** (this is the user-facing route, so it was smoke-tested rather
than assumed): `./run-kolibri1.sh --detach` came up on `127.0.0.1:8080`, the engine logged its pool warning as usual,
and one chat completion with a tool came back as a proper call:

```
tool_calls: [{"id":"call_77c7e595…","type":"function","function":{"name":"emit","arguments":"{\"a\": 7, \"b\": 42}"}}]
content: null     finish_reason: tool_calls
[server.log] this model's chat template asks for tool calls as JSON inside <tool_call>: the parser reads that form
[server.log] tool call complete: 54 of max 300 tokens, 17.9 tok/s, 5 s
```

One limit worth writing down, seen while testing it: when `max_tokens` truncates *inside* the block the reply is raw
text with `finish_reason: length` - there is no complete JSON to recover. A 64-token cap did exactly that; 300 tokens
produced the call above. Not a defect in the recovery, but a caller that caps at 64 tokens on this model will see the
raw form.

## T15 (2026-10-28): the warm-up hook lands opt-in, and the first-call cost is not code load

The hook: at load, run the LM head's own kernel (`native_mmvq` at the head's real `n_out`) once into a throwaway
buffer, behind `STRATA_KOLIBRI_WARM=1`. Measured with `rocprofv3 --kernel-trace` over an 8-token decode, three runs,
the first LM-head dispatch of each (the same probe that measured the steady state in §T1):

| arm | first LM-head call | steady | warm cost at load |
|---|---|---|---|
| shipped (no warm-up) | **697.8 µs** | 519.8-520.3 µs | - |
| warm at `n_out = 1` | 696.2 µs | 520.3-520.8 µs | 1 launch, 4 B |
| warm at `n_out = 128000` | **618.4 µs** | 519.8-520.8 µs | 2 948.3 µs + 512 KB |

Two findings, both of which change what the hook should be:

- **It is not code load.** A warm-up that launches the same kernel at one row removes nothing (696.2 against 697.8 µs):
  the kernel's code is already resident after the first launch. The hypothesis the task was built on is refuted.
- **The shape is what matters.** Warming at the head's own 128000 rows recovers 79.4 µs - 45 % of the 178 µs excess -
  for 2.9 ms of extra load and a 512 KB buffer. What is left (618.4 against 520) is the first touch of the real logits
  buffer and the weight's cold pages, which a warm-up cannot pre-pay: the buffers it would have to touch are the ones
  the first request allocates.

**Shipped default: off.** 79 µs once, against 2.9 ms added to every start, on a first token that is 62.9 ms and a
prefill that is seconds - not a trade this launcher should make for everyone, and not one it makes silently for
anyone. `STRATA_KOLIBRI_WARM=1` is there for a card or a kernel where the balance differs, and it changes no number:
`oracle.sh check` is BITWISE PASS both unset and with the knob on.

Gates after the change: oracle BITWISE PASS (knob unset, and `WARM=1`), `ctest` 85/89 with the same four box-scoped
failures.

## T10 (2026-10-28): the `PF_PARTIAL` arm, measured as specified - a slower prompt, unchanged decode

Arm exactly as the plan writes it: `STRATA_KOLIBRI_PF_STAGE=1 STRATA_KOLIBRI_PF_PARTIAL=1
STRATA_KOLIBRI_EXPERT_F16=28`, ctx 8192 (the ledger's short shape), 512-id prompt, `--prefill --cpu-threads 8`, box
idle, against §T1's baseline on the same shape and the same session protocol:

| quantity | baseline (ctx 8192) | `PF_STAGE` + `PF_PARTIAL` + `F16=28` |
|---|---|---|
| wall, n=4 / n=257 | 19.57/19.63 s / 35.49/35.57 s | 23.66 s / 39.60 s |
| decode | 62.9 ms/token = 15.90 tok/s | 63.0 ms/token = **15.87 tok/s** |
| prefill, 511 tokens (last `PFchunk`) | 4.62 s | **8.12 s** |

Decode is unchanged (the T4 record's "must never go down" invariant holds: 15.87 against 15.90 is inside the
run-to-run spread), and the **prompt is 81 % slower**. The 4.7e7-byte staging traffic that PF_STAGE adds is the
explanation worth testing next time, but it is not this task's question: as specified, the arm loses.

One correction to how this arm was described earlier: the earlier sample at ctx 131072 (62.0/73.3 tok/s prompt against
57.6/70.1 with the cache) set `PF_PARTIAL` and `EXPERT_F16` but **not** `PF_STAGE`, and it read "inside the noise". The
two arms are not comparable, and the difference between them is exactly the knob that now shows the regression - so
the earlier "inside the noise" note must not be quoted for `PF_STAGE`.

**Decision: not adopted.** `run-kolibri1.sh` keeps no staging knobs, and Task 14's re-derivation has nothing to take
from this arm. KL on the pooled path was moot: the arm does not win on speed, so it does not get to ask for a numerics
budget.

## P0 (2026-10-28): the prefill decomposition - and a correction to every prefill number this ledger quoted

**The correction first, because it invalidates a baseline the speed plan was built on.** `PFchunk` prints
`ms_since(tp0)` - **cumulative** time since the prompt started (`kolibri_engine.cu:5248`). Every prefill figure in this
ledger that read "2543+4623 = 7198 ms" (or 7166, 7336, 7247, 7185, 12985) added cumulative timers together. The real
511-token prefill is the **last** number, not the sum. Measured now, twice, same box and config:

| arm (ctx 16384, `--prefill --cpu-threads 8`, `EXPERT_F16=0`) | wall | prefill (last `PFchunk`) | rate | pool | pool share |
|---|---|---|---|---|---|
| 511 tokens, 2 chunks of 256 | 20.72 s (load 14.4) | **4.710 s** | **108.6 tok/s** | 3.728 s | 79.2 % |
| 8 192 tokens, 32 chunks of 256 | 85.08 s (load 13.7) | **69.66 s** | **117.6 tok/s** | 61.09 s | **87.7 %** |

Run 1 of the 511-token arm took 175.95 s because the 45 GB model was still coming off disk (load line 169.5 s); it is
recorded here so nobody mistakes it for a rate. The 8K arm reproduced to 0.5 % (69.33 / 69.66 s) and its pool to 0.5 %
(60.81 / 61.09 s).

**So the speed plan's "71 tok/s prefill" was wrong by 1.5x, and its "50.8 % is the pool" was wrong the other way: the
pool is 88 % of a long prompt's prefill.** The earlier percentage came from dividing the pool's 3.66 s by the summed
7.20 s. Corrected, at 511 tokens the pool is 79 %.

**The decomposition P0 was built to answer:**

| per token | 511 tok | 8 192 tok | what it is |
|---|---|---|---|
| prefill | 9.22 ms | 8.50 ms | wall between the first and last `PFchunk` |
| pool | 7.30 ms | 7.46 ms | `CNT prefill pool`, 7.0-7.5 GB/s at 316-323 GFLOP/s of a 226 GFLOP/s roof |
| **non-pool** | **1.92 ms** | **1.05 ms** | GPU kernels (~0.5 ms, §T1's rocprof) + host marshalling |

**The 6.40 ms/token "host residual" does not exist.** It was the arithmetic of a wrong prefill wall (7.20 s) minus a
pool measured over a different run. The real residual is **~1 ms/token**, and it is dominated by the GPU kernels the
prompt needs anyway, not by host serialisation. There is no fourth leg to chase.

**What this changes in the plan.** The pool *is* the prefill - 88 % at 8K, 426.69 GB streamed for 8 192 tokens
(52 MB/token) at 7.0 GB/s - so P3 (residency: the bytes stop being streamed) and P4 (MMQ: the arithmetic stops running
on eight cores) are the whole of the opportunity, and P2 (bigger chunks) is a minor term: chunk 256 against 512 at 8K
moved the wall by -0.6 % in the one comparison that differed, and both arms here ran 256-token chunks because
`g_pf_total` is unset on the CLI driver, so the K28 prefix-length rule falls back to the position rule.

Projection with the pool gone, from the measured non-pool 8.57 s and the measured 19 325 GFLOP of expert work:

| expert FLOPs on the GPU at | 8K prefill | 8K rate | TTFT |
|---|---|---|---|
| 3 TFLOPS | 15.0 s | 546 tok/s | 15 s |
| 5 TFLOPS | 12.4 s | 659 tok/s | 12 s |
| 10 TFLOPS | 10.5 s | 780 tok/s | 11 s |
| 20 TFLOPS | 9.5 s | 859 tok/s | 10 s |
| **today** | **69.7 s** | **118 tok/s** | **70 s** |

That is 4.6-7.3x, not the 2-3x the plan projected from the (wrong) 71 tok/s baseline, and it lands Kolibri within
2-3x of Swift IQ3_S's 1 865 tok/s at 7.2K instead of 18x behind it.

## P1 (2026-10-28): `prefill-standing`

Two commands, cited verbatim; every later row is held against them. Box idle, 91 GiB free, `HIP_VISIBLE_DEVICES=0`,
R9700 gfx1201, `EXPERT_F16=0`, `ATTN_MB=512`, `RES_TOP=56`, `--cpu-threads 8`, `--prefill --counters`.

| row | command | wall | prefill (last `PFchunk`) | rate |
|---|---|---|---|---|
| 511 tokens, ctx 8192 | `--tokens-file tools/kolibri/oracle_ids512.txt --n 4 --ctx 8192` | 20.20 / 20.20 s | 4.71 s | **108.6 tok/s** |
| 511 tokens, ctx 16384 | same, `--ctx 16384` | 20.72 s | 4.710 s | 108.6 tok/s |
| 8 192 tokens, ctx 16384 | `--tokens-file /tmp/ids8k.txt --n 4 --ctx 16384` | 84.69 / 85.08 s | 69.33 / 69.66 s | **117.6-118.2 tok/s** |

`/tmp/ids8k.txt` is `oracle_ids512.txt` repeated 16 times (8 192 ids, 36 144 bytes) — a timing fixture, not a text: the
content does not affect the prefill split, and routing-dependent effects would show up as run-to-run spread, which was
0.5 %.

## P2 (2026-10-28): the chunk sweep - bigger chunks are worth ~2.3x, and they are blocked by the SWA ring

`STRATA_KOLIBRI_PF_CHUNK` caps the chunk, so the sweep ran without code changes (8 192 tokens, ctx 16384, counters):

| cap | chunks | prefill | pool | non-pool | rate |
|---|---|---|---|---|---|
| 64 | 128 | 83.349 s | 65.891 s | 17.46 s | 98.3 tok/s |
| 128 | 64 | 75.531 s | 63.484 s | 12.05 s | 108.5 tok/s |
| 256 | 32 | 70.863 s | 61.070 s | 9.79 s | 115.6 tok/s |
| 512 | **32** | 69.562 s | 60.930 s | 8.63 s | 117.8 tok/s |

**`512` is not a 512-token chunk.** With `g_pf_total` unset on the CLI driver the K28 prefix-length rule falls back to
the position rule, so every chunk at an 8K prefix is 256 tokens; the 512 arm is the 256 arm (69.56 against 70.86 s is
run-to-run spread). Chunks above 256 are **unreachable today** because `kTmax = 512` and the sliding-window invariant
refuses `kTmax >= n_swa` (513).

**What the sweep says:** each extra chunk-layer costs **143.6 ms** (51.7 pool + 91.9 non-pool). At 32 chunks that is
4.6 s of the 69.7 s; at 128 chunks it is 18.4 s. And larger chunks also dedup the experts: 256-token chunks touch 111
experts/layer (427 GB for the prompt), while a 2 048-token chunk would touch the full 384 (170 GB for the same
prompt). Projected at 2 048-token chunks, from those two measured terms:

| | 256-token chunks (today) | 2 048-token chunks (projected) |
|---|---|---|
| expert bytes, 8K prompt | 426.69 GB | 170 GB |
| pool | 61.09 s | 24.2 s |
| non-pool | 8.57 s | 6.1 s |
| prefill | 69.66 s | **30.3 s** |
| rate | 117.6 tok/s | **270 tok/s (+130 %)** |

**The blocker, and the two ways through.** The layer loop is chunk-outer, layer-inner, and the SWA layers cannot see
more than `n_swa - 1 = 512` tokens per pass (the ring would wrap mid-chunk). To get a 2 048-token expert batch the
loop has to change shape - either (a) **layer-outer over a wave of W chunks**: run attention for W chunks of 512
(each ring-safe), then one pool call covering W x 512 rows for that layer - which extends the `PF_PIPE` two-chunk
wavefront that already exists; or (b) grow the SWA ring to the chunk, which changes the sliding semantics' cache shape
and costs VRAM. (a) is the one that keeps the attention contract intact.

### P2 addendum (2026-10-28): the reachable range is +2.7 %, and the pool's rate falls as chunks grow

One real defect found and fixed on the way: the chunk rule keys on the prefix length and compares it with 8192, but
`n = ids.size() - 1` (the last id seeds generation), so an **8192-token request prefilled 8191 and fell on the short
side** - 32 chunks of 256 where 16 of 512 were intended. `g_pf_total` now takes `ids.size()` (`kolibri_engine.cu`), and
an 8 192-token prompt runs 512-token chunks. Gates: the 511-token oracle is unaffected (still below the threshold) and
the 8K ids are **bitwise identical** to the chunk-256 arm on both runs (`gen: 286 3515 310 262`), so chunk 256 against
512 changes no number - the K28 claim holds between those two sizes.

| 8 192 tokens, ctx 16384 | chunks | prefill | pool | expert bytes | rate |
|---|---|---|---|---|---|
| chunk 256 (before) | 32 | 69.33 / 69.66 s | 60.81 / 61.09 s | 426.69 GB (267 MB/chunk) | 117.6 tok/s |
| **chunk 512 (after the fix)** | 16 | **68.76 / 66.91 s** | 61.72 / 59.93 s | **279.82 GB (350 MB/chunk)** | **120.8 tok/s (+2.7 %)** |

**Why so little, and what it does to the P2 projection.** The bytes fell 34 %, and the pool time did not move at all,
because the pool's own throughput fell with them: **7.0 GB/s at 267 MB/chunk against 4.5 GB/s at 350 MB/chunk**. The pool
is not bandwidth-bound; it loses efficiency as a chunk touches more distinct experts. Applying that measured trend to
a 2 048-token chunk (170 GB at ~3.5-4 GB/s) gives a pool of 43-49 s, not the 24 s the earlier projection assumed - so the
reshape's realistic ceiling is **+25-30 %, not +130 %**.

**Verdict, per this task's kill criterion (+15 % at 8K): it fires.** The reachable range (256 -> 512) is +2.7 %, and
the measured rate trend caps the >512 reshape well below the criterion. The loop reshape is therefore **not** carried
out; the fix above is kept (it is a defect fix with a bitwise gate, and it is what makes 512-token chunks reachable at
all), and the effort goes to P3+P4, which attack the pool itself rather than the shape of its work.

One loose end recorded rather than explained: the chunk-128 arm alone produced different ids
(`gen: 10 127908 263 50` against `286 3515 310 262`), while chunk 64, 256 and 512 all matched. One arm out of four, not
reproduced; the plan's two-runs-per-arm rule is what caught it, and it is why a single id comparison cannot certify
chunk invariance.

## P4 (2026-10-28): `strata::prefill::mmq` is not in this build - it is one CMake option away, and the K-quants come with it

Before any wiring: the library the plan wants to call **does not exist in this build**. `ls build/*.a` has core, engine,
kernels, kernels_cpu, ri, prefill, qwen35, spec - **no `strata_mmq`**, and `build/build.ninja` never mentions
`STRATA_KQ_DSO`. So the grouped Q4_K expert product is not missing because it is slow; it is not compiled.

It is available, though, on this backend: `CMakeLists.txt:2400` opens

```cmake
  elseif(STRATA_PREFILL_MMQ)
    # Compile only the pinned GGML MMQ and activation-quantizer sources used by prefill. This uses
    # common.cuh's native HIP path without enabling a second GGML HIP backend.
    ...
    if(STRATA_MMQ_KQUANTS)
      # the same K-quant instances as the CUDA branch: on HIP they serve the dense GGUF projections
      list(APPEND _strata_mmq_types q4_k q5_k q5_1 q6_k)
    endif()
    foreach(_t IN LISTS _strata_mmq_types) ... mmq-instance-${_t}.cu
    set_source_files_properties(${_strata_mmq_sources} PROPERTIES LANGUAGE HIP)
    add_library(strata_mmq STATIC ${_strata_mmq_sources})
```

so `-DSTRATA_PREFILL_MMQ=ON` (with `STRATA_MMQ_KQUANTS=ON`, already the default) builds the Q4_K instances for HIP. A
runtime switch is still needed on this backend: `kq_ok()` (`src/prefill/moe_mmq.cu:44`) returns `kq::available()` only
under `STRATA_KQ_DSO` (the CUDA library), so on HIP it falls back to **`STRATA_MMQ_KQ=1`** as an env opt-in.

What that means for this task, in order:

1. Build the library in a **second build directory** (`build-mmq`, same options plus `-DSTRATA_PREFILL_MMQ=ON`) so the
   working `build/` is not disturbed; the MMQ instance TUs are many and slow to compile.
2. There is no benchmark for Kolibri's shape on HIP: `mmq_expert_bench` is a CUDA-CMake target (links `CUDA::cudart`) and
   its cases are Qwen's shapes and NVFP4/Q8_0/IQ3_S formats. Kolibri needs **Q4_K at gate/up [512 x 2560] and down
   [2560 x 512]**, with the rows per expert a chunk actually produces: at 512-token chunks the counters give 1 535.8
   rows/layer-chunk over ~158 touched experts, i.e. **~10 rows per expert** - a skinny grouped GEMM, which is exactly
   where MMQ's advantage is thinnest.
3. So P4's kill criterion (300 GFLOP/s on the grouped product) has to be measured on that shape before any wiring, and
   the number to beat is the pool's **316-323 GFLOP/s at 7.0 GB/s**.

That is the next step in this session; no engine code has been touched for P4 yet.
