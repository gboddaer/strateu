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
