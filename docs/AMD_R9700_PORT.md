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
