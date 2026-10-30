# Kolibri CUDA preview qualification

The preview targets AMD gfx1201/R9700 and CUDA, but each backend needs its own evidence. A CUDA build on an AMD
host proves compilation/linking only. T4 results do not qualify RTX 3090 or other cards.

## CUDA build fixes

Two regressions in the R9700 campaign were reproduced with CUDA before fixing them:

- `grid_shape_bench` directly uses HIP APIs, but its target was created in CUDA configurations too. Guard the
  entire diagnostic target with `STRATA_ENABLE_HIP`; CUDA configurations must not request the HIP language.
- CUDA translation units use C++17. The new decode graph used C++20 `std::atomic_ref`. Host control-word helpers
  now use lock-free compiler atomic intrinsics with acquire/release ordering (MSVC uses interlocked operations),
  without changing the CUDA language standard or casting mapped storage to a `std::atomic` object.

Regression checks are in `tools/kolibri/test_build_portability.py`, including a C++17 host publication test.

Locally observed: CUDA 13.1 (`nvcc` 13.1.115), GCC 14.2, CMake 3.31, `sm_75;sm_86`: `strata-kolibri`,
`kolibri_decode_rollover` and `kolibri_decode_wait_test` compile and link after both fixes. No NVIDIA GPU is
present on this host, so CUDA runtime, race behaviour, numerical parity and speed were not tested here.

CUDA 12.8.93 failed compiler identification on this host's glibc 2.41 (`sinpi`/`cospi` exception specifications),
before compiling project code. This is a separate toolkit/host compatibility limit, not a passing CUDA 12.8 build.
The isolated official toolchains were not patched. Use a toolkit supported by the build host and target driver.

## T4 build and full-logit gate

Use a fresh CUDA build directory, separate from HIP. Name the architecture explicitly: T4 is 75; RTX 3090 is 86.
Do not rely on the generic engine's default architecture. From the source checkout, with CUDA on `PATH`:

```sh
cmake -S . -B build-kolibri-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_DSV41=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build build-kolibri-cuda --target strata-kolibri kolibri_decode_rollover kolibri_decode_wait_test -j 4

CUDA_VISIBLE_DEVICES=0 STRATA_GEMM_DETERMINISTIC=1 STRATA_KOLIBRI_EXPERT_F16=0 \
  KOLIBRI_BIN="$PWD/build-kolibri-cuda/strata-kolibri" \
  KOLIBRI_MODEL=/path/to/Kolibri-1-Q4_K_M.gguf \
  KOLIBRI_PYTHON=python3 KOLIBRI_DEC_GATE_OUT=/tmp/kolibri-t4-decode-gate \
  bash tools/kolibri/decode_gate.sh check
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build-kolibri-cuda -R '^kolibri_decode_wait_test$' --output-on-failure
```

Run from a shell without unrelated `STRATA_KOLIBRI_*` overrides. Both comparator arms use deterministic
prefill for the gate; this is a test configuration, not a production-default change. The gate compares full
finite logits on aligned forced histories, real prefix reuse and six repeats, and checks replay markers.
An equal generated-id list alone is insufficient. Also qualify the 8191-id KV boundary and 11000-id attention
transition with forced full-logit comparisons before claiming long-context decode qualification.

The committed `oracle_gfx1201.txt` is AMD-specific. Do not overwrite it or use it as a T4 acceptance oracle.
Any T4-specific oracle must identify its model, reference build and configuration, and must be derived from an
independently qualified reference rather than recorded from a candidate just to make a gate pass.

## Operator-provided T4 report (not independently rerun here)

Reported source: `82a2d5d2` plus the T4 agent's local fixes for the same two build blockers above. Those local
control-word edits used volatile accesses and fences; they are not the identical implementation now in this tree.
The T4 must be re-tested on the integrated fix before it is release qualification.

Environment reported: Tesla T4 16 GB, driver 580.196.07, Linux 6.1.0-52, NUMA node 1, 24 CPU lanes, 512 prompt
ids, 128 generated tokens, ctx 8192. Baseline source `bdd3803a`.

| Configuration | Baseline decode | Candidate decode | Interpretation |
|---|---:|---:|---|
| CPU pool, no expert cache/slots | 29.2 / 29.3 / 29.3 tok/s | 30.5 / 30.3 / 30.5 / 30.5 tok/s | Reported ~4.1% improvement |
| Prior speed environment with cache and resident slots | 24.6 / 23.7 tok/s | 24.6 / 24.5 tok/s | Within reported spread; graph arms correctly disabled |

Prefill was bimodal in both builds. Seven CTest failures were reported reproduced on the baseline:
`attn_k_parity`, `pf_wmma_parity`, `ple_parity`, `pool_tasks_test`, `pool_cores_test`, `isa_guard_check`,
`cuda_spill_guard` (210 tests: 189 passed, 7 failed, 14 skipped/skipped groups).

The reported 16 generated ids matched across builds and repeats. The supplied report does not establish
bit-identical full logits. Raw logs, exact integrated revision, forced-history parity, longer contexts and a
CUDA release-package smoke test remain pending. No RTX 3090 runtime result is available.
