# Kolibri CUDA preview qualification

See [the preview guide](KOLIBRI_PREVIEW.md) for package installation and publication status.
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

Locally observed: CUDA 13.1 (`nvcc` 13.1.115), GCC 14.2, CMake 3.31, `sm_75;sm_86`: `strata`, `strata-kolibri`,
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

## T4 qualification (measured on the card, 2026-10-30)

Replaces the earlier operator-provided summary. Host: Tesla T4 16 GB, driver 580.196.07, kernel 6.1.0-52-amd64,
PCI `82:00.0` = `numa_node 1`, `numactl --cpunodebind=1 --membind=1`. Revision **`5bbe6c4e`**, built with the recipe
above including `STRATA_Q6K_EXPERTS=ON` (`build-kolibri-cuda`, Ninja, CUDA 12.8.93, arch 75): 0 errors.

The T4 agent's earlier local control-word edits were discarded in favour of this tree's helpers, after reading both:
the integrated version uses `__atomic_*` acquire/release with lock-freedom asserted, which is stronger than the
local volatile+fence variant. The patch is not part of this tree.

### Correctness

| gate | result |
|---|---|
| `decode_gate.sh check`, prompts 16 / 511 / 512 / 513 | **PASS** - 9 records x 128,000 values bitwise identical, max \|a-b\| 0.0 |
| prefix reuse (cold 512 ids, then +64) | **PASS** bitwise, headers included |
| six repeats, activation marker | **PASS**; marker present in the candidate and absent in every control run |
| **8191-id KV-wrap boundary**, forced full logits | **PASS** bitwise, max \|a-b\| 0.0 |
| **11,000-token attention transition**, forced full logits | **PASS** bitwise, max \|a-b\| 0.0 |
| `kolibri_decode_wait_test` | **PASS** |
| `ctest` (210 tests) | 189 passed, 7 failed, 14 skipped |
| `tools/kolibri/test_build_portability.py` | **2 passed** (needs pytest; run in a throwaway venv) |

The seven failures are the pre-existing set, reproduced identically on a build of `bdd3803a` on the same box:
`attn_k_parity`, `pf_wmma_parity`, `ple_parity`, `pool_tasks_test`, `pool_cores_test`, `isa_guard_check`,
`cuda_spill_guard`. Causes are environmental (missing fixture GGUFs, wide-ISA listings, sm_75 kernel expectations), not
this change. The 11,000-token sequence was the committed 8191-id fixture's ids cycled: synthetic, but identical on both
sides of every comparison.

Conclusion: **the decode arms are bitwise against the plain decode on this card, including the KV boundary and the
attention transition.** The generated-id match reported earlier is superseded by full-logit equality.

### What the fast-prefill knob costs on this card

Prefill here needs the fp16 expert cache (`STRATA_KOLIBRI_EXPERT_F16`), not the env members the R9700 campaign credited.
Measured with 512 prompt ids, 128 greedy tokens, ctx 8192, `--prefill --cpu-threads 24 --expert-slots 0`, two runs each,
box load 1.5-3.8:

| environment | pp-512 tok/s | tg-128 tok/s |
|---|---:|---:|
| engine defaults (no `STRATA_KOLIBRI_*`) | **51.6 / 53.6** | **30.5 / 30.5** |
| `PF_STAGE=1` | 52.2 / 51.9 | 30.6 / 30.6 |
| `PF_STAGE` + `ATTN_MB=256` + `ATTN_FLASH=1` + `ROUTER_TILE=1` + `ROUTER_TJ=16` | 50.8 | 30.6 |
| `EXPERT_F16=24` | 158.6 / 162.1 | 30.3 / 29.4 |
| `EXPERT_F16=24` + `PF_PARTIAL=1` | 181.5 / 182.0 | 30.2 / 30.2 (EOS at 105 of 128 tokens) |
| the whole 2026-10-27 "speed env" | 167.5 / 170.9 | 30.3 / 30.3 |

`PF_STAGE`, `ROUTER_*` and `ATTN_*` are inert at this shape; the cache is the whole lever. And the cache changes the
numbers, on the same forced history, against the same build's plain path (`EXPERT_F16=0`, `STRATA_GEMM_DETERMINISTIC=1`):

| comparison | mean KL | median | p99 | max \|dlogit\| | top-1 |
|---|---:|---:|---:|---:|---:|
| plain vs `EXPERT_F16=24` | 0.400 | 0.0010 | 3.24 | 11.08 | 9/9 |
| plain vs `EXPERT_F16=24 PF_PARTIAL=1` | 0.168 | 0.0005 | 1.26 | 5.47 | **7/9** |

Both are bitwise identical run to run. That establishes repeatability, not the cause: rounding/accumulation
changes and deterministic implementation defects still need to be distinguished. The reported mean KL is ~2 orders
of magnitude worse than the R9700's recorded `PF_PARTIAL` figure (answers-only mean KL 0.00133, top-1 0.9725), and an
11-logit worst case is too large to accept as "fp16 rounding" without an explanation. Under
[the accuracy policy](KOLIBRI_ACCURACY_POLICY.md) that is a separate numerical qualification, not a default change.

### The accuracy-first configuration on this card

Engine defaults, expert cache off, slots 0, decode arms on: **51.6 / 53.6 tok/s prefill** and **30.5 / 30.5 tok/s decode**,
bitwise identical to the plain path at 512, 8191 and 11,000 prompt tokens. The arms are worth ~+4.1 % decode (30.5 against
29.3 on the pool path, measured on the pre-pull build) and cost nothing numerically, so they stay on.

On this box `--cpu-threads 24` also makes the engine warn that 24 expert workers share 14 physical cores
(`Recommended: --pool-workers 13`). The 13-worker variant is unmeasured, including whether it stays bitwise.

### Not tested here

No RTX 3090 or other NVIDIA card (a CUDA build on one host does not qualify another); no KV-boundary or
attention-transition parity for the fp16-cache configurations (only plain and arms paths were dumped); no long-context
speed for any configuration; no CUDA release-package smoke test; no explanation yet for the 11-logit worst case.

### Oracles

The committed `oracle_gfx1201.txt` is AMD-specific and was not overwritten. On this T4 the plain path produces the
pooled variant `... 32 50 55 58 4037 262 7771`, which `docs/AMD_R9700_PORT.md` documents as the pooled signature
(14/16 ids, first divergence at the 11th). Both `bdd3803a` and `5bbe6c4e` produce it, so it is not a regression, and
`oracle.sh check` against the committed file reads as a diff on this box whatever the build. A T4 oracle is not recorded
yet; when it is, it must name the model, reference build and configuration, and be derived from an independent reference
rather than from a candidate.
