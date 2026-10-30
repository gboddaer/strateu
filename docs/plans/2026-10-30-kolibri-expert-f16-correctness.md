# Kolibri `EXPERT_F16` correctness on CUDA (sm_75 / Tesla T4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Amended 2026-10-30, after checking every code claim against the tree:** Task 2's assertion is fp16 half-ulp (subnormals break any floored relative error), Tasks 2-4 build and run against the interfaces the tree actually exposes (`dequant_bf16_test --selftest`, `Gemm::f16_batched` via the `strata_prefill` library), Task 3's run replicates the decode gate's exact flags (`--prefill --cpu-threads 8 --n 8` — without them the cache is never consulted), and Task 4 gained a real-shape run and a slot-eviction case.

**Goal:** Decide whether the fp16 expert cache's 11-logit worst case on the Tesla T4 is a defect in the Turing path or expected fp16 loss, and either fix it or qualify it — so the 3.4x prefill gain stops being unusable.

**Architecture:** Four cheap instruments first, then one verdict experiment, then the decision. Each instrument is a committed, unit-tested tool so the numbers can be reproduced by someone else. The verdict experiment compares one expert's GEMM three ways — a float64 host reference, an f32 control, and the fp16 cache path the engine actually runs (`Gemm::f16_batched`, fp16 operands on both sides, fp32 accumulate) — and asks whether the error is inside the fp16 operand budget that Task 2 measures. If it is, the knob is "expected" and the work becomes the accuracy policy's qualification; if it is not, the error names its own root cause. A fourth case replays a slot eviction, pinning the fill-on-claim invariant this exact leg has broken in production before (`kolibri_engine.cu:3415-3425`).

**Tech Stack:** C++17 CUDA (`.cu` translation units — CMake pins `CMAKE_CUDA_STANDARD 17`), CMake + Ninja, Python 3 + numpy for the dump comparators, pytest for their tests.

**Spec:** `docs/plans/2026-10-30-kolibri-expert-f16-correctness.md` (this file, sections below), [`docs/KOLIBRI_ACCURACY_POLICY.md`](../KOLIBRI_ACCURACY_POLICY.md) (binding), [`docs/KOLIBRI_CUDA_PREVIEW.md`](../KOLIBRI_CUDA_PREVIEW.md) (the T4 measurements this starts from).

## Global Constraints

- Accuracy outranks speed. `docs/KOLIBRI_ACCURACY_POLICY.md`: "Different accumulation order, or different CPU/GPU expert arithmetic" requires separate numerical-fidelity and task-quality qualification; "No universal numerical tolerance is approved"; "Numerical acceptance and default promotion are separate decisions requiring actual results and explicit approval".
- Do **not** change any default in this plan. The accuracy-first T4 configuration (`EXPERT_F16=0`, slots 0, decode arms on) stays the default until a later, explicitly approved change.
- The plain path must stay byte-identical: with `STRATA_KOLIBRI_EXPERT_F16` unset, full logits must stay bitwise-identical to `5bbe6c4e`'s.
- `.cu` files compile as CUDA C++17. No `std::atomic_ref`, no `std::span`, no `starts_with`. Host-only helpers go in `include/`, not in a `.cu`.
- Build: `cmake -S . -B build-kolibri-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_DSV41=OFF -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON`.
- Model: `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`. Pin every engine run: `numactl --cpunodebind=1 --membind=1` (the T4 is PCI `82:00.0`, `numa_node 1`).
- The 0.400-KL configuration is `--prefill --cpu-threads 8 --n 8 --ctx 2048` (`tools/kolibri/decode_gate.sh:57`, `COMMON`); the run's own stderr records it (`/tmp/t4-cand/f16only512.bin.err`: "prefill pool: 8 lane(s)"). Re-measurements must replicate it exactly — without `--prefill` the chunked prefill, the only prefill consumer of the cache (gated on `e.prefill > 0`, `kolibri_engine.cu:6140`), never runs, and decode never consults the cache without `STRATA_KOLIBRI_DEC_RES` (`:2803`, off by default; the arms refuse it, `:5841`).
- Comment-only corrections in engine code are allowed on this branch; behaviour changes are not. Known: the comment at `:3903` ("--dump-layers disables chunking entirely") is stale — the live gate is `--prefill` alone, and the only remaining `fl == nullptr` sits inside the comment at `:6160`.
- Never overwrite `tools/kolibri/oracle_gfx1201.txt`. A T4 oracle may be recorded only from an independently qualified reference, never from the candidate.
- Keep raw logs and dumps under `/tmp/f16-<task>/`. Numbers quoted in a commit message or a doc must come from those files.
- Commit titles start with an area: `[Engine]`, `[Docs]`, `[Test]`. One problem per commit.

## Review Focus

1. **Precision mixing across experts — not within one.** The warm gate is conjunctive (`expert_f16 > 0 && supported(gu_type) && supported(d_type)`, `kolibri_engine.cu:5652-5653`) and a claimed slot dequants all three projections (`:3460`), so gate/up-in-fp16 with down-on-the-pool inside one expert cannot happen. The real mixing is across experts in one chunk: the top `res_eff = min(STRATA_KOLIBRI_RES_TOP=48, n_cached)` experts by row count run resident fp16 GEMMs while the rest run the f32 pool (`:3436`). The load banner also reports "native expert kernels for 25 / 50 layers" — pin down which layers and types differ (one `--counters` run, Task 3), because the pool path itself is not uniform.
2. **Slot wiring, not id ranges.** The warm fill dequants experts 0..N−1 into the buffer but assigns every slot −1 (`:5667`) — those bytes are never read. Residency is built per chunk by promotion of the chunk's top experts with LRU eviction (`:3438-3460`); `seed_hot` claims slots from the touch histogram only under `STRATA_KOLIBRI_DEC_RES` (`:2485`). Promotion re-dequants before claiming, so the invariant to verify is eviction and mapping — and this exact leg has prior form: the P1 comment at `:3415-3425` records the resident GEMM multiplying by the wrong expert's weights (duplicate slots at `res_top=48 > n_cached=24`, fixed by the cap). Task 4's eviction case pins it at the unit level.
3. **Chunk boundaries with the cache on.** 256-token chunk rule, 512, and the 8191 KV wrap. `PF_PARTIAL`'s ring is allocated inside the fp16 cache's scratch and the engine refuses the knob without the cache (`kolibri_engine.cu:5821`), so chunking and the cache share memory that has already faulted once in this area. Mind the stale `:3903` comment when reading this path (Global Constraints).
4. **First use, reset and cancellation with the cache on.** The policy's repeatability clause: a stale slot table after reset reproduces identically and would pass a two-run determinism check while still being wrong.
5. **Long context and attention transitions with the cache on.** Nothing has been measured there at all — only the plain and arms paths were dumped at 8191 and 11000 tokens.

---

### Task 1: Commit the fidelity comparator

**Files:**
- Create: `tools/kolibri/klcmp.py`
- Test: `tools/kolibri/test_klcmp.py`

**Interfaces:**
- Produces: `load(path: str, vocab: int) -> np.ndarray` (shape `(records, vocab)`, float32; raises `ValueError` on a size that is not a multiple of `vocab`), `summary(P: np.ndarray, Q: np.ndarray) -> dict`, `main(argv: list[str] | None = None) -> int`.
- `summary` keys, used by every later task: `records`, `bitwise`, `max_abs`, `kl_mean`, `kl_median`, `kl_p95`, `kl_p99`, `kl_max`, `kl_worst_step`, `top1`, `top1_count`, `margin_worst`, `margin_min`, `margin_min_step`. Raises `ValueError` if any value is non-finite.
- CLI: `python3 tools/kolibri/klcmp.py --ref R.bin --cand C.bin [--vocab N]`; exit 0 when comparable (identical or not), 1 on non-finite, 2 on shape/size mismatch. `--vocab` defaults to 128000 so a tiny test vocabulary is expressible.

- [ ] **Step 1: Write the failing tests**

```python
# tools/kolibri/test_klcmp.py
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("klcmp", Path(__file__).with_name("klcmp.py"))
klcmp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(klcmp)


def test_identical_dumps_are_bitwise():
    rng = np.random.default_rng(7)
    P = rng.standard_normal((4, 5)).astype(np.float32)
    s = klcmp.summary(P, P.copy())
    assert s["bitwise"] is True and s["max_abs"] == 0.0 and s["kl_mean"] == 0.0
    assert s["top1_count"] == 4


def test_kl_matches_hand_computation():
    # uniform reference over 3 values against q = [0.5, 0.25, 0.25]:
    # KL = (1/3)[ln(2/3) + 2 ln(4/3)] = 0.0566330
    P = np.zeros((1, 3), dtype=np.float32)
    Q = np.array([[math.log(2.0), 0.0, 0.0]], dtype=np.float32)
    s = klcmp.summary(P, Q)
    assert s["kl_mean"] == pytest.approx(0.0566330, abs=1e-6)
    assert s["top1_count"] == 1  # [-0.405, -1.098, -1.098]: the argmax is unchanged


def test_top1_flip_is_counted():
    P = np.zeros((1, 3), dtype=np.float32)
    Q = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    assert klcmp.summary(P, Q)["top1_count"] == 0


def test_non_finite_is_rejected():
    P = np.zeros((1, 3), dtype=np.float32)
    Q = np.array([[np.nan, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        klcmp.summary(P, Q)


def test_size_mismatch_exits_2(tmp_path):
    (tmp_path / "r.bin").write_bytes(np.zeros(3, dtype=np.float32).tobytes())
    (tmp_path / "c.bin").write_bytes(np.zeros(4, dtype=np.float32).tobytes())
    rc = klcmp.main(["--ref", str(tmp_path / "r.bin"), "--cand", str(tmp_path / "c.bin"), "--vocab", "3"])
    assert rc == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tools/kolibri/test_klcmp.py -q`
Expected: FAIL — `FileNotFoundError`/`AttributeError`, `klcmp.py` does not exist yet.

- [ ] **Step 3: Implement `tools/kolibri/klcmp.py`**

Move the working body from `/tmp/klcmp.py` (used for the 2026-10-30 numbers) into the repo, split into `load` / `summary` / `main` with the keys above, and keep the existing CLI output lines (`records …; bitwise identical: …`, `max |a-b| …; rows differing …`, `KL  mean … median … p95 … p99 … max … (step N)`, `top-1 agreement …`, `reference margin at the worst step …`). Docstring: engine-vs-engine only — it measures a change against the current engine path and is **not** the llama.cpp-referenced `tools/kolibri/kl.py`; say so, because the policy requires the other one for qualification.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tools/kolibri/test_klcmp.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Re-derive one 2026-10-30 number with the committed tool**

Run: `python3 tools/kolibri/klcmp.py --ref /tmp/kolibri-t4-decode-gate/base512.bin --cand /tmp/t4-cand/f16only512.bin`
Expected: `mean KL 0.400` ±0.001 and `top-1 agreement 1.0000`, matching `docs/findings-log.md` 0.8 (the dumps still exist; if they were removed, record that the re-derivation was not run).

- [ ] **Step 6: Commit**

```bash
git add tools/kolibri/klcmp.py tools/kolibri/test_klcmp.py
git commit -m "[Test] commit the engine-vs-engine fidelity comparator"
```

---

### Task 2: The fp16 conversion's error budget, per expert type

**Goal of the task:** a number per type for "what fp16 costs before any GEMM". The conversion is `__float2half_rn` (`src/kernels/cuda/dequant_bf16.cu:34`) — IEEE round-to-nearest, subnormals included — so a correct conversion is bounded by half an fp16 ulp per value. If Task 2 confirms that bound, an 11-logit worst case cannot come from the conversion step and Task 4 must look at the GEMM.

**Files:**
- Modify: `src/kernels/dequant_bf16_test.cpp` — add a `--selftest` mode. The file is a plain `main()` program (no gtest) whose current mode requires a real GGUF shard argument and is deliberately not a ctest ("needs the 38 GB model", `CMakeLists.txt:2666`); do not add gtest macros, and there is no synthetic-block "existing pattern" to follow — the selftest mode is the new thing.
- Modify: `CMakeLists.txt` — register the selftest as a ctest beside `router_top10_parity`'s (`:2662`), inside the same `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` guard that defines the target (`:2663-2664`).
- Read: `include/strata/kernels/dequant_bf16.hpp` (`dequant_f16`, `dequant_f32`, `dequant_bf16_supported`), `src/kernels/cuda/dequant_bf16.cu:429`.

**Interfaces:**
- Consumes: `strata::kernels::dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream)` and `dequant_f16(...)` with the same leading parameters.
- Produces: in `dequant_bf16_test.cpp`, `int selftest()` — synthetic Q4_K (12) and Q6_K (14) blocks, both kernels, per-value error in fp16 half-ulps; returns 0 on pass, 1 on violation. `main` calls it when `argv[1] == "--selftest"`; the GGUF mode keeps working unchanged.

**The bound, stated exactly (do not weaken it):** for every value, `|f16 − f32| ≤ max(2^-11 · |f32|, 2^-25)`. Subnormals are why: below fp16's min normal 6.1e-5 the spacing is 2^-24, so a value near 1e-6 rounds with relative error up to ~3e-2 on a *correct* implementation — a floored relative-error assertion (the first draft's `max(|f32|, 1e-6)` denominator) would fail a correct kernel and be read as the defect.

- [ ] **Step 1: Write the failing test first — the CMake registration**

```cmake
# CMakeLists.txt, beside router_top10_parity's add_test (~:2662), inside the CUDA/HIP guard
add_test(NAME kolibri_dequant_f16_selftest COMMAND dequant_bf16_test --selftest)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build build-kolibri-cuda --target dequant_bf16_test -j 8 && ctest --test-dir build-kolibri-cuda -R kolibri_dequant_f16_selftest --output-on-failure`
Expected: FAIL — the program does not know `--selftest` yet (it falls through to the usage path and returns 2).

- [ ] **Step 3: Implement the selftest**

Build the blocks from seeded random bytes — any byte pattern decodes, so no quantizer is needed — but set the fp16 scale fields to finite values drawn from a seeded list so no NaN/Inf scale arises (Q4_K: `d`/`dmin` at block bytes 0-3; Q6_K: `d` at bytes 0-1). Two scale regimes per type: **nominal** (finite random scales) and **tiny** (`d = 2^-20`, so a large share of outputs land in the fp16 subnormal range — the regime a wrong floored-relative assertion fails). For each block row run `dequant_f32` and `dequant_f16` on the same device blocks, then per value compute `err_ulp = |f16 − f32| / (0.5 · ulp(v))` with `ulp(v) = 2^-24` for `|v| < 2^-14`, else `2^(floor(log2 |v|) − 10)`; exact zeros have error 0. Assert `err_ulp ≤ 1.0 + 1e-3` (the slack is the f32 path's own rounding). Print per type and regime: max half-ulps, values seen, subnormal-range count; fail the tiny regime if its subnormal count is 0 — a pass that never exercised subnormals proves nothing. Print unsupported expert types as skipped.

- [ ] **Step 4: Run it to verify it passes, and record the numbers**

Run: `ctest --test-dir build-kolibri-cuda -R kolibri_dequant_f16_selftest --output-on-failure`
Expected: PASS, four lines (2 types × 2 regimes). **Record the max-half-ulp numbers** in the commit message; they are the budget Task 4 is judged against.

- [ ] **Step 5: Prove the old mode still works**

Run: `CUDA_VISIBLE_DEVICES=0 ./build-kolibri-cuda/dequant_bf16_test /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`
Expected: the same pass/fail output as before this task (the GGUF-driven comparison is untouched); skip with a note only if the model file is absent.

- [ ] **Step 6: Commit**

```bash
git add src/kernels/dequant_bf16_test.cpp CMakeLists.txt
git commit -m "[Test] measure the fp16 expert conversion's error budget per type"
```

---

### Task 3: Localise the divergence by layer

**Files:**
- Create: `tools/kolibri/layer_delta.py`
- Test: `tools/kolibri/test_layer_delta.py`

**Interfaces:**
- Produces: `load_layers(path: str, n_layer: int, n_embd: int) -> np.ndarray` (shape `(positions, n_layer, n_embd)`, float32), `first_diverging_layer(A, B, rel_threshold: float) -> tuple[int, float]` returning `(layer_index, median_relative_delta)` — `(-1, 0.0)` when nothing exceeds the threshold and `(n_layer + 1, …)` when the record count differs (call that a hard failure, not a rounding difference).
- Layout source of truth: the writer is `src/kolibri/kolibri_engine.cu:6147` (`--dump-layers`), and the existing reader in `tools/kolibri/compare_dumps.py` reads the same records with `N_EMBD = 2560`, `N_LAYER = 50`. **Read both before writing the test** and pin the test's synthetic fixture to what the writer actually emits; if they disagree, stop and report it — a mismatch means the existing layer metric compares misaligned data.

- [ ] **Step 1: Write the failing tests** — a synthetic two-layer, four-value record pair where layer 0 matches exactly and layer 1 differs by a known relative amount; assert `first_diverging_layer` returns `(1, ≈known)` and `(-1, 0.0)` for identical inputs, and that a record-count mismatch returns the hard-failure tuple.

- [ ] **Step 2: Run them to verify they fail** — `python3 -m pytest tools/kolibri/test_layer_delta.py -q` → FAIL, module missing.

- [ ] **Step 3: Implement `tools/kolibri/layer_delta.py`** with the layout confirmed from the writer.

- [ ] **Step 4: Run them to verify they pass** — `5 passed` or whatever count the test file has.

- [ ] **Step 5: Run it on the card — with the gate's exact flags**

The 0.400 KL was measured with `--prefill --cpu-threads 8 --n 8` (`tools/kolibri/decode_gate.sh:57`, `COMMON`) — and none of those are defaults: `prefill` defaults to 0 (`kolibri_engine.cu:5746`, parsed at `:5815`) and the chunked prefill, the only prefill consumer of the cache, is gated on `e.prefill > 0` (`:6140`). Without them the cache is never consulted anywhere — decode needs `STRATA_KOLIBRI_DEC_RES` (off by default, `:2803`) and the arms refuse the cache (`:5841`); the 0.400 run's own stderr says "decode acceleration off … running the plain decode path" (`/tmp/t4-cand/f16only512.bin.err`). `--dump-layers` does **not** disable chunking (the `:3903` comment is stale; see Global Constraints). `STRATA_GEMM_DETERMINISTIC` governs the Gemm-class path only (`src/prefill/gemm.cu:682`), not attention's handle (`:5942`) — keep it, but it is not a full determinism pin.

```bash
mkdir -p /tmp/f16-t3
for arm in plain f16; do
  envs="CUDA_VISIBLE_DEVICES=0 STRATA_GEMM_DETERMINISTIC=1"
  [ "$arm" = f16 ] && envs="$envs STRATA_KOLIBRI_EXPERT_F16=24"
  numactl --cpunodebind=1 --membind=1 env $envs ./build-kolibri-cuda/strata-kolibri \
    --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
    --tokens-file /tmp/kolibri-t4-decode-gate/ids512.txt \
    --force 286,3515,310,262,14292,1343,1444,46 --n 8 --ctx 2048 \
    --prefill --cpu-threads 8 \
    --dump-layers /tmp/f16-t3/$arm.layers >/dev/null 2>/tmp/f16-t3/$arm.err
done
python3 tools/kolibri/layer_delta.py --a /tmp/f16-t3/plain.layers --b /tmp/f16-t3/f16.layers --rel 1e-3
```

**Sanity gate before interpreting:** both `.err` files must show `prefill pool: 8 lane(s)`, and the f16 arm must show the cache working set allocated. If the two `.layers` files are **bitwise identical, the cache path was not exercised — do not interpret; re-check the flags** (that is the exact null result the first draft of this command would have produced). One extra `--counters` run on the plain arm pins the 25/50 native-kernel split for Review Focus 1.

Expected with the flags right: a per-layer table over 10 records (1 final prompt position + 9 generated steps — the dump rows are decode positions, `kolibri_engine.cu:6140-6160`) and one line naming the first layer over 1e-3. **Read it with the KV offset:** decode runs plain in both arms, so divergence enters the dumped rows only through the prefill KV — dumped layer *l* diverges exactly when prefill layer *l−1*'s output already diverged (KV[l] is computed from layer *l*'s input). Expect the first diverging dumped layer ≈ first-MoE-layer + 1, and report prefill layer *l−1* as the culprit, not *l*. A genuine divergence in dumped layer 0 (its KV is written from embeddings alone) means the dense/attention front half, not the experts — that would redirect the whole plan and must be reported before Task 4. Optional sanity arm: `strata-kolibriref --dump-layers` on the same prompt (check its CLI accepts the same prompt flags first); the plain arm is chunked while kolibriref equals the per-token path bitwise (`:2133`), so a nonzero plain-vs-ref delta there is accumulation order, not a defect — measure, don't assume.

- [ ] **Step 6: Commit** — `[Test] localise a Kolibri numeric divergence by layer`.

---

### Task 4: The verdict — is the fp16 expert GEMM inside its budget on sm_75?

**Files:**
- Create: `tests/kolibri/f16_expert_gemm_parity.cu`
- Modify: `CMakeLists.txt` (register beside `kolibri_router_parity`, `:1525-1529`; **link `strata_prefill`** — `Gemm` is only declared in `include/strata/prefill/gemm.hpp` and implemented in `src/prefill/gemm.cu` inside that library, `:807` — plus `strata_kernels` and `${_strata_gpu_runtime_target}`; check `strata_prefill` is in scope at the registration point)
- Read: `src/kolibri/kolibri_engine.cu:276-297` (cache layout), the `Gemm::f16_batched` call site, `include/strata/prefill/gemm.hpp:47` (the signature)

**Interfaces:**
- Produces: `int main(int argc, char** argv)` returning 0 on PASS, 1 on a violation; flags `--embd N --ff N` (default 256/256). Prints one line per case: `case … max |y−y_ref| / Σ|w·x| = … (budget …)`.
- Consumes: `strata::kernels::dequant_f32/dequant_f16` (seeded-random Q4_K blocks as in Task 2) and `Gemm::f16_batched(const uint16_t* const* Warr, const uint16_t* const* Xarr, float* Y, …)` — "Y_i[T,N] = X_i[T,K]·W_i[N,K]^T, row-major, fp32 out, fp32 accumulate" (`include/strata/prefill/gemm.hpp:47`).

**The budget, stated exactly (do not weaken it):** the resident leg runs **both operands in fp16** — activations too (`s.res_x16`) — with fp32 accumulate, so the bound is backward-error style and scale-aware: `|y_c − y_ref| ≤ 4·2^-11 · Σ|w_i x_i|`. The factor 4 = 2^-11 per operand × both operands × 2 for fp32-accumulation and product-rounding slack; state it in the test's header. The f64 reference runs over `dequant_f32` values, so the case isolates operand precision — the Q4_K quantization error is common to every path and out of scope. An output-relative error is the wrong metric: |y| can sit near zero from cancellation and a correct implementation would fail it.

- [ ] **Step 1: Write the failing test** — three cases, each against a float64 host reference over `dequant_f32` of the same seeded-random Q4_K blocks, and each run on both projection shapes (gate/up: K = n_embd → n_ff; down: K = n_ff → n_embd):
  - **(a) f32 control** — host f32 GEMM over the same weights and f32 activations. Assert `|y − y_ref| ≤ 2e-5 · Σ|w_i x_i|` (256·2^-24 ≈ 1.5e-5, worst-case f32 summation over K=256). Label it a **control, not "the plain path"**: the pool's exact arithmetic (`ExpertPool`, `include/strata/kernels/cpu/pool.hpp`) is not what this test runs; say so in the header.
  - **(b) fp16 operands, the engine's entry point** — weights via `dequant_f16` into a device buffer, activations converted to fp16, `Gemm::f16_batched` with fp32 accumulate. Assert the scale-aware budget above; print the observed ratio against it.
  - **(c) eviction/wiring** — two experts E1, E2, one slot: dequant E1 into the slot, run the batched GEMM against it, assert it matches E1's reference; then dequant E2 into the **same slot** (overwrite, as promotion does, `kolibri_engine.cu:3445-3460`), re-run, assert it matches E2's reference — not E1's. This pins the fill-on-claim invariant this exact leg has broken in production (P1, `:3415-3425`). Scope note in the header: it tests the slot/pointer mechanism at the Gemm level, not the engine's LRU bookkeeping — that stays with 5b's first-use/chunk/reset runs.

- [ ] **Step 2: Run it to verify it fails to build** — `cmake --build build-kolibri-cuda --target f16_expert_gemm_parity -j 8` → FAIL, target does not exist. Add the CMake target mirroring `kolibri_router_parity` (`:1525-1529`) — **with `strata_prefill` in `target_link_libraries`**, or it will not link. Rebuild, and confirm the test then fails on an unimplemented assertion rather than a link error.

- [ ] **Step 3: Implement the three cases.** `Gemm` is caller-owned: initialize its handle and provide the scratch/workspace buffers it documents (`init_external`, `rebind`, `scratch_elems()` in `include/strata/prefill/gemm.hpp`), mirroring the engine's own setup.

- [ ] **Step 4: Run it — toy shapes, then the model's real shapes**

Run: `CUDA_VISIBLE_DEVICES=0 ./build-kolibri-cuda/f16_expert_gemm_parity`, then `CUDA_VISIBLE_DEVICES=0 ./build-kolibri-cuda/f16_expert_gemm_parity --embd 2560 --ff <n_ff>`. Get `n_ff` once from the GGUF header (any metadata read of the first expert tensor's shape) and pass it as a flag — the test itself never reads the model. The real-shape run matters: cublas algorithm choice is size-dependent, and a Turing-specific defect can hide at real sizes while the toy case passes.

- **Any case fails its budget** → the arithmetic (or wiring) path is the defect. Report the numbers, then debug from the smallest failing case (one row, one expert) before touching the engine. Do not proceed to Task 5b.
- **All cases pass at both shapes** → the GEMM and the slot mechanism are not the source; the remaining candidates are the accumulation across experts (Task 3's layer numbers decide where) and `PF_PARTIAL`. Proceed to Task 5b.

- [ ] **Step 5: Commit** — `[Test] gate the fp16 expert GEMM against its fp16 budget`.

---

### Task 5: The decision

**5a — defect path (only if Task 4 fails its budget).**
- [ ] Find the root cause from the smallest failing case; fix it.
- [ ] Re-run Tasks 2, 3 and 4 and record the new numbers.
- [ ] Prove the plain path is unchanged: full logits with `EXPERT_F16` unset must be bitwise identical to `5bbe6c4e`'s (re-run `tools/kolibri/decode_gate.sh check`, all four prompt lengths, plus 8191 and 11000 as in `docs/KOLIBRI_CUDA_PREVIEW.md`).
- [ ] Commit `[Engine] fix …`, with the before/after error numbers in the message.

**5b — expected-loss path (only if Task 4 passes).**
- [ ] **Reference precondition, before any fidelity run:** `kl.py` does not just read dumps — it *runs* llama.cpp itself (`--lock`/`--cpu-lock`, a parity dir; `tools/kolibri/kl.py:1-15`) — and no `llama-cli` is on this box's PATH today. Locate or build a Kolibri-capable llama.cpp (the gfx1201 oracle campaign used one); if it cannot be made to run on the T4 box, record 5b as **blocked**. `strata-kolibriref` (`build-kolibri-cuda/strata-kolibriref`, already built) is a sanity reference only — it shares the engine's dequant kernels and is not the policy-grade reference.
- [ ] Numerical fidelity per policy, with a reference the engine is not a part of: `tools/kolibri/kl.py` (llama.cpp GPU as P, llama.cpp CPU as the noise floor). Report mean/median/p95/p99/worst-case KL, top-1, and rank changes against the reference's winning margin — for a 512-token prompt **and** an 8191-token prompt.
- [ ] Repeatability: two runs bitwise identical on each configuration, plus first-use, chunk boundary, prefix reuse, reset and cancellation with the cache on (Review Focus 1-4).
- [ ] Long context: 8191 KV wrap and ~11000 attention transition with the cache on, forced full-logit comparison against the plain path — measure the delta, and state it; it will not be zero.
- [ ] Write the qualification record to `docs/KOLIBRI_CUDA_PREVIEW.md` (a new subsection) and `docs/findings-log.md`, marking clearly that task-quality evaluation is still open.
- [ ] Stop and ask for approval before any default or shipped-config change. The T4 shipped configuration using `EXPERT_F16` is a promotion decision the policy reserves.

---

### Task 6: Task-quality plan (only after 5b, and only if the numbers justify it)

- [ ] Write the evaluation plan — prompts per category (reasoning, coding, factual, long-context retrieval), sample size, the maximum acceptable regression, and the statistical decision rule — **as a document, committed before any candidate output is inspected** (the policy: "Agree the maximum acceptable regression margin, sample size and statistical decision rule before seeing candidate results").
- [ ] Get that margin approved by the maintainer. No run before approval.

## Self-Review

- **Spec coverage:** the seed doc's four hypotheses map to Task 2 + Task 4 (hypothesis 1, the conversion path per type), Task 4 + Task 3 (hypothesis 2, accumulation and the layer it enters), Task 5b + Task 6 (hypothesis 3, sample breadth: several prompts, 512 and 8191), and Task 4's eviction case + Task 5b (hypothesis 4, first use, chunk boundary, reuse, reset). Task 2 supplies the budget every verdict is judged against. The policy's three qualification blocks map to Task 5b (fidelity, repeatability) and Task 6 (task quality). Default promotion is gated in Task 5b.
- **Step scan:** every step names a command or a signature; the steps that could hide a decision (Task 3's record layout, Task 4's budget and its factor) name the file that settles it and require the number to be printed.
- **Type consistency:** `summary()`'s keys are used as written; `first_diverging_layer` returns a 2-tuple everywhere; the only new C++ entry points are `dequant_bf16_test --selftest` and `f16_expert_gemm_parity`'s `main(argc, argv)`; no gtest is introduced anywhere.
- **Review Focus:** all five lines have an owning task — 1 (across-expert mixing and the 25/50 native-kernel split) in Task 3's counters run and 5b, 2 (slot wiring) in Task 4's eviction case and 5b's repeatability, 3 and 4 in 5b's repeatability step, 5 in 5b's long-context step.
- **Proportion:** no task restates engine code; the code steps that carry real algorithm (the GEMM comparison, the layer reader, the selftest) name the source of truth instead of transcribing it.
