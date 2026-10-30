# Kolibri `EXPERT_F16` correctness on CUDA (sm_75 / Tesla T4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether the fp16 expert cache's 11-logit worst case on the Tesla T4 is a defect in the Turing path or expected fp16 loss, and either fix it or qualify it — so the 3.4x prefill gain stops being unusable.

**Architecture:** Four cheap instruments first, then one verdict experiment, then the decision. Each instrument is a committed, unit-tested tool so the numbers can be reproduced by someone else. The verdict experiment compares one expert's GEMM — computed from Q4_K/Q6_K through the plain path, and from the fp16 cache through the path the engine actually runs — against a float64 host reference, and asks whether the error is inside the fp16 operand budget that Task 2 measures. If it is, the knob is "expected" and the work becomes the accuracy policy's qualification; if it is not, the error names its own root cause.

**Tech Stack:** C++17 CUDA (`.cu` translation units — CMake pins `CMAKE_CUDA_STANDARD 17`), CMake + Ninja, Python 3 + numpy for the dump comparators, pytest for their tests.

**Spec:** `docs/plans/2026-10-30-kolibri-expert-f16-correctness.md` (this file, sections below), [`docs/KOLIBRI_ACCURACY_POLICY.md`](../KOLIBRI_ACCURACY_POLICY.md) (binding), [`docs/KOLIBRI_CUDA_PREVIEW.md`](../KOLIBRI_CUDA_PREVIEW.md) (the T4 measurements this starts from).

## Global Constraints

- Accuracy outranks speed. `docs/KOLIBRI_ACCURACY_POLICY.md`: "Different accumulation order, or different CPU/GPU expert arithmetic" requires separate numerical-fidelity and task-quality qualification; "No universal numerical tolerance is approved"; "Numerical acceptance and default promotion are separate decisions requiring actual results and explicit approval".
- Do **not** change any default in this plan. The accuracy-first T4 configuration (`EXPERT_F16=0`, slots 0, decode arms on) stays the default until a later, explicitly approved change.
- The plain path must stay byte-identical: with `STRATA_KOLIBRI_EXPERT_F16` unset, full logits must stay bitwise-identical to `5bbe6c4e`'s.
- `.cu` files compile as CUDA C++17. No `std::atomic_ref`, no `std::span`, no `starts_with`. Host-only helpers go in `include/`, not in a `.cu`.
- Build: `cmake -S . -B build-kolibri-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_DSV41=OFF -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON`.
- Model: `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`. Pin every engine run: `numactl --cpunodebind=1 --membind=1` (the T4 is PCI `82:00.0`, `numa_node 1`).
- Never overwrite `tools/kolibri/oracle_gfx1201.txt`. A T4 oracle may be recorded only from an independently qualified reference, never from the candidate.
- Keep raw logs and dumps under `/tmp/f16-<task>/`. Numbers quoted in a commit message or a doc must come from those files.
- Commit titles start with an area: `[Engine]`, `[Docs]`, `[Test]`. One problem per commit.

## Review Focus

1. **Q6_K down projections.** A Q4_K_M file mixes Q4_K and Q6_K. The cache gates `gu_type` and `d_type` separately (`kolibri_engine.cu:5652`); a layer that caches gate/up in fp16 but leaves down on the pool mixes two precisions inside one expert. Check the cache's actual coverage per layer, not the flag.
2. **Expert ids outside the cache.** Only the first N experts are resident, and hot promotion moves ids in and out between requests. A prompt therefore runs some picks in fp16 and some on the pool — verify the boundary is stable and that a promoted slot never serves a stale expert.
3. **Chunk boundaries with the cache on.** 256-token chunk rule, 512, and the 8191 KV wrap. `PF_PARTIAL`'s ring lives inside the fp16 cache's scratch (`kolibri_engine.cu:5821`), so chunking and the cache share memory that has already faulted once in this area.
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

**Goal of the task:** a number per type for "what fp16 costs before any GEMM". If conversion alone is ≤0.05 % relative, an 11-logit worst case cannot come from the conversion and Task 4 must look at the GEMM.

**Files:**
- Modify: `src/kernels/dequant_bf16_test.cpp` (add one test function; the target `dequant_bf16_test` already exists, `CMakeLists.txt:2664`)
- Read for the reference of truth: `include/strata/kernels/dequant_bf16.hpp` (`dequant_f16`, `dequant_f32`, `dequant_bf16_supported`), `src/kernels/cuda/dequant_bf16.cu:429`

**Interfaces:**
- Consumes: `strata::kernels::dequant_f32(int ggml_type, const void* blocks, int64_t row0, int64_t rows, int64_t cols, float* out, void* stream)` and `dequant_f16(...)` with the same leading parameters.
- Produces: a helper in the same file, `float fp16_roundtrip_max_rel(int ggml_type)`, returning the worst per-value relative error of `dequant_f16` against `dequant_f32` on a synthetic block row (0.0 if `dequant_bf16_supported(type)` is false).

- [ ] **Step 1: Write the failing test**

```cpp
// src/kernels/dequant_bf16_test.cpp — add
TEST(DequantF16, ConversionAloneIsWithinHalfAnFp16Ulp) {
    // Q4_K (12) and Q6_K (14) are the expert types of a Q4_K_M file.
    for (int t : {12, 14}) {
        ASSERT_TRUE(strata::kernels::dequant_bf16_supported(t));
        const float rel = fp16_roundtrip_max_rel(t);
        EXPECT_LE(rel, 4.9e-4f) << "type " << t << " relative error " << rel;
        std::printf("type %d fp16 round-trip max relative error %.3e\n", t, (double) rel);
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cmake --build build-kolibri-cuda --target dequant_bf16_test -j 8 && CUDA_VISIBLE_DEVICES=0 ./build-kolibri-cuda/dequant_bf16_test --gtest_filter='DequantF16.*'`
Expected: FAIL — `fp16_roundtrip_max_rel` is not defined.

- [ ] **Step 3: Implement `fp16_roundtrip_max_rel`**

Follow the existing pattern in that file for building an input row (it already loops `dequant_bf16_supported` types and compares kernels), allocate `rows=1, cols=256`, run `dequant_f32` and `dequant_f16` on the same blocks, and return `max(|f16 - f32| / max(|f32|, 1e-6))`. Skip types whose `dequant_bf16_supported` is false by returning 0.0 — but print that they were skipped, so the log shows which expert types the cache would refuse.

- [ ] **Step 4: Run it to verify it passes, and record the numbers**

Run: `CUDA_VISIBLE_DEVICES=0 ./build-kolibri-cuda/dequant_bf16_test --gtest_filter='DequantF16.*'`
Expected: PASS, two `type … fp16 round-trip max relative error …` lines. **Record both numbers** in the commit message; they are the budget Task 4 is judged against.

- [ ] **Step 5: Commit**

```bash
git add src/kernels/dequant_bf16_test.cpp
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

- [ ] **Step 5: Run it on the card**

```bash
mkdir -p /tmp/f16-t3
for arm in plain f16; do
  envs="CUDA_VISIBLE_DEVICES=0 STRATA_GEMM_DETERMINISTIC=1"
  [ "$arm" = f16 ] && envs="$envs STRATA_KOLIBRI_EXPERT_F16=24"
  numactl --cpunodebind=1 --membind=1 env $envs ./build-kolibri-cuda/strata-kolibri \
    --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
    --tokens-file /tmp/kolibri-t4-decode-gate/ids512.txt --force 286,3515,310,262,14292,1343,1444,46 \
    --n 1 --ctx 2048 --dump-layers /tmp/f16-t3/$arm.layers >/dev/null 2>/tmp/f16-t3/$arm.err
done
python3 tools/kolibri/layer_delta.py --a /tmp/f16-t3/plain.layers --b /tmp/f16-t3/f16.layers --rel 1e-3
```

Expected: a per-layer table and one line naming the first layer over 1e-3. Record it. A divergence that appears in layer 0 means the dense/attention front half, not the experts — that would redirect the whole plan and must be reported before Task 4.

- [ ] **Step 6: Commit** — `[Test] localise a Kolibri numeric divergence by layer`.

---

### Task 4: The verdict — is the fp16 expert GEMM inside its budget on sm_75?

**Files:**
- Create: `tests/kolibri/f16_expert_gemm_parity.cu`
- Modify: `CMakeLists.txt` (register the target and `add_test` beside the other `kolibri_*` parity tests, ~line 1528)
- Read: `src/kolibri/kolibri_engine.cu:276-297` (the cache's layout comment) and the `Gemm::f16` call site that consumes `b.f16`

**Interfaces:**
- Produces: `int main()` returning 0 on PASS, 1 on a budget violation; prints one line per case: `q4k f16-gemm max relative error … (budget …)`. No model, no GGUF: synthetic blocks, `n_embd=256`, `n_ff=256`.

- [ ] **Step 1: Write the failing test** — one expert's gate/up product, three ways: (a) float64 host reference over `dequant_f32` of the blocks, (b) the plain path's arithmetic, (c) the fp16 cache's arithmetic (`dequant_f16` into a device buffer, `Gemm::f16` with fp32 accumulate). Assert (b) is within 1e-5 of (a) and (c) is within the Task 2 budget times a small factor (state the factor and why in the file's header comment); FAIL if (c) exceeds it. Print all three errors.

- [ ] **Step 2: Run it to verify it fails to build** — `cmake --build build-kolibri-cuda --target f16_expert_gemm_parity -j 8` → FAIL, target does not exist. Add the CMake target (mirroring `kolibri_router_parity`, `CMakeLists.txt:1525-1529`), rebuild, and confirm the test then fails on an unimplemented assertion rather than a link error.

- [ ] **Step 3: Implement the three-way comparison.**

- [ ] **Step 4: Run it and read the verdict**

Run: `CUDA_VISIBLE_DEVICES=0 ./build-kolibri-cuda/f16_expert_gemm_parity`

- **It fails the budget** → the arithmetic path is the defect. Report the numbers, then debug from the smallest failing case (one row, one expert) before touching the engine. Do not proceed to Task 5b.
- **It passes the budget** → the GEMM is not the source; the remaining candidates are the accumulation across experts (Task 3's layer numbers decide where) and `PF_PARTIAL`. Proceed to Task 5b.

- [ ] **Step 5: Commit** — `[Test] gate the fp16 expert GEMM against its fp16 budget`.

---

### Task 5: The decision

**5a — defect path (only if Task 4 fails its budget).**
- [ ] Find the root cause from the smallest failing case; fix it.
- [ ] Re-run Tasks 2, 3 and 4 and record the new numbers.
- [ ] Prove the plain path is unchanged: full logits with `EXPERT_F16` unset must be bitwise identical to `5bbe6c4e`'s (re-run `tools/kolibri/decode_gate.sh check`, all four prompt lengths, plus 8191 and 11000 as in `docs/KOLIBRI_CUDA_PREVIEW.md`).
- [ ] Commit `[Engine] fix …`, with the before/after error numbers in the message.

**5b — expected-loss path (only if Task 4 passes).**
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

- **Spec coverage:** the four hypotheses in the spec map to Task 3 (hypothesis 3, sample narrowness), Task 4 (hypotheses 1 and 2), Task 5b (hypothesis 4: first use, chunk boundary, reuse, reset), Task 2 (the budget every verdict is judged against). The policy's three qualification blocks map to Task 5b (fidelity, repeatability) and Task 6 (task quality). Default promotion is gated in Task 5b.
- **Step scan:** every step names a command or a signature; the two steps that could hide a decision (Task 3's record layout, Task 4's budget factor) name the file that settles it and require the number to be printed.
- **Type consistency:** `summary()`'s keys are used as written; `first_diverging_layer` returns a 2-tuple everywhere; `fp16_roundtrip_max_rel(int) -> float` is the only new C++ signature.
- **Review Focus:** all five lines have an owning task — 1 and 2 in Task 4/Task 5b, 3 and 4 in Task 5b's repeatability step, 5 in Task 5b's long-context step.
- **Proportion:** no task restates engine code; the two code steps that carry real algorithm (the three-way GEMM comparison, the layer reader) name the source of truth instead of transcribing it.
