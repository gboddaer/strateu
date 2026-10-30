# Kolibri `EXPERT_F16` correctness on Tesla T4 — Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` for native execution. Use `superpowers:subagent-driven-development` only if the operator authorizes delegation. Execute the checkboxes in order; report evidence and blockers, not assumed success.

**Goal:** Explain and, where defective, fix the large full-logit divergence with `STRATA_KOLIBRI_EXPERT_F16=24` on Tesla T4. Establish whether the resulting opt-in path is numerically and task-quality acceptable before considering its prefill benefit.

**Architecture:** Preserve a release control, reproduce the discrepancy, isolate conversion/arithmetic/cache-state effects, then fix demonstrated defects. Synthetic tests are diagnostic instruments, not a verdict on the engine. Both repaired and apparently defect-free candidates enter the same independent-reference and quality qualification gates.

**Tech Stack:** CUDA C++17, CMake/Ninja, Python 3, numpy, pytest.

**Binding references:** [accuracy policy](../KOLIBRI_ACCURACY_POLICY.md), [T4 measurements](../KOLIBRI_CUDA_PREVIEW.md), [findings log §0.8](../findings-log.md). Read `AGENTS.md`, `docs/AI_REFERENCE.md` and `CONTRIBUTING.md` before execution. This revision supersedes the earlier plan and its claims that a microtest could decide “defect versus expected loss.”

## Scope, safeguards and completion

- Priority: **accuracy → prefill → decode**. Lower-precision experts are a higher-risk accuracy project under the policy, not a scheduling optimization.
- Keep all executable defaults and shipped configurations unchanged. `EXPERT_F16=0` remains the control. Defect fixes may change the opt-in path; the cache-off path must retain bit-identical finite full logits.
- Do not pull/rebase, push, promote a default, overwrite `tools/kolibri/oracle_gfx1201.txt`, or alter the running server without explicit permission. Do not silently stop a server for GPU access: request a measurement window if necessary.
- Work in `/data2/develop/strateu-publish`, branch `kolibri-expert-f16-correctness`. Check status first; preserve unrelated changes. Use an isolated worktree if execution needs one, and configure its own build directory.
- Control engine revision: `5bbe6c4e`, with `STRATA_Q6K_EXPERTS=ON`. Existing dumps are useful only when revision, flags, environment and forced histories match. A current-binary arms-off/arms-on check alone does **not** prove equality to this revision.
- T4: sm_75, 16 GB; model `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`. CUDA translation units remain C++17.
- Store runs under `/tmp/f16-tN/`, with a manifest of model identity, source revision/dirty diff, binary hash, toolchain, hardware, all effective `STRATA_*` variables, complete argv, input hashes, stdout/stderr, exit status and dump hashes. Archive indispensable evidence outside `/tmp` before handoff; never commit the model or large dumps.
- Use one GPU job at a time. Separate the historical reproduction configuration from NUMA-pinned diagnostic/performance runs. Pin new diagnostic runs with `numactl --cpunodebind=1 --membind=1`; do not claim this exactly reproduces earlier runs whose logs show a two-node pool.
- Engineering error bounds in unit tests do **not** authorize an end-to-end accuracy tolerance. Agree qualification limits before inspecting new qualification results. Never fit a threshold to make the candidate pass.
- Commit logical changes with an area prefix. If a shared file changes, validate the supported affected backends before review; explicitly list unavailable CUDA/HIP/SYCL checks instead of claiming they passed.

**Done means one of:**
1. A reproduced defect has a regression test and fix, the cache-off release control is bit-identical, and the opt-in candidate has documented qualification results; task-quality and numerical acceptance require explicit approval.
2. The discrepancy has a supported arithmetic explanation, with the same qualification evidence; it remains opt-in pending approval.
3. A specific unresolved cause or missing reference/approval prevents a verdict. Deliver the smallest reproducer, artifacts and a precise blocked report. Do not label this fixed, safe or qualified.

A performance gain is not guaranteed. If fidelity cannot be justified, retain the accuracy-first control and reject promotion.

## Code facts and review focus (checked at `1432941c`)

Use symbols as primary references; line numbers below are navigation aids and will move.

1. **Conversion:** `src/kernels/cuda/dequant_bf16.cu` writes fp16 with `__float2half_rn`. Q4_K has `d`/`dmin` at bytes 0/2; **Q6_K's `d` is at byte 208**, not byte 0 (`group32<TYPE == 14>`). `dequant_bf16_supported()` checks format geometry, not native bf16 hardware support. Task 2 tests both formats independently of its name.
2. **Slot ownership:** `kolibri_engine.cu` allocates a whole gate/up/down cache only when both formats are supported (~5652). Initial slot ids are -1 despite the warm fill. The promotion loop (~3435–3465) maintains per-slot expert id/age and selects residents before building batched pointer arrays. A one-slot overwrite microtest does not test this policy. Task 4 checks the actual multi-slot selection and eviction path.
3. **Actual resident arithmetic:** ~3523–3530 gathers fp16 activations, runs gate/up through `Gemm::f16_batched`, SwiGLU, fp16 hidden conversion, then a down GEMM. CPU pool work can also quantize activations (`native_quant_act`, ~3372). “Pool = exact fp32 reference” is false. Task 4 compares actual expert paths, not just weight conversion.
4. **Diagnostic coverage:** `--prefill --cpu-threads 8` is needed for the measured pool-prefill configuration. `--dump-layers` does not itself disable chunking; the ~3903 comment is stale. With eight `--force` ids the chunked CLI produces **nine** decode-position records (`n_steps = forced.size() + 1`, ~6252), not ten. Layer dumps show decode outputs after prefill, not the prefill intermediates. They cannot prove an exact culprit at layer l-1. Task 3 uses them to narrow the search; Task 4 instruments the prefill leg directly.
5. **Transitions and state:** cache size/resident limit, mixed Q4_K/Q6_K, partial final blocks, multiple chunks, first use, reuse, reset and cancellation can expose faults invisible to a one-expert test. Tasks 4 and 6 own these checks. `PF_PARTIAL` is a separate arm, not a substitute for testing cache-only.

## Task 0 — Freeze the control and make the experiment reproducible

**Files:** create `docs/experiments/kolibri-t4-f16-correctness.md` as an evidence ledger; artifacts in `/tmp/f16-t0/`. No engine changes.

- [ ] Check branch/status, GPU ownership and available RAM/disk. Record any running server without changing it. Verify model and interpreter dependencies; `/tmp/pt-venv` may exist but must not be assumed.
- [ ] Record the existing candidate and control artifacts before rebuilding anything. Preserve `/tmp/klcmp.py`, `/tmp/kolibri-t4-decode-gate/` and `/tmp/t4-cand/` if present. If absent, regenerate from a recorded recipe, not an invented measurement.
- [ ] Preserve or separately build the release control at `5bbe6c4e`. Never overwrite the only control binary. Configure the working build with:

```sh
cmake -S . -B build-kolibri-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_DSV41=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_Q6K_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON
cmake --build build-kolibri-cuda --target strata-kolibri dequant_bf16_test -j 8
```

- [ ] Recover the historical argv/environment from retained scripts/logs where possible. Known gate argv includes `--prefill --cpu-threads 8 --ctx 2048`, 512 prompt ids and forced ids `286,3515,310,262,14292,1343,1444,46`. Record what cannot be recovered; do not claim exact historical reproduction from incomplete metadata.
- [ ] Put the input ids and manifest under `/tmp/f16-t0/`; pin their hashes. Write the ledger's configuration matrix: control (`EXPERT_F16=0`, `PF_PARTIAL=0`), cache-only (`24`, `0`), cache+partial (`24`, `1`). Explicitly disable `DEC_RES` and unrelated opt-in knobs. Inherited `STRATA_*` settings must be cleared or explicitly recorded/set.
- [ ] Identify the pinned Kolibri-capable llama.cpp reference and custom dump tool used by `tools/kolibri/kl.py`. Check their version/build compatibility now. This is **not** just `llama-cli`. Missing reference blocks qualification, not diagnostic work. Do not download/build external dependencies without checking the applicable project procedure.

**Exit:** preserved control, repeatable input/configuration matrix, artifact manifest and known reference availability. Commit the ledger skeleton without fabricated results.

## Task 1 — Reliable full-logit comparison and reproduction

**Files:** create `tools/kolibri/klcmp.py` and `tools/kolibri/test_klcmp.py`; update evidence ledger.

**Interfaces:** `load(path, vocab) -> float32[records,vocab]`; `summary(ref, cand) -> dict`; `main(argv=None) -> int`. CLI: `--ref PATH --cand PATH --vocab 128000 [--json]`. Report record count, bytewise equality, max absolute delta, per-step KL(P||Q), mean/median/p95/p99/max KL with worst step, top-1 agreement, reference winning margins and candidate rank of the reference winner. Exit 0 means comparable, **not accepted**; 1 means non-finite input; 2 means invalid shape/input.

- [ ] Write tests: identical finite inputs, hand-computed KL (`P=[0,0,0]`, `Q=[ln 2,0,0]`, KL≈0.0566330), top-1 flip, NaN/Inf rejection, shape/record mismatch, empty input, invalid vocab, truncated bytes and signed-zero byte inequality. Test CLI exit codes. Do not silently truncate to the shorter dump.
- [ ] Run `python3 -m pytest tools/kolibri/test_klcmp.py -q`; establish a genuine failing test before implementation. Implement the comparator with float64 stable log-softmax; reuse `/tmp/klcmp.py` only after inspecting it. Bytewise means matching float32 bits, not `np.array_equal` alone.
- [ ] Re-run the tests. Compare retained `base512.bin` and `f16only512.bin` if available; report whether the previously recorded mean KL≈0.400 and nine records reproduce. Treat disagreement as an investigation, not a reason to tune the comparator.
- [ ] Run the three-arm matrix with the preserved control binary, identical inputs/forced history and two independent runs per arm. Keep stdout/stderr and full logits. The forced sequence has eight ids: expect **9 × 128000 × 4 = 4,608,000 bytes** per full-logit file.
- [ ] Compare same-arm repeats first, then cross-arm deltas. If non-finite or unexplained same-arm variation occurs, debug that before interpreting numerical differences. Commit tool/tests and observed reproduction evidence.

**Exit:** the discrepancy is reproduced with known configuration, or its failure to reproduce is documented and investigated before proceeding.

## Task 2 — Verify conversion without confusing rounding with fidelity

**Files:** modify `src/kernels/dequant_bf16_test.cpp`, `CMakeLists.txt`; update ledger. Existing target is a standalone GGUF-driven `main()`, not gtest.

**Interface:** add `--selftest` (exit 0 pass, 1 assertion failure); preserve the existing GGUF mode. Register `kolibri_dequant_f16_selftest` after defining its executable, under the same CUDA/HIP guard.

- [ ] Before modification, run and record the existing GGUF mode on the model. Write selftest assertions for Q4_K and Q6_K, row slices/strides, zeros, normal values and subnormal-range values. Required types missing from support must fail, not silently skip.
- [ ] Register selftest and run it before implementing the dispatch. An unknown `--selftest` currently reaches the GGUF-opening path, so expect a nonzero error, **not a guaranteed usage exit 2**. Capture the actual failure.
- [ ] Build valid seeded blocks using the layout in the kernel and CPU artifact dequantizers. Q6_K scale is at **208**. Use bounded finite scales/output values (no accidental fp16 overflow); set both Q4_K scales deliberately. Include tiny non-power-of-two scales and halfway-rounding fixtures; power-of-two scales alone can produce exact half values and test no rounding at all.
- [ ] Compare GPU `dequant_f32` against the existing independent CPU block decoder first. Then compare `dequant_f16` bits against a trusted host round-to-nearest-even fp32→fp16 conversion of those f32 values. Do not derive the sole reference from the same GPU conversion. Assert finite results, correct zeros and expected bits; print error/ULP statistics and nonzero subnormal/rounding-case coverage. If CPU/GPU f32 arithmetic differs, isolate that difference before blaming fp16 conversion.
- [ ] Build, run `ctest --test-dir build-kolibri-cuda -R '^kolibri_dequant_f16_selftest$' -V`, and re-run the GGUF mode. Step 4 requires rebuilding after code edits. Deliberately corrupt an expected conversion/offset in a disposable test run to confirm the assertion detects it, then restore and rerun.
- [ ] Commit verified tests and measured conversion results. Passing this test establishes conversion behavior for covered inputs; it does **not** bound an 11-logit error through 50 nonlinear layers.

## Task 3 — Locate the first observable divergence

**Files:** create `tools/kolibri/layer_delta.py`, `tools/kolibri/test_layer_delta.py`; update ledger.

**Interfaces:** `load_layers(path, n_layer, n_embd) -> float32[records,layers,embedding]`; `layer_metrics(ref,cand) -> per-record/per-layer metrics`. CLI `--a PATH --b PATH --layers 50 --embd 2560 --rel 1e-3`. Relative threshold is a navigation aid, not an accuracy gate. Invalid/non-finite/misaligned inputs fail explicitly.

- [ ] Read the actual writer in the CLI decode loop and `tools/kolibri/compare_dumps.py`; test exact byte layout, malformed inputs, known layer differences, zero reference norms, and an outlier affecting only one record. Report both maximum and median relative L2 delta; median alone must not conceal an outlier.
- [ ] Establish failing tests, implement the reader/metrics, then run `python3 -m pytest tools/kolibri/test_layer_delta.py -q` to green.
- [ ] Repeat Task 1's control/cache-only recipe adding `--dump-layers /tmp/f16-t3/ARM.layers` **and** `--dump-full-logits /tmp/f16-t3/ARM.bin`. Include `--prefill --cpu-threads 8 --expert-slots 0`, eight forced ids, and `--ctx 2048`; record complete effective environment. For this prompt expect **9 × 50 × 2560 × 4 = 4,608,000 bytes** of layer data.
- [ ] Require successful exits, finite complete dumps and measured resident activity, not an assumed allocation banner. Inspect `--counters` output and its implementation; add opt-in activity instrumentation if needed. If identical outputs occur, verify cache activity before deciding what the result means; identical output alone does not prove inactivity.
- [ ] Record the earliest observable divergence by record/layer. Do **not** map it exactly to prefill layer l-1: decode residual propagation, attention, router sensitivity and prior KV changes mix causes. Use it to choose a prefill region for Task 4's direct instrumentation. Commit tools and evidence.

## Task 4 — Isolate arithmetic and actual cache wiring

**Files:** create `tests/kolibri/f16_expert_gemm_parity.cu`; register in `CMakeLists.txt`; add strictly opt-in diagnostics to `src/kolibri/kolibri_engine.cu` only as required; test driver `tools/kolibri/test_f16_expert_trace.py`; update ledger.

**Interfaces:** parity CLI `--embd N --ff N --rows N --batch N`. Link `strata_prefill`, `strata_kernels` and GPU runtime. Use the **actual** header declaration: `Gemm::f16_batched(..., float* const* Yarr, ...)`, not a contiguous `float* Y`. Capture actual M=32 block/padding behavior (`Scratch::kResBlk`), device pointer arrays and stream ordering.

- [ ] Write failing parity tests for Q4_K gate/up **and Q6_K down**, both toy and model dimensions (extract `n_ff` from the model), rows 1/31/32/33 and multiple batches. Cover mixed signs, cancellation, tiny operands, zero padding and deliberately distinct experts. Register a small synthetic ctest; keep expensive real-shape tests explicit. A missing target is setup failure, not sufficient proof that the numerical regression assertion works.
- [ ] Implement three separated comparisons: (a) independent CPU dequantized weights and f32 input with a float64 dot-product reference; (b) float64 dot product over the **exact rounded half operands** actually sent to cuBLAS; (c) the actual batched GPU result. Report operand-rounding error (a vs b) separately from accumulation/implementation error (b vs c). Use per-dot `Σ|w*x|`, an additive underflow term when applicable, and K-dependent accumulation reasoning (`gamma_K = K*u/(1-K*u)`, u=2^-24 for the f32 control) instead of a fixed K=256 bound reused at real dimensions. Record the applicable cuBLAS arithmetic contract and justify the synthetic assertion before using it; no invented universal threshold.
- [ ] Confirm tests detect wrong-expert pointers, corrupted Q6_K offsets and missing/padded rows using deliberate test mutations. Pass all covered shapes after restoring correct code. A pass only validates these primitive cases.
- [ ] Add a bounded opt-in trace for the first suspect prefill layer/chunk: input activations, gate/up/hidden/down intermediates, selected expert ids, slot ids/ages, batch pointers, selection/scatter indices and output. Default-off must add no arithmetic changes or synchronization. Exercise the engine's **existing** selection/promotion loop, not a hand-written replacement.
- [ ] Check cache ownership invariants across multiple chunks and requests: every resident has the intended weights; slot assignments are distinct within the active chunk; a hit later in selection cannot be evicted by an earlier miss; every selected row is scattered once to its intended token/expert; padding contributes nothing. Test fewer/equal/more touched experts than slots, and already-full caches with a mixture of hits and misses. The latter is an **unconfirmed static-analysis risk**, not a claimed bug, until reproduced.
- [ ] Replay captured rows through the actual CPU expert path and resident path, including activation quantization, SwiGLU and hidden fp16 conversion. Compare each intermediate with independent f32/f64 controls. If the primitives pass but the integrated result differs unexpectedly, continue at routing/slot selection/scatter/reduction; do not jump to “expected fp16 loss.”
- [ ] Commit diagnostic tests and a causal report: earliest explained difference, remaining unexplained effects and minimal reproducer. Before changing any arithmetic, identify which observed difference the proposed fix targets.

**Exit:** a demonstrated defect, a supported explanation covering the large observed discrepancy, or a specific unresolved investigation. A synthetic pass alone is not an exit verdict.

## Task 5 — Fix demonstrated defects and prove non-regression

**Files:** only the code implicated by Task 4, regression tests, evidence ledger.

- [ ] For each demonstrated defect, write a regression that fails on the preserved pre-fix build. Implement the smallest opt-in-path fix; rerun the failing case and Tasks 1–4's relevant tests. Do not change rounding/precision merely to make a metric look better.
- [ ] Compare the new cache-off binary against the preserved `5bbe6c4e` control on identical histories/configurations: 16/511/512/513 prompt lengths, 8191 KV wrap and ~11000 attention transition. All full logits must be finite and bit-identical. Preserve explicit record counts and input/config hashes.
- [ ] Also run `tools/kolibri/decode_gate.sh check` for current arms/control, reuse and repeatability with cache off. This complements, not replaces, the cross-revision check. Investigate failures against a same-day control and distinguish known baseline failures from new ones.
- [ ] Run affected unit/portability tests and available affected-backend builds. List passed, failed and not run separately. Commit tested fixes and update the ledger.
- [ ] Whether a fix was needed or not, proceed to Task 6. Fixing an implementation bug does not qualify the changed arithmetic.

## Task 6 — Independent fidelity and state qualification (approval-gated)

**Files:** evidence ledger; extend/create a qualification driver under `tools/kolibri/` with tests if existing runners cannot preserve the required histories/results.

- [ ] Before collecting qualification results, agree numerical acceptance criteria and a fixed prompt/history corpus with the maintainer. Include several prompts, mixed expert formats, short and 512-token cases, 8191 and ~11000 contexts, and forcing beyond the EOS region where relevant. Existing diagnostic results remain diagnostic; do not retrospectively label them preregistered qualification.
- [ ] Verify reference tooling from Task 0. `tools/kolibri/kl.py` currently fixes `STRATA_DUMP_CTX=8192`, reports only a subset of required metrics, truncates comparisons with `min(len(P),len(Q))`, caches reference dumps by filename and deletes Strata full dumps. It cannot be used unchanged as the long-context qualification harness. Adapt it with tests, or use a separate driver: explicit sufficient context on **all** executables, exact record alignment, configuration-keyed caches and retention of raw logits. `--ref-args` is how this runner passes engine-specific flags; verify the actual candidate receives prefill/cache settings.
- [ ] Generate independent llama.cpp GPU and CPU dumps on identical forced histories for control, cache-only and cache+partial comparison. Record the actual offload/precision arrangement; llama.cpp GPU with CPU experts is not an all-GPU reference. If a trusted reference is unavailable, mark qualification blocked; `strata-kolibriref` shares kernels and is a diagnostic control only.
- [ ] Report full-distribution KL (mean/median/p95/p99/worst), absolute-error percentiles/worst, top-1, reference-winner rank changes and margins, by prompt/arm/position and aggregate. Separately compare plain-vs-reference and candidate-vs-reference: the current engine is a regression control, not perfect ground truth.
- [ ] Run two repeats plus first-use, replay, prefix reuse, reset, cancellation and a clean request after cancellation, for both cache-only and cache+partial. Compare like-for-like cold/reused histories; explain state-dependent arithmetic if residency changes. Reject non-finite values, stale outputs and unexplained variability. Do not demand cross-arm bitwise identity for deliberately different arithmetic.
- [ ] Publish numerical and state findings, explicitly separating pass/fail/inconclusive/blocked. Do not infer task quality from KL or top-1. Request approval to proceed to Task 7; leave defaults unchanged.

## Task 7 — Preregister and run task-quality evaluation

**Files:** create `docs/experiments/kolibri-t4-f16-task-quality.md`, then record its raw-result manifest and analysis.

- [ ] Before examining candidate **task-evaluation** outputs, document paired reasoning/coding/factual/long-context-retrieval prompts, fixed decoding settings/seeds, reference variability, evaluation sensitivity, sample size, uncertainty calculation and statistical decision rule. Diagnostic logit inspection in earlier tasks is not forbidden by this step. A nonzero acceptable regression margin needs explicit approval.
- [ ] Stop for maintainer approval of the protocol and margin. If not approved, mark this task pending; no evaluation run before approval.
- [ ] After approval, run the preregistered evaluation and report paired differences and uncertainty. Inconclusive is not pass. An accuracy failure rejects promotion regardless of speed.

## Task 8 — Measure speed separately and deliver the decision

**Files:** evidence ledger; `docs/KOLIBRI_CUDA_PREVIEW.md`, `docs/findings-log.md`, and `README.md` only where new measured facts warrant an update.

- [ ] After fidelity/quality approval (or with explicit permission for exploratory speed work clearly labeled unqualified), run unprofiled same-day control/candidate measurements on the T4. Use equal pinned NUMA placement, worker count, model/context/slot settings, prompt and requested generation length. Use a recorded ABBA order with at least two completed runs per arm; reject incomplete/EOS-shortened runs as directly comparable fixed-length timings.
- [ ] Record load, prefill, capture and steady decode separately, actual processed/generated token counts, VRAM and full configuration. Do not mix the 8-worker diagnostic recipe with earlier 24-worker timing claims.
- [ ] Deliver the root-cause/explanation, regression results, independent numerical and task-quality findings, speed data, limits and raw artifact locations. State unresolved work rather than promising a usable speedup.
- [ ] Keep opt-in/default promotion as a separate explicit approval decision. Do not push without permission. If any evidence gate is blocked, deliver its precise unblock requirement and keep the control unchanged.

## Final review checklist

- [ ] Every claimed mechanism is backed by current code or a retained run; static hypotheses are labeled unconfirmed.
- [ ] Primitive tests cannot be mistaken for end-to-end fidelity approval; repaired and unrepaired candidates pass the same qualification gates.
- [ ] The independent runner covers long contexts, exact alignment and retained raw results; every measurement records actual configuration and completed work.
- [ ] The cache-off build is compared with the preserved release control, not only itself.
- [ ] Defaults, server and upstream/oracle state are unchanged; approval stops and untested backends are explicit.
