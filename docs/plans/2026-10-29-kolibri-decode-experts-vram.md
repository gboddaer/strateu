# Kolibri decode: host-assisted scheduling implementation plan

> **For agentic workers:** Use `superpowers:executing-plans` for native execution, or
> `superpowers:subagent-driven-development` only if the operator explicitly chooses delegation.
> Implementation was authorized through goal `mvticbbf-mb0flp`. Defaults remain unchanged.
> The segmented stage first compared slower and was saved; the user directed continued investigation, and the
> whole-stack graph built on those segments now measures 15.6%/24.4% less decode time than handoff-plus-stream.

**Goal:** Improve exact single-stream decode on the R9700 without a drafter. First transfer
Flash-Next's host-assisted scheduling while preserving Kolibri's CPU expert arithmetic. Measure the
result before choosing a separate placement or kernel optimization.

**Architecture:** Actual Flash-Next can keep CPU misses inside its execution protocol: a captured
GPU segment publishes normalized input, ids and weights; CPU work overlaps independent shared/resident
GPU work; this layer's results are combined before the next layer begins. Start with two graph segments
per layer, then consider a whole-layer-stack graph with acknowledgement waits. **CPU misses do not need
to move to PCIe weight reads to make capture possible.**

**Tech stack:** Existing C++20 host code and project HIP/CUDA compilation settings, mapped pinned
memory, existing pool and elementwise primitives, Python serve-protocol harness, CMake/CTest.

**Spec:** [docs/KOLIBRI_DECODE.md](../KOLIBRI_DECODE.md).
The historical filename is retained so existing links continue to work; this no longer starts with VRAM.

## Status (2026-10-29)

- **Task 1 — done** (`ffc1e536`): `tools/kolibri/decode_bench.py` + 46 tests, the unprofiled baseline
  (six identical runs, 0.4 % spread) and the `DEC_XEXP` arm measured as a one-variable pair (−5.2 % short,
  −4.2 % long) with marker-proven activation.
- **Task 2 — done** (`41b2a91b`): `tests/hip/kolibri_decode_handoff.cpp` proves the per-layer
  publish/wait/answer protocol with a payload that changes on every replay, plus reset, 30/30 ring visibility
  with the graph still running, and a bounded no-ring failure. Negative control run (exit 12 in 5.1 s).
- **Task 3 — done** (`eb752473`, `057d42a6`): `STRATA_KOLIBRI_DEC_HANDOFF=1` measured −10.1 % (512 ids) and
  −7.5 % (8191 ids), bitwise against the plain engine through `tools/kolibri/decode_gate.sh` at
  16/511/512/513 ids and on the reused-prefix path, six repeats, both prefill profiles unchanged. Capture
  followed as `STRATA_KOLIBRI_DEC_HANDOFF_GRAPH=1`: device-position attention and the explicit stream keep the
  eager single-pass arithmetic, PRE/POST segments retain the CPU pool, and the same gate reports DECODE GATE
  PASS against the plain engine. Unsupported segment/debug combinations are refused by name.
- **Initial segment measurement, superseded**: three alternating direct pairs at 512 prompt ids / ctx 8192 /
  96 generated tokens gave 4657.7 ms/request for handoff-plus-stream versus 4728.8 ms/request with segmented
  graphs, 1.5 % slower with every pair losing. That was a prerequisite cost, not the final design; the user
  directed continued investigation, the patch and source were saved at
  `/tmp/kolibri-dec2/rejected-segments/` and restored, and the segments were then hardened and joined. The
  pre-patch records are kept at `/tmp/kolibri-dec2/graph-v-stream/` and are not presented as the final result.
- **Task 4 — done** (`e449dfbb`): `STRATA_KOLIBRI_DEC_TOKEN_GRAPH=1` assembles one parent executable per
  attention geometry from PRE → bounded acknowledgement wait → POST over all 50 layers, with the host servicing
  every publication during replay and CPU expert arithmetic unchanged. It requires HANDOFF, STREAM,
  HANDOFF_GRAPH and all segments, warms cold mirrors/buckets through segments, and refuses layer dumps.
  Final-source three rotating-order blocks per workload measure **23.91 tok/s** (512 ids) and **17.98 tok/s**
  (8191 ids) against 20.17/13.60 for handoff-plus-stream and 19.73/13.43 for segments: 15.63 %/24.38 % and
  17.46 %/25.29 % less decode time, every generated history matching. It is off by default.
- **Final verification**: the full decode gate passes against both plain and segmented controls; a final-binary
  8191-id KV-wrap pair (two runs per arm) is bitwise; the 11000-id attention transition is bitwise; a dependent
  eight-layer publication chain, seeded sequence/generation rollover, missing publication/ACK, STOP recovery,
  mapped and parent allocation failure and a CPU producer exception all pass without emitting a failed record;
  88 CPU/no-model tests pass; both prefill profiles pass with unmodified expectations; a fresh full build passes
  and CTest reports 188 tests = 171 passed / 11 skipped / 6 failed, all six reproduced at committed baseline
  `057d42a6`. Not established: no independent reviewer ran (delegation was not authorized), CUDA and SYCL were
  not built, and the occasional prefill KV variation has a deterministic test configuration rather than a
  proven vendor fix.

## Global constraints

- Card0 only: `HIP_VISIBLE_DEVICES=0`. No other GPU, drafter, MTP, speculative proposal or extra sequence.
- No promised tok/s target. The former 30–38 tok/s forecast and 42.6 ms spike threshold are withdrawn.
- New runtime arms default off. Existing prefill, serving protocol and off-arm outputs stay unchanged.
- Scheduling-only means bit-identical **full logits**, not merely finite output, equal ids or acceptable KL.
- Keep expert format, CPU kernel, quantization, selection order and combine arithmetic unchanged first.
- Do not use profiling as the primary speed measurement. Do not subtract load-inclusive process walls.
- Never record over either existing pool or MMQ prefill expectation files. Decode has a separate gate.
- Bound every child run; reap its own process group on timeout. Poll completion, never blind-wait for a run.
  On a crash, zero-block launch or unexplained failure, stop and report it rather than adding a timing fix.
- Benchmark without logits dumps or tracing. Run correctness dumps and diagnostic traces separately.
- Builds touched by shared code need backend compilation checks before review; list unavailable backends
  explicitly. No CUDA/SYCL/Windows or other-card pass is implied by a HIP/card0 run.
- Model: `/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`; build: `build-mmq/`;
  short fixture: `tools/kolibri/oracle_ids512.txt`.

## Source audit completed for this revision

- Flash-Next reference: `src/program/generate.cpp` selects `session_loop` / `session_run_token`;
  `src/core/session.cpp` contains the per-layer and whole-stack graph protocols. `q36_engine.cu` is
  a different model engine and is no longer the primary reference.
- `src/core/layer.cpp:moe_route` publishes work; `block_layer_pre` can then execute the shared expert.
- `src/kernels/cuda/elementwise.cu` implements publication, waits and mapped-result copying, including
  HIP visibility provisions. `tests/hip/handoff.cpp` already tests changing mapped payloads and replay.
- Kolibri already has `s.pos_d`, `k_attn_prep_p`, `k_attn_p` and `STRATA_KOLIBRI_DEC_XEXP`.
  Its current `graph=true` attention is not the same path/launch shape as eager attention. Do not silently
  substitute it while claiming a scheduling-only change.
- The GGUF has Q4_K gate/up in all layers, Q6_K down in 25 and Q4_K down in 25. The current grouped
  dispatch includes Q6_K down when built with `STRATA_Q6K_EXPERTS=ON`; it lacks Q4_K down.
  Grouped-expert type expansion is not needed for the CPU-preserving first arm.

## Review focus

1. **Stale layer/token payload:** the CPU must consume this layer's input and ids; this layer's combine
   must consume its results. Capture-time literals must not freeze token, position or sequence numbers.
2. **Visibility and buffer lifetime:** publish input before announcing readiness; publish results before
   acknowledging them. Do not reuse/free mapped buffers or update slots while readers are active.
3. **Failure during a wait:** missing publication/acknowledgement, CPU failure, cancel and reset must not
   leave an unbounded GPU wait or emit a token computed from stale results.
4. **Capture changes the math or state:** attention branch, sliding-window wrap, transposed-K mirrors,
   scratch aliases and first-use initialization must behave identically before and after replay.
5. **False speed or correctness pass:** exclude loading/prefill from decode timing; reject early EOF,
   malformed protocols, unaligned histories, incomplete logits and silently inactive candidate arms.

---

### Task 1: Reliable baseline and candidate comparison

**Files:**
- Create: `tools/kolibri/decode_bench.py`, `tools/kolibri/test_decode_bench.py`.
- Reference, do not replace blindly: `tools/kolibri/bench.py`, engine `serve_loop` and its `DONE` line.
- Update measured results only: `docs/KOLIBRI_DECODE.md`.

**Interfaces:**
- `parse_done(line: str) -> dict`: fields `produced`, `prompt_ids`, `prompt_ms`, `decode_ms`,
  `reused`, `hits`, `lookups`, `tok_per_s`; validate the actual engine field positions.
- `parse_token_times(times_ns: list[int]) -> dict`: completed interval count, mean/median interval
  milliseconds and elapsed-window tok/s. At least four intervals; strictly increasing timestamps.
- CLI: `decode_bench.py --bin PATH --model PATH --tokens-file PATH --ctx N --n N --threads N
  --pairs N --out DIR [--extra TEXT] [--candidate-env NAME=VALUE ...]`.
  A pair runs baseline/candidate with candidate controls unset/set in child environments; reverse order
  on alternate pairs. Each engine uses the same prompt/prefill/sampling settings and a fresh session.
  Loading may happen per process, but it is outside the reported `DONE` decode interval.

- [ ] Write failing parser tests: a valid `DONE` with 128 produced tokens and 8000 decode ms yields
  exactly 16 tok/s; zero/negative/NaN times, missing fields, unexpected EOF and nonzero exit all fail.
  Token timestamps at 0, 60, 120, 180, 240, 300 ms yield five 60 ms intervals and 16.6667 tok/s.
- [ ] Run `.venv/bin/python -m pytest tools/kolibri/test_decode_bench.py -q`; verify the tests fail
  because the new module does not exist, then implement the parser/harness and make them pass.
- [ ] Test harness cleanup with a fake child that never sends `DONE`: deadline reached, only its own
  process group terminated, logs retained, benchmark fails. Test an inactive candidate marker and an
  early-EOS request: neither may silently count as a successful fixed-length paired benchmark.
- [ ] Capture monotonic timestamps on flushed `T` lines, retain all intervals, and report both engine
  `DONE` throughput and the T-to-T window. Drain stdout/stderr concurrently. Keep prompt, decode and
  post-request reselection accounting separate; report first-token timing separately from steady decode.
- [ ] Add optional diagnostic trace parsing with explicit phase metadata (`prompt_forwards`,
  `generation_forwards`, reuse/prefill mode). Parse with `csv.DictReader`, not comma splitting. Test
  names with commas and `void (anonymous namespace)::k_row<12>` using `csv.writer`, shuffled rows,
  overlapping streams, phase boundaries and missing markers. Report kernel interval union and gap
  distribution; label large gaps as gaps, not as established driver cost. If phase boundaries cannot
  be established, refuse a decode attribution rather than guessing from the last 17 markers.
- [ ] Run an unprofiled baseline twice first, without any candidate. Record raw intervals, actual
  token counts and protocol fields. Do not require it to match the old profiled 14–17 tok/s claim.
  Assess the existing `DEC_XEXP` arm as a separate one-variable comparison before new overlap code.
- [ ] For later candidate runs, use at least three alternating pairs with these workloads:

  | Prompt ids | Allocated ctx | New tokens | Prefix state |
  |---:|---:|---:|---|
  | 512 | 8192 | 128 | cold request |
  | 8191 | 12288 | 128 | cold request |

  Assert fixture lengths and context capacity. Keep P4/chunk settings identical between arms. Record
  `EXPERT_F16`, pool mode, threads, full environment, model/binary hash, host load and memory state.
  Longer visible histories are a later acceptance check; a large allocated ctx is not a long history.
- [ ] Commit the instrument only after its CPU tests and baseline protocol checks pass.

### Task 2: Prove the publication protocol before model capture

**Files:**
- Extend: `tests/hip/handoff.cpp` and its existing `hip_handoff` CMake wiring if needed.
- New safety test if necessary: `tests/hip/kolibri_decode_handoff.cpp`, wired in `CMakeLists.txt`.
- Reference: `src/kernels/cuda/elementwise.cu`, `include/strata/kernels/elementwise.hpp`.

**Interfaces:**
- Reuse `doorbell_publish`, `doorbell_ring`, `copy_from_mapped` and their ordering contract.
- Request identity is `(token_generation, layer)`, never a value read from host memory during capture.
  Buffers and controls stay alive until completion. Sequence reset requires a quiescent stream.
- For the first per-layer-graph arm, no device acknowledgement wait is required: the host enqueues
  the post graph after its pool call. Whole-stack waits are deferred to Task 4.

- [ ] Run the existing `hip_handoff` test on card0 under an external deadline and retain its actual
  result; its existence is not proof that this driver/configuration passed today.
- [ ] Write failing tests for a 50-layer, multi-replay changing payload with N=2560 and top-k=6.
  Encode token and layer in inputs and outputs; assert exact values, ids and selection order every time.
  Include token reset, stale sequence, delayed publication and altered weights between replays.
- [ ] Implement only the additional synthetic test/control plumbing needed to pass. Preserve the
  existing HIP fences/coherence behavior; do not replace the protocol with a volatile-only assumption.
- [ ] Prove the host can observe publication while the pre graph has not yet finished independent
  work. Report ordering separately from timing; a final stream sync must not manufacture visibility.
- [ ] Test errors and cleanup with bounded children. If the handoff is not reliable, stop before model
  integration and investigate; no global serialize-kernel or per-layer sync workaround counts as a fix.
- [ ] Commit the passing synthetic proof separately. No new expert kernel or placement policy here.

### Task 3: Two graph segments per layer, CPU arithmetic unchanged

**Files:**
- Modify: `src/kolibri/kolibri_engine.cu` (Scratch/lifecycle, decode enqueue boundaries, forward dispatch).
- Create: `tools/kolibri/check_decode.py`, `tools/kolibri/test_check_decode.py`.
- Retain existing `tools/kolibri/check_repeat.py` and both `pf_gate.sh` profiles unchanged.

**Interfaces:**
- New opt-in `STRATA_KOLIBRI_DEC_HANDOFF=1`; reject unsupported combinations explicitly before work.
  First supported combination is the CPU pool, no routed VRAM slots, no grouped-expert override and
  no FP16 decode residency. Prefill still uses its existing configuration.
- Model-specific adapter stays in Kolibri. Do not import Qwen's SessionState or its expert math wholesale.
  Per-layer order: attention/FFN norm/router → publish → independent shared expert → pool-result
  join → existing combine/residual. CPU runs the existing `pool_experts` on the published payload.
- `check_decode.py --baseline FILE --candidate FILE --vocab N --records N`: require exact byte counts,
  finite float32 values and bitwise equality at each aligned forced position. No truncating to the
  smaller record count, and no unforced cross-arm KL comparison.

- [ ] First add comparator tests: identical arrays pass; one changed bit, NaN, trailing/missing record
  and mismatched vocab each fail. Run them red, implement the comparator, then run them green.
- [ ] Establish baseline full-logit outputs for identical forced histories before editing decode.
  Write an activation test that fails when the requested arm is ignored. Record first-token and
  subsequent replay results separately; the latter is what catches frozen capture arguments.
- [ ] Audit all position-dependent state before capture: `s.pos_d`, RoPE/store indices, attention
  visible range, sliding rings, transposed-K first-fill/increment, scratch reuse and graph lifetime.
  Preserve eager arithmetic. Dynamic positions must not freeze launch bounds or cause a stale
  first-fill path. If graph buckets are needed for unchanged attention, key and invalidate them explicitly.
- [ ] Refactor only the opt-in path into capturable GPU pre/post segments. Keep the off path intact.
  Put arm GPU work on an explicitly ordered stream or use explicit events; legacy/default-stream
  side effects are not a synchronization contract. Prewarm lazy dispatch/handles without retaining
  warmup KV, counters or mirror readiness in the real request.
- [ ] Wire publication after routing, CPU computation on current-layer payload, and result transfer
  before this layer's post graph. Shared-expert work may run after publication while the CPU computes.
  Do not feed layer L results to layer L+1, or change expert quantization/combine order.
- [ ] Gate complete full logits and hidden/KV state where needed: tiny prompts; a chunk boundary;
  512- and 8191-id prompts; sliding-window wrap; extended/reused prefix; cold graph creation; repeat
  and changing requests. Explicit off must match the pre-change binary. Arm on must match aligned
  eager full logits. If it cannot, classify it as a numerical change and stop for a separate design.
- [ ] Test cancel/reset, initialization failure and allocation failure. No partial token is emitted
  as successful, buffers stay alive until GPU users finish, and only owned processes are terminated.
- [ ] Require the activation assertion and six same-arm full-logit repeats. Run both existing prefill
  gates against their unchanged expectation files, including plain oracle/reuse, then full CTest at
  the resulting test count. Reproduce and list any baseline failures; do not automatically excuse a
  newly failing test because an older report also had four failures.
- [ ] Use Task 1's unprofiled alternating pairs. Continue only when all three short-workload pairs
  favor the candidate, median decode-time reduction is at least 5%, and no tested workload's median
  regresses more than 2%. These are proposed engineering acceptance rules, not measured gains.
  If noisy/inconclusive, collect more paired data or defer; do not tune the rule after seeing results.
- [ ] Commit the opt-in arm and ledger. A default change remains a separate final acceptance decision.

### Task 4: Conditional whole-layer-stack graph, still with CPU misses

**Files:**
- Modify: `src/kolibri/kolibri_engine.cu` and the dedicated handoff/decode tests above.
- Reference: `session_capture_token` / `session_run_token` in `src/core/session.cpp`.

**Interfaces:**
- New `STRATA_KOLIBRI_DEC_HANDOFF_GRAPH=1`, requiring the proven handoff arm.
- The CPU services each layer during replay. This is not a host callback inside capture and it does
  not require misses to move to GPU weight reads. Embedding and logits/sampling can remain outside.
- A wait/abort protocol is required before capture: on CPU error or cancel, all outstanding waits can
  terminate, the token is invalidated, and no normal token output is emitted. The existing unbounded
  `doorbell_wait` alone is insufficient for deliberate missing-ack tests. Do not add a waiting graph
  before a bounded failure path is designed and tested.

- [ ] Write failing synthetic multi-layer/replay tests for acknowledgements, withheld ack, stale ack,
  CPU failure and cancel while the graph waits. Require bounded termination without freeing live buffers.
- [ ] Implement and validate the abort-aware wait/control path, then capture the already-proven
  per-layer work. Reset counters only after prior completion; prevent wraparound/stale replay values.
- [ ] Run the full aligned-logit, state, lifecycle and repeatability suite from Task 3 against both
  eager and segmented-graph execution. Keep expert arithmetic and cache placement unchanged.
- [ ] Benchmark unprofiled against **both** eager and segmented execution. A kernel-gap plot with a
  long spin kernel is not success. Keep this stage only if it improves the measured critical path.
- [ ] Build/validate touched backends and document unavailable checks; run full gates/CTest again.
  Decide deployment/default scope only after correctness and paired performance have passed.

## Follow-up decisions, not prerequisites

- Residency/grouped GPU hits: measure per-decode hit/miss deltas at each context and evaluate overlap
  with CPU misses. Support Q4_K-down correctly before claiming all-layer grouped coverage. Keep a
  separate numerical gate if compute moves from CPU to GPU. Static slot updates happen only when safe.
- Direct GPU reads of pinned expert weights: optional PCIe arm with measured card0 traffic/latency;
  do not reuse the T4's bandwidth figure or assume scattered reads achieve bulk DMA bandwidth.
- Attention/full-kernel, dense/shared fusion and CPU-format kernels remain open after the new profile.
- No drafter or speculation is included. No engine implementation, GPU benchmark or default change was
  performed while writing this reassessment.
