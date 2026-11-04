# Kolibri GPU-streamed expert prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (when delegation is explicitly authorized) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Beat the tuned CPU expert prefill workload on the T4 with bit-identical grouped GPU streaming, then demonstrate an opt-in end-to-end prefill improvement without changing generation or defaults.

**Architecture:** Stable expert grouping, tiled Q4_K/Q6_K products and two event-driven bounded batch slots. Preserve CPU SwiGLU, `native_quant_h`, and original selection-order output placement. A representative transfer-inclusive replay benchmark is the integration gate.

**Tech Stack:** C++17, CUDA sm_75, existing native AVX2/ggml-cpu kernels, CMake/CTest, Python 3/pytest, numactl.

**Spec:** `docs/superpowers/specs/2026-10-31-kolibri-gpu-streaming-design.md` (read completely).

## Global Constraints

- Bit-identical finite full logits are mandatory for unchanged arithmetic. No numerical-change approval exists.
- Reproduce the Q4_K/Q6_K accumulation contracts; CPU SwiGLU and `native_quant_h` stay unchanged.
- New flag `STRATA_KOLIBRI_EXPERT_STREAM=1`; absent or `0` leaves existing execution untouched.
- No generation path changes, no default changes, no pushes and no HTTP/network server starts.
- One GPU job at a time; poll progress in short intervals. Preserve all previous artifacts read-only.
- Separate implementation worktree, build and evidence tree.
- Protected: FP16 branch at e5f4815c, `/data4/kolibri-f16-evidence`, `/data2/develop/f16-control-5bbe6c4e`, parked `/data2/develop/native-q4q6` at 7c600d48, released defaults and their builds.
- Initial supported target: CUDA sm_75 on this model. Other backends retain their current path when the flag is off; setting it on an unsupported backend fails clearly.
- At most eight experts per batch, 32 MiB compressed weights per slot, two reusable slots, 256 MiB maximum new device allocation. Check sizes and free-memory headroom before allocation; never shrink KV or context to fit.
- No steady-state cudaMalloc/cudaFree, cudaHostAlloc/cudaFreeHost, cudaDeviceSynchronize or per-selection event/wait loop.
- Kernel/pipeline parity precedes representative timing. Replay GPU/CPU median must be <= 0.90 on both short- and long-prompt distributions before engine integration. This is a proposed engineering go/no-go margin, not an accuracy tolerance.
- Shared-file edits require backend build validation before review, or explicit NOT RUN status and no review-ready claim.
- Engine stdio `--serve` subprocesses for state tests are not network services; terminate them after each test.

## Review Focus

1. Group tails, singleton experts and groups larger than a tile: every original selection must be written exactly once without reordering top-k reduction (Tasks 1–4).
2. Reused slot buffers and skewed workloads: no overwrite until final copy completes, and no head-of-line wait on an unready slot when another can advance (Tasks 3–4).
3. Cancel/reset/context growth and final partial chunks: no stale output, use-after-free, leaked pinned memory or contaminated next request (Tasks 5–6).
4. Unsupported format/backend, conflicting flags, overflow and insufficient VRAM: fail clearly before publication, never report streamed coverage while falling back (Tasks 1, 3, 5).
5. Misleading performance evidence: no per-job thread creation, profiled-vs-unprofiled comparisons, summed worker time as wall time, cumulative-timer summation, or excluding CPU activation/copies from GPU time (Tasks 1, 4, 7).

---

## Read first and establish the handoff location

These documents are authored on the clean linked worktree `/data2/develop/perf-gap`, branch `kolibri-a1-prefill`. The implementation parent inspected for this recipe is:

```
f076741b453ce14666a868b635e04c3dfa6be3a8
```

The documentation commit containing this plan may be used as implementation base if its only changes from that parent are these design/plan files. Do not use `/data2/develop/strateu-publish` for writes: it is the protected FP16 checkout, despite being the harness's default cwd.

Implementation startup (do not execute during a documentation-only task):

```sh
BASE=$(git -C /data2/develop/perf-gap rev-parse HEAD)
git -C /data2/develop/perf-gap diff --stat f076741b453ce14666a868b635e04c3dfa6be3a8 "$BASE"
# Verify only the two handoff docs changed before proceeding.
git -C /data2/develop/perf-gap worktree add -b kolibri-gpu-streaming /data2/develop/kolibri-gpu-streaming "$BASE"
cd /data2/develop/kolibri-gpu-streaming
export ROOT=$PWD
export BUILD=$ROOT/build-stream-cuda
export EVIDENCE=/data4/kolibri-gpu-streaming
export MODEL=/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf
# Refuse an occupied branch/path. Never reset or remove someone else's worktree.
mkdir -p "$EVIDENCE"
```

If any path already contains a different campaign, choose a new suffixed path and record it; do not overwrite old evidence. Read local AGENTS.md, `docs/AI_REFERENCE.md`, `docs/KOLIBRI_ACCURACY_POLICY.md` and the spec. Use the execution, TDD and verification skills. No subagent delegation is implied by this plan; ask for authorization if desired.

Configure a NEW build, never copy CMakeCache.txt from another worktree:

```sh
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_SYCL=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=75 -DSTRATA_NATIVE_EXPERTS=ON \
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_DSV41=OFF
cmake --build "$BUILD" --target strata-kolibri -j8
ctest --test-dir "$BUILD" -N
```

Record configure/build logs and enumerate/run relevant existing Kolibri tests. Stop on unexplained baseline failure. Establish a same-day arm-off baseline before GPU implementation. CMake may fetch dependencies; reuse a compatible read-only dependency source through supported CMake options if available, not another worktree's libraries. Do not benchmark CPU kernels linked from a different build as the control.

### Existing source landmarks

- `src/kolibri/kq_a1.cuh`: proven dot mapping and fixed trees; **retain as the slow independent CUDA comparator**.
- `probe/q4k_multiblock.cpp`, `probe/q4k_parity.cu`, `probe/q6k_parity.cu`: existing real-CPU oracles; call `ggml_cpu_init()` before ggml trait use.
- `include/strata/kernels/cpu/native_expert.hpp`: `NativeFmt`, `native_gu_rows`, `native_down_rows`, `native_quant_h`.
- `src/kernels/cpu/kq_avx2.cpp`: CPU Q4_K lane arithmetic. `src/kernels/cpu/native_expert.cpp`: CPU activation expression and down dispatch.
- `src/kolibri/kolibri_engine.cu`: `a1_run_selections` around line 2990; `forward_multi`; grouping arrays `cnt`/`ord`/`pool_list`; expert dispatch near line 4015; scatter destination `A.ey_h`; `Scratch` and request lifecycle. Line numbers drift: use symbols.
- `CMakeLists.txt`: `strata_kolibri` / `strata-kolibri` targets near line 1010, native CPU library near line 1931. New CUDA-only source must not be added to HIP unconditionally.
- `tools/kolibri/state_check.py`: existing stdio lifecycle test; token checks alone do NOT prove full-logit state parity.
- `docs/experiments/kolibri-perf-gap.md`: historical record. Read, do not rewrite it to erase negatives.

## New file map and shared interfaces

All names below are new unless marked Modify. Keep each component focused; do not refactor the large engine file beyond guarded integration.

| File | Responsibility |
|---|---|
| `src/kolibri/expert_stream_plan.hpp` | CUDA-free grouping and checked capacity planning |
| `src/kolibri/expert_stream.hpp` | Host-only public layer view, config, context and statistics API |
| `src/kolibri/expert_stream_kernels.cuh` | Tiled CUDA product kernels preserving A1 arithmetic |
| `src/kolibri/expert_stream.cu` | Slot ownership, grouped launches, transfers, events and CPU activation bridge |
| `tests/kolibri/expert_stream_plan_test.cpp` | Host grouping/capacity tests |
| `tests/kolibri/expert_stream_kernel_test.cu` | Tiled dots, activation/quantization and expert parity |
| `tests/kolibri/expert_stream_pipeline_test.cu` | Scheduler/reuse/error/ownership tests |
| `tools/kolibri/expert_stream_replay.cu` | Capture reader, persistent CPU control and full GPU pipeline timing |
| `tools/kolibri/expert_stream_capture.hpp` | Versioned capture serialization shared by engine hook and replay |
| `tools/kolibri/expert_stream_campaign.py` | Manifest, isolated run environments, forced-history and timing orchestration |
| `tools/kolibri/test_expert_stream_campaign.py` | Harness tests without GPU/model |
| `tests/kolibri/expert_stream_state_test.py` | Full-logit state-transition regression tests |
| `docs/experiments/kolibri-gpu-streaming.md` | New decisions/results/negative findings ledger |
| Modify `CMakeLists.txt` | Guarded targets, linking and CTest registration |
| Modify `src/kolibri/kolibri_engine.cu` | Capture hook, guarded opt-in dispatch, Scratch-owned context and lifecycle |

Define the following host interfaces before downstream tasks. They deliberately contain no CUDA types:

```cpp
namespace strata::kolibri::stream {
struct Group { int expert; std::vector<int> selections; };
std::vector<Group> make_groups(const int* ids, int selections, int top_k, int experts);
struct Config {
    int experts_per_batch = 8;
    size_t weight_bytes_per_slot = 32u << 20;
    size_t device_budget_bytes = 256u << 20;
};
struct LayerView {
    strata::kernels::cpu::NativeFmt fmt;
    const uint8_t* weights; // expert_count blobs at weight_stride
    size_t weight_stride;
    int expert_count;
    const uint8_t* token_acts; // selection_count/top_k rows, stride fmt.act_bytes
    const int* expert_ids;    // one expert ID per selection
    int selection_count;
    int top_k;
    float* output;           // [selection_count][fmt.n_embd], ORIGINAL selection order
};
struct Capacity {
    size_t device_bytes, pinned_bytes, weight_bytes_per_slot;
    int max_selections;
};
Capacity required_capacity(const LayerView&, const Config&);
struct Stats {
    uint64_t groups, selections, weight_h2d_bytes, activation_h2d_bytes;
    uint64_t gu_d2h_bytes, hidden_h2d_bytes, output_d2h_bytes;
    uint64_t kernel_launches, slot_waits, device_allocations;
    double wall_ms, cpu_activation_ms;
};
class Context { // complete host declaration; definitions live in expert_stream.cu
public:
    ~Context(); // drains in-flight work before destroying Impl
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
private:
    struct Impl;
    explicit Context(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
    friend std::unique_ptr<Context> create(const LayerView&, const Config&);
    friend Stats run(Context&, const LayerView&);
};
std::unique_ptr<Context> create(const LayerView& capacity_view, const Config&);
Stats run(Context&, const LayerView&); // synchronous host-output publication; async internally
}
```

Keep the complete Context declaration in the host header and define its constructor/destructor out of line, so `unique_ptr<Context>` is safe in the engine without exposing Impl or CUDA types. Invalid arguments throw `std::invalid_argument`; allocation/runtime failures throw `std::runtime_error`. Catch at engine boundary, print `kolibri: expert streaming refused: <reason>`, exit 2 before publishing incomplete results. Never throw through a CUDA callback or leave work using freed buffers.

`make_groups` orders groups by ascending expert ID; selections within each group are ascending original indices. Selection count must be nonnegative and divisible by top_k; top_k/expert count positive, IDs in range. Empty input performs no GPU work. Validate `NativeFmt` types, Q8_K activations, strides/offsets/tail/rotation flags and checked-size arithmetic before allocating. Unsupported nontrivial tails/rotations are refused in this initial path.

### Task 1: Reproducible workload capture and a trustworthy CPU control

**Files:** New plan/capture headers, host test, replay executable, campaign script/tests and ledger; Modify CMake and the engine only for a disabled capture hook.

**Consumes:** NativeFmt and existing `cnt`/`ord` mapping at the pool boundary.
**Produces:** `make_groups`, `required_capacity`, capture schema v1, replay `--cpu` and campaign manifests.

- [ ] Write `expert_stream_plan_test.cpp` first. Assert for IDs `{2,0,2,1,0,1}`, top_k=2: groups are `0:{1,4}`, `1:{3,5}`, `2:{0,2}`. Assert zero selections succeeds, bad IDs/zero top_k/nondivisible counts fail, and size overflow is rejected before allocation. Add a >64-selection single-expert group to pin no fixed local pointer-array limit.
- [ ] Add the CMake host target/test `kolibri_expert_stream_plan_test`; build/run it red before implementation, green after.
- [ ] Implement plan helpers and exact capacity accounting, including both slots, descriptors, all device/pinned arrays and alignment. No CUDA dependency in this test.
- [ ] Define binary capture v1: fixed magic/version/endian, manifest plus length-delimited per-layer/chunk records containing geometry/format, original selection IDs, already-quantized token activations, unique touched blob bytes, source stride, CPU expert output and SHA256 checksums. Do not serialize pointers, padding in NativeFmt, or borrowed signs arrays. Reader rejects truncation, bad version/checksum, invalid IDs/offsets and impossible sizes.
- [ ] Write campaign tests `test_rejects_truncated_capture`, `test_no_inherited_experimental_flags`, `test_failure_is_not_a_timing`, `test_manifest_pins_model_build_and_inputs`, `test_final_pfchunk_is_not_summed`. Run pytest red, implement, run green. Parse final cumulative PFchunk time only; preserve raw logs.
- [ ] Add opt-in capture hook `STRATA_KOLIBRI_STREAM_CAPTURE=<new-directory>` around the CPU expert boundary. Capture buffers before execution and outputs after the CPU pool drains, before weighting. Outside this opt-in, no capture allocations or file operations. Capture is never enabled in speed runs.
- [ ] Implement replay `--capture DIR --cpu --cpu-threads 14 --repeat 3 --json OUT`. Build a persistent worker set once, mirroring engine expert partitioning and native calls (including <=64-row CPU call chunks); thread startup is outside repeats. Do not substitute serial CPU timing or thread-per-group timing. Compare CPU output bytes with captured output before timing. Pin same NUMA policy and CPU features.
- [ ] Capture all layer/chunk boundaries from one short and one 8191-ID forced-history CPU run. Read only touched weights per record; deduplicate blobs in the capture store and reference them by checksum. Cap disk use before starting and record actual bytes. Loader/D2D setup and disk reads are outside timed replay; GPU H2D from host is inside. Do not capture into `/data4/kolibri-perf0`.
- [ ] Record build/model/input hashes, GPU/driver, `/proc` NUMA binding, flags, chunk geometry and group-size histogram, copy bytes and CPU wall distribution. Cross-check replay CPU wall with instrumented engine expert wall; explain a mismatch before using replay as a gate. Run captures twice and require identical CPU outputs for each record.
- [ ] Commit: `[Benchmark] capture Kolibri expert groups and establish streamed replay control`.

**Verification commands:**

```sh
cmake --build "$BUILD" --target kolibri_expert_stream_plan_test expert_stream_replay strata-kolibri -j8
ctest --test-dir "$BUILD" -R '^kolibri_expert_stream_plan_test$' --output-on-failure
python -m pytest tools/kolibri/test_expert_stream_campaign.py -q
numactl --cpunodebind=1 --membind=1 "$BUILD/expert_stream_replay" \
  --capture "$EVIDENCE/capture-short" --cpu --cpu-threads 14 --repeat 3 \
  --json "$EVIDENCE/replay-cpu-short.json"
```

### Task 2: Tiled grouped dots, gated by the real CPU oracle

**Files:** New kernels header/kernel test; Modify CMake.

**Consumes:** Group layout and NativeFmt; A1 lane mapping as specification, real CPU implementations as oracle.
**Produces:** internal grouped gate/up and down launch wrappers used by Context.

- [ ] Add failing `kolibri_expert_stream_kernel_test` cases before tiled implementations. Test nb={1,2,10,32}, output rows={1,15,16,17,512,2560}, token counts={1,2,7,8,9,13,31,64,65}. Use a bounded pairwise matrix plus the exact model geometries, not an enormous Cartesian allocation. Include negative inputs, valid extreme scales, zero blocks, nonaligned source offsets permitted by the format and incomplete output tiles.
- [ ] CPU reference: Q4_K against real `kq256_rows`; Q6_K against initialized ggml AVX2 vec_dot. Also compare selected cases against A1, without treating A1-only agreement as a pass. Initialize ggml once.
- [ ] Implement grouped row/token tiling. Initial tile is 16 output rows x 4 tokens; handle tails with masks valid for every participating shuffle/barrier. Share weight bytes/decoding across tokens; keep per-output per-lane floating-point accumulation in block order. Shared-memory staging is cooperative across the CTA, including masked threads. No divergent barrier or invalid shuffle masks.
- [ ] Use signed DP4A for four-value integer products where it preserves the integer lane mapping (decoded Q4/Q6 unsigned magnitudes fit signed int8). Keep scalar reference variant for comparison. Do not use tensor-core floating-point products or reduce FP32 across block partitions. Keep explicit FMA and final trees; do not compile with fast-math.
- [ ] Test packed gate/up outputs, then CPU SwiGLU and `native_quant_h`, then mixed-format down and original-index scatter. Require finite output, exact float bit patterns and exact hidden-quantization bytes. Add mutation tests for Q4 minimum sign, Q6 scale pair and Q6 ql offset; all must be caught.
- [ ] Run Compute Sanitizer memcheck/racecheck/synccheck on bounded cases, including row and token tails. These are correctness runs, never speed evidence.
- [ ] Inspect compiler register/spill/shared-memory reports and record them. Only tune 16x4 to alternatives after parity; run the same tests after every tile change.
- [ ] Commit: `[Engine] add bit-exact tiled Q4_K and Q6_K streamed expert products`.

**Verification:** build target `kolibri_expert_stream_kernel_test`; run its `--selftest`, then `compute-sanitizer --tool memcheck`, `--tool racecheck`, and `--tool synccheck` with `--error-exitcode 1` on `--tails-only`. All comparisons must be 100% bit-identical and finite; all sanitizer error counts zero.

### Task 3: Persistent double-buffered streaming pipeline

**Files:** New public API, `expert_stream.cu`, pipeline test; Modify CMake/replay.

**Consumes:** Task 2 launch wrappers; Task 1 stable grouping/capacity.
**Produces:** `create`, `run`, Context lifecycle and Stats.

- [ ] Add failing pipeline tests for empty input, skewed groups, mixed down formats across calls, slot reuse for 100 alternating calls, and poisoned buffers between calls. CPU expected results are computed from native kernels.
- [ ] Add failure tests for insufficient capacity, >budget calculation, unsupported type/stride/rotation, invalid IDs and partial allocation failure. A test-only allocator fault injection fails each allocation ordinal and verifies all earlier allocations/events/streams are released.
- [ ] Allocate two bounded reusable slots at create time, with pinned host staging and explicit streams/events. Batch up to eight groups/32 MiB of unique weights. Validate budget against available VRAM with recorded remaining headroom; no allocation after the context is warm.
- [ ] Implement the lifecycle from the spec. Upload each expert blob once per layer/chunk. Gather token activations by `sel/top_k`; process all token tiles without reuploading expert weights. Batch GU copy-back, exact CPU expression and native quantization, hidden upload and down copy-back. Scatter only to original selection indices. No result weighting here.
- [ ] Advance any ready slot using event query; block on the next necessary event only when neither slot can advance. Keep pageable source mappings alive; copy from pinned slot staging for genuinely asynchronous DMA. All H2D/D2H of the algorithm and CPU activation time count toward run.wall_ms.
- [ ] Implement quiesce/destruction/error cleanup: stop submitting on error, drain all live events/streams before releasing borrowed inputs or pinned memory, then propagate error. `run` returns only when all original outputs are ready for the engine's existing host-to-device publication.
- [ ] Keep initial CPU activation implementation serial. Measure it. If it becomes the limiter, introduce a persistent host executor with per-token independent native_quant_h calls; test that parallel output is bit-identical. Never invoke the stream coordinator inside an occupied worker-pool callback.
- [ ] Stats must count each transfer direction and actual unique weight bytes, launches, allocations and waits. Assert warmed second run device_allocations=0, weight bytes equal sum of unique touched blobs, and kernel launches scale with batches/tiles rather than one launch per selection.
- [ ] Run pipeline parity and bounded sanitizer tests, then replay every captured record through GPU and compare exact bytes to CPU. No representative timing until every record passes.
- [ ] Commit: `[Engine] pipeline grouped expert transfers with reusable CUDA slots`.

**Verification:** `kolibri_expert_stream_pipeline_test --selftest`; sanitizer tail/reuse cases; replay `--capture DIR --compare --json OUT` for both captures. `--compare` refuses non-finite outputs and exits nonzero on first mismatching element, recording record/group/selection/row and hex bits.

### Task 4: Transfer-inclusive replay go/no-go gate

**Files:** Modify replay/campaign/tests and new ledger only; no engine execution arm yet.

**Consumes:** Validated complete pipeline, representative captures and persistent CPU control.
**Produces:** reproducible GO or NO-GO record, raw paired timings and conditional traffic model.

- [ ] Add tests pinning gate calculation: GPU/CPU={0.89,0.90} passes, 0.91 fails; missing/failed/correctness-invalid repetitions do not pass; all captured records must be covered. Store the rule before examining candidate timings.
- [ ] Extend replay flags: `--gpu`, `--compare`, `--repeat N`, `--cpu-threads N`, `--json PATH`. Campaign runs CPU/GPU in alternating order for at least three paired repetitions per distribution, one GPU process at a time. Include one separately labeled cold call and unmeasured warmups; primary gate uses steady repeats. Preserve cold result for final reporting, not discard it silently.
- [ ] Time from grouping/gather start to final output-ready scatter, including CPU activation/quantization, host staging, all DMA, waits and launches. Capacity allocation belongs to separately reported first-use cost. CPU control includes equivalent grouping and scatter; thread creation and disk I/O belong to neither timed arm.
- [ ] Disable phase timers, sanitizer/profiler/capture hooks and debug synchronizations in timed runs. Record both CPU/GPU process environments, binding and hardware load. Poll subprocess logs without long silent calls. Abort overlapping jobs rather than share the T4.
- [ ] Record total replay wall; per-record distribution; weight, activation and output bytes; cold vs warm behavior; launch count; observed copy throughput and CPU activation time. Use a separate instrumented run/trace to verify overlap and identify stalls. Label instrumented results separately; do not sum overlapping CUDA event intervals into wall time.
- [ ] Bounded tuning: measure tile 16x4 then 16x8, and batch limits 4 then 8 experts if capacity permits. Change one dimension at a time, rerun parity before speed, retain all results. Do not expand indefinitely. If none passes GPU/CPU <=0.90 for BOTH capture distributions, publish NO-GO and stop before Task 5.
- [ ] If GO, record the selected configuration and exact executable hashes. Recompute the traffic lower bound from measured bytes/copy rate; label it conditional, not a universal ceiling. Do not promise 158–175 tok/s.
- [ ] Commit: `[Benchmark] gate streamed expert integration on paired replay speed`.

### Task 5: Opt-in engine integration with unchanged off path

**Files:** Modify engine and CMake; extend host/pipeline and campaign tests.

**Consumes:** Task 4 GO configuration and synchronous-host-output run API.
**Produces:** new flag, Scratch-owned reusable context and guarded prefill dispatch only.

- [ ] Write failing startup tests for flag absent/0, flag 1, flag conflict with A1, unsupported backend, incompatible expert-cache/MMQ modes and unsupported layer format. Enumerate real conflicts by reading `forward_multi` and startup config, not by relying on a guessed list. Explicit refusal must name the conflicting flags/format and exit 2 before partial results.
- [ ] Add CUDA-only target source/linking and a compile definition `STRATA_KOLIBRI_EXPERT_STREAM_AVAILABLE` only when CUDA and native experts are available. HIP/SYCL builds must not include CUDA headers/objects via common paths. They must keep off behavior; if the flag is requested, refuse before starting a request.
- [ ] Add guarded Context ownership to Scratch with out-of-line destruction as needed. Size once for maximum supported internal chunk selection count; prevalidate request context/format. On explicit capacity growth, quiesce then reallocate outside hot loops and separately report it. Refuse budget violations; do not alter context or allocation defaults.
- [ ] Add dispatch outside `e.pool.run`: when the new flag is true, populate LayerView from `b.fmt`, `b.table`, `b.stride`, `s.act_cur`, `A.ids_h`, NKc and K and call run. Otherwise execute the existing CPU/A1 block unchanged. Do not run the coordinator on shard zero while parking the rest inside the pool.
- [ ] Preserve `A.ey_h` publication, expert weighting, top-k sum order, dense/shared expert work and the generation path exactly. Do not set `s.mmq_gout`: the new path returns host expert outputs, not preweighted GPU results.
- [ ] Add a coverage banner/counters for streamed groups, selections and per-layer formats. Assert every requested expert product used the new pipeline; no cache bypass or silent fallback. Do not increment CPU-pool work counters as if GPU FLOPs were CPU work.
- [ ] Test short and final partial chunks, both ActSet buffers, context growth and alternating formats against CPU bytes. Build CUDA; build every affected backend before review or explicitly record NOT RUN and do not call the branch review-ready.
- [ ] Commit: `[Engine] expose grouped GPU expert streaming as opt-in Kolibri prefill`.

### Task 6: Full-logit and lifecycle qualification

**Files:** New campaign orchestration/state test; modify existing state harness only if a narrowly scoped reusable hook is necessary; update ledger.

**Consumes:** Integrated engine and preserved/pinned-parent controls.
**Produces:** immutable per-history arm-off/on/repeat evidence and full-logit state checks.

- [ ] Test campaign run-environment construction without a GPU. Start from a sanitized environment; remove inherited `STRATA_*` experimental overrides and explicitly set only common deterministic settings plus the new arm flag. Same model/context/thread count/input for each pair. Never reuse the rejected phase2a runner unchanged: it sets another experimental arm.
- [ ] Copy and hash the six input histories from `/tmp/f16-t0/inputs` into the new evidence tree: ids16.txt, ids512.txt, ids512_de.txt, ids512_code.txt, ids8191.txt, ids11000.txt. Also preserve forced_ids.txt and record the existing qualification's forced output sequence. If temporary files are missing, locate immutable prior copies and verify checksums; do not invent replacements under old fixture names.
- [ ] Make one pinned-parent binary using a separate clean parent worktree/build, or verify the hash/configuration of an existing read-only parent binary. Do not rebuild protected binaries. Arm-off must match parent and applicable preserved-control full-logit files; arm-on must match arm-off on all six histories. Run arm-on twice independently.
- [ ] Add a strict comparison helper that checks expected record count/vocabulary, byte lengths, finiteness and float bit patterns (not just max error). Use 9x128000 records only where the fixture really produces that count; a short output file is a failure. Hash entire files, not hash prefixes, in manifests.
- [ ] Run full-logit dumps with the existing engine syntax, directly, not an invented `generate` subcommand. Example template below. Each run gets unique stdout/stderr/logit files; check exit status before comparison. No timing claims from these instrumented runs.
- [ ] Run existing `state_check.py` with the arm flag and inherited numactl binding. It launches the engine's stdio serve protocol; it must not bind a network port. Existing token-only PASS is supplemental evidence.
- [ ] Add full-logit lifecycle tests: first request vs replay; extended prefix-reuse vs fresh-process same extended history; reset then standard vs fresh standard; cancellation while streaming is in flight, then a clean standard request vs fresh standard. Add repeated cancel/reset loops and forced slot delays for deterministic race exposure. Compare aligned forced histories, not divergent greedy outputs.
- [ ] If stdio protocol cannot dump full logits per request, add a test-only in-process engine hook using the same forward/state API with separate dump destinations; do not claim lifecycle full-logit parity from CLI fresh runs alone. Document exactly which state transitions are covered. Keep hooks out of normal execution.
- [ ] On any finite mismatch, state corruption or variability, stop speed qualification and debug; do not replace exact gates with a KL tolerance. Repeat sanitizer reuse/cancellation cases after fixes.
- [ ] Commit: `[Test] qualify streamed Kolibri logits and request lifecycle`.

**Forced-history command template** (replace paths with copied fixtures and unique outputs; use ctx >= input length plus forced output length):

```sh
STRATA_GEMM_DETERMINISTIC=1 STRATA_KOLIBRI_EXPERT_STREAM=1 \
numactl --cpunodebind=1 --membind=1 "$BUILD/strata-kolibri" \
  --model "$MODEL" --tokens-file "$EVIDENCE/inputs/ids512.txt" \
  --ctx 2048 --n 8 --force 286,3515,310,262,14292,1343,1444,46 \
  --prefill --cpu-threads 14 --dump-full-logits "$EVIDENCE/qual/512-on-r1.bin" \
  > "$EVIDENCE/qual/512-on-r1.out" 2> "$EVIDENCE/qual/512-on-r1.err"
```

The Python campaign must sanitize all other experimental environment variables before invoking this template. For long histories select adequate context consistently in both arms. Record actual prompt IDs processed: input files and engine-reported processed counts can differ by one.

**Supplemental state command:**

```sh
STRATA_KOLIBRI_EXPERT_STREAM=1 STRATA_GEMM_DETERMINISTIC=1 \
numactl --cpunodebind=1 --membind=1 python tools/kolibri/state_check.py \
  --bin "$BUILD/strata-kolibri" --model "$MODEL" \
  --ids "$EVIDENCE/inputs/ids512.txt" --ctx 2048 --cpu-threads 14 \
  --json "$EVIDENCE/qual/state-stream.json" \
  --engine-stderr "$EVIDENCE/qual/state-stream.err"
```

### Task 7: Same-day end-to-end speed, generation regression and handoff

**Files:** Campaign/tests and ledger; update design/plan checkboxes with results, not altered acceptance rules.

**Consumes:** Task 6 exact parity/state PASS.
**Produces:** reproducible speed verdict and opt-in usage instructions, or a documented negative.

- [ ] Add campaign dry-run/command tests ensuring speed has no full-logit dumps, capture hooks, phase timers, debug parity or profiler; flags differ only by STREAM=0/1. Parse final cumulative PFchunk once; unit-test multiple chunk lines and an incomplete run. Measure load/startup separately, never subtract guessed overhead.
- [ ] At short (512 fixture), 8191 and 11000 input lengths, alternate at least three CPU/stream pairs with identical model/context/binding/threads/output length and no concurrent job. Keep the model warm-up policy identical; record cold startup and first-stream allocation separately.
- [ ] Benchmark 128 generated tokens after short and 8191-ID prompts, using the engine's real end-to-end DONE/wall timing rather than expert-only counters. Same forced continuation where supported; verify actual generated count. Context 160K is an additional memory-fit and smoke check only after shorter runs pass, not a requirement to start a network server.
- [ ] Primary engine outcome: lower median prefill time at every tested length; record each pair and range, not just best run. No generation regression beyond observed paired control spread; if close/noisy, add paired repeats and label inconclusive rather than claim success.
- [ ] Explain remaining time using exclusive wall intervals and a separately recorded trace: host grouping/gather, DMA, GPU products, CPU activation, waits, existing attention/dense/publication. Overlapping intervals are not additive. Report real PCIe bytes, launches, memory footprint and first-use cost.
- [ ] Document how to enable the opt-in, supported target/format restrictions, refusal messages, validation passed/failed/not run, limits and rollback (unset flag). New path remains opt-in even if fast; no default promotion, push or PR without instruction.
- [ ] Verify protected branches/worktrees and artifact checksums remain unchanged; no orphan engine/test subprocess or GPU allocation remains. List all new commits and artifact paths in final handoff.
- [ ] Commit: `[Benchmark] report paired Kolibri GPU-streamed prefill results`.

## Delivery checklist for the implementing agent

- [ ] Every task's red/green tests and observed commands logged; skipped/not-run checks named.
- [ ] Both representative replay gates passed before engine integration, or project stopped with NO-GO.
- [ ] CPU oracle parity, all captured expert outputs, six forced histories and lifecycle full logits finite/bit-identical.
- [ ] Arm-off parent parity and unsupported/backend refusal checks passed.
- [ ] Same-day paired prefill/load/decode results kept distinct; no best-of-only claims.
- [ ] CUDA built; HIP/SYCL impact validated or explicitly not review-ready.
- [ ] Memory/stream ownership and cancellation sanitizer checks passed.
- [ ] No protected worktree/default/artifact changed, no pushes, no network server left running.
- [ ] No numerical optimization was quietly substituted when exact streaming was difficult.

## Self-review of this recipe

Coverage: approved grouped streaming, tiling, CPU arithmetic bridge, double buffering, bounded memory, replay speed gate, guarded integration, full-logit/state gates and final speed/regression reporting each have a task. The five failure classes in Review Focus map to explicit tests. Public host API names are fixed above; kernel launch layouts remain internal so tuning does not leak into engine code. The opaque Context destructor issue is called out. Existing historical timing claims are hypotheses/context, not acceptance measurements. No implementation was performed while authoring this plan.
