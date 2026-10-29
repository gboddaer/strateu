# Kolibri decode validation progress

Plan: `docs/plans/2026-10-29-kolibri-decode-experts-vram.md`.
Spec: `docs/KOLIBRI_DECODE.md`. Goal: `mvticbbf-mb0flp`.

## Resumed scope

The operator explicitly clarified that a 1.5% intermediate regression is acceptable if it enables a
larger final gain. Elapsed time is not a stopping criterion. Investigate the final design, rather than
rejecting graph capture from one segmented implementation's timings. Correctness gates still apply.
No defaults change; no push; native execution; GPU card0 only.

Ruling: retain segmented capture as a prerequisite experiment, not as a promoted optimization.
Assess cold capture and replay separately, and compare the final combination to the best validated
baseline. A failed intermediate speed comparison alone does not stop the investigation. The cost if
wrong is additional investigation time, not an unvalidated default change.

## Validation defects reproduced and fixed

- `attn_dp_bucket`: compiled the actual host function from `kolibri_engine.cu` in a CPU-only regression
  test. Positions 8193 and 11000 incorrectly selected a 16384-entry bucket for a 11000-entry cap; a cap
  below 512 was also violated. Four regression cases failed before the fix. Clamp the bucket to the
  configured cap. Previously valid power-of-two buckets remain unchanged.
- Benchmark baseline marker: a child printing the candidate marker in both arms was incorrectly
  accepted. The regression failed before the fix. Observe markers in the baseline without requiring
  them, then reject contamination. Candidate activation is still required.
- Benchmark output units: `DONE.decode_ms` is request time, not per-token time; fix those two labels.
- Persistent reuse driver: tests reproduced acceptance of a reset (no reuse) and acceptance of a
  nonzero engine exit after DONE. Replace blocking supervisor reads with a deadline-aware queue,
  validate DONE/T counts and actual reuse, propagate failures, and kill/reap the owned process group.
- Headered serve dumps: identical empty/truncated files and identical non-finite values must not
  pass a byte comparison. Add an independent declared record/vocabulary check and finiteness gate.
  Use that comparator in the shell gate, fail a missing reuse helper, and derive forced-step count
  from the forced sequence, not a literal nine. Expected vocabulary is declared (128000 for Kolibri-1),
  not inferred from the file being tested.

CPU verification: 74 tests passed across `test_decode_policy.py`, `test_decode_bench.py`,
`test_decode_gate.py` and `test_check_decode.py`. The new defect cases were run failing first.
HIP `strata-kolibri` build passed; log: `/tmp/kolibri-dec2/validation-fix-build.log`.
CUDA compilation and independent review are not implied.

## Fresh GPU correctness check after the bucket fix

Card0, model `/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`, 8191 prompt ids from
`tools/kolibri/decode_ids8191.txt`, ctx 12288, eight CPU threads, P4 MMQ and the direct gfx1201 tuning.
Identical forced history `286,3515,310,262,14292,1343,1444,46`, nine decode positions; the visible
history crosses 8192 and exercises the previously oversized bucket.

Baseline: all `STRATA_KOLIBRI_DEC_*` controls unset.
Candidate: `DEC_HANDOFF=1`, `DEC_STREAM=1`, `DEC_ATTN_DP=1` (full prefixed names).
Both engines exited zero; P4 activation and candidate activation lines were retained.
`check_decode.py --vocab 128000 --records 9`: **bitwise PASS**, all logits finite, both files exactly
4608000 bytes. These are correctness runs, not throughput measurements.

Raw command, binary hash, selected environment, stdout, stderr and full-logit dumps:
`/tmp/kolibri-dec2/verified-long/`. This establishes this boundary case, not every long-context case.

## Additional findings during segmented replay validation

The stricter reuse check initially failed in BOTH arms. The original fixture extended only the 512
prompt ids, omitting the four ids the engine generated. It therefore did not extend the full held
prefix. A failing CPU regression now enforces inclusion of generated history. The corrected real
requests report `reused=512`, `prompt_ids=580`, `fresh=68` in both arms; the two headered full-logit
records match bitwise. Earlier byte-only reuse verdicts did not establish actual KV reuse.

Segmented capture is restored in the working source as a prerequisite, with shared-expert work after
publication inside PRE. The online-attention geometry has a distinct negative cache key: a 512-thread
online launch must never replay a 128-thread single-pass graph. A CPU regression failed before that
key was added. CUDA-neutral graph types replace HIP-only types in the engine source.

The layer-dump instrument also read the legacy stream while explicit-stream decode was still writing.
A 16-id, two-position baseline/stream comparison failed before its correction. The diagnostic now uses
compute-queue snapshots and one download after token completion, not per-layer host drains.
Temporary layer-19 buffer probes were removed. No diagnostic synchronization is a correctness fix.

One 8191-id no-graph reference and one diagnostic snapshot produced different logits. This initially
looked like a graph error; artifact hashes showed most graph runs matched the ORIGINAL independently
verified reference exactly. A fresh same-binary audit then ran two no-graph processes and one graph
process, all producing the same complete nine-record SHA256:
`93274a15a769b28b4e4ea57e5ae5289fd624d01a172f9872aaa4ac27f1105c88`.
The prefix fingerprint later located the source of that class of difference before decode: six no-graph runs
produced five identical incoming KV states and one different one, first differing in a retained full-attention
layer, so at least one anomalous reference was prefill state rather than graph scheduling. The exact vendor
internal cause is still not proven; paired decode comparisons therefore run with
`STRATA_GEMM_DETERMINISTIC=1` in both arms. Retain the raw records and do not call them explained by a
stream fix, transfer fix, or graph architecture. Raw records: `/tmp/kolibri-dec2/reference-audit/`,
`segmented-boundaries/`, `long-layer-probe/`, `prefix-repeat-audit/`, `deterministic-prefix-audit/`.

A hypothesis that moving CPU-result transfer from DMA into the compute queue would fix the mismatch
was NOT supported: its output was identical to the preceding graph output. The mapped result path is
kept only as a whole-stack-capture prerequisite (the reference protocol uses compute-queue result
reads), not as a claimed numerical fix. No vendor-library root cause is established.

A separate matched 11000-id baseline/graph pair at ctx 12288 crossed from the single-pass attention
branch to online attention; nine complete full-logit records match bitwise. This checks the distinct
cache-key/launch-geometry transition. It does not establish all-context repeatability.

## Bounded wait prerequisite (now used by the whole-token graph)

This primitive was written before model integration and is the mechanism the parent executable below
waits on. Its own guarantees are unchanged; the model-level invalidation is described in the
whole-stack section.

`DecodeWaitControl` and `enqueue_decode_wait` are in `include/strata/kolibri/decode_wait.hpp` and
`src/kolibri/decode_wait.cu`. A captured wait reads sequence and generation on EVERY replay. It rejects
future ACKs, stale generations, and an unpublished zero identity; abort or a finite device-clock tick
budget terminates it. Tick budgets are not presented as calibrated wall milliseconds. The caller must
still invalidate aborted tokens and reset affected state; this primitive alone cannot enforce that.

The dedicated GPU test was run RED against a non-waiting stub (accepted future ACK), then GREEN on
card0. A further absent-publication regression was run RED, then fixed. Final test:
`ctest --test-dir build-mmq -R '^kolibri_decode_wait_test$' --output-on-failure` passed in 1.30 s:
32 changing replays, future ACK and stale-generation rejection, correct ACK acceptance, simulated
cancellation/CPU-producer failure, withheld ACK timeout, and absent-publication timeout.

Build caveat discovered: `build-mmq` had `STRATA_BUILD_TESTS=OFF`; old CTest files/executables still
existed. Reconfigured explicitly with tests ON before building/running the new wait test. Earlier
historical full-suite counts are NOT proof that all current sources were rebuilt and tested. A fresh
full build and CTest run remain required. Current regression suite: 82 passed, including six negative engine-startup checks that require no
model or GPU allocation. Those guards were run RED before the early HANDOFF/STREAM checks were added.

## Current segmented checkpoint

The complete repaired gate was rerun on the current mapped-result/shared-in-PRE implementation:
`/tmp/kolibri-dec2/segmented-current-gate/driver.log` reports **DECODE GATE PASS**: independent
128000-value shapes at 16/511/512/513 ids, actual generated-prefix reuse with finite headered logits,
and six bit-identical repetitions. The marker required is `DEC_HANDOFF_GRAPH replay active`, emitted
only after a PRE executable is actually launched, not merely because an environment switch was set.

The wait primitive was committed separately from model integration and is now used by the whole-token
executable described in the next section; the segmented stage alone carried no performance claim.
Neither the earlier 1.5% comparison nor the reference anomalies are a reason to promote or abandon the
final architecture. The arms were later promoted to the default path (see the promotion section below); nothing was
pushed.

## Whole-stack integration and the incoming-state audit

`STRATA_KOLIBRI_DEC_TOKEN_GRAPH=1` now assembles a parent executable from retained segment graphs:
PRE → bounded ACK wait → POST for all 50 layers. CPU expert arithmetic stays unchanged. Embedding
and the logits head remain outside. Cold mirrors and new attention buckets warm through segments.
The arm requires HANDOFF, STREAM and HANDOFF_GRAPH; it rejects unsupported segment/debug combinations.
All runtime switches remain default-off. Nothing has been pushed.

Initial real-model tests passed: active replay and finite bit-identical short full logits, a forced
zero-tick timeout without an invalid logit record, and STOP during publication followed by a cold,
bit-identical next request. These three tests passed in 77.64 s; final reruns remain necessary.
Malformed or effectively unbounded tick overrides now fail before model loading. Six no-model tests
were observed RED, then GREEN; guard + benchmark unit tests currently report 60 passed.

### The long-reference discrepancy is present before decode

An opt-in `STRATA_KOLIBRI_PREFIX_FINGERPRINT=<path>` diagnostic copies/hashes only valid KV bytes
at the already-quiescent prefill/decode boundary. It adds no decode-layer drains. Six independent
**no-graph** 8191-id runs produced five identical prefixes/logits and one different prefix/logit dump.
The first differing retained cache was layer 34, before any generation forward. This proves that
at least this discrepancy is incoming prefill state, not a scheduling-only decode difference.
Artifacts: `/tmp/kolibri-dec2/prefix-repeat-audit/` (run 5 differs).

Ruling: use the existing `STRATA_GEMM_DETERMINISTIC=1` option in **both** arms of paired decode
validation/measurement — bitwise comparisons need repeatable incoming state. This changes no
production default or prefill expectation file. It is an explicit test configuration, not a fallback.
Cost if wrong: a library path could still vary outside these fixtures, so retain state diagnostics
and do not claim a general vendor-internal root cause. The option affects both GEMM handles; the
exact internal source of the earlier variation is not proven.

With that option, three no-graph and three whole-stack runs, in alternating order, had identical
prefix fingerprints and nine finite full-logit records each, crossing position 8192. Every run
confirmed MMQ activation with direct shared-expert tuning. Full-logit SHA256 remains
`93274a15a769b28b4e4ea57e5ae5289fd624d01a172f9872aaa4ac27f1105c88`.
Artifacts: `/tmp/kolibri-dec2/deterministic-prefix-audit/`.

### Fresh unprofiled three-arm measurements

Card0/gfx1201, eight CPU pool threads, Q4_K_M model, zero expert slots, EXPERT_F16=0,
ATTN_MB=48, GEMM_DETERMINISTIC=1 and the direct gfx1201 tuning table. Each workload uses
96 generated tokens, one stream, no drafter. Three rotating-order blocks contain handoff+stream,
segmented and whole-stack arms. All nine generated histories match within each workload. These
are DONE.decode_ms medians, **milliseconds per request**, excluding loading and prefill; cold
capture work encountered during generation is included. Raw token timestamps, INFO/DONE fields,
commands, environment, hashes and output logs are retained.

| prompt ids / allocated ctx | handoff+stream ms | segmented ms | whole-stack ms | whole-stack tok/s |
|---|---:|---:|---:|---:|
| 512 / 8192 | 4759.1 | 4864.8 | 4015.2 | 23.91 |
| 8191 / 12288 | 7061.0 | 7146.8 | 5339.7 | 17.98 |

Whole-stack reductions versus handoff+stream are 15.63% short and 24.38% long; versus segmented,
17.46% and 25.29%. No arm's own spread exceeds its measured difference. These are the **final-source**
numbers, taken after the cold-counter rollover patch; the earlier pre-patch block is preserved
separately at `/tmp/kolibri-dec2/three-arm-before-rollover/` and is not reported as the result.
Artifacts: `/tmp/kolibri-dec2/final-three-arm/{short,long}/runs.json` and logs.
The runner is `/tmp/kolibri-dec2/three-arm-bench.py`. These results supersede the earlier short
segmented +1.5% prerequisite comparison.

Per request that is 41.8 ms and 55.6 ms of decode per token: cold capture is inside those numbers and is
small against them (111 segment captures totalling ~72 ms plus 5.5 ms / 13.0 ms of parent instantiation,
from the profiled diagnostic trace, not a throughput measurement). Steady `T`-to-`T` medians after the
first five tokens were ~40.5 ms and ~53.8 ms.

### Final-binary 8191-id KV-wrap parity

Two plain and two whole-stack runs of the 8191-id prompt on the measurement binary each produced nine
finite 128000-value records with the identical SHA256
`93274a15a769b28b4e4ea57e5ae5289fd624d01a172f9872aaa4ac27f1105c88`; all four cross-comparisons were
bitwise, the arm marker was present only in the two arm runs, and the binary hash was unchanged before
and after. This is the case that crosses position 8192 during generation, the KV-wrap concern the
earlier audit was taken from a different build. Artifacts: `/tmp/kolibri-dec2/final-8191-parity/`.

## Subsequent lifecycle and protocol checks

The full whole-stack gate now passes on current source with GEMM_DETERMINISTIC=1 in both arms:
`/tmp/kolibri-dec2/token-current-full-gate/driver.log`. This includes four prompt/chunk boundaries,
true reused prefix with generated ids retained, six identical repetitions, finite complete full-logit
records and an actual replay marker absent from controls.

The expanded `kolibri_decode_wait_test` passes (1.48 s). Its dependent eight-layer chain replays
changing N=2560 activations and top-six ids/weights, feeds each CPU result into the next publication,
rejects future ACKs and previous generations, drains missing publication/ACK, cancel and simulated
CPU failure, then successfully resets/replays. It uses real doorbell publication and the real bounded
wait. This strengthens coverage beyond the earlier isolated-wait test.

A real-engine test-only TU (`tests/kolibri/decode_rollover.cu`) seeds private descriptor state without
adding production test knobs. It found a real defect: the overflow reset lived inside the parent-only
path, after eligibility checks, so cold segmented warmup could wrap through the unpublished zero
identity. RED: seeded nine-forward test ended at 424/424 rather than 450/450 publications. The reset
now reserves a whole token's nonzero identities before **any** decode publication. Its rare reset
quiesces the queue before descriptor reuse; this is not a per-layer timing workaround. The same TU
also seeds generation UINT32_MAX, proving the parent skips zero on generation rollover.
RED/GREEN logs: `/tmp/kolibri-dec2/rollover-{red,green}.log`.

All seven current real-model integration cases passed in 178.02 s:
normal and seeded-rollover full-logit parity, mapped allocation failure before loading, parent
executable allocation failure before replay, a CPU producer exception after publication with the
graph already quiescent on return (query only, no test-side drain), bounded missing ACK, and STOP
followed by a cold bit-identical request. Fault injection is confined to the test TU. Completed logit
records before a failure are checked against the same forced baseline; no failed-step record is
accepted. Log: `/tmp/kolibri-dec2/model-failure-tests.log`.

## Fresh broad verification (not the historical stale CTest tree)

The current 11000-id pair crosses single-pass/online attention and passes nine finite, full 128000-logit
records bitwise. Actual whole-stack replay is required. Artifacts: `/tmp/kolibri-dec2/token-11000-current/`.

Both original prefill profiles now pass, with unchanged in-repo smoke/hash expectations. The first pool
attempt failed because the explicit hipBLASLt tuning override was still set. TABLE=off disables table
auto-discovery but does not override an explicit TUNING path. Removing that explicit override restores
all 100 original pool hashes exactly. The pool gate is run with TABLE=off and **no** TUNING override;
the MMQ gate uses the direct gfx1201 tuning table. This is matching the original numerical profiles,
not updating an expectation to accept new output. Logs under
`/tmp/kolibri-dec2/final-prefill-and-ctest/`: `pool-original-no-explicit-table.log` and
`mmq-gfx1201-qualified.log` (both GATE PASS). Retain earlier failures as configuration evidence.

A fresh full build exposed unconditional links from q4k_grouped_vram_parity to CUDA-only optional
libraries absent on HIP. The actual link failed for strata_mmq_w4a4/strata_kq; target-existence generator
expressions now retain those dependencies only when their targets exist (including later-defined CUDA
targets). The full HIP build then passed. No model/runtime default changed.

The fresh full CTest run has **188** tests: **171 passed, 11 skipped, 6 failed**. Failures are
kolibri_load_test (hard-coded /data4 model path absent), vmm_test, ple_parity (missing Qwen fixture),
platform_memory_test, conversation_copy_streams_test (first stream allocation exceeds its threshold),
and isa_guard_check (two unlisted iq_avx512 public functions). These are not waived: all six were rebuilt
and reproduced in a separate worktree at committed baseline `057d42a6`, so they are pre-existing on this
box and not introduced here. This replaces the old 94-test totals, not vice versa. Full logs:
`final-prefill-and-ctest/{build-all-fixed,ctest}.log`; baseline reproduction:
`baseline-six-failures/{configure-fixed,build,ctest}.log`.

The CPU/no-model decode/tool suite now passes **88 tests** (`guard-restored.log`). A proposed guard
for `--graphs` was withdrawn after checking the actual parser: no such option exists, and legacy graphs
are selected internally only for a non-pool path. The mistaken guard and its test are both removed;
the restored engine builds. Do not describe that discarded test as an engine defect.

## Self-review (no subagent tool; delegation was not authorized)

A separate pass over the final diff, after the last task line. What it confirms and what it defers:

- `run_token_graph` resolves every layer's geometry key before launching, so a token is either fully in the
  parent executable or fully on segments; no partial mix. Embedding and the logits head stay outside, and
  `decode_aborted` suppresses logits for a discarded token.
- Abort sets `control->abort`, then one token-boundary `cudaStreamSynchronize`; the wait kernel is the only
  long-running GPU node and its budget is finite device ticks. On abort `hb.expect` adopts the observed
  publication count so a discarded token does not also report a second, misleading mismatch.
- Sequence reset moved to the token boundary in `forward()` and quiesces with `cudaDeviceSynchronize` only at
  the rare identity boundary; that is not a per-layer timing change.
- `prefix_fingerprint.inc` is env-gated, diagnostic-only, reads valid KV rows at an already-quiescent boundary,
  and adds no decode synchronization or kernels.
- Graph caches (`DecGraphArm`, `HandoffBuffers`) are process-lifetime and keyed by layer geometry; replacing a
  loaded model inside one process is not supported and is documented rather than handled.
- Deferred minor: the `DecGraphArm` parent executables are never explicitly destroyed before exit; the count is
  bounded by distinct attention geometries, so this is cleanup rather than a growth path.
- Deferred minor: `docs/KOLIBRI_DECODE.md`'s verification table keeps its historical 55-test row, marked as such,
  next to the current 88-test row.

A self-review by the author is weaker than a fresh reviewer. The goal's remaining verification gap is that no
independent reviewer has run.

## Where a token goes, and the overhead that was rejected

`STRATA_KOLIBRI_DEC_TOKEN_TIMING=1` splits each token with wall clocks only (no driver trace, no extra
synchronisation):

| Leg (median, steady tokens) | 512 ids | 8191 ids |
|---|---:|---:|
| host waiting for the GPU to publish | 16.04 ms | 29.44 ms |
| host payload work + CPU experts + acknowledgement | 23.94 ms | 24.03 ms |
| graph launch | 0.68 ms | 0.61 ms |
| token-boundary sync | 0.05 ms | 0.06 ms |
| **token** | **41.0 ms** | **54.7 ms** |

The earlier arithmetic estimate — the 717 MB/token of expert blobs streamed at 27–32 GB/s — is confirmed at
23.9 ms, and it does not grow with context. The GPU leg grows ~13.4 ms between 512 and 8191 ids, which is
attention. Summed, the driver/launch/sync legs are 0.7 ms of 41 ms.

Two attempts to cut that 0.7 ms were measured and **rejected**:

- `cudaGraphUpload` on the parent executable after instantiation: launch 0.68 → 0.69 ms short and 0.61 → 0.68 ms
  long, i.e. no help. Reverted.
- Stream-scoped sync plus a queued logits read instead of the device-wide sync and blocking copy: three
  rotating-order pairs gave 4044.3 → 4039.1 ms/request short (0.13 %) and 5375.4 → 5370.9 ms long (0.08 %), inside
  the 0.4 % noise floor, with one short pair slower. Reverted rather than kept as an unmeasured default.

Artifacts: `/tmp/kolibri-dec2/{default-hip-api,dec-split-short,dec-split-long,overhead-short,overhead-long,tail-ab}/`.
A driver trace was also taken and discarded: it is dominated by the 43 GB pinned allocation at load and by the
instrument's own serialisation, so dividing it by a token count says nothing about the decode.

## Status: complete, with stated limits

All five tracked tasks are done and every contractual gate above has a recorded, inspectable artifact.
What remains is not unfinished work but a boundary on the claims:

- **No independent reviewer ran.** Delegation was not authorized, so the final pass was a self-review
  (recorded above) by the author. That is weaker than a fresh reviewer.
- **CUDA and SYCL were not built.** `nvcc` is absent on this box and the Kolibri engine path is CUDA/HIP only.
  The `CMakeLists.txt` change is a target-existence guard and is backend-neutral, but only HIP was compiled.
- **Six broad-suite failures** (`kolibri_load_test`, `vmm_test`, `ple_parity`, `platform_memory_test`,
  `conversation_copy_streams_test`, `isa_guard_check`) were rebuilt and reproduced at committed baseline
  `057d42a6`. They are pre-existing environment/fixture issues, not waivers for a regression here.
- **Prefill KV variability** is characterized but not root-caused inside the vendor library; paired decode
  comparisons therefore run with the existing `STRATA_GEMM_DETERMINISTIC=1` in both arms. No
  synchronization workaround was added to either path.
- **The decode arms are the default path** (promoted with the user's explicit instruction) and nothing was pushed.
  Each arm keeps a `=0` off switch; unsupported configurations run the plain decode and say why, while an explicit
  `=1` is still a named startup error.
