# Kolibri exact-arithmetic optimization implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. The user explicitly says run to the end without per-task check-ins. No subagent delegation is authorized.

**Goal:** Complete ranked paths2–10 on `amdw7800probuild_optimize`, skipping attention tiling1, retaining correct speedups and tested enablers.
**Base branch:** `amdw7800probuild`; commit and push preparation documents there, then create the work branch from that tip before implementation.
**Architecture:** New paths are individually opt-in around existing arithmetic. Synthetic exact-byte tests precede full-model forced-history gates. Each task has an independent keep/reject decision and a local commit for retained code/evidence.
**Tech Stack:** C++17, CUDA/HIP-style kernels, Python offline fixtures, ROCm7.10 on W7800.
**Spec:** `docs/superpowers/specs/2026-11-08-kolibri-exact-followup.md` (read first).

## Global constraints

Verify `git branch --show-current` reports `amdw7800probuild_optimize` before implementation. Do not run implementation tasks on `amdw7800probuild`.
Copy the specification's binding constraints into the execution ledger. Do not silently reinterpret them.
No server, attention tuning, arithmetic tolerance, full-table pinning, default promotion or system setting changes.
Use `/home/gbo/models/Kolibri-1-Q4_K_M.gguf`. Runtime:V=/home/gbo/develop/Strata/.venv;
LD_LIBRARY_PATH=$V/lib/python3.13/site-packages/_rocm_sdk_devel/lib:$V/lib/python3.13/site-packages/_rocm_sdk_libraries_gfx110X_dgpu/lib;
HIP_VISIBLE_DEVICES=1. Use python3, not python.
Builds:`build-kolibri-hip` and `build-kolibri-hip-tests`. Preserve the current executable as the regression control
BEFORE rebuilding. Preserve the ignored workspace and all full-logit/raw measurement evidence.

## Common gate used by EVERY task

1. Write a failing targeted test; run and observe the intended failure. Implement the smallest opt-in path.
2. Build the touched HIP targets and run synthetic tests; reject nonfinite or bit-different outputs, not merely different argmax IDs.
3. Run legacy flag-off and candidate on aligned identical histories. Compare complete raw output bytes and validate every float.
4. Integration matrix:512input/128steps auto-cache;8input/32steps cache8;synthetic4097input/8steps;
   all demand short/long/hot regression phases with --hot; include Q4_K and Q6_K and first-use/replay.
5. For changes to cache allocation, lifetimes or long-prefix state, add16385input and65537input comparisons.
   Run a near-boundary131064input/8step pair for the final retained composition. Per-case watchdog>=10800s;
   no600s helper cutoff and no global overnight cutoff. Neither a timeout nor missing reference is a pass.
6. Timings:five alternating same-day control/candidate pairs on the512/128 and cache8 cases, per-blob stats OFF.
   Preserve profiles, context, input, forced history and binary hashes. Report all runs/median/spread. Any repeatable
   positive gain is acceptable; noise is inconclusive, not a win. Profile diagnostic runs separately.
7. KEEP if correct and faster; KEEP-ENABLER if correct and used by a named subsequent retained path; otherwise
   reject/isolate the experiment. Never keep a dangerous speculative framework just because it might help someday.
   Then run the mandatory external review gate below and act on it before marking the milestone complete.
8. Full HIP test-tree build and main+standalone builds before commit. CUDA/SYCL unavailable => NOT RUN, no cross-backend claim.

## Review focus
- Scratch-pointer reuse is not proof activation VALUES are unchanged:quantization reuse must be explicit within one operation group.
- Selected experts can repeat, be permuted, use per-selection activations, or include zero/spill slots:preserve index/stride bounds.
- Copy completion is not permission to overwrite weights still being read:keep reader and readiness fences.
- Prefill may select more experts than permanent slots:protect every current resident pick and bound staging/evictions.
- A fast warm path must still allow first use, a later miss, poison-zero diagnostics, reset/drain and mixed-format fallback.

## External review gate (MANDATORY at every milestone)

**A milestone is the completion of Task0 and of every Task2–Task10, plus the final composition.** Before a
retained change is committed or a task is marked complete, an independent review runs on `gpt-6.1-sol` and is
acted on. The reviewer never edits the tree: it runs read-only and ephemeral from the repository root.

```sh
mkdir -p .superpowers/sdd/2026-11-08-kolibri-exact-followup/reviews
pi --print --no-session --no-extensions --no-mcp --no-skills --no-prompt-templates \
  --no-context-files --no-themes --tools read,grep,find,ls \
  --model openai-codex/gpt-6.1-sol --thinking high \
  "<review brief>" > .superpowers/sdd/2026-11-08-kolibri-exact-followup/reviews/TaskN.md 2>&1
```

Reachability was verified before adding this gate: `pi --list-models gpt-6.1` lists `openai-codex gpt-6.1-sol`,
and both a plain prompt and a read-only repository review returned successfully. If the model becomes unavailable,
record the milestone review as BLOCKED; do not silently substitute another reviewer.

The brief must contain, in this order: (1) the exact plan task and its acceptance rules; (2) the base and head
commits and `git diff` scope; (3) paths to raw logs, full-logit hashes and timing artifacts; (4) the claimed decision;
(5) an explicit demand for technical evidence, not reassurance. Ask it to report findings by severity with file:line,
and to state separately: correctness/bitwise risks, lifecycle/race risks, performance-evidence gaps, and any plan
step that was skipped. Do not paste credentials, the model file, or unrelated workspace data.

**Acting on findings is part of the milestone, not optional:**

1. Read the review against the code; invoke `superpowers:receiving-code-review`. Verify each claim before changing
   anything; do not implement suggestions that are wrong or unverified.
2. Record every finding in the execution ledger as FIXED (with the failing test that proved it), REJECTED (with
   technical justification and evidence), or DEFERRED (with an explicit dependency). Unanswered findings block
   completion.
3. Critical/Important findings get one fix pass using RED→GREEN tests, then the full common gate is re-run.
   If the fix changes arithmetic, data lifetimes or public interfaces beyond the reviewed diff, request a focused
   follow-up review of that fix and act on it the same way. Do not loop indefinitely on cosmetic Minor items; ledger them.
4. The retained commit must include the review file and the resolution notes. A milestone whose review is missing,
   unanswered, or marked BLOCKED is not complete, even if tests pass.

## Task0: seal controls and fix the test-driver budget (not attention work)

Files:Modify tools/kolibri/bench_unpinned.py;Test tests/kolibri/bench_unpinned_test.py;Create workspace/progress.md.
Interface:benchmark gains --timeout(seconds,default600), --ctx(default131072), --ids-file(optional explicit token fixture).
All lengths are positive; actual input length + n<=ctx; supplied IDs are integers in[0,128000). Copy external
fixture into the run directory. Do not overwrite source profile. No engine changes in this task.
- [ ] Save baseline executable and source/binary/model identities in `.superpowers/sdd/2026-11-08-kolibri-exact-followup/`.
- [ ] RED:CLI rejects131065+8 atctx131072, nonpositive timeout and invalid token IDs before launch; existing input-preservation tests stay.
- [ ] Implement driver options/validation; GREEN CPU tests and py_compile. Do not edit the historical overnight controller.
- [ ] Capture same-day flag-off512/128 and cache8 controls using the sealed executable. Validate finite full logits.

## Task2: exact multi-selection MMVQ

Files:Modify include/strata/kernels/native_mmvq.hpp,src/kernels/cuda/native_mmvq.cu,src/kolibri/kolibri_engine.cu,CMakeLists.txt;
Create tests/kolibri/mmvq_selected_test.cpp.
Interface:
`bool native_mmvq_selected(int type,const void* base,size_t weight_stride,const int* host_slots,
 const void* xq,size_t xq_stride,float* y,int n_in,int n_out,int selections,void* stream)`.
Return false/nothing launched for unsupported formats/dispatches;initially Q4_K=12,Q6_K=14,2..8 selections only.
Host slots are copied into a fixed-size by-value kernel parameter block:do not enqueue DMA from a local vector.
xq_stride=0 means shared activation;otherwise one native_q8_1_bytes(n_in,1) column per selection.
y is contiguous selections*n_out. Caller owns valid table bounds. Reject negative/overflowing slot offsets.
Gate:new STRATA_KOLIBRI_MMVQ_SELECTED=1,defaultoff,no-pin expert calls only.
- [ ] RED synthetic test against selections independent native_mmvq calls using SAME q8_1 bytes. Cover both formats,
  both SmallK branches, n_out tails, duplicate/permuted slots, shared/per-selection inputs, unsupported no-launch fallback.
  Reuse finite fp16 scale recipe in src/kernels/mmvq_multi_parity.cpp;random scale bits can produce NaNs and fake equality.
- [ ] Add selected kernels by mechanically preserving legacy per-row loops, partial sums and warp reductions.
  Do not replace with native_expert_grouped or multi-column arithmetic. Leave legacy kernels/signatures untouched.
  Grid.y selects weights/input/output. Refuse gfx906 alternate-wave arithmetic and registered packed-Q6 dispatch
  unless the exact same dispatch has been implemented;fall back instead of bypassing them.
- [ ] GREEN synthetic tests;wire gemv's quantized expert branch after its existing quantization. Dense/unsupported calls unchanged.
- [ ] Common gate and independent commit if retained. Function stays a tested enabler for Task3 if standalone speed inconclusive.

## Task3: explicit activation quantization reuse

Files:Modify src/kolibri/kolibri_engine.cu;extend selected tests or create tests/kolibri/quant_reuse_test.cpp.
Interface:split expert gemv into an explicit quantize producer and a consumer accepting known q8_1 bytes;
`gemv_pair_quantized`(or an equivalently explicit local helper) runs gate/up on the SAME quantized input.
Gate:STRATA_KOLIBRI_REUSE_Q8=1,defaultoff. Keep global g_q8 lifetime stream-ordered. No address-based cache.
- [ ] RED compare two old quantize+gemv calls with one quantize+two consumers for gate/up;quantized bytes and both outputs exact.
  Include same address overwritten with new values between groups and per-selection down inputs to expose stale reuse.
- [ ] Implement within adjacent gate/up pairs only. Do not reuse across SwiGLU/down or across layers. Use selected API when eligible,
  otherwise run the old individual native_mmvq consumers. Do not change q8_1 scales or activation conversion.
- [ ] Test four flag combinations(legacy,selected only,reuse only,both);run common gate and commit retained composition.

## Task4: cache allocation/admission and repeated staging

Files:Modify include/strata/kernels/kolibri_hot_policy.hpp,tests/kolibri/hot_policy_test.cpp,src/kolibri/kolibri_engine.cu.
First prototype uses fixed-total existing VRAM capacity;variable per-layer arena is NOT prerequisite.
Interface:`plan_prefill_admission(...) -> vector<HotSwap>` for one layer, current requested-ID counts and resident map.
All current requested resident experts are protected. Admit most-used requested missing IDs into free/unrequested
victim slots, deterministic ties by expert/slot, bounded per-layer budget. No change to router scores/probabilities.
Gate:STRATA_KOLIBRI_PREFILL_ADMIT=1,defaultoff;requires no-pin,per-layer,hot residency.
- [ ] RED host tests:duplicate requests,current-pick protection,all requested>capacity,no victim,budget0,deterministic ties.
- [ ] Implement a conservative planner using existing hot_plan mechanics with protected current occupants;plan before mapping prefill blobs.
  Reuse already-required transfers but choose permanent destinations where safe. Publish new map only with appropriate weight-ready
  dependencies;old readers must finish before overwrites. Keep bounded fallback staging when requests exceed capacity.
- [ ] Measure bytes/repeated packs at4K,16K,65K against both demand and balanced controls. Do NOT claim balanced-to-demand savings as new gains.
- [ ] If adaptive allocation is still justified by per-layer misses, add an OFF-by-default byte-benefit planner with exact budget tests;
  no unsafe arena rewrite merely to complete a checkbox. Common gate. A proven traffic reduction may be a named enabler for Task5/7,
  but only retain integration actually used by a later kept path or with a demonstrated gain.

## Task5: redundant synchronization

Files:Modify src/kolibri/kolibri_engine.cu;extend transfer tests/lifetime synthetic harness.
Gate:STRATA_KOLIBRI_ORDERED_DECODE=1,defaultoff;first slice only no-pin legacy/blocking compute-stream case.
- [ ] RED delayed-producer/consumer checks for gate/up/SwiGLU/down/shared/sum ordering;assert unsupported stream mode falls back.
- [ ] Trace/document the two decode g_st drains near3187/3228. Replace only if explicit same-stream or proven legacy-stream
  dependencies protect both data and host argument lifetimes. Keep prefill stack-metadata drains until owned uploads exist.
- [ ] Do NOT remove wait_for(old readers),wait_on(weights),source-buffer ownership or arbitrary synchronizations wholesale.
- [ ] Common gate;retain owned-upload/stream helper as an enabler only with named consumer and tests.

## Task6: promotion fences per round

Files:Modify src/kolibri/kolibri_engine.cu,src/kolibri/expert_transfer.hpp/.cu,tests/kolibri/expert_transfer_test.cu.
Gate:STRATA_KOLIBRI_PROMOTE_BATCH=1,defaultoff.
- [ ] RED delayed old readers across multiple layers and two staging buffers;one round's final fence must cover all uploads;
  next round/generation and second consumer stream must not observe stale weights.
- [ ] Hoist a common prior-reader dependency before a move list,enqueue copies with current ownership,then publish one final
  readiness dependency. Preserve current per-move path when unsupported;never omit prior-reader protection.
- [ ] Profile promotion-heavy small-cache/long-prefix cases,not just zero-transfer warm decode. Common gate and commit decision.

## Task7: GPU cache-hit dispatch / capture feasibility

Files:if viable,src/kolibri/kolibri_engine.cu and a narrow new cache-map helper/test. No table-mode graph switch is enabled blindly.
- [ ] Measure host router D2H/idle share after earlier changes. Write the device-map generation/miss/update/reset ownership contract.
- [ ] Prototype device-side slot translation and exact selected MMVQ against CPU slot selection under forced hits/misses.
  A synchronized shadow-map helper may be kept ONLY if used by a subsequent measured hit path,not as dead speculative infrastructure.
- [ ] Only attempt hit capture if misses can safely leave/refresh it and offline reset/drain tests cover map updates and generations.
- [ ] If complete lifecycle qualification needs prohibited server runtime or invasive unproven waiting,record BLOCKED with concrete
  missing gate;remove half-implementations and continue Task8. This is a valid best-effort end state,not a pass.

## Task8: bounded CPU packing

Files:if warranted,new src/kolibri/byte_pack_pool.hpp/.cpp,CPU tests,engine wiring,CMake.
Gate:STRATA_KOLIBRI_PACK_THREADS=2,default0,byte copies only(no CPU expert arithmetic).
- [ ] RED byte-exact three-span packing test with padding,partial layouts,repeated calls,exceptions/drain and source lifetime.
- [ ] Start conservatively:two persistent workers plus caller copy independent gate/up/down spans of ONE blob,join before DMA.
  The existing transfer ring remains single-producer. No unbounded queue,whole-model copy or pinned source registration.
- [ ] Test cold4K/source-fault workload separately from131K. If no win and no concrete retained dependency,remove experiment.
- [ ] Common gate;report memory/thread cost and fallback behavior. Do not reintroduce rejected projection-major packing/read-ahead.

## Task9: host-copy ceiling diagnosis

Files:new standalone tools/kolibri transfer probe and tests only if existing probes cannot record reproducible results.
- [ ] Read endpoint/upstream topology and runtime environment;do not write sysfs or change clocks/IOMMU/firmware.
- [ ] Isolated processes compare default,HSA_ENABLE_SDMA=1/0,small pageable/pinned sources,copy sizes and source CPU memory rate.
  Cap explicit pinning at64MiB total. Verify every copied byte;time copy and verification separately. Run GPU work serially.
- [ ] Reject settings that fail correctness or lack repeatable gain. Keep any useful switch opt-in and process-local only;
  inaccessible root/system changes are BLOCKED,not attempted. No benefit claimed for a zero-transfer workload.

## Task10: all-resident exact fast path

> Task10 and the final composition are milestones: run the external review gate and act on it before committing or closing the task.

Files:Modify src/kolibri/kolibri_engine.cu,tests/kolibri targeted zero-buffer tests.
Gate:STRATA_KOLIBRI_ZERO_BUFFER=1,defaultoff,requires existing skip-zero-overflow path.
- [ ] RED compare per-layer memset overflow output with one dedicated immutable K*N zero buffer;include miss->hit->miss,
  poison-zero diagnostics, mixed types and cache-map changes. Preserve final k_moe_sum2 and its operand order.
- [ ] Allocate/zero dedicated scratch once before auto-fit. When overflow work is already legally skipped,use that buffer as
  s.yc instead of enqueueing another memset each layer. Never alias it with writable expert output scratch.
- [ ] Do not fold the final sum or change floating reduction/sign handling. Common gate and final composed-path comparison.

## Finalization
- [ ] Each Task0/Task2–Task10 and the final composition has a `reviews/` artifact, a resolution entry per finding, and a ledgered decision with raw evidence.
- [ ] Every task has a decision with raw evidence. Keep successful flags individually overrideable;do not silently add them to demand defaults.
- [ ] Rebuild all affected HIP targets,run full integration and current setup regression suites,then perform final long-context paired gate.
- [ ] Record CUDA/SYCL NOT RUN if unavailable;no release readiness inference. Self-review if independent delegation remains unauthorized.
- [ ] Commit only verified retained code/tests/docs on `amdw7800probuild_optimize`;preserve unrelated user files and raw evidence. No implementation push/merge without a further request.
