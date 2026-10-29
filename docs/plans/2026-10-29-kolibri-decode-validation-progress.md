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
The cause of the earlier anomalies is not established. Retain them; do not call them explained by a
stream fix, transfer fix, or graph architecture. Raw records: `/tmp/kolibri-dec2/reference-audit/`,
`segmented-boundaries/`, and `long-layer-probe/`.

A hypothesis that moving CPU-result transfer from DMA into the compute queue would fix the mismatch
was NOT supported: its output was identical to the preceding graph output. The mapped result path is
kept only as a whole-stack-capture prerequisite (the reference protocol uses compute-queue result
reads), not as a claimed numerical fix. No vendor-library root cause is established.

A separate matched 11000-id baseline/graph pair at ctx 12288 crossed from the single-pass attention
branch to online attention; nine complete full-logit records match bitwise. This checks the distinct
cache-key/launch-geometry transition. It does not establish all-context repeatability.

## Bounded wait prerequisite (not yet integrated into model replay)

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

The wait primitive is committed separately from model integration. It is not yet used by a whole-token
executable. No performance claim is made for the hardened segmented variant, and neither the earlier
1.5% comparison nor the unexplained reference anomalies are a reason to promote or abandon the final
architecture. All new runtime switches remain off by default; nothing was pushed.

## Remaining work

Restore and harden segmented capture; prove changing positions, bucket transitions and request reset.
Before any whole-stack graph waits, write and run the bounded acknowledgement/abort tests. Then gate
full logits/state/repeatability and measure the combined design. Do not claim the goal complete from
historical task labels or from the earlier short-only gates.
