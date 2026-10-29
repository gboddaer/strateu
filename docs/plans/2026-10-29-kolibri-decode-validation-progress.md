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

## Remaining work

Restore and harden segmented capture; prove changing positions, bucket transitions and request reset.
Before any whole-stack graph waits, write and run the bounded acknowledgement/abort tests. Then gate
full logits/state/repeatability and measure the combined design. Do not claim the goal complete from
historical task labels or from the earlier short-only gates.
