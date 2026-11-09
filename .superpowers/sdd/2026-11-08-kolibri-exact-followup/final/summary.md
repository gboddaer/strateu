# Final composition - Kolibri exact follow-up

## Retained

- Task 0: offline driver budget/artifact protection (`a7574911`).
- Task 2: exact multi-selection MMVQ for Q4_K/Q6_K, `STRATA_KOLIBRI_MMVQ_SELECTED=1`, default off (`33ed811a`).
- Task 10: immutable zero buffer for the existing skip-zero-overflow path, `STRATA_KOLIBRI_ZERO_BUFFER=1`, default
  off and only consumed with skipping enabled (`5aaee42b`).

Rejected: Task 3 quantization reuse, Task 4 prefill admission, Task 5 decode drains, Task 6 promotion batch,
Task 8 CPU packing, Task 9 host-copy switch. Blocked: Task 7 GPU hit dispatch (synthetic prototype only; no engine
map lifecycle). Each has a review artifact and a ledger entry.

## Builds and regressions

- `build-kolibri-hip` full build: OK.
- `build-kolibri-hip-tests` full build: OK.
- HIP ctests `kolibri_hot_policy_test`, `kolibri_expert_transfer_test`, `kolibri_mmvq_selected`: 3/3 passed.
- `tools/test_setup_*.py`: three scripts fail (`test_setup_dl_fix.py` missing numpy, `test_setup_download_length.py`
  module import path, `test_setup_pins.py` pin assertions). Causes are not all established and no baseline was run;
  they are outside the Kolibri engine tests. Logs are copied into `final/setup-logs/`.
- `tests/terminal` via unittest: 1654 tests, 5 failures, 166 errors, 52 skipped. The logged failures include
  component/server/package fixtures and a process-discovery assertion, not solely missing engines; no Kolibri engine
  test is among them. Causes are not all established. Full log: `final/terminal-tests.log`. This is not a green
  Terminal suite.
- `pytest` is not installed in either interpreter; unittest discovery was used.

## Final composed integration

Flags: `STRATA_KOLIBRI_MMVQ_SELECTED=1`, `STRATA_KOLIBRI_SKIP_ZERO_OVERFLOW=1`, `STRATA_KOLIBRI_ZERO_BUFFER=1`.

- 10/10 short/long/hot/profile dumps finite and byte-identical to the Task 2 control; `final/integration/compare.txt`.
- Near-boundary long context, 131064 input + 8 steps, ctx 131072:

| arm | prefill tok/s | decode tok/s | wall s | logits SHA |
|---|---:|---:|---:|---|
| flags off | 19.60 | 5.43 | 6801.9 | `5c85e7f6569fea0c...` |
| flags on | 19.66 | 6.17 | 6786.1 | `5c85e7f6569fea0c...` |

The complete dumps are identical. Prefill is flat (+0.3%); decode is +13.6% in this pair; wall is flat. One pair is
not a repeatable speed claim. These dumps hold eight complete vocabulary distributions; this is local regression
identity, not independent model-quality qualification. Targeted Task 10 miss/poison/mixed-format/flag-matrix tests
remain unrun.

CUDA and SYCL: NOT RUN (no toolchain/hardware). No server was launched. No default was promoted; flags remain
individually overrideable.
