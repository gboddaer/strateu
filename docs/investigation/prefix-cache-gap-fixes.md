# Prefix reuse and cancellation: RED to GREEN

2026-11-05, Tesla T4 16 GiB, driver 580.196.07, dual Xeon E5-2690 v4,
Kolibri-1 Q4_K_M, CUDA Release SM75 with Q6K experts, 14 lanes. No new arithmetic,
expert placement or precision. The served default temperature remains 1.0.

Baseline source: `69741cf9`; baseline qualified engine: `301352d2`, SHA256
`8eba03b8f98c6502df7ebfcd8bd437d285ef6b5eb2cf180701b0907a36b65bc8`.
Candidate: `build-gaps-cuda/strata-kolibri`, SHA256
`f0da4786034f2b2d68a4d6b2adf4ba4f713717e277c5404e9e54c5044b88b129`.
Raw logs/dumps/configs: `/data4/kolibri-sram/out/exact-prefix-cache/fix-gaps/`.

## Original assertions, not weaker replacements

All six original RED assertions were reproduced against the unchanged baseline
before changing production code. The CPU pair now also runs in normal pytest.
The model assertions were strengthened to require exact finite full logits as
well as reuse, to compare the actual cancelled history rather than independently
sampled histories, and to reject a request that fails to publish its own vector.

| Original failure | Baseline | Candidate |
| --- | --- | --- |
| Store retains boundary 512 for a 600-ID follow-up | RED, evicted | GREEN |
| Store retains boundary 1024 below an edit at 1500 | RED, evicted | GREEN |
| Engine: shortened 600-ID history | reuse 0, exact | **512 reused**, exact |
| Engine: edit at 1500 | reuse 0, exact | **1024 reused**, exact |
| Continue after token-graph abort | reuse 0, exact | **3072 reused**, exact |
| STOP after first PP of a 9216-ID prefill | reads 18/18 chunks | reads **2/18**, emits **0 tokens**, cancel |

The STOP sender and engine reader run concurrently: the second chunk can already
be in flight when STOP arrives. The contract is to finish that chunk, not abort
halfway through its writes. STOP after PP 3 likewise finished at PP 4. Repeated
prefill STOPs emitted no tokens and the next request remained exact.

## Changes and invariants

- **Retention:** keep early anchors and the newest boundary instead of evicting
  the oldest snapshot. The same 128 MiB payload budget still holds three complete
  42,024,960-byte frames. One-frame budgets remain newest-only. Pruning still
  invalidates descendants before a diverging branch overwrites absolute rows.
- **Decode abort:** the graph drains before returning. Its partially written
  current row is beyond every completed prefill checkpoint. Keep those checkpoints
  and their still-owned absolute attention prefix, trim the live list to consumed
  positions, and invalidate transpose mirrors. The next request restores a
  checkpoint, not a partially decoded ring. Without a checkpoint, retain the
  original cold-reset fallback.
- **Prefill abort:** poll STOP between completed chunks. Publish only consumed
  prompt IDs, retain completed checkpoints, report the actual fresh read count,
  emit DONE/cancel with zero generated tokens, and never begin decoding just to
  acknowledge a cancelled prompt.

This bounded retention policy is not optimal for every edit location. Reserving
old anchors can leave fewer intermediate boundaries than FIFO; an intermediate
rollback may therefore need more recomputation. The unchanged append/cold speed
measurements below do not promise identical latency for arbitrary edits.

## Accuracy and workflow checks actually passed

All compared vectors contain **128000 finite FP32 logits**, byte-identical on
identical forced histories. Greedy-ID agreement alone is not the accuracy gate.

- Full/long matrix: **24 cases**, including 8191/11000/32768 histories, edits,
  branch/reset ownership, one-token suffix, forced continuation, repeats and STOP.
  The 32K extension actually reused **32256** positions.
- Basic stress **16**, prefill STOP **4**, decode STOP **4**, one-frame budget **4**:
  all full-vector comparisons exact. Shortened/mid-edit and decode-abort rows now
  have positive reuse, not cold fallbacks disguised as hits.
- Historical degeneration: **6 streamed + 6 pool** cases pass. Stream-growth:
  both sequences, **4 long-request vectors**, remain exact.
- HTTP stress: **27 steps**, **24 comparable full vectors**, all exact on/off;
  **26 rendered prompt-ID lists** match by scenario. Actual read_file tool-call
  envelopes were validated, not just manually supplied tool history. Synthetic
  tool results, parallel calls, malformed/empty/large results, changed tool lists,
  truncated replies, refusals, streamed disconnects, one-token loops and two
  concurrent clients are covered. No generated tool request was executed.
- HTTP prefill-disconnect replay reused **1024**; decode-disconnect replay reused
  **2560**, rather than zero. A cancellation may validly retain only one completed
  512-token chunk; the HTTP hit assertion now checks completed work, not a demand
  to continue processing after cancellation.
- **24 additional baseline-versus-candidate full vectors** (12 in each cache mode)
  match byte for byte, not merely candidate cache-on against candidate cache-off.

Concurrent requests are compared by scenario, not assumed completion order, and
rendered IDs are matched to the concurrent client's unique synthetic prompt.
The final reuse probe explicitly establishes its owner before the extension.

## Same-day unprofiled speed control

Explicit `numactl --cpunodebind=1 --membind=1`, 14 lanes, 128K allocated context,
512-token chunks, streaming demanded, device activation/pairing on, F16 experts
OFF, POOL_ALL=1, prefix reuse/alignment on, 128 MiB checkpoints. Each binary uses
fresh cache-off and cache-on server processes and two cold requests per arm.

| Prompt phase | Qualified baseline | Candidate |
| --- | --- | --- |
| Cache-on cold 2706-ID prompts, tok/s | 149.9 / 150.7 | **150.7 / 150.5** |
| Cache-off cold, tok/s | 154.4 / 152.6 | 150.9 / 151.3 |
| Growing turn prompt ms, 2560 cached IDs | 2149.6 / 2169.8 | **2153.0 / 2171.3** |

These are prompt-phase times, not whole-response times. Cache-on cold means no
matching transcript, not a new model process for every request. Sampling is
explicitly greedy for comparison; it does not change the saved temperature 1.0.
No GPU clocks or global machine policies were changed.

## CPU/server verification and explicit limits

Passed: CPU tools/setup **219 passed, 22 skipped, 1 optional network check
deselected**; full server suite **1257 tests OK, 31 skipped**. CUDA build passed
with existing unused-symbol warnings; the serving suite still emits existing
telemetry ResourceWarnings.

Not an overall repository test pass: root pytest collection stops in
`tools/trellis/test_blockldl.py` because a standalone benchmark parses pytest's
arguments during import; the attempt reported **55 collection errors**, log
`root-collection.log`. A broader combined tools/setup/Terminal run timed out at
**1300 s** without a final result; its capture buffers were lost on timeout.
That command is **incomplete**, not passed. The scoped CPU and full serving
commands above were then rerun separately with durable logs and passed.

HIP/SYCL builds/runtime, other GPUs, actual 128K-token histories, forced CUDA
allocation failure and exhaustive cancellation instruction timings are not run.
No model coding-quality claim is made.

**F5 clarification:** the earlier concurrent test submitted different first
prompts. Zero reuse for both is correct; neither had a saved matching prefix.
The A→unrelated-B→A case is covered by the branch/reset matrix: returning to A
correctly goes cold and stays exact. The engine remains a bounded **single live
history**, not an independent multi-session cache. That separately documented
limit was not one of the six RED assertions and is not claimed fixed here.
