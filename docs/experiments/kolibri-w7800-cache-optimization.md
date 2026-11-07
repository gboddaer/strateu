# Kolibri mmap cache: W7800 optimization campaign

Measured on 2026-11-07 UTC, Radeon PRO W7800 (gfx1100), Debian 13, 30 GiB system RAM,
Kolibri-1-Q4_K_M, HIP. The engine allocated 131,072-token context capacity. This was not
a full-128K input or retrieval test. No server was started.

## Result

Keep the new **demand** profile for low startup traffic and reuse of requested experts.
Keep **balanced** as an eager-loading alternative. Keep **pipeline** as an optional transfer
profile; it did not consistently beat balanced with a large cache. Do not change the existing
CUDA setup profile or any default based on these AMD results.

All profiles require `--expert-no-pin`. They do nothing to other model families.

```sh
STRATA_KOLIBRI_NOPIN_PROFILE=demand \
  build-kolibri-hip/strata-kolibri --model /path/Kolibri-1-Q4_K_M.gguf \
  --expert-no-pin --expert-slots auto --expert-slots-hot \
  --expert-slots-profile kolibri.profile --ctx 131072 --vram-reserve-mib 1536 \
  --prefill --tokens-file tools/kolibri/oracle_ids512.txt --n 16
```

Use the existing ROCm library path and GPU selection for the machine. `--tokens` takes IDs,
not English. The integrated `build-kolibri-hip/strata` family-dispatch path also passed a
chat-template smoke test and answered `Paris` to the capital-of-France question.

## Speeds, with the limits attached

These runs use the same 512-token oracle input, 128 fixed-history decode steps, Q4_K_M,
128K allocation and 1,536 MiB requested reserve. The timed prompt phase processes 511 tokens;
the final prompt token is processed at the first decode step. Every step dumps complete logits.
These are fixed-history throughput measurements, not a general chat-performance guarantee.

| Run | Startup | Prefill | Decode | Startup + prefill |
|---|---:|---:|---:|---:|
| Original control, partial saved profile | 30.96 s | 11.23 tok/s | 6.18 tok/s | 76.47 s |
| Final balanced, same initial profile | 127.11 s | 24.61 tok/s | 12.83 tok/s | 147.87 s |
| Demand-equivalent switches, same initial profile | 6.47 s | 20.07 tok/s | 10.08 tok/s | 31.93 s |
| Demand, no saved profile | 1.48 s | 34.56 tok/s | 14.91 tok/s | 16.27 s |
| Demand, learned profile replay | 11.37 s | 113.61 tok/s | 17.01 tok/s | 15.87 s |
| Final rebuilt demand, learned replay | 11.37 s | 108.34 tok/s | 16.90 tok/s | 16.08 s |

**No saved profile does not mean cold filesystem cache.** The OS file cache was not purged.
The learned-profile rows replay a profile produced on this workload: they demonstrate reuse,
not the hit rate expected for unrelated prompts. Different initial profiles are called out
rather than treated as identical placement controls. Every row nevertheless produces exactly
the same complete output logits on the aligned history.

Balanced had other measured 128-step decode results of 14.80–14.88 tok/s. Its prefill ranged
roughly 25–43 tok/s in the selected long-step runs. File-fault/packing time varied substantially;
there is no universal prefill speedup factor. The dataset keeps the slower runs too.

The original control is the instrumented pre-fix engine at `376a9e9d`; `0d757d97` fixes seed
padding. The current `baseline` profile disables the subsequent experiments, but retains that
fix and the smaller shared host stage: it does not reproduce the old bug.

## What was tried

| Path | Evidence and decision |
|---|---|
| Partial-profile cache top-up | **Keep.** Previously 14,314 of 16,300 startup fills repeated expert 0. Fill unused slots with distinct experts. Full logits match. This alone increased measured 16-step decode from 6–7 to 10.7–12.2 tok/s, but increased startup reads. |
| Two-buffer pageable/pinned transfer ring | **Keep as an enabler, not a universal default.** Bounded host memory and event ownership permit overlap. Pageable staging remained available. Large-cache results were mixed. |
| Coalesced transfer fences | **Keep.** One last-copy event covers the copy queue; do not enqueue the same completed generation's wait for every layer. Tests verify second-stream dependencies too. No isolated speed claim. |
| Skip all-zero overflow calculations | **Keep.** Same fixed 128-step history: 10.33 to 12.42 tok/s; complete logits identical. Preserve the final combine and the poison-zero diagnostic. |
| Retain recent misses in existing spill slots | **Keep.** 395 to 321 weight fetches over 128 steps; no additional VRAM. Protect all current picks before selecting a spill victim. |
| Joint cache/stage fitting | **Keep.** Staging needs only the complement after admission, not another whole layer of experts. Cache capacity rose from 326 to 334 experts/layer. Actual free VRAM remained about 1,626–1,628 MiB, above the reserve. |
| Shared blocking host staging | **Keep.** About 686 MiB in 50 buffers becomes one roughly 15 MiB buffer. Blocking copies complete before reuse. This is a memory reduction/enabler, not an isolated speed claim. |
| Resident compute / missing-weight overlap | **Keep in pipeline/demand, optional for eager loading.** The warmed 8-slot stress case increased from 3.07 to 3.29–3.31 decode tok/s and 3.09 to 3.51–3.58 prefill tok/s. Large-cache eager runs did not consistently improve. |
| Exact-demand `MADV_WILLNEED` | **Reject.** No reliable end-to-end gain; page-fault variance remained large. Removed experimental code. |
| Projection-major startup packing | **Reject.** 158.5 s startup versus 102.3 s in its comparison; no demonstrated gain. Removed experimental code. |
| Standalone startup DMA batching | **Reject.** 119.8 s versus 102.3 s; no demonstrated gain outside the ring pipeline. Removed standalone switch. |
| Lazy permanent-cache admission | **Keep.** Reserve the full cache, but seed only the supplied profile and load other experts when selected. Empty-profile startup needs no expert copies. Needed experts remain in free permanent slots; existing hot replacement takes over once full. |
| Cached grouped-decode prototype | **Reject on correctness, not a timing decision.** 511,999 of 512,000 logits differed; maximum absolute difference 1.4207089. Outputs were finite, but that is not enough. The prototype was removed; no tolerance was relaxed. The cause was not separately qualified. |

Lazy admission was the important follow-up to the seed fix: eagerly reading all 40 GB of
reserved expert capacity was unnecessary. In the empty-profile run, prefill loaded 7,660
experts (18.36 GB) and decode loaded 643 more, with no repeated weight loads. The learned
replay seeded 8,303 experts (19.90 GB), then needed no weight transfers during prefill or
decode. The last diagnostic replay had zero major faults in those two phases. This working
set fits the host's file cache; eagerly scanning the full model can evict useful file pages.

## How the retained path works

- Dense weights, KV and workspace are in VRAM. Auto-fit budgets permanent slots, the zero slot,
  six spill slots per layer, temporary prefill staging, the requested reserve and allocation margin.
- `demand` requires hot, per-layer residency. Unfilled permanent slots are invalid, not duplicate
  copies of expert zero. A selected uncached expert claims a free slot and is copied there.
- When permanent capacity is full, misses use the existing spill slots. Repeated picks reuse
  their weights; the existing decayed-usage policy controls permanent promotion and eviction.
- CPU work is routing coordination, byte packing and transfers. Expert matrix products, routing
  math and attention remain GPU-side. No new quantization, CPU expert products or router bias.
- Pipeline buffers own their in-flight copies. GPU consumers wait for weight-ready events;
  overwrites wait for prior readers. Constructor failure, delayed DMA/readers and drain/replay
  have explicit tests. Shared activation scratch stays on the ordered compute stream.
- The entire model is **not pinned**. Pipeline/demand opt into two small pinned transfer buffers,
  capped at 64 MiB total (about 30 MiB for this model). `STRATA_KOLIBRI_TRANSFER_RING=pageable`
  selects unpinned transfer buffers; `off` selects the shared blocking pageable stage. Small
  existing router/control buffers are still pinned in all modes.

## Validation

Passed:

- Both integrated `strata` and standalone `strata-kolibri` HIP builds; full HIP test-tree build.
- `kolibri_hot_policy_test`, `kolibri_expert_transfer_test`, `kolibri_decode_wait_test`,
  `kolibri_window_test`, `kolibri_expert_stream_plan_test`.
- Transfer tests: delayed source reuse, delayed old device reader, two consumer streams,
  partial batches, bounds, injected allocation failure, drain/reset/replay.
- Entire `test_unpinned.py --phase all --hot` with the demand profile: decode, repeated
  prefill, 8-slot versus auto-cache full logits, 1,025-token chunk/window transitions,
  CLI profile save/reload, finite values and invalid CPU/grouped combinations.
- Six GGUF tokenizer tests, two CPU-only benchmark argument/input-preservation tests, and integrated-engine chat generation.
- All 44 successful measured runs have finite complete logits; hashes are identical within
  each of six aligned history/length classes. The largest comparison covers 16,384,000 logits.

Failed/rejected: the grouped-decode prototype failed the strict bitwise gate; OS read-ahead
and standalone startup packing/batching did not demonstrate a performance gain. Initial
RED tests intentionally failed before implementation. A missing transfer dependency in the
rollover-injection executable was found by building that target and fixed; the full tree now links.

Not run: CUDA and SYCL builds/runtime (toolchains unavailable), full-128K input/retrieval,
server/API runtime, protocol-level reset/cancellation/prefix reuse and general task-quality
campaigns. The rollover executable's CPU-handoff runtime is not a valid no-pin profile; only
its build was checked. Transfer-level reset/drain tests are not server cancellation qualification.
Independent subagent review was not run; delegation was not authorized. This is a local HIP
qualification, not a claim of package or cross-backend release readiness.

## Evidence and reproduction

[CSV of every successful measured run](kolibri-w7800-cache-optimization.csv) records phase
rates, UTC timestamps, switches, binary hashes and complete-logit hashes. The fixed initial
profile is [committed here](kolibri-w7800-cache-input.profile). Failed prototype logits, source
snapshots, raw logs, binaries and full distributions remain locally under
`.superpowers/sdd/2026-11-06-kolibri-cache-transfer-optimization/`.

```sh
python3 tools/kolibri/bench_unpinned.py --bin build-kolibri-hip/strata-kolibri \
  --model /path/Kolibri-1-Q4_K_M.gguf --out /tmp/kolibri-measure --label demand-first \
  --prompt 512 --n 128 --env STRATA_KOLIBRI_NOPIN_PROFILE=demand
python3 tools/kolibri/bench_unpinned.py --bin build-kolibri-hip/strata-kolibri \
  --model /path/Kolibri-1-Q4_K_M.gguf --out /tmp/kolibri-measure --label demand-replay \
  --prompt 512 --n 128 --env STRATA_KOLIBRI_NOPIN_PROFILE=demand \
  --profile-input /tmp/kolibri-measure/demand-first.profile \
  --compare /tmp/kolibri-measure/demand-first.bin
STRATA_KOLIBRI_NOPIN_PROFILE=demand python3 tools/kolibri/test_unpinned.py \
  --phase all --hot --bin build-kolibri-hip/strata-kolibri --model /path/Kolibri-1-Q4_K_M.gguf
```

`--stats` adds optional packing, repeated-fetch and Linux process-fault/I/O counters; it does
not add stream drains. Retain an otherwise identical run without those per-blob diagnostics.
Rejected experimental switches in historical CSV rows are not supported by the final engine.
