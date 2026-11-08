# Further Kolibri optimization paths — listing only

Updated 2026-11-08 after overnight testing, branch `amdw7800probuild`, engine source at `6c241aa4`.
**Candidate mechanisms remain unconfirmed; no new optimization implemented.** Overnight measurements below
change their priority, not their qualification. This ranking is for long-context performance on the W7800;
CUDA and short interactive requests can have different bottlenecks.

The [overnight test summary](kolibri-w7800-overnight-long-context.md) records the measured coverage and limits.
Follow-up implementation is planned on `amdw7800probuild_optimize`, branched from `amdw7800probuild`;
the historical measurements remain those of source `6c241aa4`.

## Revised ranking from the overnight evidence

| Rank | Original ID | Path | Why this priority now |
|---|---:|---|---|
| 1 | 3 | Long-attention score-buffer/head tiling | Actual prefill fell from 76.34 tok/s at 32768 input tokens to 51.29 at 65537 and 20.62 at 131000. The middle 131K probe spent 6352.48 s in prefill; packing/fault time was only 86.15 s (1.36%). This rules out CPU packing as the dominant wall cost there, but does NOT yet prove attention is the entire remainder. The source's two-query tile at full context is the leading profiling target. |
| 2 | 2 | Batch selections with the exact MMVQ row computation | Addresses repeated expert launches without deliberately changing reductions. Relevant to decode and fallback expert work. No overnight kernel trace yet: potential, not a measured gain. Do not reuse the rejected grouped-decode substitution. |
| 3 | 1 | Reuse identical activation quantization | The clearest small, arithmetic-preserving code opportunity: gate/up consume identical input but quantize separately. Lower expected ceiling than attention, lower implementation risk than a new multi-selection kernel. |
| 4 | 7 | Cache allocation/admission by marginal saved bytes | At 65537 tokens balanced packed/transferred 294.02 GB in prefill, including 120728 repeated packs; demand used 37.89 GB with 1815 repeats. This supports examining placement/repeated staging before trying faster copies. Demand already avoids much of the problem; do not claim another 7.8x gain is available in demand. |
| 5 | 4 | Remove redundant drains using precise dependencies | Long inputs execute many chunk/layer transitions. Identify actual idle gaps after the attention/expert trace, then replace only genuinely redundant host waits. Old-reader and source-lifetime safety remain mandatory. |
| 6 | 5 | Batch promotion fences per round | A narrower scheduling enabler, potentially useful during the long request's many replacement rounds. No isolated promotion-overhead measurement yet, so it stays below paths with stronger evidence. |
| 7 | 6 | GPU cache-hit dispatch / cache-hit graph | Potential warm-decode improvement, but needs a new coherent miss path and lifecycle tests. Existing CPU-table capture is not interchangeable with mmap. No server authorization or protocol qualification yet. |
| 8 | 9 | Bounded CPU packing/source-fault pipeline | Still relevant to cold, shorter prompts: packing/fault time was 55.10 of 105.29 s at 4,097 tokens. Its fraction dropped to 1.36% at 131K. Move this higher if first-response latency, rather than very-long prefill, is the objective. |
| 9 | 8 | Host-transfer ceiling/locality diagnosis | Could help startup and miss-heavy phases, but cannot explain a 6352-second phase with only 43.88 GB of expert uploads. Do not blame endpoint width: read-only sysfs reported Gen4 x16. Measure copy-engine/driver/source behavior separately after GPU tests. |
| 10 | 10 | Simplify the fully resident fast path | A plausible low-ceiling cleanup after the larger launch/attention issues. Existing zero-overflow and warm-admission fast paths already remove much of the work. |

Additional interpretation: at 65,537 tokens learned replay reduced prefill uploads to 0.929 GB, but loaded 33.34 GB
at startup. Total wall time was 1,279.4 s versus 1,282.8 s without a saved profile; most benefit moved work between
phases rather than materially improving this complete first request. Do not rank profile seeding as the next
large long-context end-to-end win. These runs used diagnostics and uncontrolled filesystem cache; they are
bottleneck evidence, not final unprofiled speed qualification.

Before the next optimization: correct the controller's prompt/output budget to match the CLI and finish the
near-boundary, interrupted and serial cases. The four exact-edge rejections were a harness error, not an
engine inference failure. Keep the CUDA independent-reference fidelity gap separate from cache-path identity.

Suggested next sequence: profile rank1, then test its existing knobs under strict bitwise gates; implement
rank3 as the first small low-risk code experiment, and rank2 as the larger launch-reduction experiment.

## Detailed candidate notes (original identifiers)

| ID | Path | Evidence / hypothesis | Required check before keeping |
|---|---|---|---|
| 1 | **Reuse identical activation quantization for gate/up.** | `kolibri_engine.cu:1943–1962` quantizes inside every `gemv`; the expert gate/up calls at `3180–3183` consume the same unchanged `s.h`. Quantize once and pass the exact q8_1 bytes to both products. Q/K/V and shared-expert pairs may offer the same opportunity. `src/kernels/cuda/shared_expert.cu` already demonstrates explicit q8_1 reuse. | Count duplicate quantization launches first. Preserve rounding, stride, stream and activation-generation ownership. Do not cache merely by pointer: scratch addresses are reused for new values. Full-logit bitwise gate; mixed Q4_K/Q6_K and spill coverage. |
| 2 | **Batch selections while retaining the exact MMVQ row computation.** | `gemv` launches `native_mmvq` separately for each selected expert (`1957–1960`). A selection/grid dimension could index different weight pointers while reusing the existing per-row reduction. Dispatch lives in `src/kernels/cuda/native_mmvq.cu:3477`. | Profile launch overhead. Preserve each selection's quantized input and each row's exact reduction; validate all formats. This is NOT the rejected cached `native_expert_grouped` substitution, which failed bitwise qualification. Reject any changed logits. |
| 3 | **Revisit long-attention score-buffer/head tiling at actual 32–128K input.** | Default score cap is 48 MiB (`173–179`); `3713–3714` makes query tile size inversely proportional to visible keys. With 48 heads and ldp=131072, the formula permits only **2 query rows/tile**. Existing `STRATA_KOLIBRI_ATTN_MB` and `STRATA_KOLIBRI_ATTN_KG` provide bounded experimental routes. More score workspace may be worth fewer expert slots at long context. | Use overnight results to locate the cliff. Measure scores/softmax/PV and launch counts separately; include workspace/cache tradeoff and reserve. These knobs already exist and are not an untried universal win. GEMM shape/algorithm changes may alter arithmetic: require strict bitwise agreement, or separate numerical approval if it fails. Do not retune during tonight's control campaign. |
| 4 | **Replace redundant host drains with explicit local dependencies.** | The slot path synchronizes `g_st` after resident work and again before shared combination (`3187`, `3228`); prefill has additional metadata-lifetime drains (`4012–4072`). Some may be redundant for one ordered compute stream, but others protect stack metadata or buffer reuse. | Trace the real dependency graph. Retain old-reader and weight-ready fences, metadata ownership and default/nonblocking-stream ordering. Delayed DMA/readers, cancellation, reset and generation rollover tests must pass. Never simply remove all synchronizations. |
| 5 | **Batch hot-promotion fences per round.** | `slot_promote_layer:5701` fences each moved expert; `slots_reselect:6005–6009` already owns the whole move list and drains the round. One prior-reader fence and one final readiness dependency might cover a batch of independent moves. | Demonstrate promotion-heavy overhead, not just warm hits. Preserve map publication, old-reader protection, two-buffer source ownership and all in-flight generations. Do not remove the safety fences added in the earlier campaign. |
| 6 | **GPU-side cache-hit dispatch / capture a cache-hit subgraph.** | Every decode layer copies router IDs to the CPU (`3066–3071`), including learned-profile replay with no weight fetches. GPU-side residency lookup and exact batched dispatch could avoid host round trips on all-hit layers. | Measure D2H/host scheduling first. Miss handling, map generations and cache replacement must remain coherent. Existing CPU-table decode capture/resident-parent modes are not automatically suitable for mmap mode. High lifecycle risk: requires explicit reset/cancel/reuse qualification, not just a warm microbenchmark. |
| 7 | **Allocate permanent slots by per-layer marginal benefit per byte.** | Cache fitting currently chooses a common slot count, while layer routing entropy and Q4_K/Q6_K blob sizes differ (`kolibri_hot_policy.hpp:29–65`; engine cache allocation and hot profiles). One additional slot need not save the same traffic in every layer. | Collect misses/bytes by layer and workload, then test a fixed-total-VRAM allocation. Include per-layer zero/spill indices and worst-case nonresident staging. This is a placement-only hypothesis, not a proposed router bias or fewer experts. Hold out prompts to avoid tuning solely to one profile. |
| 8 | **Investigate the host-transfer ceiling before inventing more packing variants.** | Earlier local probes found roughly 2 GB/s across several transfer sources. Read-only sysfs on 2026-11-07 reports both AMD endpoints at 16 GT/s, x16, matching their advertised maximum. A narrow endpoint link is therefore not established as the cause. | After overnight work, separately measure SDMA/copy-engine behavior, pageable vs pinned copy, CPU memory throughput, upstream topology, driver/IOMMU and source locality. Do not change clocks, firmware or system settings without approval. Relevant mainly to cold/miss-heavy runs; zero-transfer warm decode cannot benefit. |
| 9 | **Pipeline bounded CPU packing only where source-fault/packing time dominates.** | `pack_blob:385–398` copies three mmap spans; `np_admit:2911–2929` packs/fills sequentially on the host. The current ring overlaps DMA but is deliberately single-producer. Cold admission might benefit from bounded prepacking of already-known requests. | Separate file faults, memcpy, event waits and DMA. Preserve mmap backing and the host-memory ceiling; no whole-table copy/pinning. Threads must not concurrently acquire the single-producer ring. Exact-demand read-ahead and projection-major startup packing already failed: do not relabel and repeat them without a new workload-specific reason. |
| 10 | **Avoid unused expert-side scratch/launch work when the selected set is fully resident.** | The zero-overflow path already skips its GEMVs, but the resident/missing framework and metadata are still prepared each layer. All-hit learned replay is a useful control for evaluating a simpler exact fast path. | Count actual remaining work before coding; expect potentially small gains. Preserve selected-expert order, zero/sign behavior, final weighted sum, poison-zero diagnostics and mixed-format fallback. Do not fuse/reorder the reduction unless separately qualified. |

## Separate correctness prerequisites, not speed promotions

- Investigate the recorded CUDA GPU-only pinned-table prefill `status=7` failure. It predates the mmap change;
  the table-to-mmap decode identity check does not fix or qualify that prefill path.
- Investigate the T4 GPU-strategy/reference fidelity gap (mean KL about 1.10 versus about 0.074 for the CPU-pool
  control on the recorded short history). Different activation quantization/accumulation is a numerical project,
  not an ordinary cache optimization. Matching an existing GPU baseline is not independent model qualification.
- Diagnose the rejected cached grouped-decode prototype before reconsidering it. Its failure is not permission
  to loosen tolerance or evidence that a different fused implementation is safe.

## Decision rules

1. Rank actual bottlenecks from overnight logs before choosing the next implementation.
2. For scheduling/layout-only paths: finite, complete, bit-identical logits on aligned histories, including
   long-attention and 513-row/chunk transitions, replay and miss-heavy cases.
3. Then run same-day paired timings without per-blob diagnostics. Separate startup, prefill and steady decode;
   report filesystem/profile cache state and hardware. Enablers can be retained without an isolated speed claim.
4. Any changed arithmetic requires the separate numerical/task-quality process and explicit approval.
5. Do not launch a server for lifecycle qualification without user authorization. No new CUDA/HIP runtime
   qualification is implied by this list.
