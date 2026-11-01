# Kolibri bit-exact GPU-streamed expert prefill

## Status and intent

The operator approved the GPU-streaming direction and the design described in the conversation, then requested an implementation recipe for another agent. This document records that design. No implementation or numerical-change approval is implied by this documentation task.

Goal: make Kolibri-1-Q4_K_M prompt processing faster on the Tesla T4 by grouping routed tokens by expert, streaming compressed expert weights, and computing tiled expert products on the GPU. First prove the complete pipeline faster than the tuned CPU expert workload, then integrate it behind a new opt-in flag. Do not optimize generation in this project.

## Approaches considered and decision

1. CPU expert tiling, informed by local patched llama.cpp: plausible and potentially cheaper, but the operator explicitly chose not to pursue this route now.
2. Extend A1 selection-by-selection execution: rejected. It is bit-exact, but measured 2.02–2.49x slower at prefill because each selection incurs launches, transfers and waits.
3. Grouped, tiled, double-buffered GPU streaming: selected by the operator. Retain the existing arithmetic and CPU activation/quantization; overlap bounded batches of expert groups, not individual selections.

## Evidence, not promises

Parent implementation inspected: `/data2/develop/perf-gap`, branch `kolibri-a1-prefill`, commit `f076741b453ce14666a868b635e04c3dfa6be3a8`. It contains A1's Q4_K/Q6_K kernels and the failed A1 engine arm. Use this inspected commit, not an assumed later commit from a conversation summary.

T4 sm_75, CUDA 12.8, 2x E5-2690 v4, 755 GiB host RAM. GPU is on NUMA node 1. Model: `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`, 47,454,113,472 bytes. Measured tuned CPU prefill: 511 processed IDs in 7.989 s, approximately 64 tok/s. Longer prompts measured approximately 70 tok/s. A1: 16.164 s, 288.5 s and 344.407 s for 511, 8190 and 10999 processed IDs respectively. Those are prefill timers, not load-inclusive wall time.

Model: embedding 2560, expert intermediate 512, 384 experts, top-6, 50 layers; gate/up Q4_K; down Q4_K on 25 layers and Q6_K on 25. One measured 511-ID request streams 28.09 GB of unique touched expert blobs across its internal chunks. Existing measured host-to-device copy rate is approximately 9.56 GB/s. Dividing those numbers gives a conditional transfer lower bound, NOT a universal prefill ceiling. Actual chunking, touched experts, padding, activation traffic and overlap must be measured again.

The engine's 226 GFLOP/s denominator describes an existing CPU pool measurement, not an absolute CPU hardware roof. Existing phase counters include summed worker time and possibly overlapping wall intervals; do not turn them into exclusive wall-time percentages without reconciling them. Qwen decode statistics do not establish Kolibri prefill batch efficiency. No speculative decoding is part of this project.

## Arithmetic contract

- Reproduce `src/kolibri/kq_a1.cuh` against the real CPU kernels, not merely against another CUDA implementation.
- Preserve Q4_K's eight FP32 accumulation lanes and four minimum-correction lanes, block traversal order, explicit FMA behavior, negative minimum term and final reduction trees.
- Preserve Q6_K's lane-specific offset correction, pair of 16-value scales per 32-value chunk, 64-byte ql consumption per 128 values, and reduction tree.
- Parallelize rows/tokens and exact integer operations. Do not split FP32 accumulation across independent block ranges or use a different floating-point reduction.
- CPU computes `(g / (1.f + std::exp(-g))) * u`, compiled with the existing host arithmetic settings. Use `native_quant_h` unchanged. GPU `expf`, new activation formats, dequantization to FP16 and tensor-core FP16 GEMM are out of scope.
- Each GPU result is scattered to `ey_h[original_selection][embedding]`. Existing downstream top-k weighting and accumulation order remain unchanged.
- No non-finite outputs, stale buffers, races or unexplained repeat variability. Bit equality of NaNs is not a pass.

## Pipeline

Stable grouping maps selection `sel` to expert `ids[sel]`, and token `sel / top_k`. Reuse the engine's already quantized token activations. Transfer each touched expert blob once per layer/chunk; do not resend weights when a large token group is split into compute tiles.

Use two reusable slots. Each slot holds a bounded batch of expert groups, their compressed weights, gathered Q8_K activations, gate/up results, CPU-produced quantized hidden rows and down results. Begin with at most eight experts per batch and a 32 MiB compressed-weight capacity per slot. Flush a batch before either limit; groups larger than a compute tile stay within one batch. Derive other buffer capacities with checked arithmetic from the maximum request geometry. Total new device allocation is capped at 256 MiB; refuse an opt-in configuration that cannot fit, with required/available bytes reported. Do not evict KV, shrink context or silently fall back.

Kernels tile output rows and tokens, sharing weight decoding/loading across tokens. A batch descriptor permits different expert IDs, token counts and Q4_K/Q6_K down formats in replay tests. Engine calls normally contain one layer format. Fuse gate/up into one grouped launch if parity permits; one grouped gate launch plus one grouped up launch is an acceptable intermediate, but measure the difference.

GPU batch lifecycle:
`FREE -> H2D -> GU -> GU_D2H -> CPU_ACT_QUANT -> HIDDEN_H2D -> DOWN -> OUT_D2H -> FREE`.

Each slot has explicit non-default-stream work and completion events. The host advances whichever slot is ready; do not always block on slot zero. At the CPU boundary, another slot can transfer/compute. Slot weights remain resident until its down stage finishes. Source pinned buffers and output views remain alive until their last event completes. No steady-state cudaMalloc/cudaFree, cudaHostAlloc/cudaFreeHost, cudaDeviceSynchronize or per-selection event/wait loop. Layer completion must explicitly bridge back to the engine's consumer stream, or wait before publishing host results.

First implementation may compute CPU activation/quantization on the coordinator thread. Count its time in pipeline wall time. If it limits overlap, add a fixed host worker executor preserving per-element expressions; do not create threads per group or re-enter an occupied engine worker pool. The replay control must use a persistent tuned CPU pool, never spawn/join threads per expert.

## Scope and safety

- New flag: `STRATA_KOLIBRI_EXPERT_STREAM=1`; absent or `0` leaves existing execution untouched.
- Refuse simultaneous A1 (`STRATA_KOLIBRI_EXPERT_GPU`) or incompatible expert-cache/MMQ/other experimental prefill modes. The implementation must enumerate actual conflicts from the engine dispatch; no silent bypass or CPU fallback while claiming streamed coverage.
- Initial supported target: CUDA sm_75 on this model. CUDA/native-expert availability and supported formats/strides/geometry must be validated before work is published. Other backends retain their current path when the flag is off; setting it on an unsupported backend fails clearly.
- New source is CUDA-only with guarded integration. Shared-file edits still require backend build validation before review, or explicit NOT RUN status and no review-ready claim.
- No generation path changes, no default changes, no pushes and no HTTP/network server starts. Engine stdio `--serve` subprocesses used by the existing state-check harness are test processes, not network services; terminate them after each test.
- One GPU job at a time; poll progress in short intervals. Preserve all previous artifacts read-only.
- Separate implementation worktree, build and evidence tree. Protected: FP16 branch at e5f4815c, `/data4/kolibri-f16-evidence`, `/data2/develop/f16-control-5bbe6c4e`, parked `/data2/develop/native-q4q6` at 7c600d48, released defaults and their builds. Do not reuse their output paths.

## Acceptance and stopping rules

1. Kernel dot, CPU activation/quantization and expert-output parity: finite and 100% bit-identical on real CPU oracles, generated-valid blocks and captured real workload; include tails and both down types.
2. Representative replay: at least three paired alternating unprofiled CPU/GPU runs, same day, same workload and binding. GPU end-to-end median including all CPU activation and transfers must be at least 10% below the tuned CPU expert median (GPU/CPU <= 0.90). This 10% is an engineering go/no-go margin proposed by this recipe, not an accuracy tolerance or a promised speedup. Reproduce both short- and long-prompt group distributions. No integration until this gate passes; report a negative and stop if bounded kernel/pipeline tuning cannot pass it.
3. Engine: arm-off agrees with the pinned parent and preserved control where applicable; arm-on repeats and agrees bitwise on six existing forced histories (16, 512, German512, code512, 8191, 11000). Compare matching build/configuration and complete finite full-logit records. State tests cover first use, replay, prefix reuse, reset, cancellation and clean request after cancellation; existing token checks must be augmented with full-logit checks.
4. Final speed: paired unprofiled short/8K/11K prompt runs show lower median prefill time, separately reported from load/first-use and decode; no generation regression beyond the observed paired control spread. Inconclusive timings are not a win. Any default promotion is a separate operator decision.

If exact kernels cannot beat the CPU, stop and report. A numerical CUDA alternative needs its own approved qualification design under `docs/KOLIBRI_ACCURACY_POLICY.md`; this project does not authorize one. No speed result waives accuracy gates.

## Handoff

Implementation plan: `docs/superpowers/plans/2026-10-31-kolibri-gpu-streaming.md`. Open design choices left to implementer: register/shared-memory tile tuning within the fixed arithmetic contract, event polling strategy, and whether measured host activation pressure justifies a persistent executor. Record those decisions and the measurements supporting them in the new ledger.
