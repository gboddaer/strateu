# strateu - a Strata fork for European LLMs

Scope: run **European open models** with Strata's efficiency design. First target: **Kolibri-1**
(Q4_K_M, 50 layers x 384 experts, 2560 embd, 128K context).

Status: **v0.1.0-alpha**. Every number below was measured on one box (16 GB Tesla T4, sm_75, 755 GB DDR4),
two runs where it matters, with the box load stated.

**Read [`docs/strateu-campaign-report.md`](docs/strateu-campaign-report.md) before building or porting.**
It carries the full measured state, the accuracy evidence, the limits of this configuration (CUDA-only,
16 GB T4, pinned-RAM requirement, no cross-request KV reuse, and the long-context sensitivity of *both*
engines), the dead ends that must not be re-tried, the two remaining work packages, and a
bring-up checklist for moving to a different accelerator — the intended next target being an
**AMD Radeon AI PRO R9700** (32 GB), where several of this box's verdicts must be re-derived rather than
inherited.

## Platform support

| backend | status |
|---|---|
| NVIDIA CUDA (sm_75 measured) | **supported, measured** |
| AMD HIP / ROCm | **planned, untested** - upstream paths exist (`docs/AMD_HIP.md`), but this fork ships no AMD measurements and makes no AMD claims |

## Quickstart

```bash
./run-kolibri1.sh --detach         # serve on :8080 (needs a Kolibri-1 GGUF; see MODEL=)
./run-kolibri1.sh --print-config   # what would run, launching nothing
./run-kolibri1.sh --stop
```

The design is the point: all experts stay pinned in RAM (42.90 GiB), a bounded slice of hot experts is mirrored in VRAM,
**cache misses are computed in place by the CPU pool over the pinned table** (no PCIe round trip for them), and the prefill
runs as a staged pool-carried pipeline.

## What is measured

Measurements follow [the measurement guideline](docs/MEASUREMENT_GUIDELINE.md): every rate is a
wall-clock two-run with the harness recorded, cross-checked before it is published.

| gate | target | measured | standing |
|---|---|---|---|
| prefill, 512-token prompt | 200 tok/s | **157.0 / 160.3** (wall clock) | not met |
| prefill after 32K of context | 150 tok/s | **154.0 / 153.7** (wall clock) | met |
| decode, short context | 30 tok/s | **24.5 / 24.4** (wall clock) | **not met** |
| decode after 32K | 20 tok/s | **12.3 / 12.2** (wall clock) | **not met** |
| KL vs llama.cpp (109 answer steps) | < 0.02 | **0.00132**, top-1 0.982 | met - 0.74x of llama.cpp's own CPU-vs-GPU floor |
| plain-path oracle | bitwise | **BITWISE** | met, re-anchored after every numerics change |
| prefill determinism | two runs identical | **bitwise identical**; `compute-sanitizer racecheck` 0 hazards | met (a missing `__syncthreads` in `block_max` was the cause) |
| 100K needle, five fixtures | retrieval recall | **recall 4/5, exact 3/5** | deviations below |

## Known gaps (recorded, not hidden)

* **Short-prompt prefill is below its 200 tok/s target** (150 cold). ~17% of a cold run is one-time state (expert-cache fill,
  handles, pinned warm-up) that the serve path amortises - hence 176 warm. Five candidate levers were measured dead
  (`DENSE_F16`, the router path, `RES_TOP` either way, a per-layer promotion budget, and overlap re-scoping); the remaining
  gap needs a fast Q4_K-from-VRAM GEMM, which this tree does not have.
* **100K needle**: on two 98,591-token fixtures the shipped path misses the framed code word - one by a single letter
  (`quart` for `quartz`, on a fixture where llama.cpp is also wrong and gets the number wrong), one by answering a different
  instruction in the haystack. At that length the model's output is chaotically sensitive: changing only llama.cpp's prefill
  batch width (2048 -> 1024) turns its own answer into a different question, reproducibly. Long-context token-level
  comparisons are therefore only meaningful between *identically configured* engines.
* **No prompt/KV reuse across requests** on this engine path (`cache_n = 0`): every request re-reads its prompt.

## Model paths

Harnesses default to `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`; every one honours
`KOLIBRI_MODEL` / `STRATA_KOLIBRI_GGUF` instead.
