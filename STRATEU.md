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
| AMD HIP / ROCm | **supported, measured on a Radeon AI PRO R9700** (gfx1201, 32 GB, ROCm 7.2.2) - see [`docs/AMD_R9700_PORT.md`](docs/AMD_R9700_PORT.md) for every number and [`docs/AMD_HIP.md`](docs/AMD_HIP.md) for the build |

### The AMD floor table

Measured on host `rizo`: Ryzen 7 5700G (8 physical cores, one NUMA node), 93 GiB RAM, 89-91 GiB free
per run, GPU idle at start, `--cpu-threads 8` with the prefill pool on, **no `STRATA_KOLIBRI_*` env
set** - the shipped default, which is 3.8 s faster at this shape than the §9.3 arm baseline that
carries `EXPERT_F16=24`. Two runs each. Prefill subtracts the engine's own 13.4 s load line; the
end-to-end row does not.

| | CUDA box (16 GB T4) | **R9700 (32 GB)** |
|---|---|---|
| prefill, 512-token prompt | 157.0 / 160.3 tok/s | **87.8 / 85.4 tok/s** (26.3 / 26.2 end to end; prefill pool on, 8 threads) |
| prefill, 99,299-token prompt | - | **102.3 / 100.8 tok/s** (`ATTN_MB=512`), **105.2 / 105.8** (`ATTN_MB=1024`) |
| prefill, 127,811-token prompt | - | **94.1 / 94.2** (`512`), **100.4 / 100.8** (`1024`) |
| decode, short context | 24.5 / 24.4 tok/s | **15.8 / 15.9 tok/s** |
| decode, long context | 12.3 / 12.2 tok/s (after 32K) | **6.7 / 5.5 tok/s** (at 100K) |
| KL vs llama.cpp, answers | 0.00132 (0.74x of that box's floor) | **0.00146** plain prefill / 0.00162 pooled, top-1 0.9942 - **0.78x of this box's floor** (0.00187), measured on `strata-kolibri` itself |
| 100K needle | recall 4/5, exact 3/5 | **recall 6/6, 0 not exact** - re-verified live on six fixtures, longest 127,811 tokens |
| ctest | - | **60 / 62**, both failures box-scoped |
| pinned expert table | 42.90 GiB, needs `RLIMIT_MEMLOCK` | **42.90 GiB, no root needed** |

Three of the campaign's verdicts do not survive the move, and §9.3 said to expect that rather than
inherit it: the CPU pool is no longer a win (`POOL_ALL=0` measured faster), decode residency engages
well (61.2% of picks) and costs 11x, and `ATTN_MB=1024` is faster rather than an OOM. The full set of
re-derived verdicts, with the counters behind them, is in
[`docs/AMD_R9700_PORT.md`](docs/AMD_R9700_PORT.md).

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

Campaign II (`kolibri2-speed`) measures the same gates twice: on the **default path**, which is byte-identical to
`main` because the campaign's change is opt-in, and with **`STRATA_KOLIBRI_PF_PARTIAL=1`**, which is the campaign's
actual speed change.  Both columns are two-run wall clock from `tools/kolibri/bench.py` under
`numactl --cpunodebind=1 --membind=1`, box load stated, on driver 580.196.07 / kernel 6.1.0-52-amd64.

| gate | target | default path | `PF_PARTIAL=1` | standing |
|---|---|---|---|---|
| prefill, 512-token prompt | 200 tok/s | **156.0 / 159.6** (load 0.88 / 3.03) | **179.8 / 181.0** (load 1.32 / 2.50) | not met - ceiling 181.0, +16% over the default path |
| prefill after 32K of context | 150 tok/s | **119.6 / 119.4** | **125.7 / 122.1** (load 0.06 / 9.02) | not met - the box moved, see below |
| decode, short context | 30 tok/s | **30.3 / 30.3** (load 0.88 / 3.03) | **30.0 / 30.3** (load 1.32 / 2.50) | **met** |
| decode after 32K | 20 tok/s | **13.4 / 13.5** | **13.3 / 13.4** (load 0.06 / 9.02) | not met - ceiling 13.5, and it is bandwidth |
| KL vs llama.cpp (109 answer steps) | < 0.02 | **0.00123**, top-1 0.982 | **0.00133** | met - both under llama.cpp's own CPU-vs-GPU floor of 0.00178 |
| plain-path oracle | bitwise | **BITWISE** | **BITWISE** | met, re-anchored after every numerics change |
| prefill determinism | two runs identical | **bitwise identical**; `compute-sanitizer racecheck` 0 hazards | same | met (a missing `__syncthreads` in `block_max` was the cause) |
| 100K needle, five fixtures | retrieval recall | **recall 4/5, exact 3/5** | same | deviations below; unchanged by campaign II |

The 512-token prefill is cross-method: the serve path, a separate process with its own accounting, gives
**181.7 cold / 201.8 tok/s** warm against the wall clock's 179.8 / 181.0 - 1.1% on the cold figure, and the warm
figure higher for the documented one-time-state reason.  Decode agrees to 0.3% (serve 30.2 / 30.1).

Two of these numbers differ from the ones `main` published, and neither difference is this change.  The 32K prompt
read measured **154.0 / 153.7** on this box's earlier state and measures **119.6 / 119.4** today with the driver at
580.196.07; the 512-token figure did not move between those two states, which is what identifies the 32K shift as
box state rather than code.  The change never made it worse in any pair - the knob-on figure is at or above the
knob-off figure every time.  The short-context decode floor, published as 24.5 / 24.4, was measured with both
sockets; the same code under `--cpunodebind=1` measured 30.3 / 30.3 then and 30.3 / 30.3 now, so that floor is met
and always was once the binding is stated.

## Known gaps (recorded, not hidden)

* **Short-prompt prefill is below its 200 tok/s target** (181.0 with the campaign's knob on, 156.0-159.6 with it off).
  ~17% of a cold run is one-time state (expert-cache fill, handles, pinned warm-up) that the serve path amortises -
  hence 201.8 warm. Five candidate levers were measured dead (`DENSE_F16`, the router path, `RES_TOP` either way, a
  per-layer promotion budget, and overlap re-scoping), and campaign II measured the sixth: a grouped Q4_K-from-VRAM
  GEMM is **bitwise correct** (81920/81920 elements equal) but runs at **9.99 ms/layer against the FP16-resident leg's
  4.18**, so it is in the tree as a correctness gate and an A/B baseline, not as an engine path.  Closing the last
  19 tok/s needs a fused Q4_K tensor-core kernel that dequantises in registers into `mma.m16n8k16`; the traffic math
  says ~1.2-1.3 ms/layer is reachable, and the gate is 2.6.
* **Decode after a 32K prompt is 13.4 tok/s against a floor of 20, and the wall is PCIe, not kernels.**  Counters put
  **1.28 GB of PCIe per token** (488.7 MB device-to-host + 789.5 MB host-to-device) at **0% residency**, against a
  74.6-75.2 ms token of which the CPU pool drain is only 12.86 ms.  The residency paths that exist lose here: the
  id-based `--expert-slots` costs 17% of decode (25.0/25.1 against 30.3) because it caches experts *numbered* 0..N-1
  rather than the hot ones, and `STRATA_KOLIBRI_DEC_RES=1` costs 10% of prefill for -3% of decode.
* **100K needle**: on two 98,591-token fixtures the shipped path misses the framed code word - one by a single letter
  (`quart` for `quartz`, on a fixture where llama.cpp is also wrong and gets the number wrong), one by answering a different
  instruction in the haystack. At that length the model's output is chaotically sensitive: changing only llama.cpp's prefill
  batch width (2048 -> 1024) turns its own answer into a different question, reproducibly. Long-context token-level
  comparisons are therefore only meaningful between *identically configured* engines.
* **No prompt/KV reuse across requests** on this engine path (`cache_n = 0`): every request re-reads its prompt.

## Model paths

Harnesses default to `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`; every one honours
`KOLIBRI_MODEL` / `STRATA_KOLIBRI_GGUF` instead.
