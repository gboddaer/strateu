# Kolibri P4 shipping validation

Measured on 28 October 2026: Ryzen 7 5700G, 93 GiB RAM, Radeon AI PRO R9700 32 GB (gfx1201), card0 only.
Model: `/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`. ROCm 7.2.2, hipBLASLt 100202.

## Default and fallback

The Linux gfx1201 default in an MMQ build is P4. It requires a compatible direct hipBLASLt tuning entry for the
shared expert's FP16 gate/up GEMM. The versioned table travels in `tuning/` beside the engine. The runtime validates
its architecture, library version and actual shape. An absent or rejected table keeps the CPU pool, with a notice.
Other cards and Windows retain the old default. `STRATA_KOLIBRI_PF_MMQ=0` explicitly selects the pool.

The source installer now configures the HIP Kolibri target with MMQ, separately from a cached CUDA build, and
passes the matching source tuning table in its launch config. CUDA setup is unchanged. HIP links the K-quant
instances statically: a separate `strata_kq` DSO is a CUDA requirement, not a HIP requirement.

FP16-resident expert layers retain their pool path: otherwise MMQ's device-only output would omit the resident rows
that the host scatter normally supplies. The two-chunk pipeline and experimental dense-kernel overrides also retain
the pool. An explicit `STRATA_MMQ_KQ=0` is honored. None of these paths is advertised as an active P4 run.

## Why the previous nondeterminism verdict changed

The first differing buffer was the shared expert's dense gate output. At that boundary, indexed bit-pattern hashes
of MMQ's products, the combined routed-expert buffer, router weights, normed input and residual matched. Per-group
device drains were not a fix: the repeat test also reproduced all-zero output with them enabled. A diagnostic run
exited with an unspecified launch failure; it was not counted as a passing run.

Disabling rocBLAS's hipBLASLt backend produced repeatable runs, but was not the selected solution. The packaged,
version-matched direct hipBLASLt table also produced repeatable results and passed the independent accuracy screen.
The engine now requires this direct route for P4. The exact internal vendor-library fault is **not established**.
A standalone fixed-input native GEMM probe did not reproduce the engine's failure; no upstream defect is claimed
from that probe. All temporary buffer-dump/checksum probes were removed.

## Repeatability and accuracy

- Before the fix, `tools/kolibri/check_repeat.py` failed on changing/non-finite logits and all-zero ids.
- With direct tuning and no group drain: six short and three 8K runs had bit-identical **full** logits and ids.
- With the new automatic default, no P4/K-quant environment switches: another six short runs passed the same test.
- The final relocated package, with no tuning-path override, passed six short and two 8K full-logit repeat runs.
  Disabling the table or explicitly disabling K-quants each passed two fallback repeats, without an MMQ activation marker.
- `--dump-logits` records only the top two values. The regression uses `--dump-full-logits` instead.
- Independent synthetic MMQ tests cover the actual Kolibri gate/up (1024 x 2560) and Q6_K down (2560 x 512), three
  uneven expert buckets and permuted input/output ids. Relative L2 errors: 0.001787 and 0.000395 respectively.
- The direct shared-GEMM route passed at T=255, 256 and 512 against hipBLASEx (relative L2 below 1.1e-6).
- The eight-prompt teacher-forced screen used 200 steps. llama.cpp GPU/CPU reference files were reused from the prior
  same-model campaign (`/tmp/kl-amd24`, sequences `/tmp/parity-amd24`); they were **not** rerun for this change.
  Both Strata arms were rerun. Pool: mean KL 0.00278, top-1 98%, ratio to CPU floor 1.04. P4: **0.00236**, **99%**,
  ratio **0.89**. Answers only (155 steps): P4 KL 0.00175, top-1 98.71%, ratio 0.98. These pass the documented
  mean-KL <0.02, agreement >=90%, and floor-ratio <1.5 screens. This finite fixture set is not a general quality proof.

The arm has its own in-repo smoke/hash files and `KOLIBRI_GATE_PROFILE=mmq-gfx1201`. The hash now reads MMQ's
actual device output; the former untouched-host-buffer hash could record zeros and was not a valid MMQ gate.
The profile refuses to record/check a silent pool fallback in either smoke or hash runs, and propagates engine
failures even after partial output. A negative harness confirmed exit 2 when an engine prints ids then exits 1.
`KOLIBRI_BUILD` selects the tree that the gate builds.

## Packaged-binary timing ledger

Every row below uses the **same staged binary**, card0, eight CPU pool threads, `EXPERT_F16=0`, `ATTN_MB=512`,
`RES_TOP=56`, and 128 generated tokens. P4 is the automatic default; pool is its explicit off switch. No deliberate
competing workload was launched. Load averages include these runs' own workers; other box activity was not controlled.
The stage uses its own runtime libraries and auto-discovers its own tuning table, without a tuning-path override.

Prefill milliseconds are **engine-reported** cumulative PFchunk time, excluding model loading. Generation tok/s is
**wall-derived**, from the unbuffered CLI's `gen:` marker to the final newline, including token selection/output.
Raw JSON also records process wall time and before/after load averages. The ranges are not hidden by reporting one
favorable run. In particular, the first packaged P4 short run was slower than subsequent runs.

| run | ctx | chunk | arm | prefill tokens | prefill ms (engine) | generation tok/s (wall) | 1-min load before -> after |
|---|---:|---:|---|---:|---:|---:|---|
| initial short 0 | 8192 | 256 | P4 | 511 | 4749 | 15.52 | 1.04 -> 3.08 |
| initial short 1 | 8192 | 256 | P4 | 511 | 3141 | 15.58 | 3.08 -> 3.58 |
| initial short 0 | 8192 | 256 | pool | 511 | 6687 | 15.48 | 3.58 -> 3.75 |
| initial short 1 | 8192 | 256 | pool | 511 | 6162 | 15.45 | 3.75 -> 4.25 |
| long 0 | 12288 | 512 | P4 | 8191 | 26080 | 11.70 | 4.25 -> 4.50 |
| long 1 | 12288 | 512 | P4 | 8191 | 26018 | 11.67 | 4.50 -> 4.54 |
| long 0 | 12288 | 512 | pool | 8191 | 79968 | 11.62 | 4.54 -> 5.55 |
| long 1 | 12288 | 512 | pool | 8191 | 79973 | 11.62 | 5.55 -> 5.58 |
| warm short 0 | 8192 | 256 | P4 | 511 | 3020 | 15.57 | 1.55 -> 3.23 |
| warm short 1 | 8192 | 256 | P4 | 511 | 3044 | 15.33 | 3.23 -> 3.53 |
| warm short 0 | 8192 | 256 | pool | 511 | 4926 | 15.26 | 3.53 -> 3.74 |
| warm short 1 | 8192 | 256 | pool | 511 | 6677 | 15.17 | 3.74 -> 4.21 |

Thus the paired packaged 8K result is **314–315 tok/s versus 102.4**, about **3.07x**. Warm short P4 is
**168–169 tok/s**; pool times were variable. Earlier source-tree runs measured 317.5 versus the historical pool's
123.3 tok/s (2.58x), but that is not substituted for the fresh paired ledger. Generation has no measured regression
in these pairs and no new P4 generation optimization. There is no corrected-P4 extrapolation beyond 8K.

Reproduce with `tools/kolibri/bench_p4.py --binary /path/to/strata-kolibri --model MODEL --tokens-file IDS
--arm default --runs 2 --out /tmp/p4-default`, then `--arm pool`. Use `--ctx 12288` for the 8192-id prompt.
The repeat tool additionally supports `--require-mmq` to reject a fallback masquerading as a P4 result.

## Validation and limits

Passed: both HIP build trees; full P4 `pf_gate.sh` including plain, no-prefill, prefix, 100 hash chunk-layers, oracle
BITWISE PASS and all six reuse stages; synthetic MMQ parity; direct GEMM shapes; automatic-default repeat tests;
65 packaging/tuning tests plus 15 subtests; 79 offline AMD/Qwen3.6/Kolibri source-setup tests. The legacy pool's
full gate also passed against its unchanged in-repo gfx1201 expectations. The fresh HIP source flags configured
successfully against the installed ROCm and local pinned GGML checkout.

Full CTest was run on both trees. Final MMQ: **83 passed, 6 skipped, 4 failed (93 total)**. Pool: **79 passed, 6 skipped,
4 failed (89 total)**. Both had the same failures: `vmm_test` (HIP VMM assertions), `ple_parity` (missing Qwen PLE
fixture), `platform_memory_test`, and `conversation_copy_streams_test` (first stream/event exceeds its memory limit).
The trellis MMQ cases explicitly skip host-reference types absent from this GGML build; Kolibri's Q4_K/Q6_K cases run.

The full Terminal suite did not finish within the attempted timeouts and is **not claimed green**. An independently
reproduced UI failure is `FolderStep::test_default_folder_and_free_space_and_change` (default folder stays None).
The packaging/tuning subset above passed. No CUDA, SYCL, Windows or non-gfx1201 hardware validation was performed;
the new automatic default is restricted to Linux gfx1201. No performance result is claimed for another backend.

## Packaging

The HIP release recipe accepts **`KOLIBRI=1`** to build, collect dependencies for, and include `strata-kolibri` beside
the primary engine. It already enables MMQ and stages the exact-version tuning tables. The component manifest and
Unix executable permissions now include Kolibri. The opt-in recipe explicitly bundles `libzstd.so.1`, which this
ROCm's `libamd_comgr` imports: the first package verification correctly rejected it when it was absent. The explicit
collector path and the complete component then passed verification. With KOLIBRI absent, this extra engine and
library-selection override are not added.

A local text-only gfx1201 Kolibri component is staged with the matching ROCm libraries, filtered GPU kernel data,
licenses, tuning table and validation records. Its relocatable `$ORIGIN` runtime is tested on this box. The final
archive is written under `build-mmq/release/`; this is a local artifact, not a published or signed upstream release.
