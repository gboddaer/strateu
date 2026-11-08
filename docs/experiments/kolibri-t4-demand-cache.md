# Kolibri mmap demand cache on a Tesla T4 (CUDA)

Measured on 2026-11-07 UTC, Tesla T4 (sm_75, 16 GB, driver 580.196.07, CUDA 12.8), Linux
6.1.0-52-amd64, 2 x Xeon E5-2690 v4 (56 hardware threads, 2 NUMA nodes), 755 GB of system RAM,
Kolibri-1-Q4_K_M (47,454,113,472 B), branch `amdw7800probuild` at `fb60e705`. The CUDA build is
`Release` with `CMAKE_CUDA_ARCHITECTURES=75`; the engine allocated 131,072-token context capacity
and 1,536 MiB of requested reserve. These runs are offline: no server was started.

This fills the "CUDA not run" gap in
[the W7800 campaign](kolibri-w7800-cache-optimization.md), which measured the same profiles on
HIP only and asked that no CUDA default be changed on the strength of its results. Nothing here
changes a default either: the profile stays opt-in through the environment variable. The
selection this record supports is written up in
[the cache selection record](../KOLIBRI_CACHE_SELECTION.md).

## Switches used

The W7800 switches carry over unchanged on CUDA:

```sh
STRATA_KOLIBRI_NOPIN_PROFILE=demand \
  build-kolibri-cuda/strata-kolibri --model /path/Kolibri-1-Q4_K_M.gguf \
  --expert-no-pin --expert-slots auto --expert-slots-hot \
  --prefill --ctx 131072 --vram-reserve-mib 1536 \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 16
```

`--expert-slots-hot` is not optional in practice: the demand profile keeps per-layer hot
residency. With `--prefill` at this context the engine fits 65 slots per layer (7.26 GiB of the
50-layer map) and keeps the requested 1,536 MiB of reserve, 990 MiB spare; the decode-only gate
runs fit 77 slots. Startup packs nothing, which is the lazy admission the campaign describes.

## Speeds, with the limits attached

Same fixed 512-token oracle input, 128 forced decode steps, Q4_K_M, 128K allocation, reserve
1,536 MiB, one binary (`sha256 9e8a3f44…`), `tools/kolibri/bench_unpinned.py`. The timed prompt
phase processes 511 tokens; the final prompt token is processed at the first decode step. Every
step dumps complete logits.

| Run | Startup | Prefill | Decode | Startup + prefill |
|---|---:|---:|---:|---:|
| demand, no saved profile | 0.93 s | 63.54 tok/s | 20.73 tok/s | 8.97 s |
| demand, learned profile replay | 3.11 s | 99.86 tok/s | 21.79 tok/s | 8.23 s |
| pipeline, no saved profile | 2.93 s | 73.49 tok/s | 20.40 tok/s | 9.88 s |
| baseline, no saved profile | 3.37 s | 49.07 tok/s | 14.86 tok/s | 13.78 s |
| balanced, no saved profile | 3.18 s | 51.43 tok/s | 16.90 tok/s | 13.12 s |

Repeats of the demand row on the same machine gave 63.54 (the row above), 67.54 (the first set,
built from the same commit) and 69.83 tok/s prefill (the `--stats` run) and 20.73, 18.45 and 21.48
tok/s decode: a run-to-run spread of about 10 %. The ordering against baseline and balanced is
the finding, not a single digit. The `--stats` run measured the same logits as the rest.

`baseline` is the same no-pin path with the campaign's experiments disabled; `balanced` and
`pipeline` are the eager and transfer-overlap profiles. All five rows produce exactly the same
complete output logits (`md5 252b5822790b27a623feb154e71ad944`, 128 records x 128,000 values =
16,384,000 logits) and the same generated token ids.

Transfer accounting from the `--stats` run (Linux process counters, per-blob diagnostics; no
stream drains): startup packs nothing; prefill packs 9,428 expert blobs (22.59 GB) in 4.30 s of
packing time; decode packs 4,628 blobs (11.10 GB) with 2,617 reuse hits and 601 spill-slot hits.
In the first set's `--stats` replay run the learned profile seeded 3,207 experts (7.69 GB) at
startup and then needed 4,697 more (11.25 GB) for prefill, which is where the faster prefill
comes from.

The T4 numbers are not the W7800's: prefill and decode are faster here on a much older card, and
pipeline is close to demand rather than clearly behind. Different host, different PCIe
generation, different engine backend. Neither table predicts the other.

## A CPU-pool run on the same card

Setup's shipped CUDA profile adds a stream environment this run did not set, so what follows is the
CPU-pool leg (`--prefill --cpu-threads <node-local physical cores> --expert-slots 0`), not the
shipped streamed configuration. Measured here with the same 512-token input, 128 decode steps and
128K allocation:

| Run | Load | Prefill | Decode |
|---|---:|---:|---:|
| CPU pool, no expert residency | 37.6 s | 141.6 tok/s | 34-38 tok/s |

The prefill figure is the engine's `PFchunk 511/511 3610 ms`; the decode rate is the difference
between runs of the identical configuration at `--n 65` and `--n 1` (1.89 s for 64 tokens) and at
`--n 128` and `--n 1` (3.34 s for 127 tokens), so it carries the load variance, about 10 %.

On a host with room for the 42.90 GiB pinned table this leg wins both timed phases: the mmap
profiles above are 2.2x slower in prefill and about 1.7x slower in decode, and they win only the
load. On a host without that room the table cannot be built at all and the mmap rows are the
difference between serving and not starting — which is the W7800's case, and the reason the
profile exists.

## Accuracy and correctness

The change is scheduling and memory layout, so the gate is the one
[the accuracy policy](../KOLIBRI_ACCURACY_POLICY.md) sets for unchanged arithmetic: finite,
bit-identical full logits on identical forced histories. All of the following passed.

| Comparison | Result |
|---|---|
| demand vs learned-profile replay, 128 records | bit-identical (`252b5822…`) |
| demand vs baseline / balanced / pipeline, 128 records | bit-identical (`252b5822…`) |
| `--expert-slots auto` vs `--expert-slots 8`, prefill and decode | bit-identical |
| repeated prefill of the same 512 ids | bit-identical |
| 1,025-token input across the 513-row window and the chunk boundary | bit-identical |
| cold vs profile-seeded hot set (65 slots/layer) | bit-identical |
| mmap packing (no-pin) vs the pinned host table, 511 prompt tokens + 8 decode steps through the decode path | bit-identical (`7f420efc…`) |
| generated ids of every run | identical, `286 3515 310 262 9468 …` |
| finiteness | every dumped value finite, no NaN or inf |

The packing row is the direct test of the mmap path: the same build, the same forced history,
one arm reading expert weights out of the pinned table and one packing them from the GGUF mmap,
producing the same bytes to the kernels.

The full integration gate passed too:

```sh
STRATA_KOLIBRI_NOPIN_PROFILE=demand python3 tools/kolibri/test_unpinned.py \
  --phase all --hot --bin build-kolibri-cuda/strata-kolibri --model /path/Kolibri-1-Q4_K_M.gguf
# PASS: all forced-history full-logit checks
```

It covers the rejected CPU/grouped combinations, decode, prefill, repeated prefill, 8-slot
versus auto residency, the 1,025-token window/chunk transition, and profile save/reload.

Against the independent llama.cpp reference (`kolibri_llama_dump`, `--cpu-moe 999`, dumps in
`/data4/kolibri-f16-evidence/f16-t6/qual/ref_512.bin`), the demand prefill arm gives mean KL
1.100887, max `|dlogit|` 9.77418 and top-1 agreement 7/9. The CPU-pool control on the same history
gives mean KL 0.074338, max `|dlogit|` 5.51273 and 8/9 — about 15x lower on mean KL. Both are far
outside the 1e-3 limit the EXPERT_F16 record used, and that reference was never expected to bind
here: `--expert-no-pin` refuses `--cpu-threads` by design ("the CPU pool reads the pinned table"),
so the no-pin path runs the per-expert GPU expert kernels only, an accumulation order the reference
qualification never covered. Read it as a property of that execution strategy, not of this change:
the same-strategy comparison above is bit-identical, and the CPU-pool dump is still bit-identical to
the one committed on 2026-10-30 (`md5 20f0ab89…`), so the pull did not move that path.

## A pre-existing CUDA failure, not caused by this work

`--expert-slots auto --expert-slots-hot --prefill` **without** `--expert-no-pin` exits 1 on this
card after load:

```
strata-kolibri: scores GEMM failed: status=7 P=256 nt=256 D=128 ldp=256 G=12
```

It fails with `--expert-slots 0` and without `--expert-slots-hot`, so it is not a cache-flag
interaction. It also fails at `98545dd1`, the commit this branch was pulled from, with the same
build flags and the same message, so the eight commits above it did not introduce it. It is a
pre-existing bug in the pinned-table prefill attention path on CUDA, and it is not fixed here.
Consequence for reading this record: the pinned-table arm can only be compared on the decode
path, which is where the bit-identical packing check above comes from. The GPU-only pinned
prefill configuration was not measurable — the same limitation the prior T4 record worked
around with `--cpu-threads` and the CPU pool.

A further attempt with a raised reserve (2,560 MiB, 90 slots/layer) failed with the same status 7,
so the failure is not a free-VRAM shortfall.

## Not run

Server and API runtime, task-quality or paired evaluation, full-128K input or retrieval,
protocol-level reset/cancellation/prefix reuse, HIP and SYCL cross-check of these results, and a
pinned-table prefill comparison (blocked by the failure above). Every run here sits on a fixed
forced history; none of it is a chat-performance guarantee. The OS file cache was not purged, so
the "no saved profile" rows do not measure cold disk reads. This is one T4, one day.

## Evidence

Artifacts are under `/data4/kolibri-t4-demand-evidence/` (not in `/tmp`, which is not durable):

| path | contents |
|---|---|
| `gate/` | the ten `test_unpinned.py` runs with full logits, logs and the saved hot profile |
| `bench2/` | the five speed rows plus the `--stats` row: `.json` (phases, switches, binary and logit hashes), `.log`, `.bin`, `.profile`, `.ids` |
| `bench/` | the first speed set on the first build of the same commit, for the spread |
| `acc/` | the pinned-vs-mmap decode pair, the CPU-pool control re-run, the shipped-default runs, the llama-reference comparisons, and the failed pinned-prefill logs |

Binary hashes: `ba7771321f9affda155bc342ce58a2ceeb9adac95381417b93af5ffafa0db84c` (first build of
`fb60e705`) and `9e8a3f445a3e4b2a7706cdafcd1400a98309b983fea3a66e4e2ea8ad08f523bd` (after the
pre-pull bisect and restore). The two differ only in link bytes: identical full logits on the
same forced history. The model file and the multi-megabyte logit dumps are never committed.
