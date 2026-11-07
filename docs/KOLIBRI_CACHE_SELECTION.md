# Kolibri expert-cache configuration, by backend

Which expert-cache configuration to select on CUDA and on AMD ROCm, and the evidence behind
each choice. Written 2026-11-07 from the [W7800 campaign](experiments/kolibri-w7800-cache-optimization.md)
(HIP) and [the T4 record](experiments/kolibri-t4-demand-cache.md) (CUDA, measured the same day).
Neither backend's default changes as a result of this document. It says what to select, and why
the other backend's better-looking row does not transfer.

## The selection

| Backend | Select | Why |
|---|---|---|
| **CUDA** | The shipped default: `--prefill --cpu-threads <node-local physical cores, cap 24> --expert-slots 0`, and leave `STRATA_KOLIBRI_NOPIN_PROFILE` unset. | Source setup now picks this placement itself when the RAM holds the table (52 GiB and up for Q4_K_M) and the mmap profile below it. This is also the arm the T4 reference comparison covers. On a host that can hold the table a CPU-pool run measured 2.23x the prefill rate and about 1.7x the decode rate of the mmap profiles; that run did not set setup's stream environment, so it is the pool leg, not the shipped streamed profile. |
| **AMD ROCm** | The shipped slot cache by default; select `STRATA_KOLIBRI_NOPIN_PROFILE=demand` with `--expert-no-pin --expert-slots auto --expert-slots-hot` where the pinned table does not fit or where a local run measures it better. | Measured on the W7800: 1.48 s startup instead of 30.96 s, 14.91 tok/s decode instead of 6.18 on the campaign's control, and the only configuration that runs at all on 30 GiB of RAM. |
| **Either backend, RAM too small for the table** | The mmap profiles are the only option, and source setup selects `demand` below 52 GiB detected RAM. `balanced` is the eager-loading alternative; `pipeline` is optional and was not consistently ahead. | The pinned table is 42.90 GiB for this model. Without it the engine must pack expert blobs from the GGUF mmap, and that is what `--expert-no-pin` does. Free VRAM must still fit at least one slot: `--expert-slots auto` exits when not even one does. |

The selection is a flag choice, not a build-time one. Nothing here promotes a profile to a
default, and [the accuracy policy](KOLIBRI_ACCURACY_POLICY.md) requires explicit approval before
any promotion.

## What the two configurations are

- **Host table (the default, both backends).** Load repacks all 19,200 expert blobs into one
  pinned 42.90 GiB table. The CPU pool (`--cpu-threads`) computes routed experts from that table
  and the GPU keeps only the dense weights, KV and scratch. `--expert-slots 0` on CUDA means no
  routed-expert residency at all; on AMD the shipped settings instead add
  `--expert-slots auto --expert-slots-hot` and a learned heat profile.
- **mmap cache (`--expert-no-pin`).** No pinned table. Experts are packed from the GGUF file map
  when routing selects them and copied into per-layer slots in VRAM, 65 slots per layer on the
  T4 at 128K context and 7.26 GiB. Startup packs nothing. `STRATA_KOLIBRI_NOPIN_PROFILE` picks the
  transfer policy: `baseline`, `balanced`, `pipeline` or `demand`.

These are mutually exclusive by design. `--expert-no-pin` refuses `--cpu-threads` ("the CPU pool
reads the pinned table"), and the engine refuses `--grouped` and `STRATA_KOLIBRI_EXPERT_F16` with
it. So the mmap path never uses the CPU pool, and its expert arithmetic is the GPU kernels only.
That fact decides the CUDA question.

## Speeds

### CUDA, Tesla T4, same card, same day

Fixed 512-token oracle input (511 timed), 128 forced decode steps, Q4_K_M, 128K allocation,
reserve 1,536 MiB, branch `fb60e705`, `CMAKE_CUDA_ARCHITECTURES=75`, driver 580.196.07, CUDA 12.8,
2 x Xeon E5-2690 v4, 755 GB RAM.

| Run | Load | Prefill | Decode |
|---|---:|---:|---:|
| CPU pool, no residency, setup's stream env **not** set (`--cpu-threads 14 --expert-slots 0`) | 37.6 s | **141.6 tok/s** | **34-38 tok/s** |
| demand, no saved profile | 0.93 s | 63.5 tok/s | 20.7 tok/s |
| demand, learned replay | 3.11 s | 99.9 tok/s | 21.8 tok/s |
| pipeline, no saved profile | 2.93 s | 73.5 tok/s | 20.4 tok/s |
| baseline (mmap control) | 3.37 s | 49.1 tok/s | 14.9 tok/s |
| balanced | 3.18 s | 51.4 tok/s | 16.9 tok/s |

The pool row's prefill is the engine's own `PFchunk 511/511 3610 ms`; its decode rate is the
difference between two runs of the identical configuration at `--n 65` and `--n 1` (1.89 s for 64
tokens) and at `--n 128` and `--n 1` (3.34 s for 127 tokens), so it carries about ±10 % load
noise. The four mmap rows are `tools/kolibri/bench_unpinned.py` phase counters. This pool row is
not the shipped streamed configuration, which was not measured here.

Read the columns separately. The mmap profiles win startup outright — 0.93 s against 37.6 s,
because nothing is repacked — and they win nothing else on this host. A machine with room for the
table is better served by the table. A machine without room cannot run it, and then 63.5 tok/s
prefill with 20.7 tok/s decode is the difference between working and not starting at all.

Two limits on that reading. The page cache was not purged, so the mmap rows read from RAM, not
disk, and the T4's PCIe 3.0 link is half the W7800's. The `--stats` runs move 22.59 GB for prefill
and 11.10 GB for 128 decode steps; on a cold file cache or a narrower link both mmap columns
fall.

### AMD ROCm, Radeon PRO W7800

From the campaign record, same 512-token input and 128 steps, 30 GiB RAM, gfx1100.

| Run | Startup | Prefill | Decode |
|---|---:|---:|---:|
| Original control, partial saved profile | 30.96 s | 11.23 tok/s | 6.18 tok/s |
| balanced, same initial profile | 127.11 s | 24.61 tok/s | 12.83 tok/s |
| demand, same initial profile | 6.47 s | 20.07 tok/s | 10.08 tok/s |
| demand, no saved profile | 1.48 s | 34.56 tok/s | 14.91 tok/s |
| demand, learned replay | 11.37 s | 113.61 tok/s | 17.01 tok/s |

`balanced` had other 128-step decode runs at 14.80-14.88 tok/s and prefill between 25 and 43
tok/s, so its prefill figure is not a fixed factor above demand's. On that machine the host table
does not fit at all: 42.90 GiB of pinned memory against 30 GiB of RAM. That is the gap `demand`
closes, and it is the configuration that was measured and qualified there.

## Proof, per backend

**CUDA (measured 2026-11-07, this fork).** The change is scheduling and layout, so the gate is
finite, bit-identical full logits on identical forced histories.

| Check | Result |
|---|---|
| `test_unpinned.py --phase all --hot` with `demand` | PASS: decode, prefill, repeated prefill, auto vs 8-slot residency, 1,025-token window and chunk transition, profile save/reload, rejected CPU/grouped combinations |
| demand vs replay, vs baseline, vs balanced, vs pipeline, 128 records | bit-identical (`md5 252b5822…`, 16,384,000 logits) |
| mmap packing vs the pinned host table, same build, 511 prompt tokens + 8 decode steps | bit-identical (`md5 7f420efc…`) |
| CPU-pool control re-run vs its 30 Oct dump | bit-identical (`md5 20f0ab89…`), so the pull changed nothing on that leg |
| finiteness in every run | every dumped value finite |

**ROCm (from the campaign record).** Both integrated and standalone HIP builds, the full HIP
test tree, `test_unpinned.py --phase all --hot` with `demand`, 44 successful measured runs with
finite complete logits and identical hashes within each of six aligned history/length classes,
the largest comparison 16,384,000 logits. The grouped-decode prototype was rejected on
correctness: 511,999 of 512,000 logits differed, maximum absolute difference 1.4207089.

**Against the independent llama.cpp reference, the two strategies are not interchangeable.** On
the T4, the demand prefill arm gives mean KL 1.100887, max `|dlogit|` 9.77418 and top-1 agreement
7/9; the CPU-pool control on the same history gives 0.074338, 5.51273 and 8/9 — about 15x lower on
mean KL. Both are far outside the 1e-3 limit the EXPERT_F16 record used, and that reference was
never the gate for this change. But it rules out presenting the mmap path as the accuracy-equivalent
of the pool on CUDA: it is bit-identical to the pinned GPU path, and it is not the arm the
reference work qualified.

## What would allow a CUDA switch

Not a speed number. All three of these, in this order:

1. Numerical fidelity of the GPU-only expert arithmetic against the independent reference, at
   the level the policy asks for — KL, percentiles, ranking changes, long contexts, attention
   transitions.
2. Task quality: paired reasoning, coding, factual and long-context retrieval evaluation with the
   margin and sample size agreed before the results are seen.
3. A demonstrated end-to-end gain on CUDA hardware. On the T4 measured here there is none: the
   mmap profiles are slower than the pool in both timed phases and only win the load.

Until then the CUDA default stays, and the mmap profiles stay opt-in.

## A CUDA failure that is not the default, and not this work

`--expert-slots auto --expert-slots-hot --prefill` **without** `--expert-no-pin` exits 1 on the
T4 after load:

```
strata-kolibri: scores GEMM failed: status=7 P=256 nt=256 D=128 ldp=256 G=12
```

It fails with `--expert-slots 0` and without `--expert-slots-hot` too, and it reproduces at
`98545dd1`, the commit the branch was pulled from, so it predates the mmap work. The shipped CUDA
configuration is not that combination — it passes `--expert-slots 0` with the CPU pool and runs
(measured above) — but anyone tuning a CUDA box has to know that a routed-expert residency cache
plus the GPU-only prefill path does not run on this branch. Fixing it is a separate change with
its own tests; it is not a reason to move to the mmap profiles.

## Limits of this record

One T4 and one W7800, one day each, offline engine runs with fixed forced histories. No server or
API path, no task-quality evaluation, no full-128K input, no HIP/SYCL cross-check of the CUDA
results. File caches were warm. The T4 decode rate for the default leg is a two-run difference,
not an engine counter, and the pool path prints no per-phase timing. Numbers in the two hardware
tables do not predict each other: the host, the PCIe generation and the expert arithmetic differ.

## Reproduce

```sh
# CUDA default: no mmap profile, CPU pool, no routed-expert residency
build-kolibri-cuda/strata-kolibri --model /path/Kolibri-1-Q4_K_M.gguf --prefill \
  --cpu-threads 14 --expert-slots 0 --ctx 131072 \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 128

# mmap demand, on either backend (ROCm: build-kolibri-hip/strata-kolibri)
STRATA_KOLIBRI_NOPIN_PROFILE=demand build-kolibri-cuda/strata-kolibri \
  --model /path/Kolibri-1-Q4_K_M.gguf --expert-no-pin --expert-slots auto \
  --expert-slots-hot --prefill --ctx 131072 --vram-reserve-mib 1536 \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 16

# the gate, and the speed harness that prints per-phase rates
STRATA_KOLIBRI_NOPIN_PROFILE=demand python3 tools/kolibri/test_unpinned.py --phase all --hot \
  --bin build-kolibri-cuda/strata-kolibri --model /path/Kolibri-1-Q4_K_M.gguf
python3 tools/kolibri/bench_unpinned.py --bin build-kolibri-cuda/strata-kolibri \
  --model /path/Kolibri-1-Q4_K_M.gguf --out /tmp/kolibri --label demand --prompt 512 --n 128 \
  --env STRATA_KOLIBRI_NOPIN_PROFILE=demand

# the placement rule setup applies (no GPU, no download)
python3 -m unittest tools.test_setup_kolibri
```

`--tokens-file` takes token ids, not English. The default leg prints `PFchunk` lines but no
per-phase counters, which is why its decode rate above comes from two runs; `bench_unpinned.py`
refuses to run without `--expert-no-pin`.

## Where the evidence lives

| Path | Contents |
|---|---|
| [W7800 campaign](experiments/kolibri-w7800-cache-optimization.md) + `kolibri-w7800-cache-optimization.csv` | every measured HIP run, profile, switch and complete-logit hash |
| [T4 record](experiments/kolibri-t4-demand-cache.md) | the CUDA runs, gates and reference comparisons |
| `/data4/kolibri-t4-demand-evidence/` | T4 logs, full logits, `.cmd` lines, profile and ids files, failed pinned-prefill logs |
| `tools/kolibri/bench_unpinned.py`, `tools/kolibri/test_unpinned.py` | the harnesses that produced the mmap rows and the gate |
