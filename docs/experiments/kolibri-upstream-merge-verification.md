# Kolibri after the upstream merge: correctness, prefill, decode and prefix cache

Measured 2026-11-08 on Tesla T4 16 GB (CUDA 12.8, driver 580.196.07, sm_75), 2 x Xeon E5-2690 v4,
755 GB of system RAM, Kolibri-1-Q4_K_M, context allocation 131,072 tokens. The tree is the
`strata-merge` branch at `cfead3d2`, which merges upstream Strata `146df3c4` (v0.2.12, 1,377
commits) into this fork. Every run compares that binary with the pre-merge one built from
`main` at `2fdec617`, on the same machine and the same forced histories.

## Why two configurations

The default and the mmap cache are different engine legs and must not be mixed up:

- **default**: the pinned 42.90 GiB expert table, the CPU pool (`--cpu-threads 14`),
  `--expert-slots 0`, and the stream environment source setup writes
  (`STRATA_KOLIBRI_EXPERT_STREAM=1`, `STREAM_DEV_ACT`, `STREAM_GU`, `POOL_ALL=1`, `PF_CHUNK=512`,
  `REUSE`, `REUSE_PREFIX`, `REUSE_ALIGN`, `PREFIX_CACHE_MB=128`);
- **mmap demand**: `--expert-no-pin --expert-slots auto --expert-slots-hot` with
  `STRATA_KOLIBRI_NOPIN_PROFILE=demand`. `tools/kolibri/bench_unpinned.py` always passes
  `--expert-no-pin`, so its numbers are this leg only, never the default.

## Correctness

The merge changed 475 engine files (`+82,279 / -3,875`). On this model it changed no arithmetic.

| Check | Result |
|---|---|
| `test_unpinned.py --phase all --hot`, demand | **PASS**: decode, prefill, repeated prefill, auto vs 8-slot residency, 1,025-token window/chunk, profile save/reload |
| default, 512 tokens, 64 records (8,192,000 logits) | merged and pre-merge **byte-identical** (`md5 fe0c33a2bdfd387a4eab8074e3526cab`) |
| default, 4,096 tokens, 64 records | **byte-identical** (`md5 a965c475ad832fece3aa53effd072e1c`) |
| default, 32,768 tokens, 16 records | **byte-identical** (`md5 8f3cb77545b7c0604a81e6d28c135ae1`) |
| mmap demand, 512 tokens | **byte-identical** (`995f7101ef6c`) |
| mmap demand, 4,096 tokens | **byte-identical** (`d1619d95afd8`) |
| generated ids | identical between the two binaries at every length |

## Prefill

Default settings, the engine's own `PFchunk <done>/<total> <ms>` line:

| Prompt tokens | merged | pre-merge | per-token time |
|---|---:|---:|---:|
| 511 | **135.0 tok/s** (3.785 s) | 134.2 (3.807 s) | 7.4 ms |
| 4,095 | **152.2 tok/s** (26.91 s) | 151.3 (27.08 s) | 6.6 ms |
| 32,767 | **86.8 tok/s** (377.6 s) | 86.9 (376.8 s) | 11.5 ms |

The merge costs nothing here: every rate is within 1 % of the pre-merge binary, and 4K is the
fastest point (the pool's expert reads are amortised over more tokens). The 32K figure is lower
because attention grows with the context, not because experts are re-read.

The mmap demand leg, for comparison, measured 62.2 (merged) and 67.1 (pre-merge) tok/s at 512,
and 100.7 and 92.7 tok/s at 4,096.

## Token generation

Default settings, 512-token prompt, 63 decode steps. The pool leg prints no decode counter, so
this is a paired wall-clock difference with model load (38.9 s, from the engine's own load line)
and prefill (3.75 s) subtracted:

| | merged | pre-merge |
|---|---:|---:|
| decode phase | 2.51 s | 2.33 s |
| **tok/s** | **25.1** | 27.1 |

About 7 % lower, which is inside what a two-run difference of this shape can resolve here. It is
neither a proven regression nor a proven non-regression; a per-phase decode counter on the pool
leg would settle it. The mmap leg, which does print counters, gave 19.3 (merged) and 20.3
(pre-merge) tok/s at 512, and 16.7 / 16.0 at 4,096.

## Prefix cache

`tools/kolibri/prefix_cache_gate.py --long --cache-mb 128` against the merged binary: **24 of 24
cases pass**, each requiring a real long-prefix hit and complete logits equal to a cold read.

| Case | Reused tokens | Result |
|---|---:|---|
| `one-token-suffix` | 512 | hit, bitwise equal |
| `long-prefix-extension`, `long-prefix-edit`, `long-replay` | 2,560 | hit, bitwise equal |
| `branch-ownership-0…2` | 2,560 | hit, bitwise equal |
| `forced-continuation-0…2`, `repeat-six-0…5` | 2,560-3,072 | hit, bitwise equal |
| `swa-wrap-8191` | 7,680 | hit across the sliding-window wrap, bitwise equal |
| `long-11000` | **10,752** | hit, bitwise equal |
| `long-32768` | **32,256** | hit, bitwise equal |
| `early-edit-eviction-fallback`, `branch-ownership-3…4` | 0 | correct cold fallback, bitwise equal |
| `cancel-token`, `cancel-prefill` | - | cancel finishes cleanly, bitwise equal |

So the exact prefix checkpoints still work after the merge, at 11K and at 32K, and a cancel
during prefill or during decode leaves the cache usable.

## Limits

One T4, one day, fixed forced histories, offline engine runs. The file cache was warm. Only the
default and mmap-demand legs were measured. Not run: HIP or SYCL builds, the server/API runtime,
task-quality evaluation, the full `tests/terminal` suite (it did not finish inside ten minutes),
and the desktop app's own tests. The merge itself is validated by the build (170/170 targets,
CUDA sm_75) and by the runs above; a second card's build is not covered.

## Where the artifacts are

`/data4/kolibri-merge-check/`:

| path | contents |
|---|---|
| `default/` | the six default-config runs (512, 4096, 32768 on both binaries), full-logit dumps, logs with `PFchunk` and load lines, the paired decode runs |
| `bench/`, `bench-*.json` | `bench_unpinned.py` runs of the mmap demand leg at 512 and 4,096 on both binaries |
| `gate/` | the ten `test_unpinned.py` runs and their logs |
| `prefix-quick/`, `prefix-long/` | the prefix-cache gate's `inputs.json` and `results.json`, every case |
