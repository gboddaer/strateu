# Kolibri vs Qwen3.8-Flash-Next on the Tesla T4 — performance gap investigation

**Status: Phase 0 in progress.** Purpose: measure and explain why the Qwen3.8-Flash-Next path is faster than the
Kolibri-1 path on this card *without a drafter*, decompose the gap, and rank the levers. Accuracy is preserved or
explicitly qualified; nothing here changes a default.

Branch `kolibri-perf-gap`, worktree `/data2/develop/perf-gap`, cut from `e5f4815c`. The FP16 experiment's branch,
ledger, artifacts and preserved `5bbe6c4e` control (`74b0e257…f418`) are untouched, as is the parked native-arm
worktree `/data2/develop/native-q4q6`.

## What the engines are, as measured

| | Kolibri-1-Q4_K_M | Qwen3.8-Flash-Next UD-Q4_K_XL |
|---|---|---|
| weights on disk | 47,454,113,472 B (1 GGUF) | 4 shards, **111.3 GB** (10.9 MB + 49.86 + 49.38 + 12.09) |
| active params (nominal) | ~2.8 B (6-of-384, 50 layers) | ~6 B |
| engine | `build-kolibri-cuda/strata-kolibri` (branch build) | `/data2/develop/Strata/engine/strata` (`sha256` recorded per run) |
| pack | none — repacks 903 GGUF tensors into 42.9 GiB pinned RAM, ~37 s | pre-converted pack `/home/gbo/Strata-data/packs/unsloth-ud-q4_k_xl` (`dense.bin` 1.49 GB, `native_experts.txt`, `index.txt`, `conversions.json`, `compat-bf16.json`), largest expert blob 3.99 MB |
| KV | f16 | `--kv int8` (+ `--kv-resident 32768` in the shipped config) |

## Findings recorded before any throughput comparison (Phase 0)

1. **This Qwen pack cannot run without a drafter.** The engine refuses with: *"… is a native (IQ) pack: it needs
   `--native` SHARD1, `--spec T (T >= 2)` and `--prefill CHUNK`."* Every shipped Qwen config in
   `/data2/develop/Strata/*.json` passes `--spec 4` with an MTP head
   (`strata-unsloth-ud-q4_k_xl.json`, `strata-unsloth-q8_0.json`, `strata-iq3_s.json`). So a **no-drafter** Qwen
   number cannot be produced from these packs; the closest measurable configuration is the minimum window
   (`--spec 2 --suffix-draft 0 --lookup-chain 0`), and prefill is drafter-independent while decode is not. Every
   decode figure for Qwen in this ledger must carry that label.
2. **The engine binary takes flags directly, not a `generate` subcommand** (`unknown argument: generate`), even though
   its own help text prints `strata generate …`.
3. **Cold start dominates on this box.** With `--expert-cache auto` and the shipped 24,576-slot expert profile the
   engine began preloading experts from the 111 GB file at **~108 MB/s** (29.4 GB in, 3.26 GB per 30 s, process in
   uninterruptible disk wait, GPU idle at 5.7 GiB held). Without `--expert-cache` it still loads ~76 GB RSS before it
   prints anything. On this hardware the flash path's *startup* is tens of GB of disk reads; its steady-state
   throughput is a separate measurement and cannot be inferred from a cold run.
4. **`--kv-resident 32768` with a small `--max-context` is inconsistent** (the shipped config pairs it with
   `--max-context 262144`); the Phase 0 runs use `--max-context 16384 --kv-resident 16384` and say so.

## Phase 0 measurements

### Qwen3.8-Flash-Next, prefill (drafter-independent), 511 tokens, no residency

Command: `engine/strata --pack …/unsloth-ud-q4_k_xl --native …00001-of-00004.gguf --tokens-file ids512.txt
--max-new 128 --max-context 16384 --kv int8 --kv-resident 16384 --prefill 512 --spec 2 --suffix-draft 0
--lookup-chain 0 --pool-workers 13 --stats` (no `--expert-cache`; the `--spec` requirement was satisfied only up to
the loader's own check, which then refused — see below).

```
prefill 511 tokens in 1 chunks, 7762.4 ms (65.8 tok/s); experts streamed 6253 (6253 by DMA, host 36.1 ms),
resident 0, on the CPU 6658 (share 0.69); PLE 4682.9 ms
```

| component | ms | share of prefill |
|---|---:|---:|
| prefill total | 7,762 | 100 % |
| **PLE table reads** | **4,683** | **60 %** |
| everything else (attention + experts + dense) | 3,079 | 40 % |

Two things follow immediately. First, **the flash path's prefill is not 3x faster than Kolibri's; it is ~66 tok/s
against Kolibri's ~58-60 tok/s** (8 pool workers, measured this session: 28.09 GB streamed in 8.46-8.87 s for 512
tokens; the 24-worker documentation figure is 51.6-53.6 tok/s). Second, **60 % of the flash prefill is PLE row reads
from the SSD**, a component Kolibri does not have at all — i.e. on this box the flash path is paying a large
SSD-table cost and is *still* marginally ahead, which means its compute+expert leg is substantially cheaper than
Kolibri's: ~3.08 s for 511 tokens ≈ 166 tok/s equivalent.

`--spec` then refused: *"--spec needs the device residency table (--expert-profile, --expert-cache and the token
graph)"*, so decode cannot be measured on this pack without the expert cache (and therefore without the ~76 GB load
and the cache preload). That is the next run.



*(to be filled as the runs complete; nothing is asserted above beyond what is written here)*
