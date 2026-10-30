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

*(to be filled as the runs complete; nothing is asserted above beyond what is written here)*
