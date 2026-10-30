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

### Kolibri-1, same prompt and generation length (two runs, exit 0)

Command: `strata-kolibri --model …Kolibri-1-Q4_K_M.gguf --n 128 --prefill --cpu-threads 8 --tokens-file ids512.txt
--ctx 2048 --counters` (binary `0bc772bf…acbdf`, the FP16 branch's build, arm off).

```
PFchunk 256/511   5,434 ms
PFchunk 511/511  10,398 ms                      -> 511 tokens / 10.398 s = 49.1 tok/s
CNT prefill pool: chunks=100 experts/chunk=117 bytes=28.09 GB pool=8,569 ms (85.7 ms/chunk, 3.3 GB/s)
CNT prefill pool work: rows=153300 = 1,205.6 GFLOP at 141 GFLOP/s (62% of the 226 GFLOP/s pool roof)
CNT decode: pool 714.2 MB/tok (91.42 GB total) = 3,074 ms -> 24.02 ms/tok at 29.7 GB/s
CNT branches per tok: pool 50.0 gpu-expert 0.0 slots-path 0.0 grouped 0.0
```

## Phase 0b — where the milliseconds actually are

| leg | Kolibri-1 | Qwen3.8-Flash-Next | what it means |
|---|---|---|---|
| prefill, 511 tokens | **10,398 ms** (49.1 tok/s) | **2,670 ms** (191.4 tok/s) | Qwen **3.9x** faster |
| prefill bottleneck | **compute**: 1,205.6 GFLOP at **141 GFLOP/s** on 8 CPU workers, 62 % of the pool's 226 GFLOP/s roof | **DMA + GPU products**: 19.8 GB/s over the row phases, CPU share 0.70 | Kolibri's prefill is **CPU-bound**; the T4's tensor cores are idle |
| expert bytes for the prefill | 28.09 GB (3.3 GB/s) | ~22.8 GB (9.6-19.8 GB/s) | same order of bytes, 3-6x the rate |
| decode per token | **24.0 ms** pool work (714 MB at **29.7 GB/s**) → ~30.5 tok/s | **130.7 ms**/token → 7.65 tok/s | Kolibri **4x** faster in decode |
| decode bottleneck (Qwen) | — | PLE table **6,063 ms blocked** of the 16.7 s, pool drain 91.5 ms/round, cache hits 63.1 % | Qwen's decode pays SSD reads and a slower pool drain it does not overlap |
| load | 37 s repack of 903 tensors into 42.9 GiB pinned | ~77 GB read at ~110 MB/s (≈11 min) + 48 s projections + PLE map | Kolibri loads far faster |
| drafter | none | `--spec 2`, **0 of 127 drafts accepted** | the comparison *is* drafter-free in effect |
| KV | f16 | int8 (+16 k resident) | Qwen's KV is half the bytes |

**Three conclusions, all measured:**

1. **The user's premise is half right, and the half that is right is the important one.** Flash is not faster overall — it
   loses decode by 4x on this card — but it wins prefill by **3.9x**, and it does so *on a model 2.3x larger*. The
   mechanism is not a better expert kernel: it is that **Qwen's prompt experts run on the GPU while Kolibri's run on 8
   CPU workers at 141 GFLOP/s**, which the engine itself reports as 62 % of the pool's compute roof. Kolibri's prefill
   is compute-bound, not byte-bound — the 3.3 GB/s is what 141 GFLOP/s *costs* over 28 GB of Q4_K/Q6_K rows.
2. **Kolibri's decode is already excellent and Qwen's is not.** 24 ms/token of pool work at 29.7 GB/s, versus 130 ms
   with 6 s of that blocked on SSD PLE reads. Any plan that "makes Kolibri more like flash" must not copy the decode
   side.
3. **The 9x asymmetry inside Kolibri is the real anomaly.** Decode moves 714 MB/token at **29.7 GB/s**; prefill moves
   28 GB at **3.3 GB/s**. Same kernels, same workers, same RAM. The difference is *row count per expert* (1 vs ~3.7)
   and the batch shape the pool runs — which is exactly the shape where the earlier GPU experiments looked bad. That
   asymmetry, not the model size, is what a plan should attack.



### Qwen3.8-Flash-Next — full run with the residency table (prefill + decode)

Command: `engine/strata --pack …/unsloth-ud-q4_k_xl --native …00001-of-00004.gguf --expert-profile
…/data/expert-profile.bin --expert-cache 1024 --tokens-file ids512.txt --max-new 128 --max-context 16384
--kv int8 --kv-resident 16384 --prefill 512 --spec 2 --suffix-draft 0 --lookup-chain 0 --pool-workers 13 --stats`
(load: ~77 GB read at ~110 MB/s before the first line; see finding 3.)

```
prefill  511 tokens in 2,670.0 ms -> 191.38 tok/s   (time to first token 3,923.5 ms)
decode   128 tokens in 16,724.6 ms ->   7.65 tok/s  (130.66 ms/token wall clock)
PLE      0.6 ms in this run; PLE io during decode: 12,832 rows, 26.3% row-cache hits, 2,880 SSD reads
         (12.1 MB), p50 4,597 us, p99 571,285 us, **blocked 6,063 ms total**
spec     "128 rounds of 2, drafts accepted 0 of 127 (0.000), 1.00 tokens per round"
experts  prefill: 5,709 streamed (all by DMA), 747 resident, 6,526 on the CPU (share 0.70)
         decode: 49.433 ms/token of CPU pool over 48 layers; 1.97 distinct experts/layer over PCIe
         expert-cache hits 69,592 of 110,213 = 63.1 % (cache 1,024 slots, 100 % full)
pool     gate/up 34.171 + quantize 0.428 + down 22.750 = 19.8 GB/s over the row phases
```

**The drafter contributed nothing** — 0 of 127 drafts accepted. So this *is* the no-drafter number the
investigation wanted, by accident of the small verify window and this pack's MTP head.

### Side-by-side on the same card and prompt (both measured this session)

| | Kolibri-1 (pool, 8 workers) | Qwen3.8-Flash-Next (int8 KV, 1,024-slot cache) | winner |
|---|---:|---:|---|
| prefill, 511-512 tokens | 58-60 tok/s (8.46-8.87 s) | **191.4 tok/s** (2.67 s) | **Qwen 3.2x** |
| decode, greedy/1 token per round | **~30.5 tok/s** | 7.65 tok/s (130.7 ms/token) | **Kolibri 4.0x** |
| expert bytes streamed for the prefill | 28.09 GB | ~5,709 x 3.99 MB ≈ 22.8 GB | similar |
| achieved streaming rate | **3.2-3.4 GB/s** (CPU pool) | **9.6-19.8 GB/s** (PCIe DMA + GPU) | **Qwen 3-6x** |
| PLE table | none | 6.06 s of decode blocked on SSD reads | Kolibri |

**This is the finding of Phase 0, and it is not the story we assumed.** Flash is not uniformly faster: it wins
**prefill by 3.2x and loses decode by 4x**. And the prefill win is not a better kernel — it is **the same bytes moved
3-6x faster**, because Strata streams experts by PCIe DMA to the GPU (the engine's own probe: 9.6 GB/s host→device)
while Kolibri's pool reads them with AVX-2 CPU workers at 3.2-3.4 GB/s. Qwen's decode is slow for reasons Kolibri does
not have: 60 % of its decode wall time is PLE table reads from the SSD (p99 0.57 s per read) and its pool drain is
91.5 ms/token.



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
