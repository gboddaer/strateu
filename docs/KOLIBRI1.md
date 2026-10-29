# Kolibri 1 — running Aleph Alpha's Kolibri 1 on Strata

Kolibri 1 is [Aleph Alpha](https://huggingface.co/Aleph-Alpha/Kolibri-1)'s 78-billion-parameter
mixture-of-experts reasoning model for German and English (3.46 B active per token, Apache-2.0). Strata runs it
on the same idea as Flash-Next: every expert in system RAM, the most-used ones on the graphics card, the CPU and
GPU computing experts side by side.

## What the model is (measured from the GGUF)

- 50 layers, every layer MoE: **384 routed experts, top-6, plus one ungated shared expert** (FFN width 512,
  n_embd 2560).
- The router selects on `logits + expert_bias` and weights by the **unbiased** `sigmoid(logits)`, with **no
  renormalization** (GGUF `expert_gating_func = 5`, `expert_weights_norm = False`). This is not llama.cpp's
  DeepSeek-style sigmoid router: with the wrong router the model's top-1 agreement against a float64 reference
  drops to 10/67 (the conversion README's test); with the right one it is 63/67 at Q4_K_M.
- Attention: GQA, 48 query heads over 4 KV heads, head_dim 128, per-head q/k RMSNorm. A per-layer pattern
  (GGUF `attention.sliding_window_pattern`, 39 of 50): **four sliding-window layers (window 513, RoPE base
  10000) to one full-attention layer with no positional encoding (NoPE)**.
- Sandwich norms: an RMS norm after the attention output and after the MoE output, in addition to the pre-norms.
- Context 262,144. Because 39 of 50 layers are windowed, the KV cache stays small: 11 full layers ≈ 22.5 KB per
  token in fp16 (5.6 GB at full context), the windowed layers a constant ~20 MB.
- No MTP draft layer, no PLE/n-gram table, no linear attention, no hyper-connections.

## Files

The GGUFs are the community conversion ([Hob-forge/Kolibri-1-GGUF](https://huggingface.co/Hob-forge/Kolibri-1-GGUF)),
quantized from the FP8 checkpoint via BF16; every quant is made from that BF16 GGUF, none is a requant.

| Size | File | Bytes | Needs |
| --- | --- | ---: | --- |
| Q4_K_M | `Kolibri-1-Q4_K_M.gguf` (one file, 903 tensors: 426 Q4_K, 76 Q6_K, 401 F32) | 47.5 GB | ~56 GB RAM |
| Q8_0 | `Kolibri-1-Q8_0-0000{1,2}-of-00002.gguf` | 83.1 GB | ~96 GB RAM |

Routed experts are 42.4 GB (Q4_K_M) / 75 GB (Q8_0); one decoded token reads 6 experts × 2.21 MB × 50 layers
≈ 663 MB — the same memory-bandwidth class as Flash-Next Q2_0, so the same engine design applies.

## Running it

```
./setup.sh --family kolibri --model Q4_K_M --gguf-dir /path/to/the/folder/with/the/gguf --yes
```

(`--gguf-dir` names the folder that holds the file; setup checks its SHA-256. Setup compiles the family's own
binary `strata-kolibri` in `build/` - no ready-made engine carries this family yet.)
Setup offers Q4_K_M only. The repository's other sizes (Q2_K, Q3_K_M, Q5_K_M, Q6_K, Q8_0 - the last three in
subfolders) are not wired into setup yet.

or directly:

```
build/strata-kolibri --model Kolibri-1-Q4_K_M.gguf --ctx 32768
```

Sampling defaults (the model card's): temperature 1.0, top-p 0.97, top-k 128. Reasoning effort is per request:
`"chat_template_kwargs": {"reasoning_effort": "low"}` - `none`, `low`, `medium`, `high`; measured on the T4:
`none` answers directly ("27 x 43 = 1161.", no thinking), the default thinks first. The tokenizer is gpt2 BPE,
128,000 tokens, `pre = kolibri1` (the Qwen2 pre-tokenizer split). The model writes tool calls as
`<tool_call>{JSON}</tool_call>` next to the template's XML form. Its own chat template asks for exactly that form, so the
server reads it as structured `tool_calls` with no setting to add (measured: `get_weather {"city": "Berlin"}`,
`finish_reason: tool_calls`). The server decides this from the template it serves: a template that asks for the JSON
form has that form read as calls (`tool_call_recovery` unset), and `"tool_call_recovery": false` in the config turns
it off again.

## How the engine path works

`src/kolibri/kolibri_engine.cu` is a driver of its own beside Flash-Next's (the `strata` family switch picks it
from the file's architecture), built from Strata's kernels: dense Q4_K/Q6_K matrices via native MMVQ, the norms
and router gates as F32, the routed experts repacked at load into native `[gate | up | down]` blobs per
(layer, expert). On a card that cannot hold 42.4 GB the hybrid mode runs: every expert pinned in RAM, fixed
VRAM slots per layer, the batch's other experts on the CPU pool (ggml-cpu's Q4_K/Q6_K dot products), prompt
chunks streamed. The router is Strata's `sigmoid_logit_add` top-k kernel (384/6), checked bitwise against a CPU
reference. Full-attention layers keep fp16 KV rows; sliding-window layers keep a 513-cell ring.

## Tuning it (auto-tuner, and what was measured on an R9700)

Kolibri 1 is in the auto-tuner: `strata tune auto <config>` (and the app's Tune screen) now plans candidates for it
instead of saying there is nothing to try. The family is named by the config's `arch` - `"kolibri1"` - because this
engine path's arguments (`--model X.gguf`) look like a single-GGUF Flash-Next run. Its machine class is `K`: the expert
table lives in system RAM and every expert layer is served by the CPU pool, so the levers are that pool's lanes and the
VRAM the KV, the fp16 expert copy and the slot cache draw on together. What it tries: `--expert-slots auto`, with and
without `--expert-slots-hot`, with a 1 GiB reserve and with the fp16 cache off; `STRATA_KOLIBRI_EXPERT_F16=0`;
`STRATA_KOLIBRI_ATTN_MB` 256 and 1024; `STRATA_KOLIBRI_PF_PARTIAL`; one CPU lane per hyperthread; one lane fewer than
the config names; `--prefill` where the config lacks it. `STRATA_KOLIBRI_DEC_RES` is on the never list: it made answers
11x slower here ([AMD_R9700_PORT.md](AMD_R9700_PORT.md)).

A run costs real time on this model - it writes 16-18 tokens/s and reads a prompt at 60-120 tok/s - so a Kolibri run in
the tuner uses 2,048- and 8,192-token prompts and three 100-token decodes instead of the 4,096/96K prompts and five
200-token decodes the Flash-Next families use. With `--time` a quick tune on an R9700 lands around an hour.

Measured 2026-10-28 on this box (Radeon AI PRO R9700 32 GB, gfx1201, Ryzen 7 5700G, 93 GiB RAM, Q4_K_M, ctx 131072,
8 CPU lanes, a 512-token prompt and 128 greedy tokens per request, two requests per start, one arm per engine start).
Prompt read (pp) and generation (tg) in tokens/s, first and second request:

| arm | pp | tg |
| --- | --- | --- |
| as shipped before this measurement (fp16 cache 24/layer) | 57.6 / 70.1, 60.1 / 71.8 | 15.85 / 15.85, 16.20 / 16.18 |
| **fp16 cache off (`EXPERT_F16=0`, the new default)** | **108.3 / 116.3, 102.0 / 122.3, 74.7 / 83.4** | 15.89 / 15.78, 16.23 / 16.22, 16.07 / 16.09 |
| fp16 cache off + CUDA graphs on | 75.0 / 81.7 | 16.07 / 16.06 |
| CUDA graphs on (cache on) | 58.5 / 71.4 | 16.05 / 16.06 |
| attention tile 256 MiB (cache on / off) | 58.7 / 70.2, 75.3 / 82.2 | 16.18 / 16.15, 16.07 / 16.12 |
| 7 CPU lanes (cache on / off) | 58.2 / 71.0, 71.2 / 77.0 | 16.18 / 16.20, 16.09 / 15.97 |
| `PF_PARTIAL=1` with `EXPERT_F16=28` | 62.0 / 73.3 | 15.99 / 16.07 |
| `--expert-slots auto` (135 slots/layer, 15.08 GiB) | 59.4 / 72.3 | **13.02 / 12.99** |
| `--expert-slots auto --expert-slots-hot` | 56.6 / 68.0 | **12.82 / 12.80** |

What the numbers say on this card: the fp16 expert cache is a **T4 prefill helper that does not transfer** - with it the
prompt read 57-72 tok/s in every run, without it 75-122 tok/s, and no run with the cache reached the slowest run
without it; generation is the same either way. It is now `0` in `run-kolibri1.sh`, which is also the engine's own
default, i.e. the path the bitwise oracle measures. The slot cache works (135 slots/layer fitted into the 15 GiB of free
VRAM) and **costs a fifth of the generation speed** (16.2 -> 13.0 tok/s), so it stays off and stays an arm for the tuner
to try on other cards. CUDA graphs, the attention tile at 256 MiB, 7 instead of 8 CPU lanes and the partial-sum prefill
path were all inside the noise on this box.

### Where the speed goes, and what is left (measured 2026-10-28 on the same R9700)

A decode token costs **62.9 ms** on this card (15.90 tok/s, ctx 131072, the launcher's own command line). Of that, the
GPU is busy **16.1 ms** and the CPU expert pool **22.3 ms**; the rest is hand-off and waiting. The GPU side runs
**1 347 kernel launches per token**, which is the shape of the problem: this model is not short of arithmetic, it is
short of work per launch.

- **The prompt is almost entirely CPU pool.** A 511-token prompt reads at **109 tok/s** and an 8 192-token prompt at
  **118 tok/s**, and **88 %** of the long prompt's wall is the pool - 426.69 GB streamed for 8K tokens (52 MB/token)
  at 7.0 GB/s and 316 GFLOP/s against a 226 GFLOP/s compute roof in its own accounting. The GPU and the host together
  are ~1 ms/token. Prompt gains therefore have to come from *not* sending the touched experts through the pool at all
  (a correct earlier version of this paragraph said 70 tok/s and 50.8 %, from adding cumulative chunk timers; see
  [AMD_R9700_PORT.md](AMD_R9700_PORT.md) §P0).
- **The slot cache loses for a named reason.** At this box's VRAM it fits 214 experts per layer and serves **96.6 %**
  of picks from VRAM, which cuts the streamed leg from 714 to 291 MB/token - and still decodes at **14.21 tok/s**
  against 15.90. The reason is three launches per expert pick (gate, up, down): 875 launches and 15.6 ms of GPU time
  per token, plus 4.7 GB of PCIe traffic to fill the cache after the prompt (+14 % on the prompt).
- **Two knobs that do not speed up decode here.** The hipBLASLt tuning table does not affect decode's GEMVs.
  It is now required for P4's prompt shared-expert GEMMs, as described below. The grid-capped GEMV (`STRATA_KOLIBRI_GEMV_GS`) is +10 % at a 128 000-row shape and -29 % at the
  2 560-row shapes this model runs, and never fires at all.
- **P4 is the prompt default on Linux R9700/gfx1201 in an MMQ build.** It uses groups of 16 experts and requires
  compatible direct hipBLASLt shared-expert tuning. Without it, the engine says why and keeps the pool. Other cards
  keep their previous default. `STRATA_KOLIBRI_PF_MMQ=0` switches it off. HIP links the K-quant kernels statically;
  it does not need a separate `strata_kq` library.

  On the Ryzen 5700G / R9700, the staged package measured **168–169 tok/s** for a warm 511-token prompt and
  **314–315 tok/s** for 8 191 tokens. The same packaged binary with P4 off read 8 191 tokens at **102.4 tok/s**:
  79.97 s versus 26.02–26.08 s, about **3.07x**. Short-prompt pool times varied with box state; all runs, including
  the slower first packaged start, are in [the P4 shipping ledger](AMD_R9700_PORT.md#p4-shipping-validation).
  Settings: `--cpu-threads 8`, `EXPERT_F16=0`, ctx 8192/12288, chunks 256/512 respectively.

  P4 does not change the generation code. The paired 128-token wall measurements were **15.2–15.6 tok/s** after
  the short prompt and **11.6–11.7 tok/s** after 8K. Full-logit repeat checks passed at both lengths, without a
  per-group device drain. Against the existing eight-prompt llama.cpp references, mean KL was **0.00236**, top-1
  agreement **99%**, and KL was **0.89x** the CPU noise floor (200 teacher-forced steps).

  The earlier nondeterminism claim blamed the K-quant product too soon. The first differing buffer was the shared
  expert's dense gate output. Requiring its validated direct tuning route removes the failing configuration. The
  vendor backend's internal fault is not established. FP16-resident expert layers and the two-chunk pipeline retain
  the pool. The previous 795 tok/s projection remains retracted; there is no new extrapolation beyond 8K.
- **One real defect fixed on the way**: an exactly-8 192-token request used to take the *short*-chunk branch and run
  32 chunks of 256 where 16 of 512 were intended; it now runs the long chunk (+2.7 %, ids bitwise).
- **Attention is the second lever**, not the first: 4 % of the wall at short context, the wall itself at 100K.

All of it, with the commands and the per-arm tables, is in [AMD_R9700_PORT.md](AMD_R9700_PORT.md) §T1-T14.

## Decode

P4 left decode unchanged. Measured with `tools/kolibri/decode_bench.py` on card0 (`--serve --prefill
--cpu-threads 8`, no expert slots, ctx 8192), 96 new tokens cost 62.1 ms/token (16.1 tok/s) and 85.4
ms/token after an 8191-token prompt at ctx 12288 (11.7 tok/s); six identical runs spread 0.4 %.

Two opt-in arms (both off by default) now cut that. `STRATA_KOLIBRI_DEC_HANDOFF=1` replaces the decode's
two blocking per-layer device-to-host copies with one publication kernel plus a bounded host spin and
enqueues the shared expert while the host waits: 56.9 ms/token (17.7 tok/s, −10.1 %) and 81.5 ms
(−7.5 %). `STRATA_KOLIBRI_DEC_STREAM=1` then moves the decode off the legacy default stream onto one
explicit non-blocking stream: together they reach **49.3 ms/token (20.33 tok/s, −21.5 %)** and
**73.1 ms (−15.1 %)** after 8191 ids, three alternating pairs each, every pair favouring the arm.
`tools/kolibri/decode_gate.sh` shows the combination bit-identical to the plain engine at 16/511/512/513
ids and on the reused-prefix path, six repeats included, and the plain path's own logits are byte-identical
to the build before any arm existed. Both prefill gate profiles still pass. The earlier
`STRATA_KOLIBRI_DEC_XEXP` arm on its own measured 5.2 % and 4.2 %.

The earlier kernel trace (about 1 344 dispatches and 67.9 ms between forward markers) is a profiled
diagnostic, not an unprofiled speed baseline, and the original wall-based slot comparisons and
30–38 tok/s projections are withdrawn. [KOLIBRI_DECODE.md](KOLIBRI_DECODE.md) has the measurements, the
reassessment, and why a captured graph is not yet part of the arm: the capturable attention path forces
the two-pass arithmetic, and the full logits then differ at record 0 while the greedy ids still agree.

## Validation

Checked against the patched llama.cpp (`~/llama.cpp`, upstream + the conversion's 4-commit kolibri1 series) by
`tools/kolibri/parity.py`: the same GGUF, the same token ids, per-layer hidden states, greedy tokens, and
teacher-forced KL. Gates: mean KL < 0.02, greedy agreement ≥ 90 %, per-layer relative error < 2e-2.

Measured on this box (Tesla T4 16 GB, 755 GB RAM, Q4_K_M, experts in RAM, llama.cpp `--ngl 99 --n-cpu-moe 50`
with CUDA graphs off, per the model card's T4 note):

Parity (engine vs llama.cpp, `tools/kolibri/parity.py` + `kl.py`, 8 prompts 56…4147 tokens, 64 greedy each):

| measure | value | gate |
|---------|-------|------|
| teacher-forced KL, short prompts p00–p05 (390 steps) | mean **0.00168**, top-1 0.9846 | < 0.02 ✓ |
| teacher-forced KL, long prompts p06–p07 (130 steps) | mean **0.00229**, top-1 1.0000 | < 0.02 ✓ |
| KL ratio strata / llama-CPU floor | **0.76** short, **0.41** long | < 1.5 ✓ |
| greedy identical (up to EOS or 64 tokens) | 5 of 8 prompts (p03 24/24, p04 64/64, p05 64/64, p06 4/4, p07 5/5; p00 31/64, p01 15/59, p02 48/64) | ≥ 90 % of tokens: not met — near-tie cascades |
| per-layer hidden-state rel. err (vs llama fp16 FA/KV) | median 0.018–0.028 short, 0.077–0.081 long | < 2e-2 everywhere: not met at the tail layers |

The KL gates pass and strata sits closer to llama.cpp's GPU output than llama.cpp's own CPU run (ratio < 1);
the greedy and tail-layer thresholds are not met on this card against llama's fp16 reference — the divergence
is near-tie cascades, and the engine's forward is bitwise kolibriref (p00 8/8).

### Speed (measured 2026-10-18..21 on this box, Tesla T4 16 GB + 2x E5-2690 v4, 755 GB RAM, Q4_K_M)

Two harnesses, both on this tree: `tools/kolibri/bench.py` (the serve protocol: PP lines for the prompt rate,
the DONE line for the decode rate) and the CLI (`--prefill --tokens-file`, the PFchunk lines).  The
**speed configuration** is the passing one: `--prefill --cpu-threads 24`, the fp16 resident expert cache
(`STRATA_KOLIBRI_EXPERT_F16=28`), the router tile kernel, the adaptive prefill chunk and the prefix-dependent
POOL_ALL rule, with `numactl --cpunodebind=1 --membind=1` (the T4's DMA and the pool's workers on different
memory controllers - measured: the same config at node-0 is 20% slower at 32K).

All strateu rows are wall-clock measurements from `tools/kolibri/bench.py` on the shipped configuration; the earlier phase's
counter-derived figures (129 tok/s at 32K, 21.2 tok/s decode) are superseded and removed.  The campaign II re-anchor
(driver 580.196.07, kernel 6.1.0-52-amd64, load stated in `docs/published-figures.md` #18-#25) is what these rows now
carry; where a number moved against the previous publication, the reason is the box, and both figures are named there.

| test | strateu (shipping config, wall clock) | Strata (the exactness-first plain path) | llama.cpp (same box) | llama.cpp README (RTX 5070) |
|------|------|------|------|------|
| pp 512 (short) | **156.0 / 159.6 tok/s** | 10.0 tok/s | 159-169 tok/s | 723 tok/s |
| pp 32,768 | **119.6 / 119.4 tok/s** | 1.3 tok/s | ~157 tok/s | - |
| tg 128 (short) | **30.3 / 30.3 tok/s** | 9.3 tok/s | 19-30 tok/s (binding- and load-dependent) | 18.0 tok/s |
| tg 128 after 32K | **13.4 / 13.5 tok/s** | 0.6 tok/s | ~13 tok/s | - |
| 100K needle | **recall 4/5, exact 3/5** over five distractor-bearing fixtures (`tools/kolibri/needle_audit.py`); the reference is not usable as an oracle at this length | - | - | - |
| KL vs llama.cpp (390 teacher-forced steps) | **0.00634** (ratio 3.77) | 0.00210-0.00648 | — | — |

**Opt-in `STRATA_KOLIBRI_PF_PARTIAL=1` (campaign II).**  The prefill keeps each token's expert partial sums on the GPU
instead of round-tripping them through the host pool, and it is opt-in: with the knob off the binary is byte-identical
to `main` (plain-path oracle BITWISE, two-run ids identical).  With it on, measured on the same harness and binding:
pp 512 **179.8 / 181.0 tok/s** (serve path 181.7 cold, 201.8 warm), pp 32,768 **125.7 / 122.1 tok/s**, tg 128 short
**30.0 / 30.3 tok/s**, tg 128 after 32K **13.3 / 13.4 tok/s**, KL 0.00133 against the knob-off 0.00123 with
llama.cpp's own CPU-vs-GPU floor at 0.00178.  The +16% at 512 tokens is real and the 32K figure never went down in
any pair; what it does not do is reach the 200 tok/s target, because the expert GEMM underneath is 4.18 ms/layer and
the compressed alternative measured 9.99.  It needs the fp16 expert cache's scratch: set STRATA_KOLIBRI_EXPERT_F16
too - without it the engine refuses at startup (before the 2026-10-26 fix it died with a CUDA invalid argument at
the first publish).

**Card classes - what carries over, and what does not.**  The placement sign (the findings register's
"Slots refutation: box-scoped") is the whole story, and it is decided by ONE ratio: the host's streaming
bandwidth vs the GPU's.  On this box the host streams at 43.4 GB/s (pool_bench, 24 workers, cold blobs, the
decode's shape) while the GPU reads the pinned table over PCIe at 5.7-6.1 GB/s, so **the CPU pool is the expert
placement** and VRAM slots lose to it by 1.7x (they beat PCIe by 9x).  The class rules:

| class | the ratio | the expert placement | the expected shape of the numbers |
|---|---|---|---|
| **A - host-streaming** (this box: a T4 with a host that streams > 10x the GPU's PCIe read) | host 43.4 GB/s vs PCIe 5.7-6.1 GB/s | the **CPU pool** (the pinned table is the store; no PCIe round trip) | the pool's rate is the prefill's and the decode's floor: 15.3-16.5 ms/token of expert work |
| **B - balanced** (the host streams at roughly the GPU's PCIe read) | ~6 GB/s both | either | the two placements cross over; measure, do not inherit |
| **C - GPU-memory** (a 300+ GB/s card, e.g. the RTX 5070 class) | the GPU's memory is 50x the host's | **VRAM slots** (Flash-Next's design target) | the expert bytes live in VRAM; the pool's ceiling stops mattering |

The byte arithmetic is portable; the placement sign is not - that is why the class rule is stated as a ratio and
not as a card name.  On a class-C card the slot cache should become the primary path (the findings register
records the sign flip and the byte math behind it).

**The honest gaps on this box (class A).**  Short-context decode now sits at its floor (30.3 / 30.3 against 30) and
the 512-token prompt read is 156.0-159.6 by default, 179.8-181.0 with the campaign's knob on, against a 200 target.
Two gaps remain and both are traffic, not scheduling.  The 32K prompt read is 119.6 / 119.4 where llama.cpp on this
same box does ~157 tok/s, and the decode after a 32K prompt is 13.4 / 13.5 against a floor of 20 - that token moves
**1.28 GB of expert bytes over PCIe** (488.7 MB device-to-host, 789.5 MB host-to-device) at **0% residency**, of a
74.6-75.2 ms token in which the CPU pool drain is only 12.86 ms.  The residency paths in the tree do not close it:
`--expert-slots` caches experts by **id** rather than by hotness and costs 17% of decode (25.0 / 25.1 against 30.3),
and `STRATA_KOLIBRI_DEC_RES=1` costs 10% of prefill for -3% of decode.  The levers are less expert traffic and a
fused compressed-weight GEMM, or a card class where the placement sign flips - not another knob.

The by-id limitation now has an opt-in path.  `--expert-slots-hot` picks the resident experts by routed-pick
frequency instead of by id, `--expert-slots-profile PATH` saves that set and seeds the next start from it, and
`--expert-slots auto` sizes the cache from the VRAM that is free once the dense weights, the KV and the scratch are
allocated (`--vram-reserve-mib`, default 700, is kept clear for the runtime - deliberately not the last byte).
Measured on this T4, 160K context, 70 slots (8.38 GiB), `--prefill --cpu-threads 24`, the same 31-token prompt
twice: the id-ordered cache served **314 of 1,800** routed picks (**17.4%**), the set learned from the first request
served **682 of 1,800** (**37.9%**) - same 70 slots, same bytes, placed where the traffic is.  Re-selecting cost
669 ms between the two requests (57 experts, 6.5 GiB over PCIe).  Seeded from a saved profile, the *first* request
of a cold start already served **727 of 1,800** (**40.4%**).  `gen:` was 4688 4688 4688 4688 in the id-ordered, the
hot and a 75-slot `auto` run alike: residency moves bytes, it does not change them - true with the CPU pool off, where
the two device legs are bitwise (findings-log 0.3, P == Q); with `--cpu-threads` on, moving one expert between a slot and
the pool changes the ids (0.3: mean |dlogit| 0.855 per step, id flipped at token 8), which is why the round below is
measured by hit rate and by repeated runs, not by an unchanged `gen:`.  With `auto` at 160K the fit
chose 75 slots/layer from 9,501 MiB free, and prints the exchange rate it implies - 1 slot = 114 MiB = 5,857 tokens
of context at 20.0 KB/token, which is why `--ctx` and `--expert-slots` are one decision and not two.

**The residency round now has the shape the Qwen3.8-Flash-Next engine's adaptive tier uses** (`src/program/generate.cpp`
there: `--adapt-every`, `--adapt-swaps`, `--adapt-min-gain`, `--adapt-decay`).  A round runs every 32 **positions** (not
every N seconds), moves at most 96 experts mid-request and 2,048 once after each prompt, admits a candidate only when its
score beats the resident's by 1.5, and multiplies the scores by 0.7 after every round so the map follows the
conversation instead of freezing on the prompt.  The copies are `cudaMemcpyAsync` on the engine's stream behind one
stream sync, so a round no longer drains the device (`--expert-slots-reselect-every`, `--expert-slots-swaps`,
`--expert-slots-swaps-final`, `--expert-slots-min-gain`, `--expert-slots-decay`; 0 turns a pass off).  Measured on the
same T4, 512-token prompt, 300 greedy tokens, 160K context, `--prefill --cpu-threads 24`, two runs per arm, pinned to
NUMA node 1, box load 0.9-4.0: the after-prompt round moves 2,048 experts (4,691 MiB in 469 ms) and the first request
then serves **83.2 %** of its routed picks from VRAM (61,641/74,100), the second **88.7 %** (79,844/90,000); each
mid-decode round moves 96 experts, 218-222 MiB, in 21.8-22.5 ms.  `--expert-slots-decay 1.0` gave 84.7 % / 86.3 % and
`--expert-slots-reselect-every 0` 85.7 % / 85.5 % on shorter requests; `--expert-slots-swaps 0
--expert-slots-swaps-final 0` leaves the map as it was placed (**20.2 %** on both requests) and is the arm for
byte-identical repeats.  Two runs of an arm print the same ids - the positional trigger is why, and the 3-second clock it
replaced could not (its own paragraph above is the measurement of that clock).

Not measured: sessions long enough for the hot set to drift, and any card other than this T4; and whether 1.5 is the
right margin (only 0 and 1.5 were compared) or 0.7 the right decay (0.7 against 1.0 on this fixture only, no workload
that changes subject inside one prompt).

**The speed configuration on this T4: 128K of context, `STRATA_KOLIBRI_EXPERT_F16=20`, the speed environment, the
expert cache and prefix reuse together.**  `STRATA_KOLIBRI_PF_STAGE=1 ROUTER_TILE=1 ROUTER_TK=3 ROUTER_TJ=16
ATTN_MB=256 RES_TOP=48 ATTN_FLASH=1 EXPERT_F16=20 PF_PARTIAL=1`, `--ctx 131072 --expert-slots auto --expert-slots-hot
--expert-slots-profile ... --vram-reserve-mib 1024 --prefill --cpu-threads 24`.  Measured, two runs of two requests,
512-token prompt, 128 greedy tokens, pinned to NUMA node 1, load 0.4-1.6: **prompt read 154.0 / 158.6 and decode
24.83 / 24.93 tok/s** (published-figures #35, #36), 7 slots/layer of residency cache (0.78 GiB; the fit keeps 1024 MiB
free for the runtime), 31.4 % of picks served from VRAM on the first request and 33.1 % on the second, and the second
request of an identical prompt reads **0.0 ms** (`RESUME 511`, `f[14]=1`).

Why 20 experts/layer: the fp16 cache is what makes the prompt read fast, and it is VRAM the residency cache would
otherwise have - at 8 experts 43 slots/layer fit, at 12 thirty, at 16 seventeen, at 20 seven, and at 22 and 24 the fit
reports "not one slot" (1239 and 539 MiB free).  At 160K the same 20 experts leave 4 slots/layer instead of 7 (the
KV of 128K costs 2.5 GiB of the 16), which is why 128K is the recommendation for this card: `setup.py` now gives a
16 GB Tesla T4 128K (`recommended_context`), and an explicit `--context` / the config's `--max-context` still overrides
it.

### Speed on an integrated Radeon (measured 2026-10-26, Radeon 8060S in a Ryzen AI Max+ 395, gfx1151, HIP on ROCm 7.14.1, 123 GiB RAM, Q4_K_M)

The engine was built from this tree with `-DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151` against the ROCm
7.14.1 toolchain (setup's own source build is CUDA-only - see "Not verified" below). Same harness
(`tools/kolibri/bench.py --prefill --cpu-threads 24`), the same Q4_K_M file, 42.90 GiB of expert blobs pinned in the
shared LPDDR5X pool:

| Prompt | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **short (512 tokens)** | 23.8 tokens/s | 168.5 tokens/s |
| **32K tokens** | 9.8 tokens/s | 114.0 tokens/s |

Reading (prefill) a short prompt is faster than on the T4 above (168.5 against 156-160 tokens/s): there is no PCIe
hop, the pool and the GPU read one unified pool.  Writing (generation) is slower (23.8 against 30 tokens/s), and at
32K both directions fall behind the T4 (114.0 and 9.8 against 119-120 and 13-14): the CPU pool and the 40-CU iGPU
share one memory system.
Loading took 9 s.  No 100K prompt was measured here.  The 16-token greedy generation from
`tools/kolibri/oracle_ids512.txt` is **bitwise identical** to the RTX 3090's CUDA run, which is what makes these
numbers comparable at all.

## Not verified (v1)

- AMD (HIP) beyond the two boxes measured: a Radeon AI PRO R9700 (gfx1201, docs/AMD_R9700_PORT.md) and an integrated
  Radeon 8060S (gfx1151, the speed subsection above). Setup's own HIP build for this family is still open, so a
  one-click install on AMD is not verified.
- Q8_0 beyond a smoke run (parity is at Q4_K_M).
- Speculative decoding: the model has no MTP layer; prompt-lookup drafting is future work.
- Multi-GPU, images, the Intel Arc (SYCL) build (refused, like Qwen3.6's).
- The three open speed floors on this box, wall-clock measured: prompt read short of 32K is 156.0-159.6 tok/s by
  default and 179.8-181.0 tok/s with `STRATA_KOLIBRI_PF_PARTIAL=1` (floor 200); prompt read at 32K is 119.6 / 119.4
  tok/s (floor 150, not met - the same code measured 154.0 / 153.7 on this box's earlier state, and the 512-token
  figure did not move between those states, which is what marks it as box state); decode short is 30.3 / 30.3 tok/s
  (floor 30, **met** once the NUMA binding is stated) and 13.4 / 13.5 tok/s after 32K (floor 20, not met).  An earlier
  version of this file quoted 129 tok/s at 32K and 21.2 tok/s decode from a pre-router phase; those are superseded.
  The after-32K decode shortfall is now attributed rather than estimated: 1.28 GB of PCIe per token at 0% residency,
  with the pool drain only 12.86 ms of the token, so the levers are less expert traffic, a fused compressed-weight
  GEMM, speculation, or a card class where the placement sign flips - not another knob.
- The 32K attention runs at ~1 TFLOP/s against the T4's 32.5 TFLOPS fp32 peak
  (neither a cublas re-tiling nor a tensor-core fused flash closed it - both measured and recorded).
- A card class other than this one: every number above is measured on a Tesla T4 + 2x E5-2690 v4.  The
  card-class rule (the host's streaming bandwidth vs the GPU's PCIe read) is stated in the speed section
  above, but the class-B and class-C numbers are NOT measured.

## End-to-end acceptance (2026-10-17, Tesla T4 16 GB, 755 GB RAM, Q4_K_M)

- Fresh install: `./setup.sh --family kolibri --model Q4_K_M --gguf-dir <folder with the GGUF> --yes` -
  "All set", 13 ok marks; writes `strata-kolibri-q4_k_m.json` (exe `build/strata-kolibri`, `--model`/`--ctx
  32768`, tokenizer pack, `arch: kolibri1`) and `run-kolibri-q4_k_m.sh`.
- Server: ready on `http://127.0.0.1:8080/v1`, context 32768, `/v1/models` -> `kolibri-1-q4_k_m`.
- Chat: English "What is 27 times 43?" -> "1161"; German "Was ist 27 mal 43?" -> "27 x 43 = 1161."
- Reasoning effort: `none` answers directly; `low` thinks briefly then "1161."; the default (high) thinks at
  length (`reasoning_content` carries the `<think>` block).
- Tool call: `get_weather` with `{"city": "Berlin"}` comes back as a structured `tool_calls` entry with
  `finish_reason: tool_calls` (config `"tool_call_recovery": true` at the time; unset gives the same today, because the
  pack's own template asks for the JSON call form and the server follows the template - see the top of this page).
- Long-context needle: a 6,201-token prompt (240 numbered sentences, the needle "Das Schluesselwort ist
  Kastanienmarmelade" at sentence 120) answered exactly "Kastanienmarmelade". The 100K needle (102,551
  tokens, ctx 131072, server :8124) ran to completion 2026-10-18 00:29 with `--prefill --cpu-threads 48`
  (chunked prefill, Task 8): the prompt read in 11,534 s (mean 8.9 tok/s - 14.2 early, 4-6 late as the
  position grows), 48 tokens generated, `finish_reason: length`. The reply recites the numbered sentences
  verbatim from sentence 00000 - word-perfect recall at 100K positions, which verifies the prompt-read and
  the KV cache end-to-end at ctx 131072 - and hits the token cap before answering. At 100K the model
  recites from the context start instead of answering (a model behaviour at extreme length on this
  quantisation); the 6,201-token needle answers exactly. Without `--prefill` the same prompt reads at
  4-9 tok/s (~78 h); long context works either way, the plain path is just slow (the two-pass attention
  past the 48 KB shared-memory cap) and is the recorded speed work, not v1.
- Q8_0 smoke (two shards, 83.1 GB): 903 tensors, 75.07 GiB of expert blobs pinned, 145.9 s to load, 16
  greedy tokens generated.
- Flash-Next untouched: `git diff main -- src/core src/kernels src/program` - `src/core` has no deletions
  (purely additive); `src/kernels` extends one router predicate with (384, 6); `src/program/generate.cpp`
  appends ` kolibri1` to `--families` under `STRATA_HAVE_KOLIBRI` and generalizes the family switch (the
  qwen35 dispatch keeps its own ifdef; Flash-Next files fall through exactly as before).
- Test suite: CTest 122 tests; the 121 that can run on this box all pass (re-run 2026-10-17 21:20, 0 failures;
  3 skip on this box by design: `prefill_fused_moe_test`, `prefill_fused_iq_test`, `expert_multi_test` need
  Flash-Next model data). The one excluded test, `ple_parity`, needs a Flash-Next Q2_0 PLE shard at
  `../../Q2_0/` (a dev-machine data path this box does not have - it holds IQ3_S); it reads no Kolibri code.
