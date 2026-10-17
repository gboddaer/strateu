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

The GGUFs are the community conversion (Seraphiel102/Kolibri-1-GGUF), quantized from the FP8 checkpoint via
BF16; every quant is made from that BF16 GGUF, none is a requant.

| Size | File | Bytes | Needs |
| --- | --- | ---: | --- |
| Q4_K_M | `Kolibri-1-Q4_K_M.gguf` (one file, 903 tensors: 426 Q4_K, 76 Q6_K, 401 F32) | 47.5 GB | ~56 GB RAM |
| Q8_0 | `Kolibri-1-Q8_0-0000{1,2}-of-00002.gguf` | 83.1 GB | ~96 GB RAM |

Routed experts are 42.4 GB (Q4_K_M) / 75 GB (Q8_0); one decoded token reads 6 experts × 2.21 MB × 50 layers
≈ 663 MB — the same memory-bandwidth class as Flash-Next Q2_0, so the same engine design applies.

## Running it

```
./setup.sh --family kolibri --model Q4_K_M --gguf-dir /path/to/ggufs --yes
```

or directly:

```
strata --native Kolibri-1-Q4_K_M.gguf --max-context 32768 --prefill auto
```

Sampling defaults (the model card's): temperature 1.0, top-p 0.97, top-k 128. Reasoning effort is per request:
`"chat_template_kwargs": {"reasoning_effort": "low"}` — `none`, `low`, `medium`, `high`.

## How the engine path works

`src/kolibri/kolibri_engine.cu` is a driver of its own beside Flash-Next's (the `strata` family switch picks it
from the file's architecture), built from Strata's kernels: dense Q4_K/Q6_K matrices via native MMVQ, the norms
and router gates as F32, the routed experts repacked at load into native `[gate | up | down]` blobs per
(layer, expert). On a card that cannot hold 42.4 GB the hybrid mode runs: every expert pinned in RAM, fixed
VRAM slots per layer, the batch's other experts on the CPU pool (ggml-cpu's Q4_K/Q6_K dot products), prompt
chunks streamed. The router is Strata's `sigmoid_logit_add` top-k kernel (384/6), checked bitwise against a CPU
reference. Full-attention layers keep fp16 KV rows; sliding-window layers keep a 513-cell ring.

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

Speed (`tools/kolibri/bench.py`, the plain exactness-first path; grouped GPU experts, VRAM expert slots,
split-K decode attention, fp16 prompt attention and decode graphs are the recorded speed work, not built):

| test | Strata kolibri1 | llama.cpp (same box) | llama.cpp README (RTX 5070) |
|------|-----------------|----------------------|------------------------------|
| pp 512 (short) | 10.0 tok/s | 126.1 tok/s | 723 tok/s |
| tg 128 (short) | 9.3 tok/s | 19.1 tok/s | 18.0 tok/s |
| pp 32768 | 1.3 tok/s | 157.5 tok/s | — |
| tg 128 after 32K | 0.6 tok/s | 21.2 tok/s | — |

Decode short is within 2× of llama.cpp's (the experts stream over PCIe from RAM: 663 MB/token); prompt-read
and long-context decode are far behind — the plain path reads every expert blob over PCIe per token and its
full-attention decode is O(n) per position with a two-pass softmax past the 48 KB shared-memory cap.

## Not verified (v1)

- AMD (HIP) — the code builds for HIP by the q36 pattern, validated on NVIDIA only.
- Q8_0 beyond a smoke run (parity is at Q4_K_M).
- Speculative decoding: the model has no MTP layer; prompt-lookup drafting is future work.
- Multi-GPU, images, the Intel Arc (SYCL) build (refused, like Qwen3.6's).
