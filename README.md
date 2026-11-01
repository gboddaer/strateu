<h1 align="center">Strateu, EU fork of Strata</h1>

**English** · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC</b><br>
NVIDIA or AMD graphics card (12 GB or more) · Windows or Linux · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC. It also runs **[Kolibri 1](docs/KOLIBRI1.md)** (Aleph Alpha's 78B
German & English MoE, 3.46B active) on 56 GB of RAM: `./setup.sh --family kolibri` (Q4_K_M - the only size setup
offers so far).

## About this fork

**strateu** forks [Strata](https://github.com/Niko1221/Strata) for **European open models**, starting with
**Kolibri 1** (`src/kolibri`): 50 layers, 384 routed experts per layer, top-6 routing and mixed sliding-window/full
attention. Experts are held in system RAM; CPU-pool misses need no expert-weight PCIe round trip. Residency and
prefill placement depend on the configuration and card.

**Kolibri preview preparation:** see [the preview guide](docs/KOLIBRI_PREVIEW.md) for installation, evidence and
publication status. The extracted Linux R9700/gfx1201 candidate measures **23.94 tok/s** after 512 prompt ids and
**17.75 tok/s** after 8191 (Ryzen 7 5700G, eight CPU lanes, Q4_K_M, 96 generated tokens). It uses MMQ prefill and a
host-assisted decode graph. CUDA sm_75/sm_86 compilation passes; the T4 contributor has recorded integrated
full-logit parity, but CUDA package qualification is still pending. No RTX 3090 runtime or speed result is claimed.

Regression parity is not a general accuracy guarantee. Read [the accuracy policy](docs/KOLIBRI_ACCURACY_POLICY.md)
and the preview's known failures/reference limits. Earlier T4 and AMD measurements, including unsuccessful arms,
remain in [the campaign report](docs/strateu-campaign-report.md), [the AMD port log](docs/AMD_R9700_PORT.md) and
[the findings log](docs/findings-log.md). Historical detector results do not qualify the new decode graph.

## How fast is it?

We measured it on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

In earlier campaigns, this fork also measured Aleph Alpha's **[Kolibri 1](docs/KOLIBRI1.md)** (Q4_K_M) on a **Tesla T4 (16 GB)** with
755 GB of RAM holding the expert pool:

| Prompt | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **short (512 tokens)** | 30 tokens/s (NUMA binding, expert cache off) | 156-160 tokens/s (opt-in `PF_PARTIAL`: 180-181, warm server 202) |
| **32K tokens** | 13-14 tokens/s (NUMA binding) | 119-120 tokens/s |
| **100K tokens** | — | 97 tokens/s |

The same file on the integrated Radeon of a Ryzen AI Max+ 395 (Radeon 8060S, Strix Halo, `gfx1151`, HIP on
ROCm 7.14.1) with 123 GiB of RAM holding the expert pool:

| Prompt | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **short (512 tokens)** | 23.8 tokens/s | 168.5 tokens/s |
| **32K tokens** | 9.8 tokens/s | 114.0 tokens/s |

No 100K prompt was measured on the iGPU.

Two runs each on the bitwise-deterministic binary, box load stated; the 100K row is a single cold run. Those rows were
taken at a 256K context with a **hand-sized** expert cache. The current build sizes the cache itself and can place
experts by how often the router picks them:

- `--expert-slots auto` measured the room left after the dense weights, the KV and the scratch, then took
  **75 slots/layer (8.49 GiB)** at a **160K** context, keeping 700 MiB clear for the CUDA runtime. It prints the
  exchange rate it implies: 1 slot = 114 MiB = 5,857 tokens of context, because context and cache draw on the same
  VRAM. In the shipped Tesla T4 configuration (`--ctx 131072`, `EXPERT_F16=20`, `--vram-reserve-mib 1024`) the same
  fit takes **7 slots/layer (0.89 GiB)**: the fp16 expert cache and the residency cache draw on the same 16 GB.
- `--expert-slots-hot` fills those slots by routed-pick frequency instead of by expert id, **one map per layer** (the
  default since 2026-10-27), and `--expert-slots-profile` carries the learned set across restarts. The set is re-chosen
  in **rounds**: every 32 positions at most 96 experts move while a request runs, and one larger round of up to 2,048
  runs after each prompt; a candidate has to beat the expert it evicts by 1.5 decayed counts, and the counts decay by
  0.7 a round. On this T4 at a 160K context the hit rate went from **21.4%** (by id) to **83.2%** on the first request
  and **88.7%** on the second (two runs each); the same code at 128K with `EXPERT_F16=20` serves **31.4% / 33.1%**,
  because the fp16 cache leaves 7 slots/layer instead of 75. Decode with sampling stays at 24.4-26.3 tokens/s; greedy
  is ~30% faster but loops on this model (`Chudnov Chudnov Chudnov ...`), which is why the server samples by default.
- The engine takes whatever `--ctx` gives it; for a 16 GB Tesla T4 `setup.py` now recommends **128K**, because the KV
  of 128K costs 2.5 GiB of the 16. An explicit `--context`, or the config's `--max-context`, overrides it.

**The T4 prefill rates above are not free of accuracy cost, and that is measured.**  On the T4 the fast prefill comes
from the fp16 expert cache (`EXPERT_F16`), which is on in the shipped T4 configuration. Measured 2026-10-30 against the
plain path on identical forced histories (512-id prompt, 9 forced steps, full 128,000-value logits): mean KL **0.400**
and a worst single-logit difference of **11.08** for the cache alone, and mean KL **0.168** with top-1 agreement on only
**7 of 9** steps once `PF_PARTIAL` is added. Two runs of either configuration are bitwise identical, but repeatability
alone does not establish the cause or rule out a deterministic implementation defect. The reported KL is two orders
of magnitude worse than the R9700 figure, and neither configuration is qualified under [the accuracy policy](docs/KOLIBRI_ACCURACY_POLICY.md). The accuracy-first T4
configuration is therefore the engine's own default with the expert cache off: **51.6 / 53.6 tokens/s** reading a
512-token prompt and **30.5 / 30.5 tokens/s** writing, bitwise identical to the plain path at 512, 8,191 and 11,000
prompt tokens. The fp16 cache is being investigated on its own branch; the numbers and the open questions are in
[docs/KOLIBRI_CUDA_PREVIEW.md](docs/KOLIBRI_CUDA_PREVIEW.md).

The cost of a round is bounded: 96 experts is 218-222 MiB over PCIe in ~22 ms mid-request, and the after-prompt round
of up to 2,048 is 4,691 MiB in ~470 ms (the campaign's unbudgeted round moved 57 experts, 6.5 GiB, ~690 ms). Residency
moves bytes, and with a CPU pool in play that does change them - an expert in a slot is read in fp32, the same expert
on the pool in q8_1 (findings-log 0.3: mean |dlogit| 0.855, the id flips at token 8); with the pool off the two device
legs are bitwise. Shipped T4 speed, two runs pinned to NUMA node 1: **154.0 / 158.6 tokens/s** reading a 512-token
prompt, **24.83 / 24.93 tokens/s** writing, and a repeated prompt reads in **0 ms**. The shipped T4 configuration runs
the fp16 expert cache, so those prefill rates carry the accuracy cost noted above. Box spec, full tables and context:
[the campaign report](docs/strateu-campaign-report.md).

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Strata is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. It needs **12 GB of VRAM or more**. |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **Apple Silicon Macs (M1 to M5)**, built from source by `./setup.sh` (the Metal engine): [Mac](docs/MAC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strateu on this PC for me: https://github.com/gboddaer/strateu - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### Or do it yourself

[Download Strateu](https://github.com/gboddaer/strateu/archive/refs/heads/main.zip) and unzip it (or `git clone` it).
**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

**Mac (Apple Silicon, M1 to M5, macOS 15 or newer):** install Apple's Command Line Tools once
(`xcode-select --install` in Terminal; Xcode itself is not needed). Then open Terminal in the Strata folder and run
**`./setup.sh`**. It builds the Mac engine on your Mac (about 7 minutes), asks the same questions, downloads the model
and starts it. Next time, `./setup.sh` starts it right away. Which size fits your Mac and how fast it is:
[docs/MAC.md](docs/MAC.md).

The steps are the same for NVIDIA and AMD. The installer finds your card and sets up the right engine for it. It
asks you a few questions:

- which model and which size,
- how much context (how much text the model keeps in mind),
- whether it should read pictures.

Press Enter each time for the recommended answer. Then it downloads the model (about 70 GB) and starts it. If the
download stops, run it again: it continues where it left off. Your browser opens the Strata app at
`http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window. The window shows what Strata is doing.

**Next time**, run `START-HERE.bat` (or `./setup.sh`) again. It starts right away and downloads nothing twice. Close
its window to stop the model. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

**Is it fast on your PC?** Run `BENCH.bat` (`./bench.sh`): about 5-10 minutes, it measures chat, 4K/16K prompts and an agent-style session with your own settings and writes a report to share ([how](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Which model should I pick?

The installer recommends one for your RAM. The same model comes in several sizes, compressed more or less. Smaller
sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL and Q8_0](docs/MODELS.md#unsloths-files-ud-iq4_xs-to-q8_0):**
  4- to 8-bit, closer to the full model (93-95% the same next token as Q8_0). Fast where every expert fits in RAM
  (~88 GB for UD-Q4_K_XL up to ~136 GB for Q8_0); with less RAM Strata reads part of them from the SSD while it
  answers, which is much slower (7-8.5 tokens/s for UD-Q4_K_XL on a 64 GB PC).
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, run
`SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

### Optional: more speed on one card when the whole model fits in RAM

Two opt-in settings (by RubenTorney, #1904 and #1905). They help when you run one graphics card and your RAM holds
every expert (Strata's arena mode, for example Q2_0 or IQ3_XXS on a 64 GB PC). Open your config,
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)), and add:

- **Faster decode:** `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread
  instead of pausing the answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long
  documents. Answers can differ slightly from the default (they stay the same from run to run).
- **Faster prompt reading on Windows:** `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block. Windows can lock only about
  half the RAM for the graphics card; this locks the experts the card actually needs first. Measured on the RTX 5070:
  +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata (`START-HERE.bat` or `./setup.sh`) after the change. To undo, remove the two entries.

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues where
  it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder. Found a
security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was
compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few
parts and every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support its development:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
