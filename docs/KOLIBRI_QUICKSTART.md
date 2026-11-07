# Kolibri 1: install, start and use

Kolibri 1 is Aleph Alpha's 78-billion-parameter German and English model (3.46B active per token). Strateu runs
it with its own engine path. This page takes you from nothing to a working model on your own PC, in plain
steps. The measured detail lives in [the reference](KOLIBRI1.md); this page is the short way in.

## What you need

| | |
|---|---|
| RAM | about 56 GB all in. Below 52 GiB setup switches the engine to a mode that keeps less in RAM (see [Which memory mode](#which-memory-mode)) |
| Graphics card | NVIDIA (CUDA) or AMD (ROCm). The dense weights and the KV cache live in VRAM; 16 GB runs it at 32K context |
| Disk | about 50 GB for the model file, plus room for the engine build |
| System | Linux or Windows, a source checkout of this fork (the installer does not carry this family yet) |
| Tools | `cmake` and `ninja`, and either `nvcc` (CUDA) or ROCm's `hipcc`. Setup compiles the engine once |

## 1. Install and download

```sh
./setup.sh --family kolibri
```

Setup asks which size (Q4_K_M is the only one offered), how much context you want, and then:

1. downloads `Kolibri-1-Q4_K_M.gguf` (47.5 GB) from the pinned revision of
   [Hob-forge/Kolibri-1-GGUF](https://huggingface.co/Hob-forge/Kolibri-1-GGUF) and checks its SHA-256;
2. compiles the family's own engine, `strata-kolibri`, into `build/` (CUDA) or `build-kolibri-hip/` (AMD);
3. writes the settings and a start script next to setup.py (the checkout you ran it from).

The context choices are 8K to 256K. Setup recommends 32K up to 16 GB of VRAM, 64K up to 24 GB, 128K above —
the KV cache and the expert cache share the card, so a bigger context leaves fewer experts in VRAM.

Unattended, install only, and start nothing:

```sh
./setup.sh --family kolibri --yes --no-start
```

Already have the file? Point setup at the folder that holds it; setup still checks the hash:

```sh
./setup.sh --family kolibri --gguf-dir /path/to/the/folder --yes
```

Other useful flags: `--data-dir` (put the model on another drive), `--context 131072`, `--port 8080`.

## Which memory mode

The expert weights are 42.9 GB. Setup reads your RAM and picks one of two placements, then prints what it
picked and why:

- **52 GiB of RAM or more:** the experts are repacked once into a pinned table in RAM and the CPU keeps them
  fed. On CUDA this is `--expert-slots 0` with the CPU pool; on ROCm it is the slot cache
  (`--expert-slots auto --expert-slots-hot`). Startup repacks about 43 GB, so the first start takes a while.
  Setup says so: `expert table: 43 GiB of experts in RAM for this card's pool, and this PC has 64 GiB`.
- **below 52 GiB:** setup writes the mmap cache instead — `--expert-no-pin --expert-slots auto
  --expert-slots-hot` with `STRATA_KOLIBRI_NOPIN_PROFILE=demand`. Nothing is repacked at startup, the model
  file stays on disk as a mapped file, and missing experts are copied into VRAM as routing asks for them. It
  needs much less RAM and starts in about a second; it is slower per token than the table on machines that can
  hold it.

The 52 GiB is a recommendation, not a hard limit: free VRAM and disk speed still decide whether the cache fits
and how fast it runs. If VRAM cannot hold even one expert slot, the engine says so when it starts and tells you
how much context to cut.

## 2. Start it

```sh
./run-kolibri-q4_k_m.sh          # Linux, from the folder setup wrote it in
run-kolibri-q4_k_m.bat           # Windows
```

Setup prints the exact script name when it finishes. The script starts the server on port 8080 and opens the
browser. Stop it with Ctrl+C in that window; run the script again to start it again. In a source checkout the
script sits beside `setup.py`, so `./run-kolibri-q4_k_m.sh` from the checkout works.

## 3. Use it

- **In a browser:** the page the server opens (default `http://127.0.0.1:8080`).
- **OpenAI-compatible:** `http://127.0.0.1:8080/v1`; the model name is `kolibri-1-q4_k_m`. Check it with
  `curl -s http://127.0.0.1:8080/v1/models`. (Setup's closing summary says "model name: anything" - the server
  takes any name, and `/v1/models` reports the real one.)
- **Anthropic-compatible:** `http://127.0.0.1:8080/v1/messages`.
- **Your coding apps and agents:** `strata connect` writes the address, key and model name into their settings;
  `strata connect <app> --check` sends one prompt and one tool call the way that app does. See
  [the Terminal guide](TERMINAL.md).

Two things are worth knowing when you use it:

- **Thinking level.** It thinks before answering by default. `"reasoning_effort": "none"` answers directly,
  `low`, `medium` and `high` spend more thinking tokens. Measured on a T4: `none` answered `27 x 43 = 1161.`
  with no thinking. The levels and their budgets are in [Thinking levels](KOLIBRI1.md#thinking-levels).
- **Sampling and tool calls.** Served Kolibri requests default to temperature 1.0 (the model card's number);
  explicit request values win, and temperature 0 is still greedy. Tool calls come back as structured
  `tool_calls` with no setting to add.

## 4. What to expect

Measured on a Tesla T4 (16 GB, CUDA 12.8, driver 580.196.07, 2 x Xeon E5-2690 v4, 755 GB RAM), Q4_K_M, a
512-token prompt and 128 fixed decode steps. These are fixed-history runs, not a chat-speed guarantee:

| Mode | Startup | Prefill | Decode |
|---|---:|---:|---:|
| table, CPU pool (`--cpu-threads`, `--expert-slots 0`) | 37.6 s | 141.6 tok/s | 34-38 tok/s |
| mmap demand, first run | 0.9 s | 63.5 tok/s | 20.7 tok/s |
| mmap demand, learned profile | 3.1 s | 99.9 tok/s | 21.8 tok/s |

The mmap numbers are the engine's own phase counters. The table row's decode rate is a two-run difference
(about ±10 %). A machine with room for the table is faster; a machine without room can still run. Other cards
and other prompts differ — [the reference](KOLIBRI1.md) has the rest, each with the hardware it was measured on.

## If something goes wrong

| What you see | What it means |
|---|---|
| `not enough free disk space in …` | the model needs about 50 GB; `--data-dir` puts it on another drive |
| the file `has the wrong SHA-256 …: deleted` | the download was damaged; setup removed it, run setup again to fetch a good copy |
| the engine build cannot find a compiler | the CUDA toolkit (`nvcc`) or ROCm (`hipcc`) is missing; install one for your card's vendor |
| `not one slot fits` | VRAM is too small for the expert cache at that context; lower `--context` |
| no start script was written | run setup without `--no-start`, or start the engine yourself from the config setup printed |
| It answers slowly | look at which mode setup printed; the mmap mode trades speed for RAM |

## Where the details are

- [Kolibri 1 reference](KOLIBRI1.md) — model facts, thinking levels, engine settings, all measurements.
- [Accuracy policy](KOLIBRI_ACCURACY_POLICY.md) — what a Kolibri change must prove before it ships.
- [Cache selection by backend](KOLIBRI_CACHE_SELECTION.md) — why the table stays the default where it fits.
- [W7800 cache campaign](experiments/kolibri-w7800-cache-optimization.md) and
  [T4 demand record](experiments/kolibri-t4-demand-cache.md) — the measurements behind both modes.

The model file and the engine are not committed to this repository; setup fetches and builds them.
