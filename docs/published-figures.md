# Published figures manifest

Every speed and correctness number quoted in `README.md`, `STRATEU.md`, `docs/strateu-campaign-report.md` and
`docs/KOLIBRI1.md` is listed here with the ledger row it came from, the measurement method, and the machine load.  The rule
(amendment 7): a headline rate needs **two independent methods agreeing within ~10%**, new numbers live in the ledger first,
and this manifest is what a staleness check diffs against — run `python tools/kolibri/check_published_figures.py`.

A number that is not in this file has no business being quoted.  If a doc and this table disagree, **this table is the
measurement** and the doc is stale.

| # | figure | value | method | load / scope | ledger row | quoted in |
|---|---|---|---|---|---|---|
| 1 | prefill, 512-token prompt | **157.0 / 160.3 tok/s** | wall clock, `bench.py short` | load 5.43 / 5.57 | "CORRECTION (post-audit)" | STRATEU, report §1/§3.1, KOLIBRI1 |
| 2 | prefill after 32K context | **154.0 / 153.7 tok/s** | wall clock, `bench.py after` | load 7.49 / 14.10 | "CORRECTION (post-audit)" | STRATEU, report §1/§3.1, KOLIBRI1 |
| 3 | decode, short context | **24.5 / 24.4 tok/s** | wall clock, `bench.py short` | load 5.43 / 5.57 | "CORRECTION (post-audit)" | STRATEU, report §1/§3.2, KOLIBRI1 |
| 4 | decode after 32K context | **12.3 / 12.2 tok/s** | wall clock, `bench.py after` | load 7.49 / 14.10 | "CORRECTION (post-audit)" | STRATEU, report §1/§3.2, KOLIBRI1 |
| 5 | decode short, `numactl --cpunodebind=1` | **30.3 / 30.3 tok/s** | wall clock, `bench.py short` | load 6.05 / 6.63 | "amendment 6, binding is part of the number" | report §13 |
| 6 | prefill 512, `numactl --cpunodebind=1` | **132.6 / 135.1 tok/s** | wall clock, `bench.py short` | load 6.05 / 6.63 | "amendment 6, binding is part of the number" | report §13 |
| 7 | prefill 512 / decode short, both sockets | **150.4-161.7 / 24.5-24.6 tok/s** | wall clock, `bench.py short` | load 7.42 / 6.93 | "amendment 6, binding is part of the number" | report §13 |
| 8 | KL vs llama.cpp, answers | **0.00132** (top-1 0.982, 109 steps) | `kl.py`, parity fixtures | n/a (not a rate) | "KL gate" | STRATEU, report §1/§4 |
| 9 | KL vs llama.cpp, all steps | **0.00238** (top-1 0.9779, 136 steps) | `kl.py` | n/a | "KL gate" | report §4 |
| 10 | plain-path oracle | **BITWISE** | `pf_gate.sh` on `oracle_ids512.txt` | n/a | "FIXED: block_max barrier" | STRATEU, report §1/§4 |
| 11 | prefill determinism | **two runs bitwise identical; `racecheck` 0 hazards** | `racecheck` + two-run diff | n/a | "FIXED: block_max barrier" | STRATEU, report §1/§4 |
| 12 | 100K needle | **recall 4/5, exact 3/5** | `needle_audit.py`, five fixtures (33,364 / 65,876 / 98,591 / 98,591 / 101,415 tokens) | shipped path | "amendment 4" | STRATEU, report §4, KOLIBRI1 |
| 13 | serve path, 415-token request | **150.6 cold; 176.2 / 170.8 warm tok/s** | server `timings.prompt_ms` | ctx 8192, `cache_n = 0` | "serve path at the floor's shape" | report §3.3 |
| 14 | VRAM at 100K prefill | **peak 15,241 MiB** (`ATTN_MB=512`); `ATTN_MB=1024` OOMs | sampled `nvidia-smi` | 101,414-token prefill | "queued measurements settled" | report §6.2 |
| 15 | pinned expert table | **42.90 GiB** | engine load report | n/a | "design" | STRATEU, report §2.1/§6.2 |
| 16 | expert coverage, 512-token corpus | **top-24 50.4%, top-80 79.2%, top-128 90.4%** | `STRATA_KOLIBRI_EXPERT_HIST=1` + `--counters` | instrument-time (coverage, not a rate) | "C0 coverage curve" | report §7/§C0 |
| 17 | expert coverage, 32K corpus | **top-24 42.9%, top-80 70.3%, top-128 83.0%** | same | instrument-time | "C0 coverage curve" | report §C0 |

## Retracted figures (must not be quoted again)

| figure | value | why it was withdrawn |
|---|---|---|
| decode short | ~~48.0 / 48.3 tok/s~~ | counter-derived: the reciprocal of the pool's per-token drain (20.70 ms), not a wall-clock token.  See #3. |
| decode after 32K | ~~48.3 tok/s, 2.4x the floor~~ | same error; the "stretch 45-60 reached" claim falls with it.  See #4. |
| prefill short | ~~149.7 / 151.9 tok/s~~ | a loaded-box sample; superseded by #1 (157.0 / 160.3) with its load stated. |
| prefill after 32K | ~~153.9 / 154.9 tok/s~~ | superseded by #2 (154.0 / 153.7) from the same corrected protocol. |

## Historical and comparison figures (quoted only in `docs/KOLIBRI1.md`)

These are not this fork's current measurements.  They are either (a) an earlier phase's figures, kept only where a
supersession note names them, (b) the plain-path / llama.cpp / upstream-README columns of `KOLIBRI1.md`'s comparison table, or
(c) floor *targets*, which are not measurements at all.  Every value carries its unit so the checker can see it.

| figure | value(s) | what it is |
|---|---|---|
| pre-router phase, prefill 32K | 129 tok/s | earlier phase, superseded by #2 (154.0 / 153.7 tok/s) |
| pre-router phase, decode | 21.2 tok/s | earlier phase, superseded by #3-#4 (24.5 / 12.3 tok/s) |
| plain-path prefill, plain-path decode | 10.0 tok/s, 1.3 tok/s, 9.3 tok/s, 0.6 tok/s | the exactness-first per-token path, quoted for contrast |
| llama.cpp on this box | 126.1 tok/s, 157.5 tok/s, 19.1 tok/s, 21.2 tok/s | the reference engine (itself unstable at 98.6K - report §6.5) |
| llama.cpp README (RTX 5070) | 723 tok/s, 18.0 tok/s | upstream marketing figure, not comparable to this box |
| upstream "reads your prompt" claim | 60 tok/s | the upstream project's own number, quoted in `README.md`'s speed section |
| rounded restatements of #1-#4 in prose | 160 tok/s, 158 tok/s, 162 tok/s, 154 tok/s, 169 tok/s, 157 tok/s, 30 tok/s, 13 tok/s | same measurements as the rows above, rounded in prose |
| plain path at 100K | 8.9 tok/s, 4 tok/s, 9 tok/s | the per-token path's 100K mean, for contrast (11,534 s) |
| floor targets | 20 tok/s, 30 tok/s, 45 tok/s, 60 tok/s, 150 tok/s, 200 tok/s, 300 tok/s, 400 tok/s | **targets, not measurements** |
