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

## Campaign II figures (branch `kolibri2-speed`, Tesla T4, driver 580.196.07, kernel 6.1.0-52-amd64)

Same harness and binding as #1-#4 (`bench.py`, `numactl --cpunodebind=1 --membind=1`, `--cpu-threads 24`,
`STRATA_KOLIBRI_EXPERT_F16=24`).  Two groups: the **default path**, which is byte-identical to `main` (the campaign's
knob is opt-in), and the path with **`STRATA_KOLIBRI_PF_PARTIAL=1`**, which is the campaign's actual speed change.
Rows #1-#4 above stay in the table because they are what `main` measured on the box's *earlier* state; where the same
code now measures differently, the difference is box state and both numbers are shown with their date.

| # | figure | value | method | load / scope | ledger row | quoted in |
|---|---|---|---|---|---|---|
| 18 | prefill 512, default path (knob off) | **156.0 / 159.6 tok/s** | wall clock, `bench.py short` | load 0.88 / 3.03 | "G1 (campaign II)" | STRATEU, KOLIBRI1 |
| 19 | prefill 512, `PF_PARTIAL=1` | **179.8 / 181.0 tok/s** | wall clock, `bench.py short`; cross-checked 181.7 cold / 201.8 tok/s serve-path tok/s | load 1.32 / 2.50 | "G1 (campaign II)" | STRATEU, KOLIBRI1 |
| 20 | prefill 512, serve path, `PF_PARTIAL=1` | **181.7 cold / 201.8 tok/s** (warm) | engine serve-pipe `DONE` timings, separate process | load 0.72 / 2.67 | "G1 (campaign II)" | STRATEU, KOLIBRI1 |
| 21 | prefill after 32K, default path | **119.6 / 119.4 tok/s** | wall clock, `bench.py after` | load stated in ledger | "G5 (campaign II)" | STRATEU, KOLIBRI1 |
| 22 | prefill after 32K, `PF_PARTIAL=1` | **125.7 / 122.1 tok/s** | wall clock, `bench.py after` | load 0.06 / 9.02 | "G5 (campaign II)" | STRATEU, KOLIBRI1 |
| 23 | decode short, `PF_PARTIAL=1` | **30.0 / 30.3 tok/s** | wall clock, `bench.py short`; serve path 30.2 / 30.1 tok/s | load 1.32 / 2.50 | "G5 (campaign II)" | STRATEU, KOLIBRI1 |
| 24 | decode after 32K, default path | **13.4 / 13.5 tok/s** | wall clock, `bench.py after` | load stated in ledger | "G4b (campaign II)" | STRATEU, KOLIBRI1 |
| 25 | decode after 32K, `PF_PARTIAL=1` | **13.3 / 13.4 tok/s** | wall clock, `bench.py after` | load 0.06 / 9.02 | "G5 (campaign II)" | STRATEU, KOLIBRI1 |
| 26 | KL vs llama.cpp, knob off / on | **0.00123 / 0.00133** (llama.cpp CPU-vs-GPU floor 0.00178) | `kl.py`, `ATTN_FLASH=1` both arms | n/a (not a rate) | "G1 (campaign II)" | STRATEU, KOLIBRI1 |
| 27 | expert GEMM, FP16-resident leg | **4.18 ms/layer** | `--counters` per-layer, instrument-time (a component, not a rate) | n/a | "G3 (campaign II)" | report |
| 28 | expert GEMM, Q4_K-from-VRAM arm | **9.99 ms/layer** (2.39x the FP16 leg; gate was 2.6) | same harness as #27 | n/a | "G3 (campaign II)" | report |
| 29 | decode PCIe traffic after 32K | **1.28 GB/token** (488.7 MB d2h + 789.5 MB h2d), 0% residency | `--counters`, instrument-time (a component) | n/a | "G4b (campaign II)" | STRATEU, KOLIBRI1 |
| 30 | expert-cache fill per 512-token prompt | **12.20 GB** (1551 promotions) | `--counters` `fill=`, instrument-time | n/a | "G4 arithmetic" | report |
| 31 | decode short with `--expert-slots 8 / 16` | **25.0 / 25.1 tok/s** (vs 30.3 without) | wall clock, `bench.py short` | load 4.73 / 6.67 | "G4 (campaign II)" | report |
| 32 | decode short with `STRATA_KOLIBRI_DEC_RES=1` | **29.5 / 29.2 tok/s**, prefill 144.9 / 141.8 tok/s | wall clock, `bench.py short` | load 3.79 / 4.48 | "G4 (campaign II)" | report |
| 33 | 100K needle, campaign II build | **recall 4/5, exact 3/5** (exact PASS on 32k90, 64k90, anchor; d90b recalled one letter short `quart-willow-8` vs `quartz-willow-884`; d90 answered a different instruction) | `needle_audit.py` on the venv interpreter, five fixtures | shipped path; unchanged from #12 | "G5 (campaign II)" | STRATEU, KOLIBRI1 |
| 34 | prefill 512, knob off, quiet box | **184.9 / 183.6 tok/s** with `PF_PARTIAL=1` against **159.0 / 158.8** off (+16.0%) | wall clock, `bench.py short` | load < 1 | "G1 (campaign II)" | report |


The AMD R9700 table that follows numbers its own rows from 18; the two tables are per-box and number independently.

## AMD Radeon AI PRO R9700 (gfx1201, 32 GB, ROCm 7.2.2) — re-derived, not inherited

A different box, so a different floor table. Every row below is two-run wall clock from
`docs/AMD_R9700_PORT.md` on host `rizo`: Ryzen 7 5700G (8 physical cores, 1 NUMA node), 93 GiB RAM,
90-91 GiB free at the start of each run, GPU idle at the start, `--cpu-threads 8`, shipped config
(`EXPERT_F16=24`, `RES_TOP=56`, `PF_STAGE=1`, router tiles) unless the row names a change. Prefill
rows subtract the engine's own load line (13.4 s warm); the end-to-end row does not, and says so.

| # | figure | value | method | load / scope | ledger row | quoted in |
|---|---|---|---|---|---|---|
| 18 | prefill, 512-token prompt, prefill pool ON, 8 threads | **87.8 / 85.4 tok/s** | wall clock, `--n 4` run minus the engine's 13.4 s load line, `--prefill` passed | idle, load average 0.08, 89 GiB free, 47 GiB buff/cache | "The prefill row was wrong before; re-derived as a four-arm sweep" | STRATEU, report §9.6 |
| 19 | prefill 512, end to end (load included), pool ON, 8 threads | **26.3 / 26.2 tok/s** | same runs, nothing subtracted | same | same | report §9.6 |
| 19a | prefill 512, prefill pool OFF, 8 threads | 16.8 / 16.8 tok/s | same shape with `--prefill` omitted (the pool is the 5.2x lever) | same | same | - |
| 19b | prefill 512 end to end, pool OFF, 8 threads | 11.6 / 11.6 tok/s | same | same | same | - |
| 19c | prefill 512, pool ON, 24 threads | 84.8 / 79.3 tok/s | same shape, `--cpu-threads 24` | same | same | - |
| 19d | prefill 512 end to end, pool ON, 24 threads | 25.6 / 25.1 tok/s | same | same | same | - |
| 19e | prefill 512, pool OFF, 24 threads | 7.1 / 7.3 tok/s | same shape, both arms off | same | same | - |
| 19f | prefill 512 end to end, pool OFF, 24 threads | 6.0 / 6.1 tok/s | same | same | same | - |
| 19g | decode, 253-token window, pool ON, 8 threads | 15.8 / 15.9 tok/s | `--n 257` wall minus `--n 4` wall | same | same | STRATEU, report §9.6 |
| 19h | decode, pool OFF, 8 threads | 15.9 / 15.9 tok/s | same | same | same | - |
| 19i | decode, pool ON, 24 threads | 7.6 / 7.6 tok/s | same; 24 lanes oversubscribe 8 physical cores | same | same | - |
| 19j | decode, pool OFF, 24 threads | 8.7 / 8.7 tok/s | same | same | same | - |
| 20 | decode, short context | **15.8 / 15.8 tok/s** | wall clock, tokens 130-257 of an `n=257` run | same | same | STRATEU, report §9.6 |
| 21 | prefill, 99,299-token prompt (ctx 105024) | **102.3 / 100.8 tok/s** (`ATTN_MB=512`); **105.2 / 105.8** (`ATTN_MB=1024`) | wall clock, load subtracted | same | "long context, same engine" | STRATEU, report §9.6 |
| 22 | prefill, 127,811-token prompt (ctx 131072) | **94.1 / 94.2 tok/s** (`ATTN_MB=512`); **100.4 / 100.8** (`ATTN_MB=1024`) | wall clock, load subtracted | same | same | report §9.6 |
| 23 | decode at 100K context | **6.7 / 5.5 tok/s** | wall clock, `n=129` minus `n=1` on the 99,299-token fixture, `ATTN_MB=1024` | same | "decode at 100K" | STRATEU, report §9.6 |
| 24 | KL vs llama.cpp, answers | **0.00182** (top-1 0.9884, 344 answer steps) | `kl.py` against `/fast/llama-server-kolibri/llama.cpp` `f5027fb21`, built here as `build_rocm_rizo` | n/a (not a rate) | "The KL gate" | STRATEU, report §9.6 |
| 25 | llama.cpp CPU-vs-GPU floor on the same set | **0.00187** (top-1 0.9884) | `kl.py` | n/a | same | report §9.6 |
| 26 | 100K needle | **recall 5/5, 0 not exact** | `needle_audit.py --run`, five fixtures of ~99.3K tokens rebuilt by `tools/kolibri/make_needle_fixtures.py` | shipped path | "The 100K needle audit" | STRATEU, report §9.6 |
| 27 | ctest on the HIP build | **60 / 62** | `ctest` on the gfx1201 build | the two failures are box-scoped: `ple_parity` (missing Qwen3.8 Q2_0 file) and `platform_memory_test` (POSIX `mlock` vs `ulimit -l` 8 MiB) | "HIP bring-up" | STRATEU, report §9.6 |
| 28 | pinned expert table on AMD | **42.90 GiB, no root** | engine load report at `ulimit -l` 8 MiB | `hipHostMalloc` succeeds without `RLIMIT_MEMLOCK`; the T4-era pinning requirement does not carry over | "HIP bring-up" | report §9.6 |

## Campaign-report figures outside the numbered table

These are quoted in `docs/strateu-campaign-report.md` but are not headline rates, so they carry no
numbered row. Listed here so the staleness checker accounts for them and a reader can tell what each
one is: a rounded restatement, an instrument-time sweep, a forward estimate, or a withdrawn figure.
None of them may be quoted as a headline measurement.

| figure | value | what it actually is |
|---|---|---|
| decode short, rounded in prose | 48 tok/s | the **retracted** counter-derived figure (see Retracted figures); quoted only in the sentence that retracts it |
| serve path, rounded in prose | 176 tok/s | rounded restatement of #13 (176.2 / 170.8 tok/s) |
| cache neutrality sweep | 80.4 / 80.0 tok/s | §7 instrument-time A/B (6,366 / 6,397 ms against 6,357 / 6,337 ms), a neutrality check, not a floor |
| `RES_TOP=8` sweep | 122 tok/s | §7 sweep arm showing the promotion budget's cost; superseded by the shipped `RES_TOP=56` |
| §8.1 arithmetic best case | 270 tok/s | **estimate**, not a measurement - the paper ceiling §8.1 derives before discounting |
| §8.2 expected gain | 180 tok/s | **estimate** of the +10-20% the cheap partial was expected to give |
| pool vs GPU decode, this box | 16.1 tok/s | the GPU-expert-path arm §9.3 cites against the pool's 22.7 tok/s; the AMD re-derivation of this pair is #18-#20 and report §9.6 |

## Retracted figures (must not be quoted again)

| figure | value | why it was withdrawn |
|---|---|---|
| decode short | ~~48.0 / 48.3 tok/s~~ (quoted in prose as ~~48 tok/s~~) | counter-derived: the reciprocal of the pool's per-token drain (20.70 ms), not a wall-clock token.  See #3. |
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
| rounded restatements of #1-#4, #13 and #18-#25 in prose | 160 tok/s, 158 tok/s, 162 tok/s, 154 tok/s, 169 tok/s, 157 tok/s, 30 tok/s, 13 tok/s, 176 tok/s, 156 tok/s, 180 tok/s, 181 tok/s, 126 tok/s, 122 tok/s, 120 tok/s, 119 tok/s, 13.5 tok/s, 202 tok/s, 16 tok/s, 25 tok/s, 29 tok/s, 145 tok/s | same measurements as the rows above, rounded in prose |
| plain path at 100K | 8.9 tok/s, 4 tok/s, 9 tok/s | the per-token path's 100K mean, for contrast (11,534 s) |
| floor targets | 20 tok/s, 30 tok/s, 45 tok/s, 60 tok/s, 150 tok/s, 200 tok/s, 300 tok/s, 400 tok/s | **targets, not measurements** |

## Dead-lever and projection figures (quoted in the report's §7/§8)

Rows here are either measurements of a **rejected** configuration (kept so nobody re-tries it) or **projections**, which are
labelled as such and are not measurements.

| figure | value(s) | kind | ledger row |
|---|---|---|---|
| `DENSE_F16` with the expert cache released | 80.4 tok/s, 80.0 tok/s (control 80.5 tok/s, 80.8 tok/s) | measured, rejected (neutral) | §7.1 `DENSE_F16` null |
| `RES_TOP` sweep at 512 tokens | 122 tok/s (8 slots, -20%), 152 tok/s (24, shipped), 151 tok/s (56) | measured, swept | §7.3 `RES_TOP` |
| slots path vs CPU pool, decode | 16.1 tok/s vs 22.7 tok/s | measured, rejected | §9.3 re-derive list |
| Option A arithmetic best case | 270 tok/s | **projection** | §8.1 Option A |
| Option A realistic outcome | 185 tok/s to 200 tok/s | **projection** | §8.1 Option A |
| Option C expected gain | 165 tok/s to 180 tok/s | **projection** | §8.2 Option C |
| pp-512 gap left after campaign II | 19 tok/s | **derived**, not measured: the difference between #19's 181.0 and the 200 target | "G5 (campaign II)" |
| fused-kernel traffic estimate | 1.2-1.3 ms/layer against a 2.6 ms/layer gate | **projection** from byte math (82 MB read per projection at the rate the pool already sustains), never measured | "G3 (campaign II)" |

## Agent-turn figures (branch `main`, Tesla T4, driver 580.196.07, kernel 6.1.0-52-amd64, 2026-10-26)

`tools/kolibri/agent_bench.py --turns 6 --chunk 384`, numactl node 1, `--prefill --cpu-threads 24`,
ATTN_FLASH=1, two runs per setting, load 4.6 rising to 8.1 across the four runs. 8,184 prompt tokens total.

| # | figure | value | measured on |
|---|---|---|---|
| 26 | six agent turns, reuse off | 153.9 / 153.7 s wall | 8,184 tokens read, no reuse |
| 27 | six agent turns, reuse on | 59.4 / 45.8 s wall | 2,808 read, 5,376 reused |
| 28 | tokens read per six-turn session | 8,184 -> 2,808 | 5,376 reused, every turn after the first |
| 29 | prefill rate over the tokens read | ~55 tok/s | both settings (`agent_bench.py`) |
| 30 | per-turn reuse, six turns | 0, 256, 768, 1024, 1536, 1792 | identical across runs |
