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
| 17 | expert coverage, 32K corpus | **top-24 42.9%, top-80 70.3%, top-128 83.0%** | `STRATA_KOLIBRI_EXPERT_HIST=1` + `--counters` | instrument-time | "C0 coverage curve" | report §C0 |

## AMD Radeon AI PRO R9700 (gfx1201, 32 GB, ROCm 7.2.2) — re-derived, not inherited

A different box, so a different floor table. Every row below is two-run wall clock from
`docs/AMD_R9700_PORT.md` on host `rizo`: Ryzen 7 5700G (8 physical cores, 1 NUMA node), 93 GiB RAM,
90-91 GiB free at the start of each run, GPU idle at the start, `--cpu-threads 8`, shipped config
(`EXPERT_F16=24`, `RES_TOP=56`, `PF_STAGE=1`, router tiles) unless the row names a change. Prefill
rows subtract the engine's own load line (13.4 s warm); the end-to-end row does not, and says so.

| # | figure | value | method | load / scope | ledger row | quoted in |
|---|---|---|---|---|---|---|
| 18 | prefill, 512-token prompt | **43.4 / 51.7 tok/s** | wall clock, `n=1` run minus the engine's 13.4 s load line | load 90-91 GiB free | "Wall-clock rates on the AMD path" | STRATEU, report §9.6 |
| 19 | prefill 512, end to end (load included) | **20.3 / 22.0 tok/s** | same runs, nothing subtracted | same | same | report §9.6 |
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
| rounded restatements of #1-#4 and #13 in prose | 160 tok/s, 158 tok/s, 162 tok/s, 154 tok/s, 169 tok/s, 157 tok/s, 30 tok/s, 13 tok/s, 176 tok/s | same measurements as the rows above, rounded in prose |
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
