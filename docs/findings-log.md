# Findings log — what was measured, and what was ruled out

This is the publishable half of the campaign ledger. Each entry below is one measurement campaign, replayed onto this
history from the private working ledger (which is not published, because it contains internal hostnames and machine
paths). The entries are kept in order because the *order* is the argument: several conclusions were corrected by later
measurements, and a summary would hide that.

Read `docs/strateu-campaign-report.md` first for the consolidated state; this log is the audit trail behind it.

## instruments cannot attribute pp-short: nsys inflates 15x on 15k launches and the phase marks double the pool's chunks; only wall-clock A/B survives

*Replayed from worktree commit `70a24da3`.*


## Instrumentation cannot attribute pp-short's remaining 1.33x - both instruments perturb it

Two attempts to split the 512-token wall, both self-defeating:

* **nsys** (`--trace=cuda`): 10,258 `dequant_wide_kernel` launches, 97% of them `grid=640` (one 2560x512 / 512x2560 matrix
  each), which *looked* like 1,087 ms = 60% of kernel time. But the traced run spanned **50.5 s against a 3.4 s wall** (15x
  inflation, because per-kernel overhead scales with the 15,590 launches), so those per-kernel times are launch-overhead
  artefacts, not compute.
* **the engine's own phase marks** (`STRATA_KOLIBRI_ATTN_PHASES=1`, which `cudaStreamSynchronize`s at every mark): per layer
  `pool=24.1 experts=15.1 post=4.7 dense=1.5 attn=1.0 ms`, i.e. the expert legs are 84% of the layer - but the same run reports
  `chunks=100 bytes=41 GB promotes=1986` against the un-instrumented `chunks=50 bytes=19 GB promotes=1004`, so enabling the marks
  **doubled the pool's work** by breaking the schedule it was measuring.

**What survives, because it is wall-clock A/B and nothing else:** the pool leg is **1.2 s of the 3.4 s wall** (measured
un-instrumented, 19 GB at 15.6 GB/s); `POOL_ALL=1` is 3.9x at this length; `RES_TOP=24` is optimal (8 is -20%, 56 is flat);
`DENSE_F16` is OOM-with-cache and exactly neutral without it; the router path is a null; and `ATTN_MB=1024` does not fit at 100K.

**Conclusion for the floor.** `pp short` is 150 tok/s against 200, and the remaining 1.33x is ~2.2 s of the wall that no
non-perturbing instrument currently attributes. The candidate levers are all kernel work of unknown size (a fast Q4_K-from-VRAM
GEMM, cheaper cache fills, fewer/bigger dequant launches), i.e. the kind of change that must be *built* and A/B-measured rather
than located by a profiler at this workload size.  That is the honest state, and it is the point at which continuing costs more
than the floor is worth without a direction from the goal owner.

## per-layer promotion budgets: measured null at 512 tokens, reverted; five levers now dead at short context

*Replayed from worktree commit `86da7f9e`.*


## The fifth dead lever: per-layer promotion budgets do not pay (and the fill is not where the time is)

C0's 24%-98% per-layer coverage spread suggested a data-driven promotion rule: in `seed_hot`, stop promoting once an expert
would add less than a share of that layer's rows, so low-skew layers fill fewer slots and high-skew ones keep theirs.  Swept at
512 tokens, two runs each, against the stock uniform budget:

| promotion rule | wall | tok/s |
|---|---|---|
| stock (uniform `RES_TOP=24`) | 3,469 / 3,488 ms | 147.6 / 146.8 |
| marginal share >= 0.5% | 3,404 / 3,431 ms | 150.4 / 149.2 |
| marginal share >= 2% | 3,476 / 3,480 ms | 147.3 / 147.1 |

All inside the run-to-run spread.  Reverted rather than kept: a cache-policy knob that buys nothing is complexity with a
maintenance cost and a numerics surface (it changes which experts are resident, hence which arm serves a pick).  With this,
**five levers are measured dead** at short context - `DENSE_F16`, the router path, `RES_TOP` (both directions), the
marginal-share promotion rule, and re-scoping the overlap - and the only surviving wall-clock facts are the pool's 1.2 s of a
3.4 s wall and the 3.9x that defaulting `POOL_ALL` recovered.

`pp short` remains **150 tok/s against a floor of 200**.  Everything cheap has been tried and recorded; what remains is kernel
work whose size cannot even be located with the profilers available on this box, because both of them perturb the workload
they measure (nsys by 15x on a 15k-launch workload, the phase marks by doubling the pool's chunks).

## serve path at the pp-short shape: cold 150.6 / warm 176.2 tok/s, with cache_n 0 (one-time state, not KV reuse)

*Replayed from worktree commit `9f892bc8`.*


## The serve path at the floor's shape: cold 150.6, warm 176.2 tok/s

Three identical 415-token chat requests through `run-kolibri1.sh` (ctx 8192, shipping config), reading the server's own
`timings`:

| request | prompt_ms | prompt tok/s |
|---|---|---|
| 1 (cold, first after load) | 2,756.3 | **150.6** |
| 2 (warm) | 2,355.5 | **176.2** |
| 3 | 2,430.1 | 170.8 |

with `cache_n = 0` on all three, so this is **not** KV reuse - it is one-time process state (the expert-cache fill, cuBLAS
handles, the pinned-table warm-up) amortising.  That matters for how the pp-short floor is read: the benchmark is a cold
process, while the serve path - which is what the box is actually for - pays that once and then runs at ~176 tok/s.  Still
short of 200, but the gap in the *use case* is 12%, not 33%, and it is the same fill that C1's blob residency would shrink 3.5x.

Also worth recording: the `prompt_ids` field is not accepted as a prompt by `/v1/chat/completions` (a request carrying 500 ids was
answered as a 45-token chat prompt), so prompt-shape measurements through the API need real text.

## ledger: campaign closed under amendment 5 - final floor table, deferrals with estimates, and the hand-off state

*Replayed from worktree commit `a8b93011`.*


## CAMPAIGN CLOSED (amendment 5) — final state, and what the next campaign inherits

The goal owner re-scoped the pp-short floor to the measured cold-run figure and closed the campaign on this box.  Final
ledger entry, so the next campaign starts from facts rather than from a summary of facts.

**Floors, final (two runs each, load stated, deterministic binary):**

| floor | target | measured | standing |
|---|---|---|---|
| pp short (512) | 200 | 149.7 / 151.9 cold; 176.2 / 170.8 warm (serve) | **met at the cold figure**; 200 and the 400 stretch deferred |
| pp after 32K | 150 | 153.9 / 154.9 | met |
| tg short | 30 | 48.0 / 48.3 | met, stretch 45-60 reached |
| tg after 32K | 20 | 48.3 (28.9 at load 12.1) | met, 2.4x |

Accuracy: KL **0.00132** on 109 answer steps (0.74x of llama.cpp's own CPU-vs-GPU floor, top-1 0.982); oracle **BITWISE**,
re-anchored after each numerics change; prefill **bitwise deterministic** across runs with `racecheck` 0 hazards after the
`block_max` barrier fix; needle **recall 4/5, exact 3/5** with both deviations recorded (amendment 4).

**What is deferred, with its estimate and its gate order, in `docs/strateu-campaign-report.md` of the published export:**

* **Option A** (the only route to 200): kernel parity test first, then a Q4_K-from-VRAM GEMM, then `C1` blob residency.
  ~1 s of short-prompt fill and ~7 GB of VRAM; 2.5-4 focused days; ~60% odds of landing 200.
* **Option C** (cheap partial): batched fill conversions + the MoE sum on the CPU; ~1 day; +10-20%; not the floor.
* Five levers that must not be re-tried, with numbers: `DENSE_F16`, the router path, `RES_TOP` either way, per-layer
  promotion budgets, overlap re-scoping (the `max(pool, GPU)` ceiling is unreachable - residual dependency).

**State of the tree and the export at closure:**

* worktree: `kolibri1-speed` clean, gates PASS (build / ids / PF_HASH 100 chunk-layers / oracle BITWISE).
* export: `the published export`, branch `main`, tip `64ae7743`, 2,341 commits, 53.5 MB tracked, 35 MB `.git`,
  report linked from `STRATEU.md`, zero private-path / hostname / >50 MB leaks. The earlier copy is at
  `strateu-publish-previous` because the refresh regenerated the rewritten history (hashes differ; `--force-with-lease`
  if an earlier push happened).
* next box: AMD Radeon AI PRO R9700 (32 GB) - re-derive, do not inherit: the pool-beats-GPU verdict, decode residency at
  0 of 0 picks, `DENSE_F16` being blocked, and the `ATTN_MB`/context limits. Re-anchor the floors against llama.cpp there.
* open defect carried forward (off the shipped path): `RES_TOP=0` exits 1 with a misattributed
  `dequant launch: invalid configuration argument`.

## CORRECTION (post-audit): the decode figures were the pool's drain time, not wall-clock - two floors FAIL

*Replayed from worktree commit `4ac4d787`.*

## CORRECTION (post-audit): the decode figures were the pool's drain time, not wall-clock — two floors FAIL

An independent audit rejected the completion claim, and it was right.  **I reported decode from the pool's own
`ms/tok` counter instead of from wall clock.**  `714.2 MB/token / 34.5 GB/s = 20.70 ms`, and "20.72 ms/tok -> 48.3 tok/s"
is its reciprocal; the number never measured a token, it measured the pool's slice of one.

**Wall-clock re-measurement** (`tools/kolibri/bench.py`, the campaign's own harness, shipping config, load stated):

| case | run 1 | run 2 | floor | verdict |
|---|---|---|---|---|
| pp 512 | 157.0 tok/s (3,261 ms) | 160.3 (3,194 ms) | 200 | unmet, as recorded |
| pp 32768 | 154.0 (212,765 ms) | 153.7 (213,263 ms) | 150 | **met** |
| tg 128 short | **24.5** (5,228 ms) | **24.4** (5,255 ms) | 30 | **UNMET** |
| tg 128 after 32K | **12.3** (10,420 ms) | **12.2** (10,488 ms) | 20 | **UNMET** |

Loads: 5.43/5.57 for the short pair; 7.49/14.10 for the after pair.  The audit's own runs (29.5-30.2 short, 13.4/13.5 after,
at load 4.8-11.7 with `numactl` single-socket binding) bracket these: the spread between bindings is real, and neither
binding reaches 30 at short context or 20 at 32K.

**What this invalidates, beyond the floor table:**

1. **The floor table in amendment 5 and everything published from it** - `docs/strateu-campaign-report.md`, `STRATEU.md`,
   the `README.md` block.  Corrected below and in the export.
2. **B0's conclusion that "the pool is 99% of the decode wall".**  The pool drain is ~20.7 ms of a ~41 ms token, i.e. ~50%,
   and the other ~50% is the GPU legs plus the handoffs - serialised by the same residual dependency the overlap section
   identified.  So the decode wall is the *sum*, which is why tg is 24.4 and not the 48 the drain alone would allow.
3. **C0/C1's decode pricing** ("absorbed rows -> decode seconds").  With the pool at half the token, a cache hit cannot buy
   the seconds the curve implied; the VRAM-headroom argument stands, the throughput argument does not.
4. **The "stretch 45-60 reached" claim** in the floor table: unsupported, at either binding.
5. **The claim that these were "two-run, load stated" numbers.**  They were two-run *counter* readings, which is not the same
   thing; the wall-clock pair above is what that phrasing was supposed to mean.

**What survives unchanged:** pp-32K 154.0/153.7 (met, and the audit's lower figure was a binding/load difference, not a
method one); pp-512 157-160 (still short of 200, as recorded); KL 0.00132; oracle BITWISE; the barrier fix and its
racecheck evidence; the needle audit and its deviations; every "measured dead" list, because those were all wall-clock A/B
sweeps between configurations and their *differences* were real even where the absolute decode figure was not.

**Process rule adopted:** a decode or prefill rate is only ever quoted from a wall-clock measurement of a token run
(`bench.py` or the server's own `timings`), never from a phase counter, and the harness invocation is recorded with the
number.  Counter-derived ms/tok is a *component* measurement and must be labelled as such.

## G1 (campaign II) — prefill partial sums: +16.0% at pp-512, and a harness trap that voided three gates

**What landed.** `STRATA_KOLIBRI_PF_PARTIAL=1` (default off) keeps the resident expert rows on the device instead of
shipping them to the host and back.  The old leg wrote `res_ey` -> host (27 MB at ~85% coverage), scattered on the
host, then pushed the whole `NKc x N` block (31.5 MB at Tc=512) back up.  The new leg pushes only the 64 KB selection
map down, computes the *miss* contributions on the host into a `Tc x N` partial, and combines on the device in one
kernel (`src/kolibri/pf_combine.cu`).  Summation order is fixed by indexing the shared slot by pick number, so the
combine adds in ascending `k` and matches the reference term for term.

**Wall clock, two runs per arm, `bench.py --case short`, `--extra="--prefill --cpu-threads 24"`, no `--counters`,
`numactl --cpunodebind=1 --membind=1`, binary `build/strata-kolibri`, kernel 6.1.0-52, driver 580.196.07:**

| arm | pp-512 | box load |
|---|---|---|
| knob off | 159.0 (3221 ms), 158.8 (3224 ms) | 0.27, 2.85 |
| knob on | **184.9 (2770 ms), 183.6 (2789 ms)** | 2.12, 4.35 |

**+16.0%**, against a floor of 165 and a 2%-kill criterion.  The byte counters (added under the same `e.counters`
guard as the decode legs, because the prefill staging was invisible to them) account for it: d2h 1,600,061,440 ->
107,520,000 B and h2d 1,963,008,000 -> 584,763,904 B, i.e. **2.87 GB removed**, against 333 ms of instrumented wall
delta = **8.6 GB/s implied**, inside the envelope the pool itself measures on this link (12.2 GB/s pinned) since
these transfers partially overlap compute.  Instrumented runs read lower on both arms (155.0 -> 172.4, +11.2%), which
is the counters' own cost and is why the rate above is taken without them.

**Accuracy.** `kl.py` with `STRATA_KOLIBRI_ATTN_FLASH=1` (the deterministic prefill config — see the trap below),
answers-only mean KL over the same 109 answer steps: llama.cpp CPU floor 0.00178, knob off **0.00123**, knob on
**0.00133**; top-1 0.9725 on both the floor and the knob-on arm, 0.9633 knob off.  `oracle.sh` is BITWISE PASS with
the knob on and off.  `pf_gate.sh` passes under the knob against **re-anchored** PF_HASH (100 chunk-layers, identical
across two runs); against the Campaign-I expectations all 100 lines differ, which is the expected consequence of
re-associating the sum, and the oracle shows argmax is unmoved by it.

**Harness trap, and it cost three gates.** `/tmp/kolibri_bench_node1.sh` execs
`/data2/develop/strata-speed-wt/build/strata-kolibri` — the Campaign I worktree binary, which contains **zero**
occurrences of `PF_PARTIAL`.  Any gate run through that wrapper silently tests a binary that ignores the knob.  It
produced a confident "PF_HASH bitwise identical, gate PASS under the knob" that was vacuous, and a CNT A/B with
byte-identical counters and no speed change.  `strings` on both binaries settles it in one line and is now the
pre-flight check: `strings build/strata-kolibri | grep -c PF_PARTIAL` -> 2, old binary -> 0.  **Rule adopted: before
any A/B, assert the knob string exists in the binary the wrapper execs.**

**Two things this box no longer reproduces, neither caused by the change.**
1. **pp-32K baseline drift.** Knob-off today is 119.6 / 119.4 tok/s against the published 154.0 / 153.7, at equal or
   lower load, so load does not explain 22%.  pp-512 is unaffected (159.0 today vs 157.0/160.3 published).  The prime
   suspect is the driver module: Campaign I ran with 580.178.04 loaded; after the mismatch fix the loaded module is
   580.196.07.  Consequence: "pp-32K >= 150 retained" cannot be judged against a baseline this box no longer produces;
   on a same-day A/B the knob is never negative there (124.7 and 159.0 vs 119.5).
2. **The non-flash prefill is nondeterministic.** `pf_gate.sh` says outright that PF_HASH needs `ATTN_FLASH=1` for a
   deterministic prefill.  The first KL A/B was run in the `speed_config` env without flash and produced 0.00205 off
   vs 0.00170 on — a comparison of two draws from a nondeterministic generator, not an A/B.  With flash the same arms
   read 0.00123 and 0.00133.  **Rule adopted: any KL A/B runs with `ATTN_FLASH=1`.**

## G3 (campaign II) — the grouped Q4_K-from-VRAM GEMM: measured, and it is a negative result

**What was built.** `q4k_grouped_gemm_vram` (src/kolibri/q4k_grouped.cu): reads each expert's Q4_K blob from VRAM,
dequantizes it to an FP16 scratch in VRAM, and multiplies with the engine's own `Gemm::f16_batched`.  The parity gate
written before it (G2) is **bitwise green**: 81920/81920 values equal, max abs diff 0.0, mean row KL 0.0.

**The stage harness.** `q4k_grouped_vram_parity --bench`, the staged expert set (batch=118, T=35, the gate/up/down
projections), CUDA events, best of 5 after warmup, on the Tesla T4, kernel 6.1.0-52, driver 580.196.07.

| arm | ms/layer | what it is |
|---|---|---|
| Q4_K-from-VRAM arm (this stage) | **9.99** (best-of-5 across runs 9.99-11.68) | read Q4_K, dequant to FP16, GEMM |
| **FP16-resident leg** | **4.18** | what the engine does today: weights promoted to FP16 once, then multiplied |
| G3 gate | 2.60 (stretch 1.95) | |

**The result is negative and it must be read that way.** The new arm is **2.39x slower than the path it would
replace**, not faster.  The earlier "13.9x better" reading came from comparing against the 154 ms/layer figure in
docs/strateu-campaign-report.md:42,:289, which describes a *different, previous-default* configuration - apples to
oranges.  Measured against the mechanism actually in use, the arm loses.

**Why, priced.** The FP16-resident leg pays its weight traffic **once, at promotion**: the expert sits in VRAM as FP16
and every later product reads 164 MB per projection with no unpacking.  The Q4_K-from-VRAM arm re-reads 82 MB of
compressed weights *and writes 164 MB of FP16* on **every call**, so it pays 246 MB where the resident leg pays 164.
That is the whole 2.39x.

**What the 2.6 ms gate therefore asks for.** It is **1.6x faster than cuBLAS on FP16 weights**, not "faster than the
CPU pool".  Nothing that materialises FP16 weights can reach it: the gate needs **half the weight traffic of the best
existing path**, which only a fused kernel gets, by dequantizing Q4_K sub-blocks in registers into `mma.m16n8k16` and
reading 82 MB with no FP16 ever written.  At the ~105 GB/s this card demonstrably sustains that is ~1.2-1.3 ms/layer,
inside the gate - so the gate is reachable, but only by authoring that kernel, which is a separate commissioned piece
of work, not a tuning pass on this one.

**Two dead ends recorded so nobody re-walks them.**
1. *The MMQ path is not available on CUDA.* `strata::prefill::moe_mmq` describes exactly this job ("reads the ~1.4-2 MB
   expert once" instead of writing ~10 MB of FP16), and covers Q4_K under `STRATA_MMQ_KQUANTS`.  But
   `STRATA_PREFILL_MMQ` **requires HIP** (CMakeLists.txt:89 hard-errors `STRATA_PREFILL_MMQ requires
   STRATA_ENABLE_HIP=ON`), and the CUDA K-quant DSO `strata_kq` is not built here.  A CUDA configure refuses it; this
   is a backend limit, not a flag someone forgot.
2. *`STRATA_MMQ_KQUANTS` is a CMake option, not a runtime knob* (CMakeLists.txt:79).  Setting it in the environment
   does nothing; `mmq::supported(12)` stays false.

**Not shipped.** The arm is behind no engine knob and is not wired into `forward_multi`, because wiring a 2.4x
regression into the prefill would make the engine slower.  It stays in the tree as the parity gate's subject and as
the A/B baseline the fused kernel will be measured against.
