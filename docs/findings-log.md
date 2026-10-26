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

## G4b (campaign II) — the after-32K decode wall attributed: it is PCIe streaming at 0% residency

Wall clock first, because a rate is only ever a wall clock: `bench.py --case after`,
`--extra="--prefill --cpu-threads 24"`, `numactl --cpunodebind=1 --membind=1`, load 8.5, two runs today:
**tg-128 after 32K = 13.4 / 13.5 tok/s = 74.6-75.2 ms/token** (published campaign I figure was 12.3/12.2, so this
box is slightly better than the number the 81 ms token was quoted from).

Then the attribution, **instrument-time** (`--counters`; counters are components, not a rate):

| counter | value | what it says |
|---|---|---|
| `CNT branches per tok` | `pool 50.0 gpu-expert 0.0 slots-path 0.0 grouped 0.0` | **all 50 layers of every token take the pool branch**; nothing takes a resident path |
| `CNT decode residency` | `0 of 0 picks served from the fp16 cache (0.0%)` | **decode has no residency at all** |
| `CNT serve` d2h | `488.7 MB/tok` | device->host per token |
| `CNT serve` h2d | `789.5 MB/tok` | host->device per token |
| `CNT pool` | `streamed 714.2 MB/tok`, work `12.85 ms/tok`, `55.6 GB/s` | the CPU pool's own work |
| `CNT pool phases` | `drain 1646 ms = 12.86 ms/tok`, wait_park 0, repark 0 | the drain the campaign attributed (20.7 ms there, 12.86 here) |

**Where the ~60 unattributed ms/token went.** 1.28 GB crosses PCIe per token (488.7 d2h + 789.5 h2d). A token
touches 8 experts across 50 layers = 400 expert rows, and a Q4_K expert blob is ~2.1 MB, so ~840 MB of weights
must be present per token - the same order as the measured 1.28 GB once the activations and the resident-row
round-trip are added. At the 11.5-12.2 GB/s this link sustains, that volume is ~105-111 ms of transfer time per
token, partially overlapped with compute, which is why the wall lands at 75 ms rather than the sum.

**So the wall is bandwidth, not kernel speed.** The pool's drain is only 12.9 ms of the 75. The other ~62 ms is
moving expert weights from RAM to the GPU **again on every token**, because decode serves 0% from cache. That
reframes G4b's floor: reaching 20 tok/s (50 ms/token) does not need a faster expert kernel, it needs the 1.28 GB/token
to stop existing - i.e. residency. A token whose experts were already resident would move activations only, which
is three orders of magnitude smaller.

**Consequence for G4, and it cuts the other way from what the G3 result suggested.** G4's C1 (a slot is a raw Q4_K
blob) is usually argued as a VRAM/fill win; this says its real prize is decode, where today's FP16 cache serves
nothing. But it also means G4's decode gate (tg-128 short >= 30) is a residency question first and a kernel
question second: with 0% residency today, the fastest expert kernel in the world still pays 1.28 GB/token.

**Not claimed.** tg-128 after 32K >= 20 tok/s is **not** claimed, per the stage's own rule that the floor is not
claimed before this attribution exists. The attribution now exists; the work it points at (decode residency) is
G4's, and G4 is itself waiting on the fused-kernel ruling because the same blobs have to be multiplied cheaply once
they are resident.

## G4 (campaign II) — blob residency already exists in the tree, and on this box it is a regression

C1 ("a slot is a raw Q4_K blob, a promotion a plain H2D blob copy") is not new work: `--expert-slots` already
allocates a **device** blob table (`b.dev`, kolibri_engine.cu:217) and the decode expert leg reads it with
`native_mmvq` - `gemv(gu, s.h, s.eg2, s.ids, K, 0, b.dev, b.stride)` - so no FP16 is materialised.  Loading it
says so plainly: `expert cache: 24 slots/layer = 2.87 GiB VRAM` (2.4 MB per slot, i.e. raw Q4_K, not 7.86 MB of
FP16).  A second path, `STRATA_KOLIBRI_DEC_RES=1` (K29), seeds each layer's hot experts into the FP16 cache.

Neither is enabled by the campaign's bench config, which is why G4b measured `branches per tok: pool 50.0 /
slots-path 0.0`.  Enabling them, two runs each, `bench.py --case short`, `--extra="--prefill --cpu-threads 24"`,
`numactl --cpunodebind=1 --membind=1`, load stated per row:

| arm | pp-512 | tg-128 short | verdict |
|---|---|---|---|
| baseline (no residency) | 156.0 / 159.6 (load 0.88 / 3.03) | **30.3 / 30.3** | reference |
| `--expert-slots 8` | 161.1 / 148.4 (load 4.73 / 5.37) | **25.0 / 25.0** | **-17% tg** |
| `--expert-slots 16` | 167.1 / 163.9 | **25.1 / 25.2** | **-17% tg** |
| `STRATA_KOLIBRI_DEC_RES=1` | 144.9 / 141.8 (load 3.79 / 4.48) | **29.5 / 29.2** | **-3% tg, -10% pp** |
| `--expert-slots 24` | - | - | **OOM**: `cudaMalloc(&p, n * 2) failed: out of memory` at kolibri_engine.cu:1531 |

**Why the slots path loses, named in the code.** It decides residency by **expert id**, not hotness:
`if (id < S) ++e.cnt.slot_hits` (line 2001) and `s.slot_ids[k] = id < S ? id : S`.  So `--expert-slots 8` makes
experts *numbered 0-7* resident, which are not the hot ones - the hit rate is near zero, and what remains is the
cost of splitting every layer between a GPU side and a CPU-pool overflow side plus the stream sync between them
(`cudaStreamSynchronize(g_st)` after the slot gemv run).  You pay the bookkeeping and still stream the 1.28 GB.
The prefill cache learned this lesson already (line 235: "the cache must hold the HOT experts, not ids 0..N-1"),
and the decode slots path never got the fix.

**Why DEC_RES loses too.** It does seed hot experts, but the seeding costs prefill time (pp-512 drops from ~159 to
~143, -10%) and the decode gain is negative (-3%).  And it seeds **FP16**, which is the 7.86 MB-per-slot form G4
was trying to move away from.

**Why the traffic gates cannot be met by a flag either.** `--expert-slots 24` *adds* 2.87 GiB of blob slots on top
of the FP16 resident cache that is still allocated, and OOMs at 4.74 GiB free.  Freeing the >= 6 GB requires
**replacing** FP16 promotion with blob promotion, and once the resident experts are blobs the prefill's resident
leg can no longer use `f16_batched` - it needs the blob multiply, which is the G3 arm measured at 2.39x slower
than the FP16 leg (9.99 vs 4.18 ms/layer, c77adbfd).

**G4's outcome, stated as a measured ceiling with its priced reason.** On this box, with the residency mechanisms
that exist, decode residency is not a win: the id-based path misses the hot experts and costs 17% of tg, the
hot-seed path costs 10% of pp for -3% tg, and the size that would free the VRAM does not fit alongside the FP16 it
would have to replace.  G4b's diagnosis stands - the after-32K wall is 1.28 GB/token of PCIe at 0% residency - but
closing it needs two things this campaign does not have: **hotness-based blob residency** and a **blob multiply
that is not slower than FP16**, i.e. the fused Q4_K kernel G3 could not deliver.

## G5 (campaign II) — the floor table, re-anchored on what this box actually does today

Every figure is a two-run wall clock from `tools/kolibri/bench.py --bin /tmp/kolibri_bench_pub.sh`, run under
`numactl --cpunodebind=1 --membind=1`, on kernel `6.1.0-52-amd64`, driver `580.196.07`, with
`STRATA_KOLIBRI_PF_PARTIAL=1` and `STRATA_KOLIBRI_EXPERT_F16=24`.  Box load is the 1-minute average read from
`/proc/loadavg` immediately before each run.

| campaign floor | run 1 | run 2 | load (run 1 / run 2) | status |
|---|---|---|---|---|
| pp-512 >= 200 tok/s | 179.8 | 181.0 | 1.32 / 2.50 | **not met - ceiling 181.0** |
| pp-32K >= 150 tok/s | 125.7 | 122.1 | 0.06 / 9.02 | **not met - ceiling 125.7** |
| tg-128 short >= 30 tok/s | 30.0 | 30.3 | 1.32 / 2.50 | **MET** |
| tg-128 after 32K >= 20 tok/s | 13.3 | 13.4 | 0.06 / 9.02 | **not met - ceiling 13.4** |

Accuracy on the same build: oracle **BITWISE PASS** (`pf_gate`: ids smoke PASS, PF_HASH PASS 100 chunk-layers,
oracle BITWISE), KL **0.00133** with the knob on against **0.00123** with it off, both under the llama.cpp-CPU noise
floor of **0.00178** (the literal <=0.00132 in the original contract is marginally exceeded; the recorded ruling in
bf6c3f58 supersedes it and the gap sits inside the noise floor).  Needle recall **3/5** - see below.

**Each unmet floor, with its price.**

*pp-512: 181.0 against a floor of 200.* G1 landed and is worth +16% (159.0/158.8 -> 184.9/183.6 when measured on a
quiet box; 179.8/181.0 here at load 1.3-2.5).  The remaining 19 tok/s is not in the plumbing: the expert GEMM is
4.18 ms/layer on the FP16-resident leg and the gate for the compressed leg was 2.6 ms/layer, which needs a fused
Q4_K tensor-core kernel that does not exist on CUDA (G3, skipped at c77adbfd - the Q4_K-from-VRAM arm measured
9.99 ms/layer, 2.39x slower than the FP16 leg it would have to replace).  Price: a hand-written `mma.m16n8k16`
kernel for sm_75, days of work with profiling between attempts.

*pp-32K: 125.7 against a floor of 150.* This floor was anchored on a box state that no longer exists.  Knob-off
today measures 119.6/119.4 where the same prompt measured 154 before the driver moved from 580.178 to 580.196;
pp-512 is unaffected (159 against 157/160), which is what identifies it as box drift and not as the change.  The
change never made pp-32K worse in any pair (on >= off every time, 124.7/159.0 against 119.6/119.4).  Price: none
payable in code - the floor has to be re-anchored to the box, which is what this table does.

*tg-128 after 32K: 13.4 against a floor of 20.* G4b attributed the wall completely: **0% decode residency, 1.28 GB
of PCIe per token** (488.7 MB d2h + 789.5 MB h2d), with the CPU pool drain accounting for only 12.86 ms of the
74.6-75.2 ms token.  It is bandwidth, not kernel speed.  Closing it needs two things this campaign does not have -
hotness-based blob residency (the existing `--expert-slots` decides residency by expert id and measured -17% of tg,
G4 at ca1fbd44) and a blob multiply that is not slower than FP16 (the missing fused kernel again).

**Needle: the published standing holds - recall 4/5, exact 3/5 - and the campaign did not move it.**  Scored with
`tools/kolibri/needle_audit.py` on the five fixtures against our build: `needle_32k90` (33,364 tok) exact PASS,
`needle_64k90` (65,876 tok) exact PASS, `needle_anchor` (101,416 tok) exact PASS, `needle_d90` (98,591 tok) FAIL,
`needle_d90b` (98,591 tok) FAIL.  The harness counts **exactness** (its own line reads `N answer(s) audited, M not
exact`, and it exits 0 only when every answer is exact), so those five results are **exact 3/5**, which is the
published figure.  Recall is 4/5, not 3/5: the `needle_d90b` answer is `quart-willow-8` against a needle of
`quartz-willow-884` - the code word was found and reproduced one letter short, which is recalled-but-not-exact -
while `needle_d90` answered a different instruction from the haystack and is the one true miss.  That is exactly the
pair of failures `STRATEU.md` already describes, so the needle row is **unchanged by campaign II**, which is what the
byte-identical default path predicts.

Two harness facts worth recording, because both cost time here.  The audit needs the `regex` module and the system
`python3` does not have it (`ModuleNotFoundError: No module named 'regex'`), while `/data2/develop/Strata/.venv/bin/python`
does - so the audit must be run with the venv interpreter.  And without `--run` the audit skips its own `audit_gen.txt`,
which is why a first scoring pass appeared to find nothing: the generated answers were on disk but not being read.

## The last prefill chunk rewrites K/V rows at 128-aligned positions (found during the reuse work)

*Measured 2026-10-25/26 on the Tesla T4 box (driver 580.196.07, kernel 6.1.0-52-amd64, numactl node 1,
`--prefill --cpu-threads 24`, ATTN_FLASH=1). Instrument: `STRATA_KOLIBRI_KV_DUMP` (added in `c978b540`),
which records one layer's K and V at the first decode step of a request.*

Working the prefix-reuse divergence backwards, layer 0's K/V was dumped for three requests in one process and
compared row by row (513-row sliding ring, so rows 9..512 map to the same positions in every record):

| comparison | rows that differ |
|---|---|
| two requests with the **identical** prompt `ids[:513]` | **none** (0 of 512) |
| `pos=512` (turn 1) vs `pos=521` cold | 128, 256, 384, 512 |
| `pos=512` (turn 1) vs `pos=521` reused | 128, 256, 384 (512 correctly identical) |

Two readings matter here.

1. **The engine is deterministic.** The same prompt twice in one process produces byte-identical K/V on every
   row. So the differences below are not run-to-run noise.
2. **Rows 128, 256 and 384 change when the final prefill chunk is processed**, even though that chunk writes
   positions 513..520 and those rows sit far behind it. This happens in the *cold* arm as well as the reused
   arm, so it is a property of the chunked prefill, not of the reuse path. Every affected row is a multiple of
   128; no other row moves.

Position 512 differing only in the cold arm is expected and is not a defect: the prefix request writes it
through the single-token decode path, a cold read writes it inside a prefill chunk.

Consequence for the reuse work: a resumed read cannot be made bitwise-equal to a cold read while the last
prefill chunk perturbs 128-aligned K/V rows, because the two arms cut their final chunk differently (9 rows
against 8). The reuse path's own state handover was separately wrong and is fixed in `aa9f0e0e` (a stale K/V
transpose mirror), which moved the divergence from 6.8e-01 to 2.1e-02 relative at a 512 prefix.

Not yet explained, and the next thing to look at: why a 128-row boundary is special in the K/V write, and
whether the same perturbation happens for every chunk or only for the final one.

### Which chunk does it: the K/V dump per prefill chunk

The same cold read of a 522-token prompt, with K/V snapshotted after every prefill chunk (magic `KVCH`):

  records: p0c=0/Tc=256, p0c=256/Tc=256, p0c=512/Tc=9   (all 513 ring rows each)

| comparison | rows that moved |
|---|---|
| after chunk 2 against after chunk 1 | 127, 255, 256, 383, 384, 512 |
| after chunk 3 against after chunk 1 | 127, 128, 129, 255, 256, 257, 383, 384, 512 |

Chunk 2 writes positions 256..511, i.e. slots 256..511. Every row it is *supposed* to write did change, as it
must. The problem is the rows it is not supposed to write: **127 and 255** (both written by chunk 1) changed
anyway, and so did 512, which no chunk has reached yet at that point. After chunk 3 the moved set grows to
127..129 and 255..257 — the last row of one 128-row tile and the first rows of the next.

So the chunk that perturbs is not the last one: chunk 2 already moves rows behind itself. 127, 255 and 383 are
exactly the last row of each 128-row tile, and 128/256/384 the first row of the next, which is what the
cold-against-reused comparison at the decode step was already showing.

Still not explained: `k_attn_prep_multi` writes `slot = pos % swa` for its own chunk's positions only, so
nothing in that store should touch row 127 when processing positions 256..511. The next thing to check is a
concurrent write that overlaps the tile edges - the KV write and the ring gather were previously caught in a
stream race in this file (see the STREAM RACE note at the grouped-expert leg), and the dump syncs `g_st`
only, so a write issued on another stream could land between the snapshot and the store.

### A one-row second chunk changes 192 earlier K/V rows

The sharpest form of it. Prompt of 258 ids, so the prefill runs chunks `p0c=0/Tc=256` then `p0c=256/Tc=1`,
with K/V snapshotted after each (layer 0, a sliding layer, stride 513):

    the second chunk writes position 256 and nothing else
    rows 0..255 that changed afterwards: 64, 65, 66, ... , 255        count = 192

Rows 64..255 move; rows 0..63 do not. Deterministic, and reproduced on every run (two identical cold reads
give 0 differing rows in every snapshot, so this is not a race and not the instrument).

So a chunk that writes a single position perturbs 192 rows behind it. That is not a store of the wrong slot -
`k_attn_prep_multi` writes `slot = pos % swa` for its own chunk's positions only - so something else is
writing K/V over a range derived from the chunk, and the range starts 64 rows into the buffer.

Consequence, and why this is the blocker for the reuse work: a resumed read cuts its final chunk differently
from a cold read (8 rows against 9), so this perturbation lands differently and the two arms cannot agree
bitwise, at any amount of fixing in the reuse path. It reproduces in a single cold request with no reuse
involved. What the reuse path itself got wrong (a stale K/V transpose mirror, fixed in aa9f0e0e) is separate
and is done.

### RETRACTION: the "128-aligned K/V rows differ" and "one-row chunk changes 192 rows" findings were wrong

Both came from my comparator, not from the engine. A K/V row is `D` = 128 halves; the comparator indexed
rows as `HKV * D` = 512 halves, so "row r" was really **position 4r**, and past position 128 it ran off the
end of head 0 into heads 1..3. Every claim built on that indexing is withdrawn: the 128-alignment pattern,
the 192 rows moving, and the "second writer over a range derived from the chunk".

Redone with correct per-head indexing (`offset = (h * rows + p) * D`), the cold-against-reused comparison on a
522-id prompt at a 513-id prefix gives:

    positions whose K/V differ (any head): [512]      count = 1 of 513

One position. Every other K/V position is byte-identical between a cold read and a resumed read. Position 512
is the boundary row, and its provenance legitimately differs: the prefix request writes it through the
single-token DECODE path, while a cold read of the longer prompt writes it inside a prefill chunk.

So the remaining gap is a single row with two different computation paths, not a broad perturbation. The
alignment experiment (quantising the reusable prefix down to a chunk boundary) is not the fix: at a 513 prefix
it changed the request's chunk geometry (prefill starting at 513 with 9 rows, decode at 522) and the logits
divergence got worse, 1.4e-01 against 4.1e-02. Withdrawn with the rest.

## Agent turns reuse the prefix: 153.8 s -> 52.6 s over six turns

*Measured 2026-10-26 on the T4 box (driver 580.196.07, kernel 6.1.0-52-amd64, numactl node 1, `--prefill
--cpu-threads 24`, ATTN_FLASH=1). `tools/kolibri/agent_bench.py`, six turns of 384 new fixture ids plus the 8
tokens each turn writes, 8,184 prompt tokens in total. Two runs per setting, wall-clock, load 4.6 rising to 8.1
across the four runs.*

| setting | wall | tokens read | reused |
|---|---|---|---|
| reuse off (`STRATA_KOLIBRI_REUSE=0`) | 153.9 s / 153.7 s | 8,184 | 0 |
| reuse on (default) | 59.4 s / 45.8 s | 2,808 | 5,376 |

Every turn after the first reuses (256, 768, 1024, 1536, 1792 tokens), and the reused count is identical across
runs on both settings. Prefill runs at ~55 tok/s over the tokens actually read, which is the same rate the cold
figures show - the saving is entirely in not reading them again.

Not the whole prefix: the bitwise fix rewinds up to one chunk, so a turn re-reads up to 255 tokens it already
had. At 384 new ids a turn that is a large fraction of the read, which is why 8,184 - 5,376 = 2,808 rather than
the ~2,304 the new content alone would cost. The rewind buys bitwise equality with a cold read and it is the
honest trade to state.

Two defects had to be fixed before this number existed, and neither was visible in a two-request test: the
reuse path never set `s.pos`, so the prefill wrote its chunk one position too far (`ee12e9b7`), and a reusing
request appended its prefix to `live` twice, so the next turn's prefix check failed and reuse worked every
other turn (`a2996634`).

### What the reuse work did NOT verify on this box

Two gaps, stated so nobody reads the gate as covering them.

**The gate's PF_HASH stage is default-path only.** Running `pf_gate.sh check` with `STRATA_KOLIBRI_PF_PARTIAL=1`
reports `2. PF_HASH FAIL (differs from 100 recorded lines)` while `3. oracle PASS (BITWISE PASS)` and
`4. reuse PASS`. That is the gate's reference being recorded on the default path, not a defect: the partial-sums
knob deliberately changes how the expert leg accumulates, so its per-chunk checksums differ in the last bits while
the generated ids stay bitwise - which is what Campaign II measured and why the knob is opt-in. `pf_gate` is
therefore a default-path contract, and its reuse stage is meaningful on either path.

**The full server path was not exercised end to end here.** `run-kolibri1.sh` is the AMD box's launcher: given
`MODEL`, `CTX` and `PORT` overrides it still refuses this machine, because its card detection finds none
(`no card matching 'Tesla' was found ... Cards present:`), so it never starts the server. What was verified
instead, at the boundary that matters: the engine emits `RESUME <n>` before the first `PP` and fills the DONE
reuse/read fields (measured directly over the protocol: a 522-id request extending a 512-id prefix prints
`RESUME 512` then `PP 521 522` with the DONE carrying reused 512 / read 10), and `serve/server.py`'s parsing of
exactly those two fields is covered by `serve/test_request_accounting.py`, which asserts
`(timings["cache_n"], timings["prompt_n"]) == (5, original - 5)`. The launcher path on a box whose card it
recognises still owes a two-turn `/v1/chat/completions` run asserting `cache_n > 0`.

### Through /v1/chat/completions the reuse did not fire (cache_n 0) - and why that is not the engine

Started the server on this box against the real model and drove two turns (a turn-1 chat, then the same
conversation plus the assistant reply plus a new user line), reading the server's own `timings`:

    turn 1: prompt 59 tokens, cached_tokens 0
    turn 2: prompt 72 tokens, cache_n 0, prompt_n 72, prompt_ms 953.9

So through HTTP the second turn read all 72 tokens. The engine never printed `RESUME` for those requests, i.e.
it correctly declined to reuse because the prompt it was handed did not extend what it held - the same code that
does reuse, bitwise, when the ids are resent (the gate and `agent_bench.py` both prove that).

The likely cause, not yet proven: the reply goes back to the client as TEXT and is re-tokenized on the way in,
and detokenize->retokenize is not the identity at a turn boundary - a BPE merge can span the boundary, and the
chat template appends its own marker after the assistant content. The Qwen path in this repo solves that with a
token-level conversation cache on the server side rather than relying on the client to reproduce tokens; the
Kolibri path has no such cache.

What would settle it, and what it needs: a token-id round trip that can be compared (a `/tokenize` endpoint, or
a server-side cache of the ids it last sent). Neither is engine work, and neither is in this goal's scope - it is
the server layer that owes it, together with the two-turn `cache_n > 0` assertion.

### pf_gate on the partial-sums path: PF_HASH is not emitted there at all

`STRATA_KOLIBRI_PF_HASH` lines are produced only under
`if (b.f16 && s.pf_nblk > 0 && !pf_partial_on())` (kolibri_engine.cu:2793), so with `STRATA_KOLIBRI_PF_PARTIAL=1`
there is nothing to compare and the stage reported FAIL for the absence of an instrument. It now prints
`SKIP (STRATA_KOLIBRI_PF_PARTIAL is on: the engine emits no PFHASH lines on that path)`.
With that, the gate passes on both paths with the oracle unchanged:

    default path:  build PASS, ids smoke PASS, PF_HASH PASS (100 chunk-layers), oracle BITWISE PASS, reuse PASS
    PF_PARTIAL:    build PASS, ids smoke PASS, PF_HASH SKIP, oracle BITWISE PASS, reuse PASS
