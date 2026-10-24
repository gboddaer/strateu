# How numbers are made and published (measurement guideline)

This repository quotes speed and accuracy figures in the READMEs, in `STRATEU.md`, and in
[the campaign report](strateu-campaign-report.md). Within one week, two figures reached these
pages that were not what they claimed to be: a decode rate taken from a counter instead of a
clock, and a correctness verdict scored against a reference that was itself wrong. This
document is the rule set that makes both failures impossible to repeat. It is short on
purpose: every rule exists because a specific measurement failed.

## 1. A rate is a clock, not a counter

A tokens-per-second figure is quoted **only** from a wall-clock measurement of a real token
run - `tools/kolibri/bench.py`, the server's own `timings`, or `date` around a run whose ids
are checked. Never from a phase counter, a bandwidth times a byte count, or a profiler.
Counter-derived numbers are **component measurements**: real, useful, and always labelled as
such ("pool drain", "streamed bytes"), never written as a rate.

Why: `714.2 MB/token / 34.5 GB/s = 20.70 ms` is the pool's slice of a token. Its reciprocal
(48.3 tok/s) looks like a rate, compares like a rate, and was published as one - and it never
measured a token. The wall clock said 24.4-24.5.

## 2. Every rate is two-run, load-stated, and deterministic

Two runs, both quoted; the box load at run time printed next to them; on the shipped binary
the generated ids must be identical across the two runs. A single run is a preview, not a
figure. NUMA binding is part of the setup and is stated: the same binary reads 24.5 tok/s
with the default binding and 29.5-30.2 with single-socket `numactl` - a setup property, not
noise, and wide enough to matter next to a floor.

## 3. Cross-method before publication

A rate leaves the ledger only after two **independent** methods agree within about 10%: the
wall-clock run and a derivation from components (bytes, bandwidth, kernel time). If they
disagree, the disagreement is itself a finding - investigate it, do not average it and do not
round it away. The 48.3 error would have died here in two minutes, because any single
wall-clock run of the shipped binary refutes it.

## 4. Validate the judge before the score

Any oracle, reference, or criterion used to grade output must itself pass a check before its
verdict is recorded: reproduce the reference under a no-op change (a batching width, a seed,
a re-run) and score against ground truth, not against the judge's tokens. At 98.6K context,
llama.cpp answered a code-word question with a number that appears nowhere in the prompt and
changed its whole answer under a prefill batch width - it could not grade anything, and for
close to a day it did.

## 5. Instruments measure components; they cost throughput

`*_PHASES`, `--counters`, and profiler runs synchronize or trace, so they change what they
measure: the prefill phase marks **doubled the pool's work**; `DEC_PHASES` serialises decode
and halves its observed speed; a profiler inflated kernel times 15x. An instrumented arm is
labelled instrument-time and used for *attribution* (which leg eats the time), never as a
published rate or a floor measurement.

## 6. Numbers land in the ledger first, in docs later

New figures live first in `docs/findings-log.md` and the working ledger. The published pages
(READMEs, `STRATEU.md`, the report) receive them in a separate pass, after rule 3. Nothing
publishes in the same session that measured it.

## 7. A published-figures manifest

Every figure that appears in a README, `STRATEU.md`, or the report is listed in one place
with the ledger row it comes from, and the stale-figure check is a mechanical diff. The
figure is greppable **in the units it is written in**: a sweep for "48.3" missed table cells
that said "48 tokens/s" - which is how a killed figure survived, in seven languages at once.
Ranges and roundings are different claims: "24-25" (our two runs) and "24-30" (across NUMA
bindings) must never be used interchangeably.

## 8. Determinism of the measurer is not validity of the measurer

Two runs returning identical numbers prove the instrument is deterministic. Two identical
counter readings created the 48.3 error and looked exactly like the two-run discipline
passing. Determinism is necessary; validity comes from rules 3 and 4 only.

## 9. Bridge arithmetic is an estimate

A conversion from component milliseconds to a rate ("hence tg") is an estimate. It is written
with the word estimate and never enters a floor table, a README, or a goal amendment. Floors
come from clocks only (rule 1).

## 10. Retraction is part of the same mechanical discipline

When a published figure dies, every artifact quoting it is corrected in the same pass -
including translations and tables derived from tables - and the correction stays visible
where the wrong figure stood: log files append a correction, published pages carry an explicit
correction note. A correction that fixes one language or one file is how a killed figure
survives in the others.

## The failures that priced these rules

| date | what broke | rules that now guard it |
|---|---|---|
| 2026-10-23 | the needle was scored against llama.cpp at 98.6K, which answers a code-word question wrongly and flips its whole answer under a prefill batch width | 4 |
| 2026-10-24 | decode published as 48.0/48.3 tok/s - the reciprocal of the pool's drain counter; wall clock is 24.4-24.5 (short) and 12.2-12.3 (after 32K); two floors were reported met and were not | 1, 3, 7, 10 |
| 2026-10-24 | the post-audit sweep missed the same stale figure in seven languages (the search pattern did not match the units the tables used) | 7 |
| 2026-10-23 | the phase-marks instrument doubled the pool's work; `DEC_PHASES` halves decode speed; a profiler inflated kernel times 15x | 5 |

What was measured as wall-clock A/B survived every re-measurement: the prefill figures, the
five measured-dead levers, the determinism work, the KL and accuracy audits. The failures
cluster exactly where a cheap signal stood in for an expensive one. These rules make the
expensive check happen first, where it costs minutes.
