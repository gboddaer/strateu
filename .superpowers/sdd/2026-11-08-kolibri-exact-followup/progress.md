# Kolibri exact-followup execution ledger

Branch: `amdw7800probuild_optimize` (base `amdw7800probuild` `dd5f7e6e`; docs gate commits `076829df`, `932a5c7e`).
Spec: `docs/superpowers/specs/2026-11-08-kolibri-exact-followup.md`.
Plan: `docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md`.

## Binding constraints (copied; not reinterpreted)
- No server. Offline CLI/unit tests only.
- Accuracy first; unchanged arithmetic needs finite bit-identical complete logits. No tolerance changes.
- No attention tuning, PF chunk change, expert quantization/routing change, reduction-order change,
  clock/firmware/driver/system change, full-table pinning, or global default promotion.
- Keep mmap backing, bounded transfer buffers, 1536 MiB reserve. New switches default off.
- HIP locally; CUDA/SYCL NOT RUN unless a toolchain really exists.
- Serial GPU work. Per-process watchdogs allowed; no global cutoff.
- Reviewer: `openai-codex/gpt-6.1-sol`; fallback `ollama/glm-5.3-flash:cloud` only for unavailability;
  both unusable => recorded SKIPPED, continue. A negative review never switches reviewers.

## Baseline identities
`.superpowers/sdd/2026-11-08-kolibri-exact-followup/baseline/identities.json`
- Branch head `932a5c7e`; base source `6c241aa4`.
- Sealed engine SHA `8a4c1fc1...c63c2` (42,937,712 bytes) == `build-kolibri-hip/strata-kolibri`.
- Model `/home/gbo/models/Kolibri-1-Q4_K_M.gguf` SHA `c2ac1301...562e` (47,454,113,472 bytes).

## Task status

| Task | Decision | Evidence | Review |
|---|---|---|---|
| 0 driver budget | KEEP (driver correctness) | 17 tests + 10800 s controls | 3 reviews, resolved |
| 2 selected MMVQ | KEEP + KEEP-ENABLER | full-logit gates + 5 timing pairs + integration matrix | 3 reviews, resolved |
| 3 quantization reuse | REJECT (no repeatable gain, no named consumer) | exactness + 5 timing pairs + integration | review 1 agrees |
| 4 cache admission | REJECT (no gain over demand; not correctness) | 4K/16K/65K + demand control + integration | review 2 agrees |
| 5 synchronization | REJECT (no repeatable speedup; removed) | gate + GPU ordering + 10 pairs + integration | review 3 agrees |
| 6 promotion fences | REJECT (12 ms, no end-to-end gain demonstrated) | batch test + 389-move round + diff | review agrees |
| 7 GPU hit dispatch | BLOCKED (prototype qualified; engine lifecycle deferred) | counters + DEC_PHASES + hit/miss/update/reset test | 4 reviews agree |
| 8 CPU packing | REJECT (two lanes slower) | byte-exact test + 2 alternating pairs | pending |
| 9 host-copy diagnosis | REJECT (2 GB/s ceiling; no setting gain) | probe, default/SDMA on/off | pending |
| 10 zero buffer | KEEP opt-in (+5.6% decode, exact; 5/5 pairs) | 5+5 pairs + 10/10 integration | pending |
| final composition | not started | | |

## Task 0 — driver budget

RED observed: 8 failures in `tests/kolibri/bench_unpinned_test.py` (unknown `--timeout`/`--ctx`/`--ids-file`,
no context-budget rejection, no watchdog message, no fixture copy/collision guard).
GREEN: `python3 -m unittest discover -s tests/kolibri -p 'bench_unpinned_test.py' -v` -> 10 tests OK.

Controls: `.superpowers/sdd/2026-11-08-kolibri-exact-followup/controls/` (sealed engine, no new switches).
- 512/128 auto: 65,536,000-byte dump, 16,384,000 finite floats, SHA `6ffefa8f...8721` in both the 7200 s run and the
  compliant 10800 s rerun.
- 8/32 cache8: 16,384,000-byte dump, 4,096,000 finite floats, SHA `466f48f2...c3d6` in both runs.

### Task 0 review 1 (`gpt-6.1-sol`, exit 0, 2026-11-08T10:03Z-10:06Z)

| Finding | Sev | Resolution |
|---|---|---|
| Source fixture/profile can alias another artifact or hard link; profile/ids deleted or truncated before validation | Important | FIXED: both sources checked with `samefile` + resolved path against ids, ids-source, profile, log, bin, json before writes; fixture copied; profile bytes read before deletion. Tests: cross-artifact 4 cases, hard link, preflight-writes-nothing. |
| `nan`/`inf` timeout passes positivity and disables the watchdog | Important | FIXED: `math.isfinite` required. Test: nan/inf rejected. |
| Timeout kills only the wrapper, descendants survive | Important | FIXED: `start_new_session=True`, `killpg(SIGKILL)` on timeout. Test records descendant PID and proves it is gone. |
| `ctx` above the engine's `int` range passes driver validation | Minor | FIXED: reject above 2147483647. Test added. |
| Missing profile / malformed `--env` discovered after destructive writes | Minor | FIXED: profile bytes read and env assignments validated before any artifact write; missing/unreadable sources are usage errors. |
| Controls used 7200 s watchdog vs plan 10800 s | Procedural | FIXED: reran both cases at 10800 s; hashes identical. |
| Inherited `STRATA_KOLIBRI_*` not recorded | Evidence gap | FIXED: JSON now records `env_inherited_strata`, Python version, dump bytes/floats. |
| Reviewer could not run tests/diff (read-only tools) | Limit | Recorded. Local evidence: 15 tests OK, py_compile OK. |

### Task 0 review 2 (`gpt-6.1-sol`, exit 0) — fix verification

| Finding | Sev | Resolution |
|---|---|---|
| Only `--ids-file`/`--profile-input` protected; `--compare`, `--model`, `--bin`, oracle unprotected | Important | FIXED: all consumed paths now checked (direct, symlink, hard link) against all six output artifacts before any write. Tests cover ids/profile/compare/model/bin aliases. |
| Collision tests were false positives (nonexistent bin or invalid tokens exited 2 anyway) | Important | FIXED: tests use a valid executable stub plus valid fixtures, assert `must differ`, assert no launch marker, and assert preserved contents. Symlink and hard-link cases added. |
| `out.mkdir()` runs before preflight failures | Minor | FIXED: directory creation moved after source reads, env validation and binary check; test asserts a nonexistent output directory is not created on failure. |
| Inherited env serialized by prefix, not allowlist | Minor | FIXED: explicit `EVIDENCE_ENV_KEYS`; unit test excludes `STRATA_KOLIBRI_UNUSED_SECRET`. Explicit `--env` values outside the allowlist are recorded as `<redacted>`. |

### Task 0 review 3 (`gpt-6.1-sol`, exit 0) — confirmation

Decision: Important findings resolved; no blocking finding. All six checks confirmed with file:line evidence.
Remaining Minor: two legacy preservation tests used the nonexistent default binary and lacked collision-message
assertions. FIXED: both now use the marker stub, assert `must differ`, assert no launch, and preserve contents.

Final Task 0 evidence: 17 tests OK; `py_compile` OK for tool and tests; controls rerun at 10800 s with identical
hashes. Decision: **KEEP** (test-driver correctness fix; no engine change, no speed claim).

## Task 2 — exact multi-selection MMVQ

Evidence: `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task2/summary.md`, `mmvq_selected_test.log`,
`integration-off/`, `integration-on/`, timing JSONs.
Full logits: off/on 512/128 `6ffefa8f...`, cache8 `466f48f2...`, 4097/8 `241969a0...`; integration 10/10 cases identical.
Timing: 512/128 decode 5/5 faster, paired median +20.9%; cache8 inconclusive (+2.8% paired median, one -13.3%).
Decision: KEEP (local W7800 decode) + KEEP-ENABLER for Task 3.

### Task 2 review 1 (`gpt-6.1-sol`, exit 0)

| Finding | Sev | Resolution |
|---|---|---|
| Q6 refusal keyed on env flag, not per-pointer registration; a directly packed matrix could bypass | Important | FIXED: `q6_packed_registered()` checks each selected pointer; packed-Q6 sentinel test added. |
| gfx906 default-wave mode would run the parity test and fail by design | Important | FIXED: gfx906 parity test runs with `STRATA_MMVQ_WAVE=0`; separate `--expect-refusal` test registered only for gfx906. Not runnable on this gfx1100 host. |
| Required short/long/hot regression matrix and profile replay missing | Important | FIXED: `test_unpinned.py --phase all --hot` off/on; 10/10 complete dumps byte-identical. |
| Unaligned base throws while derived returns false; contract unclear | Minor | FIXED: header states the distinction; tests cover unaligned base throw and unaligned derived false. |
| Cache8 median misreported as +0.1% | Reporting | FIXED: summary now +2.8% paired, +0.47% ratio of medians, still inconclusive. |
| RED evidence and ctest log incomplete | Evidence | FIXED: RED compile error, GREEN test log and integration hashes recorded. |
| ctest `decode_wait` failure | Environment | Device-0 mismatch: test is registered with `HIP_VISIBLE_DEVICES=0`; same binary passes on device 1, and the failure reproduces without this change. |

### Task 2 review 2 (`gpt-6.1-sol`, exit 0) — fix verification

| Finding | Sev | Resolution |
|---|---|---|
| Integration report claimed a SHA list that did not exist; off/on equality not auditable | Important | FIXED: `task2/integration-compare.txt` has per-case size, float count, finiteness, SHA-256, equality, input/profile hashes and run-time executable hash. 10/10 finite and identical. |
| gfx906 refusal test inherited `STRATA_MMVQ_WAVE` | Minor | FIXED: refusal test forces `STRATA_MMVQ_WAVE=1`, parity forces `=0`. |
| Packed-Q6 sentinel could pass under inherited `STRATA_Q6_PACKED=1` | Minor | FIXED: test unsets it at entry; only the later of two selections is registered, exercising the per-pointer loop. |
| cache8 summary still said +0.7% | Reporting | FIXED: +0.47% ratio of medians, +2.8% paired median, inconclusive. |
| Integration runner used a 300 s watchdog | Procedural | FIXED: `--timeout` default 10800. The completed artifacts were not rerun; every case finished well inside 300 s, so no timeout was mistaken for a pass. Recorded in the manifest. |

Third focused review requested for evidence closure.

## Task 3 — explicit quantization reuse (REJECTED)

Evidence: `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task3/summary.md`,
`task3/integration-compare.txt`, timing JSONs and logs.
Four flag combinations on 512/128 all kept SHA `6ffefa8f...`; cache8 both `466f48f2...`; 10/10 short/long/hot/profile
dumps byte-identical. Reuse alone 512 paired median -0.1% (ratio of medians -10.2%), cache8 neutral; both flags vs
selected alone -0.8% with a 4.36 tok/s outlier. No named retained consumer.
Review 1 (`gpt-6.1-sol`, exit 0) agreed with REJECT. Helper, wiring, flag, test and CMake target removed; evidence kept.

### Review findings

| Finding | Sev | Resolution |
|---|---|---|
| Reuse integration exact but no repeatable gain and no justified enabler | Decision | REJECTED and removed as proposed. |
| Test calls existing primitives, not `gemv_pair_quantized`; no production RED | Medium | Recorded as a skipped qualification; the experiment was rejected, so no retained code depends on it. |
| Task 3 evidence lacks a 4097/8 comparison and pinned-table numerical qualification | Medium | Recorded as skipped/not qualified; not claimed as passed. The integration runner is no-pin only. |
| Short `g_ids_host` fallback in `gemv()` can index out of bounds | Medium | PRE-EXISTING legacy defect exposed by the removed guard; no normal-path trigger found. Recorded for follow-up, not introduced by this task. |


## Task 4 - prefill cache admission

Evidence: `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task4/summary.md`,
`task4/integration-compare.txt`, timing JSONs/logs.
Traffic: 4K 19.78 -> 4.69 GB, 16K 78.31 -> 8.35 GB, 65K 313.35 -> 19.88 GB; zero staging in the on-runs. Full logits
identical at each size and 10/10 integration dumps identical. Wall: 4K noisy, 16K -5.4%/-27.6%, 65K -9.7%/-11.7%;
prefill +11.7% to +62%. Decode unchanged at 65K, can fall at 16K.
Decision: **REJECT** after the demand-profile control (demand off 25.15 GB / 1246.0 s; demand on 25.11 GB / 1263.2 s,
zero evictions, same SHA). The integration has no demonstrated incremental gain in the tested demand configuration and
no retained consumer. Removed planner, wiring and Task4 tests; existing demand free-slot admission untouched.
Not completed/passed: balanced controls, repeated-pack counts, five-pair gates, eviction-lifecycle qualification.
Baseline-only gains (19.78/78.31/313.35 GB staging -> 4.69/8.35/19.88 GB admission) are preserved as evidence.

## Task 5 - redundant decode synchronization (REJECTED)

Narrow gate + GPU ordering test + full-logit/integration equality all pass. First five 512 pairs: +6.7% median (3/5);
second five: -6.4% median (0/5), combined about -5.9%. The first series was noise. Gate, wiring and tests removed; both drains and ownership fences restored; evidence kept. Correctness was not the reason; no repeatable gain was demonstrated.


## Task 6 - promotion fences (REJECTED)

Implemented the narrow ring-only batch and its two-buffer/delayed-reader/second-generation test. A 389-move
promotion-heavy round fell 487.8 -> 475.6 ms (12.2 ms, 2.5%) with identical logits; no end-to-end improvement was
demonstrated. Removed the gate and test addition, kept the diff; per-move fences stay. Performance rejection; the
test was not a full lifecycle qualification and exposed no defect.

## Task 7 - GPU hit dispatch (BLOCKED)

The slot-path router copy was uncounted; a probe measured 0.567 ms/layer (~28 ms/token vs 68.79 ms measured).
A device-slot selected-MMVQ prototype passed bit-exact forced-map tests, and a GPU translation test passed
hit -> miss -> map update -> map reset against CPU translation with complete outputs. It was not connected to the
engine: partial residency still needs host ids for `np_admit` and misses, and all-resident does not fit (42.90 GiB
experts vs 39,860 MiB after dense+KV+scratch; auto-fit 326 slots/layer). Experimental code removed; diff kept.
BLOCKED, not a pass. No server.

## Task 8 - bounded CPU packing (REJECTED)

Two-lane packer was byte-exact (0/1/37/4099-byte spans, gaps, repeats, source lifetime, drain) and full logits stayed
`6ffefa8f...`. On two 512/128 pairs packing was not faster: startup 70.8/148.0 s off vs 89.7/156.6 s on; prefill
6.8/25.8 vs 21.3/19.3 s; decode 0.94/2.04 vs 2.43/10.56 s. Packing is mmap/IO-bound; a second lane competes. Code,
engine wiring and test removed; sources/diff/logs kept. No server.

## Task 9 - host-copy ceiling (REJECTED, no switch)

New standalone `tools/kolibri/transfer_probe.cu`: 64 MiB cap, pageable/pinned, 1/4/16/64 MiB, best of five,
every byte verified. All rows ~2.0 GB/s; pinned 64 MiB 33.618/33.614/33.618 ms for default/SDMA=1/0 (33.618/33.618/33.614 ms at 64 MiB). Engine decode
traffic ~70 MB/s is far below the ceiling. No engine or system change; probe kept as a diagnostic. No server.

## Task 10 - zero buffer (KEEP opt-in)

First attempt had a pointer bug (downstream `yc = s.ey` overwrote the buffer) and diverged; corrected with an
`if (zero_used)` override. 512/128 and cache8 full dumps match `6ffefa8f`/`466f48f2`; 10/10 integration identical.
Five pairs: 13.79 -> 14.14 decode tok/s median (+5.6% paired, 5/5 positive); cache8 neutral. All-resident does not fit (42.90 GiB
vs 39,860 MiB free; 326/384 slots). Opt-in only, no default promotion. No server.
