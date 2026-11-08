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
| 0 driver budget | in progress | tests + controls below | pending |
| 2 selected MMVQ | not started | | |
| 3 quantization reuse | not started | | |
| 4 cache admission | not started | | |
| 5 synchronization | not started | | |
| 6 promotion fences | not started | | |
| 7 GPU hit dispatch | not started | | |
| 8 CPU packing | not started | | |
| 9 host-copy diagnosis | not started | | |
| 10 zero buffer | not started | | |
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
