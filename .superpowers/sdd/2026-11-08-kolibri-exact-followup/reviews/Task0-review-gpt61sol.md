## Decision: KEEP blocked

No Critical findings. The following are static-analysis findings; I did not execute reproducers.

### Important

1. **Source fixture/profile preservation is incomplete.**  
   `tools/kolibri/bench_unpinned.py:94–105`

   Checks cover only selected same-output aliases, not every input against every writable artifact.

   With `--out /tmp/run --label case`:
   - `--ids-file /tmp/run/case.profile` passes validation, then line 103 deletes the source fixture—even if engine launch subsequently fails.
   - `--profile-input /tmp/run/case.ids` passes validation, then line 100 overwrites the source profile before line 105 reads it.
   - A fixture named `case.log` is truncated at line 119.
   - Hard links bypass the `resolve()` comparisons entirely.

   Validate both source files against **all** output paths, including existing-file identity, before writing anything. Add cross-artifact and hard-link preservation tests.

2. **Nonfinite timeout values disable the watchdog.**  
   `tools/kolibri/bench_unpinned.py:65`

   `--timeout nan` and `--timeout inf` pass `<= 0` validation and launch the engine. On this POSIX Python implementation, neither produces an expiring deadline for a hung process. Require `math.isfinite(a.timeout)` and positivity before any writes or launch.

3. **Timeout cleanup kills only the immediate process, not its descendants.**  
   `tools/kolibri/bench_unpinned.py:119–123`  
   `tests/kolibri/bench_unpinned_test.py:111–120`

   If `--bin` is a wrapper that launches the engine without `exec`, `subprocess.run` kills the wrapper while the engine can survive, retaining GPU resources and writing artifacts. The timeout test checks only the driver’s exit/message—not descendant termination.

   Use process-group/session cleanup and test that a recorded descendant PID actually terminates. For a directly launched executable, the current timeout does return a clear nonzero result.

### Minor

4. **`--ctx` accepts values the engine cannot represent.**  
   `tools/kolibri/bench_unpinned.py:67–68,109`

   Python integers are unbounded, but `src/kolibri/kolibri_engine.cu:6883` parses context into an `int` with `atoi`. For example, `--ctx 4294967296 --prompt 1 --n 1` passes driver validation but overflows the engine’s context representation. Reject unsupported numeric ranges before launch.

5. **Other preflight failures occur after destructive writes.**  
   `tools/kolibri/bench_unpinned.py:100–105,114–115`

   A missing/unreadable `--profile-input` is discovered after token outputs are overwritten and the existing output profile is deleted. Malformed `--env` is also discovered afterward. Read source inputs and validate environment assignments before modifying run artifacts.

## Correctness risks

- Source preservation currently violates acceptance.
- No context off-by-one found: the engine uses `input length + steps <= ctx` at `src/kolibri/kolibri_engine.cu:7363–7366`; the driver’s forced history gives exactly `n` steps. `131064 + 8` is correctly accepted.
- Explicit token parsing normalizes validated integers into comma-separated decimal IDs; no token/argument injection path found.

## Lifecycle/race risks

Surviving wrapper descendants can overlap subsequent GPU runs and continue modifying reused artifact paths. The current watchdog test does not establish cleanup.

## Controls and performance-evidence gaps

- Both control JSON files record the sealed SHA `8a4c1fc1…c63c2`, matching `baseline/identities.json`. Commands and logs consistently show 512/128 and 8/32 controls, with no new engine CLI switches.
- Under ordinary, non-optimized Python execution, JSON is emitted only after the complete-length and finite-float assertion at line 134. Expected dump sizes are **65,536,000** and **16,384,000 bytes**.
- I could not independently recompute dump sizes, finiteness, or hashes. Logs alone do not prove finite complete logits.
- Inherited environment is not recorded (`bench_unpinned.py:113,137`); empty `env_overrides` does not establish the effective flag-off configuration or runtime environment.

## Skipped steps / review limits

- Both controls record **7200 seconds**, below the plan’s **10800-second minimum** (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:30`). They completed, so this is a procedural deviation—not a timeout mistaken for success.
- Available tools cannot execute commands. I read both files, the plan, engine guard, and evidence, but **could not run `git diff`, tests, compilation, or hash validation**. Consequently, engine-change exclusion and preservation of existing tests remain unverified. Git metadata does confirm the requested branch and HEAD.
