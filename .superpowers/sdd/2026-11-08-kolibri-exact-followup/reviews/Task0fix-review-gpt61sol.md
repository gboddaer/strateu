## Decision: still blocked

### Important

1. **Not every input is protected from output aliases.**  
   `tools/kolibri/bench_unpinned.py:117–120,145,177–178`

   Collision checks cover only `--ids-file` and `--profile-input`. They omit `--compare`, `--model`, `--bin`, and the implicit oracle fixture.

   Concrete case: `--out /tmp/run --label case --compare /tmp/run/case.bin` lets the engine overwrite the reference, then compares the new dump against itself. Hard-linked references likewise bypass protection. A model aliasing `case.ids` is overwritten at line 135.

   Extend collision checks to every consumed file and test direct, symlink, and hard-link aliases.

2. **Preservation tests can pass without the collision protection.**  
   `tests/kolibri/bench_unpinned_test.py:15–17,121–150`

   These tests retain the default nonexistent binary. Removing the collision checks would still produce exit 2 before artifact writes, satisfying their assertions. Additionally, the two cross-artifact `--ids-file` cases use `keep me\n`, which fails token parsing before collision validation.

   Use a valid executable stub and valid token fixtures; assert the collision-specific error, no launch, and preservation of all existing artifacts. The current tests do not establish the claimed regression coverage.

### Minor

3. **Preflight failures still modify the filesystem.**  
   `tools/kolibri/bench_unpinned.py:94–95,121–134`

   `out.mkdir()` precedes profile reads, environment validation, and the missing-binary check. With a nonexistent `--out`, any of those failures creates directories.

   Artifact contents remain untouched for these failures, but “write nothing” is not literally satisfied. Move directory creation after preflight. The test at `tests/kolibri/bench_unpinned_test.py:143–150` uses an existing directory and checks only `case.ids`.

4. **The new inherited-environment field has no secret-safe allowlist.**  
   `tools/kolibri/bench_unpinned.py:151,179–189`

   Every variable beginning `STRATA_KOLIBRI_` is serialized verbatim to JSON and stdout—even unsupported variables such as `STRATA_KOLIBRI_UNUSED_SECRET`. Ordinary unrelated variables outside that namespace are excluded, but the prefix alone does not guarantee relevance or absence of secrets.

   Use an explicit allowlist of supported benchmark/engine settings and test exclusion of unrelated secret sentinels. This is a conditional disclosure path, not evidence that existing artifacts contain credentials.

## Seven requested checks

| # | Result | Code evidence |
|---|---|---|
| 1 | **FAIL overall.** Explicit fixture/profile protection passes, including hard links; other inputs remain unprotected. | `bench_unpinned.py:103–120`; finding 1 |
| 2 | **PASS, static.** Nonfinite/nonpositive timeouts and context outside `1..2147483647` fail before filesystem writes or launch. | `bench_unpinned.py:67–74` |
| 3 | **PASS, static.** Launch creates a session; timeout sends `SIGKILL` to its process group. The test records a descendant PID and requires its disappearance—not merely driver failure. | `bench_unpinned.py:153–165`; `bench_unpinned_test.py:163–181` |
| 4 | **FAIL literally; artifact-content protection passes.** Profile-read errors, malformed assignments, and missing binary precede artifact writes, but follow directory creation. | `bench_unpinned.py:95,121–140` |
| 5 | **PASS.** Forced history contains `n−1` tokens, giving exactly `n` engine steps; both reject only when input length plus steps exceeds context. Equality is accepted. | `bench_unpinned.py:92,141–147`; `src/kolibri/kolibri_engine.cu:7363–7366` |
| 6 | **FAIL the strict guarantee.** Namespace filtering excludes ordinary unrelated environment, but does not prevent arbitrary prefixed secrets from being recorded. | finding 4 |
| 7 | **FAIL—remaining defects exist.** | findings above |

**First review’s Important findings:** all three reported production defects are fixed for the originally identified fixture/profile, nonfinite-timeout, and wrapper-descendant cases. Their resolution is **not fully regression-proven**, because the preservation tests are false positives. The broader every-input guarantee remains unmet.

**Review limits:** read-only inspection only; no tests or commands executed. Git metadata confirms HEAD `932a5c7e2c61efab193c405dbb1134a8a540ba14`; available tools cannot produce a working-tree-versus-HEAD diff, so exact change attribution was not independently verified.
