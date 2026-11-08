## Decision: Important findings resolved; no blocking finding remains

Read both prior reviews and the updated resolution table.

1. **Confirmed: all listed consumed inputs are collision-checked.**  
   `tools/kolibri/bench_unpinned.py:119–146` checks IDs, profile, comparison, executable, model, and the oracle when consumed against all six artifacts. `samefile()` detects hard links; resolved-path equality detects direct and symlink aliases (`:127–136`). Checks precede directory creation and artifact writes (`:161–167`).

2. **Confirmed for the strengthened regression tests.**  
   `tests/kolibri/bench_unpinned_test.py:125–140,149–173` uses an executable marker-writing stub, valid token fixtures, collision-specific `must differ` assertions, no-launch assertions, and content-preservation assertions. Removing collision protection would fail these tests rather than pass through unrelated preflight errors.  
   The executable-alias case separately asserts its specific error and preserved executable contents (`:141–147`). Oracle protection is established by inspection, not a dedicated collision test.

3. **Confirmed: preflight rejection precedes output-directory creation.**  
   Source reads, collision checks, environment validation, and binary existence validation all precede `mkdir()` at `tools/kolibri/bench_unpinned.py:161`. The nonexistent-directory regression assertion is at `tests/kolibri/bench_unpinned_test.py:183–189`.

4. **Confirmed: explicit allowlist and redaction.**  
   `tools/kolibri/bench_unpinned.py:26–46` defines the allowlist, filters inherited values, and redacts explicit override values outside it. These helpers feed the emitted evidence at `:178,206–207`. Secret-sentinel assertions appear at `tests/kolibri/bench_unpinned_test.py:191–196`.

5. **No newly introduced correctness, data-loss, deadlock, or secret-disclosure defect found by static inspection.**  
   In particular, comparison references and executable/model inputs now receive the same preservation checks as fixtures.

6. **Context/step and watchdog behavior remain correct.**  
   Finite positive timeout and representable context checks remain at `tools/kolibri/bench_unpinned.py:90–97`. The context guard accepts equality (`:115–116`); forced history contains exactly `n−1` tokens (`:168–174`), matching the engine’s step calculation and bound at `src/kolibri/kolibri_engine.cu:7363–7366,7469`. Boundary tests cover rejection and equality (`tests/kolibri/bench_unpinned_test.py:55–70`).  
   Launch establishes a new session; timeout kills its process group and reaps the launcher (`tools/kolibri/bench_unpinned.py:181–192`). The descendant test checks PID disappearance (`tests/kolibri/bench_unpinned_test.py:209–227`).

### Remaining finding

- **Minor, nonblocking test cleanup:** the two legacy preservation tests still use the nonexistent default binary and lack collision-message assertions (`tests/kolibri/bench_unpinned_test.py:109–123`). Individually they remain weak, but the strengthened tests above resolve the Important regression-coverage finding. No blocking work is warranted.

**Limits:** No files edited or tests executed. Git metadata confirms HEAD `932a5c7e2c61efab193c405dbb1134a8a540ba14`; available tools cannot run `git diff`, so exact working-tree-versus-HEAD change attribution remains unverified.
