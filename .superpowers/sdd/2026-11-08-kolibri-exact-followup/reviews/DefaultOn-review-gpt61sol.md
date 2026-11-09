**Keep default-on for the locally qualified W7800/HIP Kolibri path.** The evidence supports the composed default; it does not establish cross-backend release readiness.

### 1. Evidence supports keeping it
The timing JSON confirms the same executable (`68f2c1fb…`) for the short default-versus-explicit-off comparisons. Both five-pair series favor defaults, with matching complete-logit hashes. The 10/10 integration comparisons and near-boundary equality provide meaningful regression coverage—not merely matching generated tokens.

Keep **+35.2% paired median** separate from **+19.2% ratio of medians**. These establish a composition gain, not an isolated gain attributable to each knob.

### 2. Explicit-off behavior is preserved
In `src/kolibri/kolibri_engine.cu`, the three relevant gates are at lines **1959, 3233 and 4974**.

`env_on_default_on` returns false for explicit `"0"`; empty and nonnumeric values also remain false under its existing parsing. `env_on` is unchanged.

One reporting correction: the helper is **not used only three times**. Existing graph, per-layer-slot and pool-pin controls already use it. Say **“three newly changed call sites”**, not “only three uses.”

### 3. Scope and less-obvious effects
Static inspection found no unintended arithmetic-path change:

- Selected MMVQ requires no-pin mode and eligible selections; unsupported dispatches fall back.
- Overflow skipping requires no-pin mode, zero misses and no poison diagnostic.
- The immutable zero operand still reaches the unchanged final combine; the corrected `zero_used` override remains present.

Two effects must nevertheless be explicit:

- `ZERO_BUFFER` allocates `K*N*sizeof(float)` during **every Kolibri scratch initialization**, including pinned/non-consuming configurations. Allocation precedes auto-fit, so at a tight memory boundary it could change fitted cache capacity or allocation success.
- The skip gate now bypasses profile-derived defaults. An unset flag permits skipping even with the `baseline` profile. **`baseline` alone is therefore not an all-off control.**

The source is also not restricted to the measured W7800 backend.

### 4. Limits that must remain recorded
- Targeted miss→hit→miss, poison, mixed-format and individual flag-matrix tests remain unrun.
- CUDA/SYCL qualification and server reset/cancellation/prefix-reuse qualification remain absent.
- Five pairs are limited evidence; startup/prefill vary substantially. Cache8 includes a pronounced slow first off-run.
- The near-boundary reference uses a **different executable hash**, unlike the short same-executable pairs. Its eight decode distributions establish regression identity, not a repeatable long-context speed claim.
- Measurements include counters/full-logit dumping; do not present them as universal production throughput.
- Existing setup/Terminal failures remain unresolved, without an established baseline.

**Verdict: keep default-on, with those scope qualifications and reporting corrections.** This was an artifact/source review, not a fresh runtime validation.
