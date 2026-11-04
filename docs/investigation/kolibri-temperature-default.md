# Kolibri default temperature: 1.0

## Decision

After the [65536-token notepad run](large-budget-notepad.md) at temperature 0.2
ended in a passage loop with no answer, the user explicitly requested changing
**temperature alone to 1.0 and making it the default**, then committing/pushing.
The model's GGUF metadata also recommends temperature 1.0.

This is a user-approved sampling-policy change, not an inference-arithmetic
optimization or a declaration that coding quality or loop behavior is fixed.
No new generation at temperature 1.0 is claimed by this change. The earlier
measurements and their immutable experiment request/config files remain at 0.2.

## Scope

- `serve/server.py`: add only `temperature: 1.0` to Kolibri's model-card defaults
  (`arch: kolibri1`), for both thinking and non-thinking requests.
- Do not change top_p, top_k, min_p, repetition/presence/frequency penalties,
  their windows, seeds, thinking budgets or recovery settings.
- Keep precedence: explicit request > shared Chat settings > run-config sampling
  > model-card defaults. Explicit temperature 0 still selects greedy decoding.
- Other model families and the low-level native diagnostic/benchmark protocol
  are unchanged. Normal Kolibri setup already writes `arch: kolibri1`.
- The active local `/data4/kolibri-sram/serve-config.json` also gains only
  `sampling: {"temperature": 1.0}`. Its old contents were backed up to
  `/data4/kolibri-sram/out/temperature-default/local-config-before.json`.
  All pre-existing config keys compare equal. No arch change or reclassification
  of that older profile was made, so its other inherited defaults are untouched.
  This hardware-specific profile is local, not committed into the repository.

Clients/presets that explicitly send 0.2 retain that value. They must stop sending
it, or explicitly request 1.0, to use the new temperature. Restart an already
running server to load changed source/config; the experiment server was stopped.

## Validation

Four new tests exercise real HTTP requests and the engine sampling-key bridge:

1. Kolibri supplies exactly temperature 1.0 in both modes, no other new sampler.
2. Explicit greedy and a complete explicit sampled profile remain unchanged.
3. Config and shared Chat values override the new family default.
4. `/props` advertises temperature 1.0 without adding other sampling parameters.

Before the production change, default-wire and props assertions failed as
expected (`tests-red.log`). After adding the Kolibri temperature entry, all
10 model-card sampling tests passed (`tests-green.log`). Existing telemetry
ResourceWarnings are present and unrelated to this change.

The Kolibri Python-tools suite passed: **181 passed, 7 skipped**
(`tools-tests.log`). The full serve suite completed: **1257 tests OK, 31
skipped**, 689.164 seconds (`serve-tests.log`). Artifacts:
`/data4/kolibri-sram/out/temperature-default/`.

No C++ or kernel source is changed, so no engine rebuild, backend runtime
qualification or full-logit arithmetic claim is part of this patch.
