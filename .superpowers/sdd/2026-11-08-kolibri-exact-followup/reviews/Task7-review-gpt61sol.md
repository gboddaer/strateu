**BLOCKED is not justified by the current evidence. Task 7 is incomplete, not proven infeasible.** Deferring capture is reasonable; skipping the bounded offline prototype on these grounds is not.

### Important findings

**1. The counters miss an actual per-layer router D2H copy.**

`src/kolibri/kolibri_engine.cu:3080–3088` copies `s.ids` into `g_ids_host` before host slot selection. Neither the synchronous copy nor its async-plus-synchronize alternative increments `d2h_n` or `d2h_bytes`.

The recorded run explicitly uses plain decode and reports 50 slot-path layers/token. Therefore, static inspection indicates **6,400 uncounted router-copy calls** over 128 tokens. With six selected experts, that is only **153,600 bytes**, but potentially significant synchronization overhead. Small byte volume does not establish small latency.

The recorded 128 calls correctly match the full-logit copies at `src/kolibri/kolibri_engine.cu:7495–7496`. They do **not** establish absence of router round trips.

Also:
- Each logit copy is **512,000 bytes = 500 KiB**, not 512 KiB.
- `dec_layers` is not pool-only: it increments at `src/kolibri/kolibri_engine.cu:3401–3404` when `STRATA_KOLIBRI_DEC_PHASES` is enabled. This run did not enable it. The pool-specific D2H timing still does not measure the GPU slot-path router wait.

**Consequently, the required router D2H/idle-share measurement remains unfinished.**

**2. Server runtime is not demonstrated to be a prerequisite for an offline prototype.**

The relevant ownership already resides in engine functions:
- startup map/fill: `alloc_slots`, around `5915`;
- demand admission: `np_admit`, around `2926`;
- replacement/map mutation: `slots_reselect`, around `6005`;
- request reset: `reset_state`, around `5372`.

Crucially, `reset_state` resets KV/request state; it does **not** clear expert residency. A contract must distinguish request reset from expert-cache invalidation/recreation.

No complete device-map lifecycle harness exists in the inspected files. But building one is feasible offline, using existing testing patterns:
- `tests/kolibri/expert_transfer_test.cu:20–73` covers delayed transfers/readers, second-stream dependencies, drain and reuse.
- `tests/kolibri/decode_wait_test.cpp:36–78` demonstrates offline generation, replay and cancellation testing.

These tests do not qualify the proposed map automatically. They contradict the claim that these lifecycle properties inherently require launching a server.

A bounded test can exercise **hit → miss → upload/map publication → hit → replacement → drain/reset/reuse**, reject stale generations, and compare exact Q4_K/Q6_K outputs against CPU-selected slots. Ordinary host-controlled stream/event boundaries suffice initially; no device waiting loop or capture is required.

### Plan compliance

The plan explicitly requires the ownership contract and forced-hit/miss translation/MMVQ prototype (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:191–195`).

It does **not** require implementing through a demonstrated safety or authorization blocker. However, “the harness does not exist yet” and “device dispatch needs another interface” are implementation work, not established blockers here.

Nor must Task 2’s existing ABI be broken: a separate experimental device-slot entry point can preserve it. Capture remains conditional on successful lifecycle qualification.

### Artifacts / correctness risks

No Task-7 device-map helper, dispatch flag, kernel or test wiring was found in the inspected sources. `task7/` contains measurement artifacts and the summary—no apparent half-implementation. Without a Git diff, I cannot independently certify the broader “no code changed” statement.

There is no new arithmetic/race defect to assess because no prototype exists. The missing ownership contract and lifecycle tests are **unperformed qualification**, not failed qualification.

Correct the ledger’s “128 D2H/tok” to **128 counted D2H calls per run**.

### Verdict

**Do not close Task 7 as BLOCKED on this rationale.** Correct the counter interpretation, measure the router synchronization cost, write the ownership contract, and attempt the bounded offline prototype. Capture may remain unattempted. BLOCKED becomes appropriate only if that work identifies a concrete prohibited or unsupported dependency.
