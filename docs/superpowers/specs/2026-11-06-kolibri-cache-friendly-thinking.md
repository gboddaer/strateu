# Cache-friendly thinking levels for Kolibri

Status: accepted. Branch `fix/kolibri-cache-friendly-thinking`, base `7e6ec5ec`.
Date: 2026-11-06.

## Problem

The exact-prefix cache only reuses an exact token prefix. The reasoning-effort sentence
sits in the system message at the top of the prompt, so changing the level changes a token
near the start and the whole conversation is read again. Measured on a 4434-token prompt on
the T4 (temperature 0, `high`):

| request | reused | fresh | prompt time |
| --- | --- | --- | --- |
| same level, different last words | 4096 | 338 | 2.4 s |
| different level | 0 | 4434 | 26.9 s |

Every level switch therefore costs a cold read of the conversation, and users switch levels
per task.

## Decision

Make the end layout the **default** on the Kolibri prefix cache and move the level instruction to a short
trailing system turn - the existing `effort_position: "end"` layout (#458) - without the engine's
`--tail-role-token`. With the instruction at the end, all levels share the whole conversation prefix, so a
switch reuses the newest 512-token checkpoint and reads at most one block plus the trailing turn. That is the
same cost as a normal next turn (the first row above). `"effort_position": "start"` in the run config keeps the
older top-of-prompt behaviour.

The maintainer chose this default knowingly: [CONTRIBUTING.md](../../CONTRIBUTING.md) otherwise asks for a new
feature to be opt-in and the default prompt to stay byte-identical. Only Kolibri configs with the prefix cache
on change; every other family keeps the prompt it had.

## Design

- The layout is the default for `arch: kolibri1`: an unset `effort_position` means `end` there and
  `start` for every other family; an explicit `start` opts out. Setup writes the key into the
  Kolibri config. The prefix-cache size does not change the prompt, so a cache-off arm renders
  the same prompt (that keeps the on/off logit gate a fair comparison).
- `serve/server.py:effort_end_args` returns `[]` (layout enabled, no engine arguments) for
  `arch: kolibri1` when the engine lacks `--tail-role-token`, and prints that a level switch
  reuses the newest 512-token checkpoint. Other families keep today's answer: `None` and the
  0.1.39 message.
- `svc.effort_end = effort_end is not None` so the empty list still enables the layout; the
  replica and split paths append the empty list harmlessly.
- `render_prompt` and `head_ids` already implement this layout (#458) and need no change:
  default and `high` keep today's prompt, `low`/`medium` render the conversation with the
  template's default sentence at the top plus a trailing `system` turn, and off keeps the
  pre-closed empty thinking block at the very end. The top sentence is the template's default
  (`high`) for every level; that is existing #458 behaviour and stays.
- No engine change, no template change, no arithmetic or chunk change.

## Tests

1. `effort_end_args`: Kolibri -> `[]` (enabled, no engine arguments) with any prefix-cache
   setting and with the key unset; Kolibri + `start` -> `None`; another family + `end` without the
   flag -> `None` and the existing message; a bad value still raises.
2. The existing `EffortAtTheEnd` layout tests keep passing unchanged (prompt strings, off's
   pre-closed block, reuse of the whole conversation in the simulated engine).
3. GPU cache check on the T4, 4434-token prompt, `high -> low -> none -> medium -> high`:
   `cache_n` 4096, fresh at most 512 plus the trailing turn (about 338), roughly 2.4 s, and
   every request's full 128000-value logits byte-identical to a cold read of the same prompt
   (`STRATA_KOLIBRI_LOGITS_DUMP`).
4. Quality: the prompts change, so the fixed task set from
   `tools/kolibri/code_quality_probe.py` plus one tool-call case, start vs end layout at
   `low`, `medium` and `high`. Report every case; make no equivalence claim.
5. Docs: `docs/KOLIBRI1.md`, the "Changing the level costs the saved prompt" subsection, gains
   the setting and the measured numbers.

## Risks and limits

- The model sees the template's default sentence at the top and the actual level at the end.
  That is what test 4 measures. If it hurts, the top sentence has to leave the template
  altogether - a separate, larger change that also alters the default prompt.
- The trailing turn adds about 30 prompt tokens.
- Reuse lands on the last 512-token checkpoint, so a switch still reads up to 511 tokens.
  Exact reuse needs the engine work that this change deliberately avoids.
- Off is unaffected in content: the pre-closed block stays at the very end, so off to on (and
  back) is also a tail-only difference.
- Opt-in for other families, default for Kolibri: their prompts and caches are untouched.
- Single-GPU serving is the tested path.

## Out of scope

An engine checkpoint at the trailing turn; template changes; levels that differ only by
request-side budgets.
