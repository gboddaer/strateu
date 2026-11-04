# Notepad retry: temperature 1.0 only

## Scope

The user requested retrying the full notepad test after approving temperature
1.0 as the default. This is one new cold generation, not a repair conversation.
It follows the [temperature 0.2 large-budget run](large-budget-notepad.md), which
used all 65536 tokens without producing an answer.

The new runner checks before submission that the entire request equals the
previous saved request with **only `temperature` changed from 0.2 to 1.0**. It
also checks the server/engine config for equality and the engine executable's
SHA256 against the previous experiment. Server-side default-policy source has
since changed, but explicit request sampling values take precedence. No new
engine binary or model weights are substituted.

Settings retained: same system/user text, thinking on, 65536-token total output
allowance, 131072 context, top_p 0.9, top_k 40, min_p 0.05, repetition penalty
1.05 over 64 tokens, presence/frequency penalties 0, seed omitted (engine default
0), no separate thinking cap, no forced reasoning closure, loop recovery or
repeat guard. One fresh server on loopback port 8210; no previous conversation
or cached prompt. The original run's request/config/output remain untouched.

Artifacts: `/data4/kolibri-sram/out/large-budget-notepad-temp1/`. The runner
refuses a second POST if its submission marker exists. It records exact inputs,
raw SSE, progress, separate reasoning and answer, final usage and compiler
results. It stops its own server after completion.

## Evaluation

Compile any final answer unchanged. If compilation succeeds, run the calibrated
PTY checks from the previous run against the unmodified binary. These cover
actual save/quit input, edits, the join cursor, scrolling, loading, failed save
and termios restoration. Do not score a handwritten reference or repaired
program as model output. A missing final answer is not a compilation failure.
This is one paired sampled trajectory, not proof of overall coding accuracy.

## Interim observations

At 402.1 seconds the server reported 7936 generated tokens; at 1299.5 seconds,
22934; at 2539.3 seconds, 39034. All were still thinking, with no final-answer
text, no thinking budget and zero recoveries. The prompt remains 569 tokens.

A partial analysis at 92227 reasoning characters found 134 paragraphs of at
least 120 characters and 134 distinct paragraphs after whitespace normalization:
no exact repeats in that sample. It contained code fences and `int main`
mentions, unlike the temperature-0.2 trace. These are observations about drafts
inside reasoning, not a final program or proof of code correctness. The complete
reasoning/answer must be assessed after generation ends.

## Final generation result

The request completed naturally on 4 November 2026, without filling the cap:

| Measurement | Temperature 0.2 baseline | Temperature 1.0 retry |
| --- | --- | --- |
| Allowed output tokens | 65536 | 65536 |
| Generated tokens | 65536 | **45442** |
| Request wall time, excluding load | 88.87 minutes | **52.13 minutes** (3127.68 s) |
| Finish | `length` | **`stop`** |
| Cached prompt tokens | 0 | 0 |
| Reasoning characters | 271403 | 170882 |
| Final-answer characters | 0 | **7068** |
| Final C++ | none | **213 source lines** |

Both SSE streams completed normally. For the retry, there is no failure record,
no thinking budget and no recovery/forced-close event. The final source compiles
unchanged with the original C++17 command, with zero compiler errors or warnings.
All ten includes are in the prompt's allowed list. The extracted source is the
model's final code block, not a repaired or assembled reasoning draft.

Reasoning analysis (`reasoning-analysis.json`): 326 paragraphs of at least 120
characters, 312 distinct after whitespace normalization. The maximum exact
repeat count is 4, involving code-fragment drafts. This is unlike the previous
six-paragraph planning cycle repeated 71–72 times. Temperature 1.0 escaped that
observed cycle and delivered an answer in this one trajectory; do not generalize
to guaranteed loop-free or correct output.

## Runtime evaluation

**The unmodified generated editor passed 0/11 PTY scenarios.** This is not eleven
independent defects: most scenarios share the same save/quit failure.

The local checker was audited before execution. It had required an initial draw
before input, which the prompt does not explicitly require. In the retry's copy
of `pty_check.py`, initial draw is informational; actual screen redraw during
the key sequence is still required. Expected edits, saved bytes, input keys,
exit and restoration checks are otherwise unchanged. The handwritten reference
passed **11/11** under that checker (`pty-calibration/`), before the model binary
was tested. No model pass/failure depends on the unsupported initial-draw
assumption. No generated C++ was changed.

Observed failures (`pty-model/results.json`, per-case `screen.bin`):

- The program leaves input flow control enabled and retains Ctrl+S/Ctrl+Q as
  VSTOP/VSTART. Thus the terminal consumes save/quit rather than delivering
  them to the editor under the normal IXON-enabled PTY state. Files are not
  saved; most runs hang until the harness timeout. A normal quit/restoration
  therefore does not occur in those cases. This is the shared primary failure.
- The bare-ESC case **does** exit naturally with termios restored. Its whole
  scenario still fails because the preceding save did not write the file.
- The long-document scenario aborts with SIGABRT, rather than merely hanging.
- No scenario passes after excluding initial draw from grading.

A separate diagnostic, not added to the eleven-case denominator, confirms an
independent crash using no numeric alias keys: type `ab`, Enter, `abcdef`, Up,
then `X`. Moving to the shorter first line leaves column 6 unchanged. Insertion
then throws uncaught `std::out_of_range`:

```text
basic_string::insert: __pos (which is 6) > this->size() (which is 2)
```

The process exits via SIGABRT and does not restore termios. Evidence:
`pty-diagnostics/up-to-shorter-line/{result.json,screen.bin}` and
`pty-diagnostics.log`. This is actual execution of the unmodified model binary,
not inferred from the common flow-control failure.

Static inspection also finds the known join-cursor error (moves to the merged
line's end instead of the join), literal digit handling instead of control-byte
1/4/5/6 aliases, and an extra empty buffer line after loading a newline-terminated
file. These were not isolated with repaired code; keep them separate from the
runtime-confirmed defects above. The numeric alias interpretation is the
previously documented control-byte convention consistent with 17/19, not an
extra unstated escape-sequence requirement.

## Conclusion and limits

**Temperature 1.0 improved answer delivery in this paired trial, but did not
produce a correct editor.** It gave a complete, compiling answer instead of no
answer, and avoided the earlier exact passage loop. The program still fails
basic save/quit and safety requirements. Compilation alone is not correctness.
No repairs, continuation prompts, feedback loops or extra generations were used.

This is one sample per temperature with common explicit settings and the same
engine binary, not a statistical coding-quality qualification or proof of an
inference-engine fault. The comparison changes only the saved request's
temperature; it does not test the model's complete recommended sampling profile,
other seeds or a different reasoning effort. The request times reflect different
generation lengths/histories, not an engine speedup measurement. No additional
production-default changes or kernel changes were made during this retry.

## Artifacts and completion

`evaluation-summary.json` aggregates exact input-difference checks, usage,
compilation, headers, runtime results and file hashes. Other evidence includes
`request.json`, `serve-config.json`, `comparison.json`, `provenance.json`,
`response.sse`, `result.json`, `reasoning.txt`, `answer.txt`, `notepad.cpp`,
`notepad.bin`, compiler/run logs, status snapshots and PTY artifacts. The previous
run's request/config and binary fingerprints were compared before sending.

Generated source SHA256:
`126454cf7776bde9e8cfe766567688d6caef91a8e213c7c6c3dca3e662edd4ad`.
The inference server and runner stopped; GPU usage returned to **0 MiB**.
