# Strateu full-notepad generation with a large output budget

## Request

The user asked for one repeat of the whole-program notepad task, with thinking
on and a very large budget, to see whether the model can deliver correct code.
Do not run other model prompts, provide compiler feedback or repair the answer.

This follows the [code-quality investigation](generated-code-quality.md) and
the [hosted thinking run](tesseracted-code-quality.md), which exhausted the
hosted service's unknown output cap without producing a final answer.

## Experiment

One cold served request using the existing Strateu executable, Kolibri-1 Q4_K_M,
T4 GPU, 14 CPU threads and a 131072-token context. The original full notepad
system/user messages are copied unchanged from `code_quality_cases.json`.

- Output allowance: **65536 generated tokens**, shared by reasoning and answer.
- Thinking enabled. No reasoning-budget limit (`reasoning_budget_tokens: 0`).
- Sampling: temperature 0.2, top_p 0.9, top_k 40, min_p 0.05,
  repetition penalty 1.05 over the last 64 tokens; presence/frequency penalties 0.
- Seed omitted, matching the original sampled request; engine Sampling defaults
  to seed 0. No seed sweep or repeatability claim is made for this one run.
- A private experiment config disables `answer_room`, `reasoning_close_retry`,
  `reasoning_loop_recovery` and `repeat_stop_tokens`. Thus the model is not
  forcibly taken out of its thinking or stopped by the repeated-token guard.
  The finite 65536-token allowance remains the bound if it loops.
- Loopback-only server on port 8210; no changes to production defaults, engine
  arithmetic, model files or the original serve config.
- The client writes a submission marker before the POST and refuses to resend
  if that marker exists. Exactly one model request is authorized by this run.

This is a practical sampled generation test, not an isolated proof that budget
alone changes quality: earlier direct comparisons also used greedy sampling,
and hosted settings are not fully known. No universal coding-quality conclusion
can follow from one answer.

## Artifacts and evaluation

Artifacts: `/data4/kolibri-sram/out/large-budget-notepad/`.
The runner (`run.py`) saves the exact request/config, executable fingerprint,
initial service status, raw SSE stream, periodic progress, reasoning and final
answer separately. Progress delta-event counts are **not** token counts; final
usage provides the generated-token count. It stops its own server afterwards.

Compile the final answer unchanged with C++17 and record all diagnostics. If it
compiles, exercise the editor in an isolated PTY with real keyboard input and
files. Do not use the old false-positive bare-ESC check or its incorrect
long-document newline expectation. Do not turn a patched program into a model
pass. A no-answer/capped run is not a compilation failure.

## Validation preparation

Prepared `pty_check.py` before a final answer was available. It drains output
while typing (avoiding a blocked PTY output buffer), requires the program alive
before input and after save, checks natural exit and termios restoration, and
compares exact saved bytes. Its 11 cases cover typing/Home, arrow/backspace,
Up/End, line join followed by insertion at the join, tabs, actual bare-ESC quit,
control-byte Home/End, loading/appending, an empty file, a failed save and a
60-line scrolling document with the required final empty line.

Numeric aliases 1/4/5/6 are tested as **control bytes**, consistently with 17/19;
printable digits must remain ordinary insertions under the stated 32..126 rule.
The old test's optional ESC-[5~/6~ sequences are not imposed as an unstated
requirement. The scrolling case checks that the final screen cursor is within
the PTY window. These are concrete behavior checks, not exhaustive correctness
proofs for every file, terminal size, fragmented escape sequence or error path.

A separately handwritten Python reference editor, explicitly **not model output**,
passed all **11/11** checks (`pty-calibration.log`, `pty-calibration/results.json`).
The reference uses the literal expected outcomes and is only an instrument
calibration. It is never substituted for or used to repair generated C++.

## Interim observations

Live `/status` samples confirm max_tokens 65536, no thinking budget and zero
reasoning recoveries. At 1199.8 seconds the server reported 21192 generated
tokens; at 2585.2 seconds, 39354; at 3360.4 seconds, 47653. All were still in
the thinking phase, with no final-answer text. The original prompt is 569 tokens.

This is already a passage loop, not just lengthy productive planning. One
partial-text analysis at 165391 reasoning characters found 275 paragraphs of
at least 120 characters, but only 26 distinct paragraphs after whitespace
normalization. Several long paragraphs occurred 42 or 43 times. These paragraph
counts are not token counts. That snapshot had no code fences and no `int main`
occurrence. Exact data is retained in `reasoning-partial-analysis.json`.

Do not infer a new arithmetic bug solely from a generated loop. This is a cold
single request under the recorded sampling, with recovery deliberately disabled;
long-history aligned comparison against a reference engine is not part of this
one-request experiment.

## Final result

The one request completed its full finite allowance on 4 November 2026:

| Measurement | Result |
| --- | --- |
| Prompt tokens | 569 |
| Cached prompt tokens | 0 |
| Generated tokens | **65536 of 65536 allowed** |
| Total prompt + generated tokens | 66105 |
| Request wall time | **5332.25 seconds / 88.87 minutes**, excluding server load |
| Finish reason | `length` |
| Stream completed normally | yes, `[DONE]` received |
| Reasoning text | **271403 characters** |
| Final-answer text | **0 characters** |
| Compiler / generated-editor PTY execution | **not run: no final program** |

This was not a transport failure, cancellation or early guard stop. The server
log ends `done: 65536 tokens ... (length, cancel=False)` and explicitly reports
that the reply reached max tokens while still thinking. The live status samples
showed no thinking budget and zero recoveries. There is no `failure.json`, and
no compile result was created. The original serve config and all engine/model
files were not modified by this experiment.

Final text analysis (`reasoning-analysis.json`): 451 paragraphs of at least 120
characters, only 26 distinct after whitespace normalization. Four long
paragraphs repeat 72 times each and two others 71 times. The full trace has no
code fence and no `int main` occurrence, so this is not a complete program
hidden inside the reasoning either. Paragraph counts are not token counts.
The loop repeatedly revisits blocking versus nonblocking input and bare-ESC
handling without progressing to an implementation.

**Conclusion for this run:** increasing the output allowance to 65536 tokens
with these settings did not yield a correct answer; it yielded no final answer
at all. Record this as an answer-delivery/passage-loop failure, **not** a C++
compilation failure. More budget alone did not solve the observed problem.

This is one sampled trajectory at the user's temperature 0.2 configuration,
not a statement about all seeds, model-recommended sampling, other reasoning
efforts or engine implementations. The cold prompt excludes cross-request
prefix reuse as the mechanism here, but no long-history aligned reference
comparison was performed to attribute this particular loop to model arithmetic,
backend differences or sampling. No new inference bug is proved by the loop.

## Completion and reproducibility

Artifacts contain `request.json`, experiment-only `serve-config.json`,
`provenance.json`, `submission-start.json`, `initial-status.json`, timestamped
live status samples, `response.sse`, `reasoning.txt`, empty `answer.txt`,
`result.json`, `reasoning-analysis.json`, `run.log` and `server.log`.
The executable fingerprint and source commit were recorded before the POST.

The runner's own server stopped after completion; no test server or runner
process remained, and GPU usage returned to **0 MiB**. Exactly one model request
was sent, with no continuation, repair or sampling change mid-run. The separate
handwritten reference calibration passed 11/11 tests; those are **not model test
passes**, since no generated editor existed to execute. No further generation
or production-default change was performed as part of this experiment.

Afterwards, the user separately approved temperature 1.0 as the served default:
[scope and tests](kolibri-temperature-default.md). All recorded request/config
artifacts for the large-budget run remain unchanged at temperature 0.2.
