# Generated-code quality: engine or model?

## Status and scope

Investigation completed on 4 November 2026, after committing the cache corrections
as `6c2ef8e4` and merging the newer remote branch. Push succeeded at `9ba42dce` on
`origin/kolibri-sram-activation`. No remote work was overwritten.

**Conclusion:** the reproduced residual semantic/API errors are demonstrably
model-side reliability failures under these prompts: they also occur in
llama.cpp and persist at Q8 precision. They are not explained solely by Strata
or by Q4 quantization. Numerical differences between engines remain real, and
this investigation does not exonerate every possible inference path or attribute
every extra Strata error. No new engine or sampling change was made.

The question is whether the remaining C++ defects come from inference errors,
serving policies, the model/quantization, or the evaluation itself. A same-GGUF
comparison can identify Strata-specific regressions, but cannot separate base
model limitations from quantization without a higher-precision reference.

## Controls and planned measurements

- Rebuild the merged CUDA source. Same T4, Q4_K_M model file, CPU thread count
  and 32768 context for both engines.
- First inspect the existing generated sources and PTY oracle. Do not treat
  compiler diagnostic count as a count of independent defects.
- Compare direct engine generation with llama.cpp using **identical token IDs**,
  greedy sampling, cold requests and no server reasoning-budget intervention.
- Use the model's own template with thinking disabled for complete-answer
  comparisons; retain this limitation when interpreting results.
- Compile unedited output and run deterministic behavior checks. Compare a
  complete notepad task and smaller isolated coding tasks.
- Where text diverges, inspect complete next-token distributions on **identical
  forced histories**, including prefixes relevant to a concrete code defect.
  Generated-history divergence alone does not establish a faulty forward pass.
- Report observations, not population quality rates. No numerical changes or
  default promotions are planned by this investigation.

Raw artifacts will be under `/data4/kolibri-sram/out/code-quality/`.

## Initial source inspection (not yet newly reproduced)

The earlier 4000-token-cap source sets `O_NONBLOCK`, then immediately exits the
input loop on `read(...) <= 0`. This likely explains exiting before the PTY test
sends its first key. It also leaves IXON enabled, has incorrect escape-sequence
length/byte checks, incorrect Home/End semantics, and does not erase the character
in its backspace branch. These are static findings until individually tested.

The earlier 8192-token-cap source has several distinct C++ defects: invalid
`string::insert(col, char)`, unscoped initialization across switch labels, and
an undefined `redraw()`. Its six compiler diagnostics are not six independent
errors.

The old oracle also needs audit: its long-document expected bytes may omit the
final empty buffer line, even though the task says every buffer line is saved
with a newline. A natural empty-document split leaves a final empty line after
60 Enter keys. The bare-ESC case currently ignores the saved bytes and does not
prove the editor was alive until ESC. Do not use that case alone as a quality
pass.

## 1. Audit results

`oracle-audit.json` records that the old 4K generated editor **exits with code 0
before any keyboard input**, measured 0.3 seconds after start. The previous
bare-ESC pass was therefore a false positive. A literal simulation of the
specified line splitting and save rule gives 61 buffer lines and 421 bytes after
60 Enter keys, versus 420 bytes in the old long-document oracle. That expectation
would reject the specified final empty line. The earlier 7/8 failure count is
not a reliable independent defect count.

A second calibration issue was caught in this investigation's first raw-mode
micro-oracle: it required the IXON bit to be cleared. Linux also permits disabling
VSTART and VSTOP with zero; that implementation correctly delivers Ctrl+S/Q with
IXON still set. The initial oracle incorrectly rejected Strata's valid alternative.
Replaced the mask-only assertion with actual byte delivery through a PTY. Both
known-correct alternatives pass, while leaving VSTART=17/VSTOP=19 with IXON on
fails. The initial data is retained in `*-cases-initial.json` and `paired/`;
corrected evaluations are in `audited-evaluation/`. No model output was edited.

## 2. Direct, matched-ID greedy comparison

Rebuilt merged source at `9ba42dce` (`build.log`). Used
`tools/kolibri/code_quality_probe.py` and `code_quality_cases.json`. Strata ran
its direct `GEN` protocol with reuse disabled, default greedy sampling and a
32768 context. llama.cpp used `/completion`, the same 32768 context and 14 CPU
threads, temperature zero, top_k 1, no repetition penalties and cache_prompt
false. The same local GGUF was used on both. No answer reserve, reasoning-budget
wrap or retry was involved. For **all five tasks**, local tokenizer IDs exactly
matched llama.cpp `/tokenize`, and that identical ID list was supplied to both
engines. This is stronger than the earlier prompt-count comparison.

The template explicitly disabled thinking for this first comparison. The raw
responses and code are retained under `paired/`; the corrected micro-oracles
were applied to the same unedited generated text under `audited-evaluation/`.

| Targeted task | llama.cpp | Strata |
| --- | --- | --- |
| Insert a character at a string index | compiles, behavior passes | compiles, behavior passes; identical code |
| Split a line and backspace-join it | compiles, behavior fails | compiles, same behavior failure; identical code |
| Raw-mode Ctrl+S/Q delivery | compiles, PTY delivery fails | compiles, PTY delivery passes |
| Nonblocking read: EAGAIN/EOF/unsigned byte | compiles, behavior passes | compiles, behavior passes; identical code |
| Full notepad, 6000-token generation cap | 149 source lines, does not compile | 180 source lines, does not compile |

Specific common defects:

- Both line-edit functions set the cursor to the **combined line's end** after
  joining, instead of the join point. They also incorrectly make Enter at
  row=column=0 a no-op. The observed assertion fails at the join-position check.
- Both full notepads call **`std::getline(FILE*, std::string&)`**, which is not a
  C++ overload. The task allowed C stdio headers, not a file-stream abstraction.
- Strata's full notepad additionally captures `lines` in a lambda before its
  declaration, and uses invalid iterator/count/string overload combinations.
  Seven compiler diagnostic lines are not seven independent defects.
- llama.cpp's raw-mode function preserves active flow control on Ctrl+S/Q;
  Strata disables the flow-control characters and passes the actual PTY check.

These results already refute attributing every remaining code defect to Strata:
identical incorrect functions and an identical invalid API choice also appear
under llama.cpp. They do **not** prove Strata's forward pass is universally
faithful, or that its extra notepad errors are inherently unavoidable. Only a
small descriptive sample was tested, not a statistically qualified benchmark.

Four independently written reference functions compiled and passed their own
oracles before interpreting model results. The diagnostic instrument tests also
pass (14 tests), including actual PTY acceptance of the two valid raw-mode
implementations and rejection of intercepted Ctrl+S/Q (`instrument-tests.log`).

## 3. Thinking enabled, without a forced budget closure

Repeated the line-edit and raw-mode tasks with `enable_thinking=True` and the
same template's `reasoning_effort=low`. Both engines received identical IDs,
were greedy, cold and capped at 3072 generated tokens. No server force-close
or rescue was used. Artifacts: `thinking/`, `thinking-cases.json`, `thinking.log`.

| Task | llama.cpp Q4 | Strata Q4 |
| --- | --- | --- |
| Line editing | cap reached inside reasoning; no answer | cap reached inside reasoning; no answer |
| Raw mode | final function does not compile: invented `termios.vmin` and `.vtime` members | compiles but puts `IXON` in the `c_lflag` mask, leaving actual input flow control enabled; fails PTY delivery |

The model is not made reliable merely by enabling thinking. The no-answer cases
are token-cap observations, not scored as incorrect C++ programs. The raw-mode
answers contain real API/field errors without any serving-policy intervention.
This still does not establish behavior at every larger budget or sampling setting.

## 4. Aligned complete distributions, Q4 Strata versus Q4 llama.cpp

Used `tools/kolibri/code_quality_distributions.py`. Nine cold forced histories:
five initial task prompts, two first generated divergence prefixes and two
prefixes near a code defect. Each comparison covers **all 128000 token IDs**.
Strata's raw finite FP32 logits are converted to log-softmax; llama.cpp provides
pre-sampling full-vocabulary log probabilities. Histories are identical within
each comparison, not the engines' separate divergent continuations. Raw llama
responses are retained as compressed JSON and Strata vectors as a binary dump
under `distributions/`; no top-k tail was dropped.

| History | KL(llama.cpp || Strata), nats | Total variation | Cold top-1 agrees? |
| --- | --- | --- | --- |
| insert-char start | 0.026580 | 0.061949 | yes |
| line-edit start | 0.000647 | 0.010643 | yes |
| before final join-cursor line (next token is indentation) | 0.000044 | 0.002094 | yes |
| raw-mode start | 0.032260 | 0.081372 | yes |
| raw-mode first generation divergence prefix | 0.010476 | 0.047515 | yes |
| nonblocking-read start | 0.000291 | 0.004963 | yes |
| notepad start | 0.000855 | 0.006785 | yes |
| notepad first generation divergence prefix | 0.006727 | 0.043482 | yes |
| notepad before invalid getline call | 0.000737 | 0.002190 | yes |

These are **descriptive numbers, not an approved numerical-fidelity gate**.
The distributions are not bit-identical; top-1 agreement on nine positions does
not prove full engine correctness or equal task quality. One repeated cold
Strata start vector was finite and byte-identical to its first read.

The strongest defect-specific observation: with the same forced notepad prefix
ending in `FILE* f ... while (`, llama.cpp assigns probability **0.994949** to
`std` and Strata **0.997122**. Both actual programs go on to the invalid
`std::getline(f, line)` call. This is shared model output, not a Strata-only
parser corruption. The indentation point is reported explicitly rather than
presenting it as a measurement of a semantic token choice.

An important limitation surfaced too: the raw-mode cold forced prefix chooses
`0` in both engines (probability 0.537241 in llama.cpp and 0.510752 in Strata),
whereas the earlier continuing llama.cpp decode chose `17`. A cold prefill is not
the same execution geometry as replaying a decode trajectory. Do not treat this
as established nondeterminism or infer that all generation paths now agree.

## 5. Higher-precision control

Local Q8_0 shards are available (44.68 + 38.46 GB). Their metadata identifies
Kolibri 1 version 1 and the same Aleph Alpha Kolibri-1-BF16 base as Q4_K_M; this
is not a tensor-by-tensor proof of conversion provenance. The first Q8 shard
was parsed and the llama.cpp server loaded both shards. Metadata is retained in
`model-metadata.json`.

The initial six-minute readiness wait expired while loading continued; the
server subsequently became ready at about **447.9 seconds**, as recorded in
`llama-q8-server.log`. This was a load/readiness delay, not an inference failure.
No model files or protected FP16 artifacts were changed, and nothing downloaded.

Q8_0 llama.cpp then generated the same five tasks on the same prompt IDs, context,
thread count and greedy settings. `--llama-only` avoided loading another Strata
instance. The corrected oracles were used from the outset for this control.
Artifacts: `q8-reference/` and `q8-reference.log`.

| Task | llama.cpp Q4_K_M | Strata Q4_K_M | llama.cpp Q8_0 |
| --- | --- | --- | --- |
| String insertion | pass | pass | pass |
| Line split/backspace join | semantic failure | same semantic failure | same semantic failure |
| Raw-mode Ctrl+S/Q delivery | failure | pass | pass |
| Nonblocking read | pass | pass | pass |
| Full notepad | does not compile, 1 diagnostic | does not compile, 7 diagnostics | does not compile, 6 diagnostics |

The Q8 line-edit code still places the cursor at the combined line's end, not
at the join point. Its notepad still uses `std::getline(FILE*, std::string&)`
and also redeclares several local variables. All three direct notepad replies
end normally below the 6000-token cap: llama.cpp Q4 generates 1408 tokens,
Strata Q4 1683, and llama.cpp Q8 1911. Q8 produces 201 source lines. None of the
three direct notepad programs compiled, so no full-editor functional pass rate
is assigned to these runs. Compiler diagnostic counts are descriptive only.

Q8 does improve the raw-mode example relative to llama.cpp Q4, confirming that
quantization/numerical trajectory can affect individual outcomes. It does not
remove the shared line-edit or full-notepad defects. Q8 is still quantized;
no unquantized BF16/FP16 generation was performed, so a definitive attribution
to the original BF16 weights rather than every possible quantization effect
would overstate this evidence.

The same nine forced histories were also evaluated under Q8 (`q8-distributions/`).
All full-vocabulary comparisons had finite scores and complete ID coverage.
KL(Q8 || Q4 llama.cpp) ranged from **0.000298 to 0.058586 nats**; cold top-1
agreed on all nine. These are not acceptance thresholds. In the fixed notepad
history before the invalid getline call, Q8 still assigns probability
**0.996220** to `std`, versus Q4 llama.cpp 0.994949 and Strata Q4 0.997122.
The shared bad library choice is not a special Strata failure or just a 4-bit
artifact. The full conditional continuations, not the isolated token alone,
establish that the actual generated call is invalid.

## 6. Attribution and what remains unresolved

| Observation | Supported attribution | Not established |
| --- | --- | --- |
| Earlier repetition collapse after cached long replies | engine cache bug; fixed separately | all possible natural repetition eliminated |
| Identical wrong line-edit function in both Q4 engines, same error at Q8 | model/task reliability limitation, not Strata-specific or Q4-only | impossibility of generating correct code with feedback, other prompts or full precision |
| Invalid stdio/iostream API combination in all three direct notepads | shared model output under the same constraints | a universal compile-failure rate |
| Different raw-mode outcomes and different extra compiler errors | numerical/generation-path sensitivity is real | a new structural inference bug responsible for the extra Strata errors |
| Missing answers while thinking at a finite cap | generation/budget issue also present in llama.cpp | no answer at all larger caps |
| Previous bare-ESC pass and long-file expectation | evaluation defects | correctness of the old editor |

There is no evidence here supporting another engine-arithmetic change as a cure
for the demonstrated shared code errors. A compile/test/repair loop is the
practical next control to evaluate; it was **not** run as part of this study.
Do not label the model inherently incapable of programming: several functions
pass, and the failing examples do not measure performance with feedback.

### Limits

- Five targeted tasks, plus two thinking-enabled controls, not a representative
  benchmark or a statistical non-regression qualification. No population rate,
  confidence bound or acceptable numerical-error threshold is asserted.
- Greedy generation was deliberately used for attribution. The user's sampled
  0.2 settings and multiple random seeds were not freshly paired here. GGUF
  metadata recommends temperature 1.0, top_p about 0.97, top_k 128; those settings
  were not evaluated. These results are not a maximum-capability assessment.
- Thinking-disabled comparisons isolate complete answers; the two low-thinking
  controls do not cover every reasoning budget. No forced server close/retry
  was involved in any of these direct comparisons.
- Only nine cold, aligned probability vectors and one same-arm cold replay were
  checked. Not a full decode trajectory, long-context campaign, full-precision
  reference, graph-off A/B or comprehensive race/non-finite investigation.
- T4 only; CUDA rebuilt. No HIP/SYCL runtime or build qualification implied.
- No protected FP16 branches, worktrees or evidence were modified. No model
  downloads, model rewrites, numerical default promotions or penalty changes.

## 7. Validation, provenance and artifacts

- Merged CUDA source rebuilt successfully (`build.log`), with unused-variable
  warnings. Engine behavior source was not changed during this quality study.
- Diagnostic/oracle tests: 14 passed initially (`instrument-tests.log`).
- Full Kolibri Python-tools suite: **181 passed, 7 skipped**, including a final
  rerun after the Linux portability guards: 9.60 s (`tools-tests-final.log`).
- C++ micro-oracles were calibrated with known-correct reference functions.
  Linux PTY checks accept both valid flow-control implementations and reject
  intercepted Ctrl+S/Q. No generated answer was hand-fixed to produce a pass.
- Raw artifacts include complete requests/cases, input IDs, output IDs and text,
  unedited extracted source, compiler logs, failed assertions, dense logits,
  complete probability responses and the model metadata.
- All servers started for this work were stopped; GPU reported 0 MiB used.
- `provenance.json` records binary/template/harness/fixture SHA256 values,
  exact common server arguments, software versions and model paths. Full
  model-weight hashes were not recomputed. Tested engine SHA256:
  `31740ad1aba0cb6f47ae204cbe19be3cdc5f4ea204a0b37fd6b2645bc4ba99a5`.
  llama.cpp reports 0.7.0-dev, build 11693, commit `a30fb4924`; compiler GNU
  12.2.0; Tesla T4 driver 580.196.07.
- The cache fix and remote integration were committed/pushed at `9ba42dce`.
  These subsequent investigation documents, probes and tests accompany the
  user-approved [temperature-default follow-up](kolibri-temperature-default.md).

Investigation source files are `tools/kolibri/code_quality_probe.py`,
`code_quality_distributions.py`, `code_quality_cases.json` and
`test_code_quality_probe.py`. Additional orchestration and Q8-to-Q4 comparison
scripts are preserved beside the local raw artifacts. The initial overstrict
raw-mode oracle and its failed results remain available; corrected results are
reported above rather than erasing the initial mistake.

### Subsequent hosted control

The user subsequently ran the same five prompts in Tesseracted's browser demo,
labelled `Official FP8`, and pasted the responses for local evaluation. Its full
notepad also fails compilation with the identical invalid `std::getline(FILE*,
std::string&)` call, plus a noncapturing lambda referencing local state. Its
line-edit response repeats the cursor-join error. This adds a hosted control,
not an independently verified FP8 fidelity qualification; the hosted template,
sampling, revision and backend remain uncontrolled. Full results, formatting
caveats and the blocked automated attempt are in
[the hosted investigation](tesseracted-code-quality.md).

### Subsequent large-budget Strateu run

The user then requested only the full-program task again with thinking on and a
very large allowance. One cold served Q4 request at the user's temperature 0.2
sampling consumed **65536 generated tokens in 88.87 minutes** without producing
any final-answer text. Automatic thinking wrap-up and recovery were disabled in
an experiment-only config. Long planning paragraphs repeated 71–72 times.
This is a no-answer/passage-loop outcome, not a compilation failure, and does not
by itself attribute the loop to a new engine bug. Full configuration, evidence
and limits: [large-budget notepad](large-budget-notepad.md).

### User-approved temperature default

After the large-budget run, the user requested temperature 1.0 as Kolibri's
served default, with no other sampling change. Explicit request/config values
remain overrides. This is not a claim that loops or coding correctness have
been fixed; the earlier results retain their recorded sampling. Scope and tests:
[temperature default](kolibri-temperature-default.md).

## 8. Reproduce

From `/data2/develop/kolibri-sram`, after starting llama.cpp on 8200 with the Q4
model, `-c 32768 -ngl 99 --n-cpu-moe 50 -t 14`, the model template and a loopback
host (full invocation is in the experiment's server log/provenance):

```sh
/data4/kolibri-gpu-streaming/venv/bin/python tools/kolibri/code_quality_probe.py \
  --bin "$PWD/build-sram/strata-kolibri" \
  --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --tokenizer /data4/kolibri-sram/servepack/tokenizer \
  --cases tools/kolibri/code_quality_cases.json \
  --output /data4/kolibri-sram/out/code-quality/reproduce-paired

/data4/kolibri-gpu-streaming/venv/bin/python tools/kolibri/code_quality_distributions.py \
  --bin "$PWD/build-sram/strata-kolibri" \
  --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --tokenizer /data4/kolibri-sram/servepack/tokenizer \
  --pairs /data4/kolibri-sram/out/code-quality/reproduce-paired \
  --output /data4/kolibri-sram/out/code-quality/reproduce-distributions

KOLIBRI_BIN="$PWD/build-sram/strata-kolibri" \
  /data4/kolibri-gpu-streaming/venv/bin/python -m pytest tools/kolibri -q
```

Use new output directory names. For the Q8 control, stop the Q4 server, start
llama.cpp on 8201 with
`/data4/models/Kolibri-1-Q8_0/Kolibri-1-Q8_0-00001-of-00002.gguf`, then run the
first command with that `--model`, `--llama http://127.0.0.1:8201`,
`--llama-only` and a different output directory. The second shard loads
automatically. For thinking controls, use copies of the line-edit and raw-mode
case records with `enable_thinking: true`, `reasoning_effort: "low"` and
`cap: 3072`; the resulting case file is retained as `thinking-cases.json`.
