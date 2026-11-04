# Kolibri degeneration investigation

## Status

Root cause reproduced: prefix reuse read sliding-window KV rows overwritten by
an earlier reply. A conservative cold fallback removes the observed collapse.
Boundary testing also found incomplete prefill-chunk alignment. Both corrections
now pass the six-case full-logit gate and the original served reproduction.
The existing close-and-continue retry remains only a mitigation, not the cause
of the quality improvement. This fixes the reproduced cache corruption; it does
not establish universal model quality or parity with llama.cpp.

Work is in `/data2/develop/kolibri-sram`, branch `kolibri-sram-activation`,
starting at `35b45422`. The separate FP16 and native-Q4/Q6 worktrees and evidence
are not modified.

## Reported symptoms and evidence limits

Earlier runs reported repeated punctuation, repeated words and incoherent
reasoning under Strata, including the `code-fn` and `plan-3` greedy probes in
`/data4/kolibri-sram/greedy_cmp.py`. Existing artifacts are under
`/data4/kolibri-sram/out`. Those reports are leads, not newly reproduced results.

The prior briefing overstated several conclusions:

- Equal prompt-token **counts** do not prove equal token IDs or templates.
- Two affected prompts out of five do not establish a population failure rate.
- Repetition counts from whitespace splitting are not engine-token counts.
- Divergent generated text does not measure logit error on aligned histories.
- The 4K and 8K runs changed the automatic reasoning budget as well as the cap;
  the cap cannot be dismissed as a cause of differing trajectories.
- A single divergent unseeded run does not establish forward-pass nondeterminism.
- No answer from llama.cpp at the cap is not evidence that Strata has better
  model fidelity; the serving policies differ.

## Hardware and configuration

Existing setup: Tesla T4 16 GB; Kolibri-1 Q4_K_M GGUF at
`/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`.
Strata executable: `build-sram/strata-kolibri`, 14 CPU threads, 131072 context.
Comparison llama.cpp uses CPU MoE offload and 32768 context. These are confounders
until controlled.

## Investigation checklist

- [x] Reproduce the two reported greedy failures; preserve raw responses.
- [x] Trace request sampling defaults and cache/reset boundary; prompt parity with llama.cpp remains unproven.
- [x] Compare default and explicit expert-stream-off regression runs.
- [x] Identify the first failing boundary and test one hypothesis at a time.
- [x] Add a failing regression test before changing implementation.
- [x] Verify the fix, repeatability and unaffected behavior; record limits.

## Accuracy requirements

Follow `../KOLIBRI_ACCURACY_POLICY.md`. Scheduling/layout-only changes require
finite, bit-identical full logits on identical forced histories. Numerical
changes require separate qualification and explicit approval before default
promotion. Do not compensate for a forward-pass bug by increasing penalties.

## Experiment log

### Initial inspection

- GPU reported `Tesla T4, 0 MiB` before experiments.
- Worktree started clean and synchronized with origin at `35b45422`.
- `StrataEngine.sampling_keys` omits temperature when it is zero. Whether this
  really means greedy must be checked against the engine's per-request defaults;
  the previous briefing did not establish that boundary.

### A. Reproduction and reuse-off control

Artifacts: `/data4/kolibri-sram/out/degeneration/{baseline,no-reuse}.json`
contain complete requests and responses; corresponding `*-server.log` files
contain server logs. Same executable, model, prompt order, sampling and caps.
Only `STRATA_KOLIBRI_REUSE=0` differs in the second server process.

| Prompt, 700-token cap | Baseline | Reuse disabled |
| --- | --- | --- |
| short-math | 263 tokens, answer `391`, cached 0 | same 263 tokens, answer `391` |
| short-cpp | 700 tokens, no answer, cached 45 | 583 tokens, coherent answer, cached 0 |
| code-fn | 700 tokens, starts with irrelevant prose, ends `The. The. ...`, cached 45 | 700 tokens, coherent C++ reasoning, capped before answer |
| plan-3 | 700 tokens, repeated prose, ends `a is a ...`, cached 45 | 700 tokens, coherent termios reasoning, capped before answer |

This reproduces the collapse and removes it without changing expert streaming,
penalties or arithmetic. It does not prove every reported loop has this cause.
An initial reuse-off server launch failed because port 8199 was still occupied
while the previous process shut down; no request result came from that failed
launch. The replacement process was checked ready before the control run.

### B. Cause: token-prefix validity confused with KV availability

Static trace, supported by the failing regression below:

- `Sampling smp` is constructed per request; its temperature default is zero.
  Omitting `temperature=0` therefore really is greedy on this engine path.
- `serve()` finds the longest common token prefix in `live`, capped by `s.pos`.
- Sliding layers write K/V into `position % n_swa`; this model has 513 slots.
- A 700-token reply overwrites the early rows. Matching the first 45 IDs of a
  later request does not restore those rows. Rewinding to 45 uses unrelated KV.
- Chunk alignment can rewind even an otherwise valid extension/replay after the
  ring has wrapped, likewise requiring rows no longer present.

### C. Failing regression before the fix

Added `tools/kolibri/degeneration_gate.py`. It uses the existing engine protocol,
fixed prompt IDs, a resetter request for the cold reference and
`STRATA_KOLIBRI_LOGITS_DUMP`. All logits are required finite; the final forced
prompt position is compared byte-for-byte, not on divergent generated histories.

Command (from the worktree):

```sh
/data4/kolibri-gpu-streaming/venv/bin/python tools/kolibri/degeneration_gate.py \
  --bin /data2/develop/kolibri-sram/build-sram/strata-kolibri \
  --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --output /data4/kolibri-sram/out/degeneration/gate-before
```

**Failed as expected, exit 1:**

| Case | Cold first ID | Reused first ID | Claimed reuse | Full finite logits equal? |
| --- | --- | --- | --- | --- |
| 45-ID common prefix after consuming 640 IDs | 589 | 58 | 45 | no |
| replay of a 640-ID prompt | 58 | 58 | 512 | no |

Raw dumps and results: `gate-before/{logits.bin,results.json}`; console output:
`gate-before.log`. Matching a generated ID in the second case hid the defect.

### D. Minimal correction under test

After the existing chunk-alignment calculation, reject reuse if it rewinds
(`reuse < s.pos`) and the previous state has exceeded the sliding capacity
(`s.pos > n_swa`). Run the existing cold reset/prefill instead. Appends without
rewind and reuse before any ring overwrite remain available. This is a
conservative cache-validity correction, not a numerical change. No sampling,
expert arithmetic or recovery defaults are changed. Full-logit verification and
serving comparisons follow below.

### E. First correction: numerical and serving checks

The initial two-case gate passed after rebuilding CUDA, with exact cold logits
and zero claimed reuse for both unsafe cases (`gate-after`). All six unaffected
cold/resetter records in the before/after dumps stayed byte-identical (128000
finite logits per record). The two broken records previously differed from cold
by maximum absolute logit errors of 12.0450 and 5.7824 respectively.

Expanded full-chunk controls (`gate-final` and `gate-stream-off`) also passed:
513-position prefix extension and replay both retained 512 cached positions and
matched full cold logits exactly. Both runs' complete 18-record dumps matched.
This tests the default configuration and `STRATA_KOLIBRI_EXPERT_STREAM=0`, not
universal parity between arbitrary expert kernels. The protocol gate logs report
a CPU-pool fallback where streaming selections exceed the available slots.

Fresh llama.cpp comparison: `fixed-comparison.json` contains all five prompts,
complete requests/responses, both engines. Identical greedy sampling and the
same prompt sequence/caps as reproduction; original context-size difference
remains. `code-fn` and `plan-3` now have coherent on-topic reasoning in Strata,
as in llama.cpp. At 700 tokens both engines still cap those replies before an
answer. Strata answers `391` and `17:05`; llama.cpp answers `391` and caps the
time problem. This is not a claim that their logits or reasoning must match.

At a 2048-token cap, alternate `code-fn` and `plan-3` three times each, with
explicit seed 42 (still greedy). All three message dictionaries per prompt are
identical, including reasoning. `code-fn` stops at 1085 generated tokens with a
normal summation function; `plan-3` stops at 974 with `ICANON`, `ECHO`, `ISIG`.
Artifact: `fixed-repeats.json`.

Longer coding check, using the original notepad request and user sampling
(temperature 0.2, top_p 0.9, top_k 40, min_p 0.05, repetition penalty 1.05):

| Cap | Generated tokens | Wall time | Result |
| --- | --- | --- | --- |
| 4000 | 3925, stop | 225.8 s | Complete fenced 267-line file; compiles; fails 7/8 PTY behavior cases |
| 8192 | 7959, stop | 509.1 s | Complete fenced 267-line file; six compiler errors; not executed |

Artifacts: `fixed-notepad.json`, `notepad-{4000,8192}.cpp`, compile logs and the
4000 PTY log. Across these 13 served requests (five probes, six repeats, two
notepad requests), the server recorded **zero** repeat guards or close/retry
recoveries, and five ordinary thinking-budget wraps. No rescued mid-file
answer was needed. Model coding correctness remains limited. Longer valid
outputs take longer; no speedup is claimed over the earlier corrupted replies.
These serving checks used the first correction; final alignment changes require
a fresh serving confirmation below.

### F. Boundary discovery: skipped zero alignment

An expanded control initially used a 257-ID prefix, incorrectly assuming the
current default chunk was 256. Inspection of `pf_chunk_at` confirmed the default
is **512**. The old alignment code computed boundary zero but refused to apply
it (`aligned > 0`); a shorter common prefix could also remain unaligned when
the previous request's final chunk started after it. Thus old partial-chunk or
decode-written rows were reused instead of recomputing them in the cold shape.

Do not dismiss this as harmless rounding: the 257-ID extension produced the
same first ID but all 128000 logits differed, maximum absolute error 0.162802.
The evidence is retained in `gate-boundaries`, rather than replacing the failed
measurement. The gate's valid reuse control was corrected to the actual default
512-row boundary, and the 257-ID case was retained as a separate regression.

Before rebuilding the alignment correction, `gate-alignment-before` failed on
both one-token-prefix and 257-ID-prefix extensions, while all four other cases
passed. The one-token case exposed the existing `prev_end > 1` exclusion too.

Correction: align the minimum of the matched reuse boundary and previous final
chunk boundary, including zero; allow a one-position previous state to align to
zero as well. This repairs recomputation boundaries, not model arithmetic. It
combines with the ring-retention check; no new kernels or sampling changes.

### G. Tests and setup failures recorded so far

- CUDA Release build, architecture 75: passed, with existing unused-variable
  and unused-function compiler warnings (`build.log`).
- Full serve suite: **1253 tests OK, 31 skipped**, 683.059 s (`serve-tests.log`).
  Existing `ResourceWarning` messages appear from telemetry file reads.
- Kolibri CTest selection: **13 passed, 1 skipped** (`kolibri_res_top_zero_test`),
  22.00 s (`cpp-tests.log`); test targets built first.
- Python Kolibri tools: **158 passed, 7 skipped** with `KOLIBRI_BIN` set to the
  tested `build-sram/strata-kolibri` (`python-tests-final.log`).
- First Python collection failed for missing numpy in four modules:
  `test_klcmp`, `test_layer_delta`, `test_phase2a_run`, `test_qualify`.
  Installed numpy 2.4.6 into the existing GPU-streaming venv, not a protected
  FP16 environment. Then three `test_decode_guards` cases failed because their
  default `build-mmq/strata-kolibri` did not exist: unsupported-config fallback,
  explicit-zero restore, malformed-wait-budget rejection. Setting `KOLIBRI_BIN`
  to the actual build resolved these setup failures without altering the tests.
- The older `reuse_gate.py` initially failed **accounting only**: it demanded
  reuse after a wrapped-ring alignment rewind. Token comparisons and
  cancellation passed. Its accounting now explicitly requires the safety cold
  rebuild (or a non-rewinding append with alignment explicitly off). The new
  full-logit gate independently requires retained safe reuse, so disabling all
  reuse cannot silently pass. Updated old gate passed (`reuse-gate-final.log`).
  Also corrected its stale strict-prefix-default assumption; no prefill hash,
  pool expectation or numerical tolerance was changed.

### H. Final-source verification (4 November 2026)

Final changes are limited to request-cache handling in
`src/kolibri/kolibri_engine.cu`, the new `degeneration_gate.py`, updated
`reuse_gate.py` accounting and this log. No sampling defaults, kernels, model
weights, prefill hash expectations or protected FP16 artifacts were changed.

**Passed on the final rebuilt CUDA executable:**

- CUDA Release / SM75 rebuild (`build-final.log`).
- Six-case gate, default configuration (`gate-complete`) and explicit
  `STRATA_KOLIBRI_EXPERT_STREAM=0` (`gate-complete-stream-off`). Both passed;
  complete 32-record dumps are byte-identical between these two runs. Each
  compared last-prompt-position vector contains 128000 finite FP32 logits.
- Original reuse/cancellation gate (`reuse-gate-complete.log`). Its long-prefix
  cases rebuild safely and retain the cold generated IDs; cancellation passes.
- Kolibri CTest selection: **13 passed, 1 skipped**, 21.85 s
  (`cpp-tests-complete.log`).
- Python tools suite: **158 passed, 7 skipped**, 6.71 s
  (`python-tests-complete.log`).

| Final full-logit gate case | Outcome | Cached positions |
| --- | --- | --- |
| 45-ID common prefix after ring wrap | bit-identical to cold | 0 |
| 640-ID replay after ring wrap | bit-identical to cold | 0 |
| one-token prefix extension | bit-identical to cold | 0 |
| 257-ID partial-chunk extension | bit-identical to cold | 0 |
| 513-ID prefix extension, ring not overwritten | bit-identical to cold | 512 |
| 513-ID replay, ring not overwritten | bit-identical to cold | 512 |

Fresh final-source served replay (`complete-probes.json`,
`complete-server.log`) used the same five requests/order as the earlier
head-to-head. All four prompts shared with the original reuse-disabled control
produced **identical message dictionaries** to that control, including reasoning.
No repetition guard or close/retry rescue fired.

| Prompt | Final Strata tokens / finish | Result |
| --- | --- | --- |
| short-math | 263 / stop | `391` |
| short-cpp | 583 / stop | coherent definition of `std::string::npos` |
| code-fn | 700 / length | coherent C++ reasoning, capped before answer |
| plan-3 | 700 / length | coherent termios reasoning, capped before answer |
| think-math | 540 / stop | `17:05` |

The earlier six longer-cap repeats and notepad checks in section E ran on the
ring-only correction, not the final alignment correction; they were not repeated
on the final binary. The full serve suite is unchanged Python code and was run
during this investigation, not rerun after the final C++ rebuild.

Provenance: `provenance.json`, `tested-final.patch`, plus the source and harness
in this worktree. Final executable SHA256:
`1da3371364916ac932f075d34fe54888297944a0d6a50868fa5594a53c9bacce`.
Driver 580.196.07; Intel Xeon E5-2690 v4; model file 47,454,113,472 bytes.
All investigation servers stopped; GPU reported 0 MiB used afterwards.

## Costs, untested cases and remaining questions

- Correctness costs cached-prefill speed: after ring wrap, an aligned rewind
  now rebuilds the prompt. The 3137-ID gate reports reading all 3137 IDs instead
  of claiming stale reuse. Restoring safe long-prefix reuse would require
  retained snapshots or a different cache design, not weakening this guard.
- No complete llama.cpp-vs-Strata aligned-distribution fidelity campaign was
  done. Prompt-ID parity between the engines is still unproven. Their differing
  greedy reasoning is not itself an error measurement.
- No universal degeneration rate is claimed. This reproduces and removes the
  cache-induced collapse; repetition can still occur naturally in generation.
- No non-greedy seed sweep, arbitrary chunk-cap sweep, long-context accuracy
  campaign, reset/cancel race stress or reference-engine comparison was done.
  The old cancellation gate passed, but is not exhaustive cancellation testing.
- Generated notepad code remains functionally deficient. Removing degeneration
  does not fix model-generated program bugs.
- Only T4 / Q4_K_M runtime tested. HIP and SYCL builds were **not run**: hipcc,
  icpx and dpcpp are unavailable here. No qualification on those backends is
  implied. The entire 217-test CTest project was not built/run; the Kolibri
  selection was built and run. Other architectures and quantizations untested.
- At the close of the cache investigation, the work was local on
  `kolibri-sram-activation` and origin was at `35b45422`. It was subsequently
  committed as `6c2ef8e4`, merged with newer remote work and pushed at `9ba42dce`.
  The next investigation is [generated-code quality](generated-code-quality.md).

## Reproduction commands

From `/data2/develop/kolibri-sram` (choose unused output directory names):

```sh
cmake --build build-sram --target strata-kolibri -j 4

/data4/kolibri-gpu-streaming/venv/bin/python tools/kolibri/degeneration_gate.py \
  --bin "$PWD/build-sram/strata-kolibri" \
  --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --output /data4/kolibri-sram/out/degeneration/reproduce-default

STRATA_KOLIBRI_EXPERT_STREAM=0 \
/data4/kolibri-gpu-streaming/venv/bin/python tools/kolibri/degeneration_gate.py \
  --bin "$PWD/build-sram/strata-kolibri" \
  --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --output /data4/kolibri-sram/out/degeneration/reproduce-stream-off

KOLIBRI_BIN="$PWD/build-sram/strata-kolibri" \
  /data4/kolibri-gpu-streaming/venv/bin/python -m pytest tools/kolibri -q

ctest --test-dir build-sram -R '^kolibri_' --output-on-failure
```

The CTest command requires the Kolibri test executables built first (see
`cpp-test-build.log`). Raw measurement artifacts are local, not checked into git;
the new gate is included in this source change and uses the existing tracked
prompt-ID fixtures.
