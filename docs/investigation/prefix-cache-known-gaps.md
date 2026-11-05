# Prefix-cache stress findings: what fails, and the RED tests for it

**Historical RED baseline (`69741cf9`).** F1–F4 and all six RED assertions are now
fixed; see [the RED-to-GREEN qualification](prefix-cache-gap-fixes.md). The values
below are retained as the original evidence, not current failure claims. F5 is a
single-live-history limit, not a demonstrated failure to reuse matching first
prompts across unrelated clients.

Date 2026-11-04. Branch `fix/kolibri-exact-prefix-cache`, engine commit `301352d2`
plus the test-only additions in this change. Build `build-cache-cuda/strata-kolibri`,
SHA256 `8eba03b8f98c6502df7ebfcd8bd437d285ef6b5eb2cf180701b0907a36b65bc8`.
Tesla T4 16 GiB, driver 580.196.07, two Xeon E5-2690 v4, Kolibri-1 Q4_K_M, 14 lanes,
128 MiB checkpoint budget unless stated. The service was stopped for the campaign;
every gate starts and stops its own process. Raw artifacts:
`/data4/kolibri-sram/out/exact-prefix-cache/stress-*`, `known-gaps-engine/`.

Accuracy held everywhere. Each of the **28 engine-level comparisons** matched a cold
read of the *same* forced IDs, and each of the **23 comparable HTTP steps** matched
the cache-off arm's logits for the same request: byte-identical finite
128000-value vectors in every case. What follows are reuse and cancellation
failures: correct answers, paid for twice, or a cancelled prompt read that will not
stop.

## What ran

| Gate | Cases | Result |
| --- | --- | --- |
| `prefix_cache_stress_gate.py --group basic` | 16 | all exact; 11 hit, 2 cold by design, 2 gaps |
| `prefix_cache_stress_gate.py --group prefill` | 4 | all exact; abort recovers, checkpoints kept |
| `prefix_cache_stress_gate.py --group token` | 4 | all exact; 3 lose the held state (gap F3) |
| `prefix_cache_stress_gate.py --group budget --cache-mb 48` | 4 | all exact; 2560/3072 reuse from one frame |
| `prefix_cache_http_stress_gate.py` | 26 HTTP steps | all exact; one gap (F3) |
| `known_gaps.py` (`--store`, `--engine`) | 6 | **RED by design**, see below |

## Failure list

### F1 - a shorter exact prefix reuses nothing

Held history 3072 ids; follow-up 600 ids, an exact prefix of it. Reuse **0**
(`truncated-target`, exactness true). A cold read of the follow-up is ~4 s at the
measured ~150 tok/s; the held boundaries at 512 and 1024 were evicted by the
newest-first FIFO (three frames fit), so nothing at or below 600 survived.

A client re-sends a shorter transcript when it regenerates a turn with fewer
messages, drops the last exchange, or applies a summary. That request is an exact
prefix and today pays full price.

### F2 - an edit in the middle reuses nothing

Held history 3072 ids; the follow-up edits one id at position 1500 and appends.
Reuse **0** (`middle-edit`, exactness true). Same cause as F1: the FIFO kept
1536/2048/2560, and the edit needs a boundary at or below 1500, i.e. 1024, which is
gone. An edit near the end (position 2800) does hit, which is the common agent
case; an edit of an early instruction or a retried early tool result does not.

### F3 - a STOP during decoding discards the held history and every checkpoint

Prefix held, request extended by 8 ids, STOP after 3 generated tokens
(`[259, 735, 12416]`, `finish=cancel`). The next turn, whose prompt is the prefix
plus exactly those three tokens, reuses **0** (`token-stop-then-continue`,
`token-stop-after-hit`, `token-stop-repeated`, all exactness true). The engine
reports `DEC_TOKEN_GRAPH cancelled: graph drained; token discarded`, sets its
abort flag, and the reset path clears the KV and `prefix_cache`.

This also reproduces over HTTP: a streamed request closed after the first real
token delta (1.99 s in the cache arm, 17.81 s in the cache-off arm because the
prompt was read first), then the identical prompt re-sent, reused **0**
(`abort-token-replay`, exactness true, and the only failure in the HTTP gate).

It is a race, and the race decides how much work survives: an earlier HTTP run
closed the socket a few milliseconds before the next decode step and that replay
reused 2560. Both outcomes are exact, which is why no accuracy gate caught it; the
cost is a full re-prefill after every Stop, disconnect or aborted tool call.

### F4 - a STOP during the prompt read does not stop the read

A 9216-id prompt, STOP requested after the first `PP` progress line: all **18 of
18** chunks were still read (`prompt_read_continued: true`, `finish=cancel`). The
STOP is honoured only around the first decoded token; `serve()` has no
cancellation check inside the prefill loop. On this box that is ~60 s of GPU and
pool work spent after the client left, and at a 32K prompt roughly 3.5 minutes.

The checkpoints built by that aborted read are valid and are reused afterwards
(`reuse 8704` on the following request, exact), so the work is not always wasted -
but a client that disconnects pays for the whole prompt before anything can reuse
it, and there is no prompt-only mode to stop cleanly.

### F5 - interleaved clients get no reuse (documented limit, measured)

Two different clients' long prompts run at the same time: both report
`cache_n 0`, both exact. The store follows one live ID history and branches reset
it, so a second client's prompt always rebuilds cold. This is the documented
single-history limit, recorded here because a coding agent with a subagent or a
second tab hits it immediately; it is not a correctness defect.

## Test-design corrections (not product failures)

Recording these because the first runs reported them as failures:

- `boundary-512` reused **512** and was scored against a hard 1024 threshold. 512
  is the right answer: one completed canonical chunk. The gate now takes the
  expected minimum per case.
- `token-stop-branch` reused **3072** and was expected to be cold. A branch that
  diverges after position 3072 may legitimately reuse the boundary at 3072. The
  gate now demands only exactness plus "reuse never exceeds the true common
  prefix" for such cases.
- The prefill gate called `engine.run(...)[0]` on a dict, and the HTTP gate asked
  for the first SSE frame containing `"content"`, which matched the empty role
  delta: that aborted during the prompt read instead of during decoding. Both are
  fixed; the `abort-token` measurement above is from the corrected run.
- The HTTP gate's rendered-ID parity compared only the rows that carried an
  explicit ID list (2 of 25). All 25 rendered prompts were compared offline from
  the per-arm dumps and are identical (`stress-http-v2/prompt-id-parity-offline.json`),
  and the gate now records IDs for every completed call.

## RED tests

`tools/kolibri/known_gaps.py` asserts the behaviour we want and therefore exits 1
while the gaps stand. It is deliberately outside the passing suites.

```sh
python tools/kolibri/known_gaps.py --store      # no GPU, ~1 s
python tools/kolibri/known_gaps.py --engine --output /tmp/kolibri-gaps-$(date +%s)
```

Current output (exit 1):

```
RED    shorter-history-keeps-an-older-boundary  (a 600-id follow-up found no boundary at or below 600)
RED    edit-in-the-middle-keeps-the-boundary-below-it  (an edit at 1500 kept no boundary at or below 1500)
RED    truncated-target  (reused 0 tokens, exact=True)
RED    middle-edit  (reused 0 tokens, exact=True)
RED    token-abort-continue  (produced [259, 735, 12416] before cancel, next turn reused 0, exact=True)
RED    prefill-stop-interrupts-read  (STOP after PP 1, 18 of 18 chunks still read, finish=cancel)
```

The store-level cases are the minimal reproduction of F1/F2 in
`tests/kolibri/prefix_cache_gaps_test.cpp` (three-frame budget, captures at
512..2560, then `best_position(600)` and an edit at 1500). They assert retention
of a boundary below the new prefix; an LRU-plus-oldest policy, a spread of
retained boundaries, or a size-aware replacement would satisfy them. No engine
change is attempted here.

When a gap is closed, its line turns GREEN and the runner exits 0; remove the
corresponding entry from `GAPS` and update this document and `docs/KOLIBRI1.md`.
Exactness stays mandatory in the same run, so a "fix" that raises reuse while
breaking the cold comparison still fails.

## Not covered

Real 128K-token histories (the longest actual prompt here is 9216 ids in the
engine gate and 2745 HTTP ids), HIP/SYCL builds, other GPUs, a forced GPU
allocation failure, and exhaustive cancellation scheduling (the measured F3
outcome is a race and only two outcomes were observed). No coding-quality claim is
made: this document is about prefix reuse and cancellation, not model output.
