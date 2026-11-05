# Exact long-prefix reuse for Kolibri coding-agent turns

## Problem and approach

The merged engine could match a transcript's IDs but could not safely rewind its
513-slot sliding KV ring after a reply overwrote earlier rows. Chunk alignment
also needs to rebuild partial/decode-written rows. Disabling those protections
made cache counters rise without proving correct logits. Independent review
rejected that attempted shortcut; it is not part of this change.

Keep bounded checkpoints at **completed full prefill boundaries** instead. Save
all sliding K/V ring bytes to owned host buffers before later work overwrites
them. Restore the newest checkpoint not after the exact token-ID common prefix,
invalidate derived mirrors, and recompute the unstable suffix. Absolute full-
attention rows below that boundary remain in the same active history and need
not be copied. A branch prunes descendant checkpoints before overwriting rows;
a reset clears them. Never snapshot a decode-era suffix as canonical prefill.

Default payload budget: **128 MiB**, `STRATA_KOLIBRI_PREFIX_CACHE_MB=0` disables
checkpoints. Actual tested engine geometry reports **42,024,960 bytes** per
checkpoint, so three fit (120.23 MiB plus small metadata/allocator overhead).
Sizes come from the actual layer classifier, not documentation's inconsistent
39/11 layer count. FP16 rings are device allocations: `halloc` means *half*
allocator. Transfers are explicit synchronized CUDA D2H/H2D copies, not CPU
pointer dereferences. The CPU store publishes only complete owned snapshots.

CUDA source setup now persists the accurate streamed profile: prefill, node-local
CPU lanes, slots 0, device activation/gate-up pairing, F16 experts off, POOL_ALL=1,
512-token chunks, reuse/prefix/alignment on, checkpoint budget 128 MiB and served
temperature 1.0. The old T4 setup preset that enabled rejected F16 expert-cache
arithmetic is removed. HIP keeps its own prefill profile; non-Kolibri families
are unchanged. Existing explicit user overrides are not blanket-overwritten.

## Validation environment and controls

Tesla T4 16 GB, driver 580.196.07, two E5-2690 v4 CPUs, 14 native lanes, Kolibri-1
Q4_K_M, CUDA Release SM75 with Q6K expert support. Isolated worktree based on
capacity fix `3f482202`, itself based on `main` `15dac194`. The operator explicitly
stopped the service and freed the machine for experiments. No protected evidence
worktree was changed. Raw local artifacts: `/data4/kolibri-sram/out/exact-prefix-cache/`.
A bounded, committed [results manifest](exact-prefix-cache-results.json) records
source/binary fingerprints, controls, outcomes, measurements and outstanding checks.

The gate demands streaming with F16 expert cache off and alignment on. References
use the candidate's exact forced histories, not separately sampled histories.
Every compared vector has all **128000 finite FP32 logits** and must match byte
for byte. Positive hits beyond the ring are required; falling cold cannot pass
as a cache success. Cache-off, undersized-budget and unsupported-route controls
are separate runs. Environment/config and actual INFO records are retained.

## Passed GPU checks

- Baseline capacity-fixed engine had correct cold logits but **0 reused tokens**
  for the long extension: the new positive-hit regression failed as intended.
- New streamed matrix: **24 comparisons pass**, including extensions, edits,
  replay, a one-token suffix, 8191/11000/32768-ID histories, branch/descendant/
  return-to-old-branch sequences, unrelated-owner reset, forced continuations,
  six same-arm repeats and two verified cancellation outcomes. Reuse reaches
  **2560, 3072, 7680, 10752 and 32256 tokens**, depending on the checkpoint.
- The pool control completed **20 passing comparisons** before its overall
  1500-second process timeout. It is an **incomplete command**, not an overall
  gate pass. Its remaining two cancellation cases were then run separately and
  passed, with actual `finish=cancel` and forced cold-logit comparisons.
- Same-binary cache-off, 1 MiB budget and adaptive POOL_ALL=2 controls pass with
  zero long-prefix reuse. Unsupported profiles report checkpoints disabled.
- Historical degeneration gate: all **six** full-logit cases pass with streaming
  and all six with the pool. The one-token and 257-ID unsafe partial-prefix cases
  still rebuild cold; safe ring-capacity cases retain 512 reused positions.
- Stream-growth gate passes again: four complete long-request vectors match
  cold after short/long/small and repeated-growth sequences.

## Real HTTP turns and prefill speed

Same-day loopback HTTP experiment, Q4_K_M, 128K allocated context, 14 threads,
no unrelated GPU work. Synthetic coding-session facts and a tool-result transcript
are used, not a claim of model coding accuracy. Sampling is greedy for these
comparisons; the saved served default remains the user-approved temperature 1.0.

| Cold 2706-token prompt, three requests per arm | Checkpoints off | Checkpoints on |
| --- | --- | --- |
| Prompt tok/s | 154.2 / 150.7 / 149.7 | **150.0 / 149.2 / 149.7** |

The following comparisons are entirely within the cache-on arm. Each cold
control follows an unrelated reset request, then resends the identical request.

| Measurement | Cold control | Matching checkpoint |
| --- | --- | --- |
| Same growing request, cached tokens | 0 | **2560** |
| Prompt tokens actually read | full 2736–2742 | **176–182** |
| Prompt milliseconds | 18180.9 / 18336.0 / 18349.6 | **2212.4 / 2243.1 / 2249.3** |
| Tool-result history, same request prompt time | 19470.3 ms | **3210.1 ms**, 2560 reused |

These are three cold **requests**, not three model-process cold loads per arm;
each arm uses one fresh server process. The label means no reusable transcript
prefix for that request. Model load and reply time are excluded from prompt_ms.
Raw wall time and response usage are retained separately in `http-{off,on}/`.
In the primary three-request campaign, cold rates differed by about one percent;
no faster-kernel claim is made. Later unbound public-gate cold requests measured
155.1 off / 151.5 on, then 156.4 off / **139.9 on**. Those worse results are
retained, not folded into the primary campaign. Per-process placement was not
captured during the outlier, so its cause is not established. Cached small suffixes report 78–81 tok/s because
fixed overhead is divided over fewer tokens, but finish their prompt phase in
about 2.2 s instead of 18.3 s. Do not interpret their lower numeric tok/s as
re-reading the whole conversation on the CPU pool.

All **16 corresponding on/off HTTP requests** have identical request objects
and finite byte-identical full logits. Each arm also compares growing/tool
requests to explicit reset/cold controls. Tool transcripts are synthetic; the
harness never executes generated tool requests. The public reproducible HTTP
gate passed two separate one-repeat campaigns (eight cross-mode comparisons
each). The final campaign additionally records and verifies exact parity of all
eight rendered HTTP prompt-ID lists.

The existing setup topology helper selects 14 workers and
`/usr/bin/numactl --cpunodebind=1 --membind=1` on this host. A further public-gate
campaign used that explicit generated-profile binding: cache-off cold rates
**156.9 / 152.9**, cache-on **153.8 / 153.3 tok/s**; cached growing reads
**2136.0 / 2157.2 ms**; tool-result **3075.1 ms** versus **18882.6 ms** cold.
All 12 bound on/off HTTP request-ID lists and full logit vectors matched.
Eighteen identical-request full-logit vectors also matched between the earlier
unbound and bound campaigns. Binding is preserved as an explicit launch setting;
no GPU clock or global machine policy was changed.

A real thinking-budget wrap request also completed with a final answer. Its two
native segments reported reuse **0 then 2560**, confirming that server-native
close-and-continue takes the checkpoint path. This integration check records
finite vectors and segment accounting, not an additional cold-equivalence claim
for that particular server-generated continuation; the forced-ID continuation
matrix supplies the exactness checks.

## CPU, setup and serving checks

- Final focused tools/setup suite: **218 passed, 22 skipped, 1 deselected**
  (the optional network HEAD check was excluded). This includes real C++17
  compile/run of byte ownership, budget/eviction, fork pruning, transport
  callbacks and exception/non-publication checks; CUDA profile generation and
  HTTP diagnostic safety checks.
- Full serve suite: **1257 tests, OK, 31 skipped**. Existing telemetry
  ResourceWarnings were observed; no unrelated cleanup was included.
- Fresh CUDA SM75 Release compilation passed. HIP/SYCL tools were unavailable;
  those checks remain pending, not inferred from CPU/CUDA success.

## Commands

```sh
python tools/kolibri/prefix_cache_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/long" --long
python tools/kolibri/prefix_cache_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/pool" --stream off
python tools/kolibri/prefix_cache_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/off" --cache-mb 0 --quick
python tools/kolibri/prefix_cache_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/small" --cache-mb 1 --quick
python tools/kolibri/prefix_cache_gate.py --bin "$BIN" --model "$MODEL" --output "$OUT/adaptive" --pool-mode 2 --stream off --quick
python tools/kolibri/prefix_cache_http_gate.py --config "$CONFIG" --output "$OUT/http" --repeats 3
```

Each output directory must be new. Gates start and close their own engine/server.
HTTP diagnostics refuse an occupied port, bind loopback only and do not copy API
credentials or remote tool configuration. Do not benchmark alongside production.

## Failed experiments, review and limits

An early GPU prototype failed before comparison because it attempted a host
copy of device-backed FP16 ring pointers. It was replaced with explicit CUDA
transfers, with an adapter regression check. This failed artifact is retained as
`candidate-quick.log`; the successful successor is `candidate-quick-v2/`. The
complete tests above use the corrected binary. The rejected zero-alignment
shortcut is separate from this implementation and remains removed.

Independent read-only review supports boundary ownership, transfer ordering,
bounded payload and reset/branch handling. Its adaptive-route warning was fixed:
checkpoints disable for POOL_ALL modes other than 1, alignment off, packed INT8,
resident/F16 experts, grouped/pipelined or non-pool configurations. This is not
permission to call those other arithmetic modes accuracy-qualified.

HIP/SYCL compilation and runtime, other GPUs, max-length 128K actual histories,
real concurrent-client/multi-session workloads and forced CUDA allocation failure
are **not run**. CPU tests inject an exception during snapshot copying to check
non-publication; that is not a forced GPU allocator failure. Cancellation checks
cover STOP during prompt/token phases and a correct following request, not every
possible instruction-level scheduling race.

The cache holds one active ID history. Earlier edits can evict/miss all retained
boundaries and legitimately go cold; a rewritten early system message is not an
identical prefix. It does not retain arbitrary sessions or every generated reply
token. Short sub-chunk prefixes may intentionally have no exact reusable boundary.
New checkpoint defaults are CUDA-only until other backends are qualified. Keep
PRs draft where required backend checks remain unavailable. No universal coding-
quality, numerical-tolerance or hardware-speed claim is made.
