# Kolibri exact-arithmetic follow-up specification

The user authorizes best-effort execution to the end of the ranked optimization list in this order:
2 exact multi-selection MMVQ;3 quantization reuse;4 cache allocation/admission;5 synchronization;
6 promotion fences;7 GPU hit dispatch/capture;8 bounded CPU packing;9 host-copy diagnosis;
10 fully resident fast path. Skip item1(attention tiling). Explanations and interfaces must be clear enough
for a smaller agent. Items may be kept as tested enablers of a named next step without an isolated speed win.

Every milestone (Task0 and Tasks2–10, plus final composition) requires an independent read-only review by
`openai-codex/gpt-6.1-sol` before it is committed or declared complete, and every review finding must be acted on:
fixed with a proving test, rejected with technical evidence, or deferred with an explicit dependency. If that model
is unreachable or does not reply, use `ollama/glm-5.3-flash:cloud` once; if both fail, the review is SKIPPED. A
negative review never triggers a reviewer switch. Reachability and read-only repository reviews were verified for
both models before this rule was written. Both are remote: never send credentials, the model file or unrelated data.
A SKIPPED review is recorded with both attempt logs and does not block the milestone; an unexplained missing review does.

## Binding constraints
- Execute on amdw7800probuild_optimize, created from amdw7800probuild after committing/pushing the preparation documents.
  Preserve pre-existing untracked files. Do not implement on the base branch. No implementation merge/push is requested.
- Do not start a server. Use direct offline CLI and unit tests only.
- Accuracy first: unchanged-arithmetic changes require finite bit-identical complete logits on identical
  histories. Never loosen tolerance, change oracle expectations or disable assertions.
- GPU strategy agreement is not independent model qualification. The existing CUDA reference-KL gap and
  pinned-table GPU-only prefill failure are not solved by this project.
- Do not change expert quantization, routing probabilities, reduction order, attention implementation,
  attention workspace cap, PF chunk512, clocks, firmware, drivers or global system settings.
- Existing defaults remain off for new experimental switches; do not promote a global default from this AMD host.
- Retain mmap backing, bounded transfer buffers and the1536MiB test reserve. No full expert table pinning.
- Build/test HIP locally. Attempt CUDA/SYCL only if toolchains really exist; otherwise record NOT RUN and
  leave the result a local qualification. Do not call a HIP compile a CUDA build.
- Execute serial GPU work. No numerical/model-quality acceptance margin is authorized.
- No artificial campaign/context cutoff; individual processes still need watchdogs against hangs.

## Evidence and decisions

Baseline source6c241aa4; current engine matches sealed overnight SHA8a4c1fc1f4eece04fee923f763887ec12643ebfdaaae6198f9250b86dd0c63c2.
The overnight audit found49 finite complete runs,27 bitwise comparisons and9/9 needle answers through131000
actual input tokens. The exact-edge failures were a controller bug: CLI requires prompt+steps<=ctx, not
prompt+steps-1. Preserve the old controller/raw data unchanged.

Rank4 evidence:65K balanced prefill uploaded294.02GB versus demand37.89GB. Prefer reducing repeated staging
before accelerating copies. Rank8 evidence:packing/fault time55/105s at4K but86/6352s at131K, so it is a
cold/shorter-context target, not an explanation of the longest runtime.

A task ends KEEP(measured gain), KEEP-ENABLER(named dependency and tests), REJECT(correctness/performance),
or BLOCKED(explicit missing hardware/authorization/unsupported execution contract). Best effort means trying
bounded prototypes and finishing the remaining tasks, not forcing an unsafe half-implementation to ship.
Every kept path is opt-in. A failed arithmetic gate rejects that prototype immediately.
