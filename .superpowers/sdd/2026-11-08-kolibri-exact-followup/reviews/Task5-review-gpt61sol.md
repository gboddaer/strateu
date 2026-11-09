**REJECT is not justified by the stated technical rationale.** Keeping the code unchanged is safe, but the summary overlooks the plan’s deliberately narrow legacy-stream slice. That slice has a credible small replacement: gated omission of the two drains while preserving existing transfer/source-ownership waits. It still needs the required tests and qualification.

Read-only static review; no runtime tests performed.

## Important

### 1. The rejection assumes unsupported branches must be solved together
**`.superpowers/sdd/2026-11-08-kolibri-exact-followup/task5/summary.md:20–25`**

The plan explicitly limits the first slice to **no-pin, legacy/blocking compute-stream execution**, with unsupported stream modes falling back (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:171–175`).

The claimed mandatory coverage of non-blocking decode, handoff and CPU-pool consumers is therefore overstated:

- No-pin rejects the CPU pool at `src/kolibri/kolibri_engine.cu:6644`.
- Slot configurations disable automatic decode scheduling arms, or reject explicitly requested arms, at `:6964–6966`. Those arms include stream, handoff and graph modes.
- Ordinary eager slot execution consequently reaches `layer_step()` with `st = 0` at `:4946`.

For that narrow execution mode, the ordering argument is:

```text
gate/up on blocking g_st
 → SwiGLU on legacy default stream
 → down on blocking g_st
 → overflow work
 → shared expert
 → sum on legacy default stream
```

`g_st` is created as a blocking stream (`:6971`), and `gemv()` explicitly documents this legacy-stream dependency (`:1949–1953`).

An ownership-object refactor is **not demonstrably necessary** here. A narrowly gated omission of `:3202` and `:3243`, retaining all existing reader/readiness/source-buffer protections, is a reasonable candidate for qualification. Extra events are unnecessary where legacy ordering already provides the dependencies.

This does **not** justify removing the drains for arbitrary non-blocking callers: final event waits alone would not repair their earlier gate/up → SwiGLU → down ordering.

## Medium

### 2. The producer/consumer and lifetime trace contains substantive errors
**`summary.md:9–16`**

- **`:3243` is before the shared expert**, not after it. It follows overflow GEMVs or the CPU-result upload. Shared work runs later at `src/kolibri/kolibri_engine.cu:3397`; the sum follows at `:3398`. The source comment at `:3243` is stale.
- **The CPU pool does not copy/read `ey2`.** It reads quantized host activations and weights and writes `s.ey_h` (`:2539–2552`). The slot branch uploads that result to `s.ey_cpu` (`:3223`).
- **`g_q8` is device scratch, not a host-side vector** (`:1909`, `:6999`). Its quantization and MMVQ accesses are ordered on `qst`.
- Quantized `gemv()` consumes `g_ids_host` synchronously to construct weight pointers (`:1972–1975`). Selected MMVQ copies slots into by-value launch parameters, not asynchronous DMA (`include/strata/kernels/native_mmvq.hpp:160–162`). Reassigning this host vector does not require waiting for GPU completion.

The summary correctly identifies the distinction between quantized GEMVs on `g_st` and SwiGLU on `st`, but does not establish a host-argument lifetime obstacle for the planned slice.

### 3. RED omission is disclosed, but does not satisfy Task5
**`summary.md:27–28`**

No-change rejection does not require invented timing or full-logit results. However, the plan explicitly requires delayed-producer/consumer checks and unsupported-stream fallback testing (`docs/superpowers/plans/2026-11-08-kolibri-exact-followup.md:172`).

Omitting them is acceptable as a **recorded skipped step**, not as completion of the planned investigation—especially when the static rejection rests on the incorrect scope argument above. No candidate has been experimentally disproven.

## Low

### 4. The drain inventory is incomplete beyond the two target calls
**`summary.md:5`**

The two targeted unconditional `g_st` synchronizations were both found. Other synchronization points in the surrounding path include:

- Staging-source protection: `src/kolibri/kolibri_engine.cu:2939`, `:3160`, `:3186`.
- Host router-ID read: `:3084–3086`.
- Slot/pool host reads: `:3207–3212`.
- Diagnostic phase drains: `:2957`, `:3079`, `:3401`.
- Token-final device synchronization: `:4950`.

These are **not additional removal recommendations**. Preserve the transfer reader/readiness fences at `:3147–3148`, `:3172–3173`, `:3190–3200`, and prefill metadata drains such as `:4063` and `:4087`.

### 5. “Whole-device” and “legacy blocking stream” are imprecise
**`summary.md:5,16`**

Both target calls are `cudaStreamSynchronize(g_st)`, not device-wide synchronizations. `g_st` is a **created blocking stream**, distinct from the legacy default stream; it implicitly orders with that default stream.

## Bottom line

**Do not approve the summary’s correctness-based REJECT as written.** No unsafe change was introduced, but the narrow candidate intended by Task5 remains untested—not shown impossible. Revise the trace and either qualify that candidate or record the task as stopped without completing the required experiment.
