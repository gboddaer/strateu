**REJECT is justified—as “no demonstrated end-to-end benefit,” not “proved unmeasurable.”** I reviewed `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task6/summary.md` and the implementation/test. This is static review; I did not rerun measurements.

1. **Ordering is correct within the gated contract.**  
   In `src/kolibri/kolibri_engine.cu:6028–6042`, the dependency chain is:
   old `g_st` readers → copy-stream wait → all uploads → final readiness wait on `g_st`.  
   `acquire()` still protects staging-buffer reuse. Because uploads share one stream, the final upload event covers the entire round. Generation tracking makes the following submissions require a fresh readiness dependency; changing consumer streams also triggers a wait. Event re-recording does not retroactively change already-enqueued waits.

   **Important boundary:** readiness on a second consumer stream does **not** automatically protect that stream’s reads against a subsequent overwrite. That requires `wait_for(other)` or joining its work into `g_st`. The engine’s narrow single-consumer contract avoids this issue.

2. **The timing supports rejection, but not the summary’s strongest wording.**  
   The observed promotion-span difference is 12.2 ms/2.5%. One profiled pair does not establish repeatability, and the confounded 259 s versus 132 s walls establish neither an end-to-end win nor its absence. Prefer:
   > “Observed 12.2 ms lower promotion time; no end-to-end improvement demonstrated.”

   The previously small promotion share makes stopping reasonable. This is not a completed performance qualification.

3. **Remove the engine gate; KEEP-ENABLER is not warranted.**  
   The plan requires a **named subsequent retained consumer**, not possible future usefulness. None is identified here. Preserve evidence and the experimental diff. Removing the experiment-specific test is reasonable, although a strengthened general transfer-lifecycle regression test could independently remain useful.

4. **The test has coverage limits—not a demonstrated batching bug.**
   - `tests/kolibri/expert_transfer_test.cu:89–98`: initialization uses default-stream `cudaMemset`, followed by reads on a nonblocking stream without an explicit initialization join. Synchronize initialization to avoid a test-side race.
   - The generation boundary is host-synchronized. It does not stress pending event re-recording, or overwrite after delayed reads on the second consumer stream.
   - Tiny uploads without deliberate copy delay make missing-readiness failures timing-dependent. Mutation testing should show that removing either round fence fails.
   - It exercises the transfer primitive, not engine fallback eligibility, multi-layer packing/publication, or interrupted-round recovery.

**Bottom line:** no ordering defect found in the narrow engine path. Rejecting and restoring per-move fences is a sound scope/performance decision; do not present it as a correctness rejection or proof that 12.2 ms could never be measured end to end.
