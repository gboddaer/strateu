**REJECT is not justified by this measurement. The bounded offline prototype is still required under the plan; capture remains conditional.**

Read `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task7/summary.md` and its logs.

### 1. Important: `d2h=0.0` does not measure this path

In `src/kolibri/kolibri_engine.cu`:

- **`:3079` synchronizes before the router copy**, moving outstanding GPU completion into `pre` and perturbing the normal execution schedule.
- **`:3080–3088` performs the router-ID D2H copy.**
- **`:3245–3249` updates `dec_d2h_ns` only inside the non-slot CPU-pool branch.** The measured run uses 50 slot-path layers/token and zero pool layers, so it never reaches this update.
- **`:3401–3403` charges the slot-path interval—including the router copy—to `post`.**

Thus `d2h=0.0` is an **unpopulated counter for this configuration**, not an unresolved small measurement.

Even for the CPU-pool branch, this label does not encompass everything suggested by “ids+h D2H”: the activation copy occurs after its timestamp boundary.

### 2. Important: the `<0.05 ms` ceiling is invalid

`%.1f` rounding could constrain a **populated average**, not individual layers or the uninstrumented synchronization cost. Here the accumulator is never updated, so **neither `<0.05 ms/layer` nor `<2.5 ms/token` follows**.

The injected synchronizations also prevent treating this diagnostic split as a clean upper bound on potential device-dispatch savings.

ABI, generations and overflow handling are legitimate implementation costs, but they do not establish a performance rejection. Correctly instrument the slot-path copy/wait, write the ownership contract, and attempt the bounded forced-hit/miss prototype. No capture or speculative infrastructure must be retained if it loses.

### 3. The uncounted-copy correction is accurate

For this recorded configuration:

- Counted logits: **128 × 512,000 B = 65,536,000 B**, or **500 KiB/copy**.
- Uncounted router IDs: **50 × 128 = 6,400 calls**.
- Six IDs/call: **6,400 × 6 × 4 = 153,600 B**.
- Hit rate: **38,107 / (38,107 + 293) = 99.237%**, correctly rounded to **99.2%**.

Small transfer volume does not establish small synchronization latency.

### 4. Half-implementation / remaining work

No Task-7 device-map helper, flag, dispatch kernel or test wiring was found in the inspected sources. **No apparent code half-implementation remains.** This is static inspection, not a Git-diff certification.

However, the ownership contract and bounded prototype remain **unperformed plan work**. Both `task7/summary.md:15–27` and the Task-7 ledger entry should withdraw the invalid timing ceiling and performance conclusion.

**Bottom line: do not close Task 7 as REJECT on this evidence. The bounded prototype is still required; capture is not yet required.**
