**REJECT is justified: keep the current settings, make no engine/system change, and retain the probe as a standalone diagnostic.** The evidence supports that decision, but not a completed root-cause diagnosis or the claim that engine copies cannot be limiting.

Reviewed `tools/kolibri/transfer_probe.cu` and the summary/three CSVs under `.superpowers/sdd/2026-11-08-kolibri-exact-followup/task9/`.

### 1. Timing and verification

**Sound for a bounded, warm-buffer H2D measurement.** Events bracket the copy in the same stream, completion is synchronized, and readback/verification are outside the copy timing. The 64 MiB result is internally consistent:

`67,108,864 bytes / 33.618 ms ≈ 1.996 GB/s`.

The size scaling and agreement across runs make **approximately 2.0 GB/s credible for this tested path**. It is not necessarily the hardware’s maximum bandwidth.

Limitations:
- Pageable asynchronous copies can involve host staging/blocking; event timing is not a clean measurement of complete host-side cost. Add synchronized wall-clock timing if investigating further.
- Best-of-five measures favorable performance, not variability. Preserve all samples and repeat processes before making statistical “within noise” claims.
- Verification checks the **final device contents once per size**, not each timed repetition. All 24 CSV rows pass, but identical repeated payloads cannot detect a stale earlier copy.
- The CPU memcpy measurement is single-shot, cache/order-sensitive, and should explicitly consume its result to prevent optimization concerns. It is not a reliable CPU-memory ceiling benchmark.

**Fix before retaining:** verification failure currently prints `ok=0` but still exits successfully. Return nonzero on any mismatch.

### 2. SDMA and topology

**Separate processes are appropriate**, because ROCr/HIP may consume environment settings during initialization. The reported separate-process execution meets that requirement.

However, identical timings do **not prove that different engines actually executed**. The setting may not affect this particular runtime/copy path. Engine tracing would be needed for that stronger claim.

For a completed diagnosis, record:
- HIP/ROCr versions and effective environment;
- selected GPU identity;
- endpoint **and upstream bridge/root-port** negotiated links;
- CPU/GPU NUMA locality and source-memory placement;
- read-only IOMMU/driver information.

Endpoint Gen4 x16 alone does not clear the entire upstream path. These checks are **not prerequisites for rejecting an unhelpful switch**, but remain unfinished diagnostic work.

### 3. Unsupported or overstated claims

In `task9/summary.md`:

- **“No engine copy is limited by this probe’s ceiling” is unsupported.**  
  Average traffic of approximately 70 MB/s rules against sustained bulk-bandwidth saturation; it does not exclude burst bandwidth, small-copy overhead, staging, or synchronization costs. At 2 GB/s, 5.4 MB/token represents approximately **2.7 ms/token of bulk transfer time**, not a bound on total copy-related time.

- **“Within noise” is not statistically established** by three minima without their distributions. Prefer “no improvement demonstrated.”

- **“CPU memcpy rates were much lower” is misleading:** the reported 4.5–63 GB/s exceeds the approximately 2 GB/s H2D rate.

- The setting order is misstated in `progress.md`. CSV values are:
  - default: **33.618 ms**
  - SDMA=1: **33.618 ms**
  - SDMA=0: **33.614 ms**

### Decision

**REJECT: yes.** No setting shows a meaningful demonstrated benefit. Keep the small standalone probe, fix its failure exit status, and narrow the documentation to an **observed transfer rate on this configuration**, not a universal ceiling or resolved topology diagnosis.
