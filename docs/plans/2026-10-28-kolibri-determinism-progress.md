# Kolibri P4 delivery checklist

Requested: fix determinism, land independent accuracy gates, package the arm, then make it the default where faster.
Card0 only. No other GPUs were used.

- [x] Reproduce changing and non-finite full logits before the fix.
- [x] Localize the first differing buffer: shared-expert dense gate, not the routed-expert result.
- [x] Require compatible direct shared-expert tuning; retain the pool otherwise. Remove per-group device drains.
- [x] Repeat full logits/ids, including automatic activation and the relocated package at 511 and 8191 tokens.
- [x] Independently compare to the prior llama.cpp GPU/CPU fixtures; rerun both Strata arms (200 steps).
- [x] Record separate in-repo MMQ smoke/hash expectations; hash actual device output and reject fallback/failed runs.
- [x] Pass full P4 and legacy-pool gates, including reuse and plain-path oracle BITWISE PASS.
- [x] Enable the qualified Linux gfx1201 default with an explicit off switch. Other defaults remain unchanged.
- [x] Fix fresh AMD source setup's CUDA configuration and pass matching tuning in the Kolibri launch config.
- [x] Add opt-in Kolibri release packaging, runtime dependency collection and manifest/executable coverage.
- [x] Run paired staged-binary prefill and wall-derived generation measurements; retain cold and variable runs.
- [x] Build both HIP trees; run full CTest and document the four shared baseline failures and six skips.
- [x] Run packaging/tuning and source-setup tests. Document the incomplete full Terminal-suite attempts.

Final local artifact verification and commit IDs are reported in the delivery message. No public upstream release,
CUDA/SYCL/Windows verification, or vendor-library internal root-cause proof is claimed.

The ledger and exact limits are in `docs/KOLIBRI_P4_SHIPPING.md`. Raw artifacts are under `/tmp/kolibri-det`, with
small validation records also included in the local package. The initial plain XOR hash was replaced during
investigation because duplicate padding rows could cancel; all temporary hash/dump probes were removed afterward.
A standalone fixed-input GEMM probe did not reproduce the failure and is not claimed as proof of a vendor bug.
