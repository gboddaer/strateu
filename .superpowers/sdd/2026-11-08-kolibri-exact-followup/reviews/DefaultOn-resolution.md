# Default-on review resolution

- Wording corrected: "three newly changed call sites", not "only three uses".
- Recorded that ZERO_BUFFER allocates on every Kolibri scratch initialization, before auto-fit.
- Recorded that the skip gate bypasses profile defaults, so `baseline` alone is not all-off.
- Recorded that the code is not restricted to W7800; only HIP/gfx1100 was measured.
- Kept +35.2% paired median separate from +19.2% ratio of medians.
- Long-context reference uses a different executable; treated as regression identity, not a repeatable speed claim.
- Limits retained: targeted Task 10 tests, CUDA/SYCL, server reset/cancellation, setup/Terminal failures.
