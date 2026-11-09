# Final review resolution

Review: `Final-review-gpt61sol.md`.

- Task 2's stale KEEP-ENABLER label removed from `progress.md` and `task2/summary.md`; Task 2 is a standalone KEEP and
  Task 3 was rejected.
- Task 9's retained standalone probe is now stated; it is a diagnostic, not an engine change.
- Reviews are no longer marked pending; a review-disposition section records fixed/rejected/deferred findings.
- Task 8's IO-causality wording changed to a hypothesis; the early-stop coverage gap (cold 4K, planned worker count,
  delayed-worker lifetime, exception cleanup) is recorded.
- Task 9's SDMA ordering is stated as default/SDMA=1/SDMA=0 = 33.618/33.618/33.614 ms, and the "within noise" wording
  was removed.
- Final integration comparison now lists per-case bytes and SHA-256; build, ctest and setup logs were copied under
  `final/`.
- Final summary no longer calls the setup/Terminal failures unrelated; it says the causes are not all established and
  the suites are not green.
- Long-context wording now says eight complete vocabulary distributions, local regression identity only.
- Task 10 targeted tests remain explicitly unrun; no claim of completed qualification or default promotion.

No code change was needed for this resolution. No finding was left unresolved without a recorded dependency.
