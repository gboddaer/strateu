# Kolibri AMD and CUDA preview release

## Scope agreed with the operator

Execute jobs 1–5: rebuild/package the current engine; verify installation and serving; preserve evidence;
reconcile public docs; publish a GitHub preview in `gboddaer/strateu`. AMD qualification is Linux gfx1201/R9700,
card0 only. The operator will independently test the current CUDA source on Tesla T4. RTX 3090 is not qualified
by a T4 result. Preserve the CUDA source/defaults for that test. No further performance experiment is included.
Follow [the accuracy policy](../KOLIBRI_ACCURACY_POLICY.md). Do not change expectation files.

## Ordered jobs and completion checks

1. **Package.** Build from the recorded source revision with Release HIP/gfx1201/MMQ flags. Stage the current
   binary, matching runtime libraries/kernel data, tuning table and license notices. Check dependency coverage,
   relocatability and the complete component manifest. Run decode and both prefill gates on the staged or
   extracted executable; verify long-context parity too. Do not ship the old P4-only archive.
2. **Install and serve.** Extract outside the build folder, verify checksums and test without build-tree library
   or tuning overrides. Exercise chat, streaming, tools, cancellation and extending-prefix reuse through the
   actual Python server with the packaged engine. Record tokenizer and server dependencies separately from
   the engine package. Bind only to localhost. Bound and reap all owned processes.
3. **Evidence.** Preserve sanitized commands, configuration, raw results, source/model/binary identities,
   build information and known failures in a checksummed release evidence archive. Historical evidence must
   be clearly distinguished from qualification performed for this release.
4. **Docs.** Publish one preview guide and release notes. Remove contradictory current-state descriptions;
   retain historical findings as historical. State accuracy/reference limits, test failures and platform scope.
5. **GitHub.** Commit and push the release documentation. Create a preview tag/release with packages,
   checksums and evidence only after corresponding gates pass. CUDA artifacts/qualification remain pending
   operator results; never claim that AMD tests qualify CUDA. Prefer a draft until both platform decisions are
   resolved. Never upload weights, credentials or unrelated `.pi/` state.

## Progress and rulings

- Start: engine source revision `82a2d5d2d9ef635ee32c2c9a4bbea70e9bf0b338`; tracked changes only in unrelated
  `.pi/` state. Existing local package identifies `3b6cc70d` / `82104171`, before decode acceleration.
- Preflight: SSH git push works. No `gh` executable, `GH_TOKEN`/`GITHUB_TOKEN`, GitHub CLI authentication file
  or git credential helper was found. GitHub release creation/upload needs API authentication; continue jobs
  1–4 and prepare assets while publication is blocked. Do not ask the operator to paste a token into chat.
- Ruling: keep CUDA arithmetic/defaults unchanged for the operator's test. The later operator-requested
  build compatibility fixes below are necessary for compilation, not a new CUDA optimization. No NVIDIA result
  is inferred from R9700. Cost: broad preview publication waits for explicit CUDA qualification or clearly scoped exclusion.
- Job 1 initial candidate: fresh HIP Release build, extracted component manifest/dependency checks passed.
  Extracted-binary decode gate and both prefill profiles passed; all expectation files stayed unchanged.
- Scope addition from operator: investigate CUDA build failures before publication. Reproduced an unguarded
  HIP benchmark at CMake generation and C++20 `atomic_ref` in C++17 CUDA TUs. Fixed the target boundary and
  introduced lock-free acquire/release control-word helpers. Both regression tests were observed RED then GREEN.
- CUDA compile check: official unmodified CUDA 13.1.115, GCC 14.2, CMake 3.31, sm_75 and sm_86; engine,
  rollover TU and wait test compile/link successfully. CUDA 12.8.93 compiler identification failed against
  glibc 2.41 before project compilation; retain as a separate failed environment check. No NVIDIA execution here.
- AMD after portability fix: HIP engine/rollover/wait/benchmark build PASS; wait/chain and grid tests 2/2;
  focused unit suite 93 passed; full decode gate PASS; 8191 full-logit SHA256 remains
  `93274a15a769b28b4e4ea57e5ae5289fd624d01a172f9872aaa4ac27f1105c88`, identical to the pre-fix reference.
  Initial package must be rebuilt and requalified with this integrated fix, not shipped unchanged.
- Operator-provided T4 report received: independently found the same build blockers; reported ~4.1% pool decode
  gain and matching generated ids. Its local volatile/fence fix differs from our atomic intrinsics. Integrated
  revision and full-logit qualification remain pending; see [CUDA preview](../KOLIBRI_CUDA_PREVIEW.md).
- Ruling: use existing packaging/gate machinery rather than introduce a new installer or optimization.
  The subsequent CUDA build request required the compatibility fixes above; no further performance feature
  was added. Documentation and release artifacts need consistency/manifest checks, not fabricated RED/GREEN
  implementation tests. Self-review only; delegation was not requested.

## Release qualification outcome

- **Job 1 complete (AMD):** integrated source `5bbe6c4e`, fresh HIP build and current extracted package. Decode
  gate and both prefill profiles PASS; all expectation hashes unchanged. Nine-record full-logit comparisons at
  8191 and 11000 ids PASS, with the historical hashes unchanged. Final relocated archive has the identical
  qualified executable, full manifest/dependency checks PASS, default MMQ and graph replay active without
  tuning-path or library-path overrides.
- **Job 2 complete (AMD):** clean `git archive 5bbe6c4e`, fresh venv, pinned runtime requirements installed;
  tokenizer extracted from the GGUF, 16 round-trip strings PASS. Real HTTP models/chat/SSE/tool JSON/cancellation
  and recovery PASS. Strict default chat reuse had zero cached tokens in the tested conversation; the separate
  existing opt-in prefix mode reused 768. This limitation is documented, not hidden by changing the default.
- **Current tests:** 93 focused units PASS; seven real-model tests PASS; packaging/source-setup subset 175 PASS,
  3 skipped, 12 subtests PASS. Full HIP build PASS. CTest 188: 170 PASS, 12 skipped, six FAIL, matching the six
  previously baseline-reproduced names. Full Terminal suite not claimed green.
- **Current package performance:** three rotating-order pairs, 96 generated tokens, same extracted binary,
  explicit-off plain vs default graph. Short median 6001.1 -> 4009.2 ms (16.00 -> 23.94 tok/s); long median
  8214.5 -> 5409.1 ms (11.69 -> 17.75 tok/s). Every pair favoured default; all six generated histories match per
  workload. One short graph run was 4165.8 ms; retain the variability, not just medians.
- **Benchmark ruling:** the older `decode_bench.run_pairs` removes candidate environment keys from its control.
  With now-default arms this produced an active graph in the supposed baseline and was correctly rejected by
  its marker check. No score was taken. Use its bounded `run_request` directly with explicit arm environments
  in the archived paired harness. Cost: callers of the older paired CLI must not treat unset controls as off.
- **Job 3 complete:** sanitized evidence archive contains current commands, fixtures, full logits, hashes,
  gate/HTTP/build/test/paired-run records and labelled historical accuracy/baseline records. Model identity was
  freshly SHA256-checked. No weights, tokenizer vocabulary, credentials or unrelated `.pi/` state are included.
- **Job 4 complete:** `KOLIBRI_PREVIEW.md` is the installation/qualification guide; README, AGENTS, model/decode/P4
  docs point to it. `docs/releases/kolibri-preview.1.md` records draft notes and publishing handoff. Historical
  accuracy targets and current finite-fixture limits remain disclosed.
- **Job 5 blocked, prepared:** final AMD engine archive, evidence archive and SHA256SUMS are prepared under
  `build-mmq/release/kolibri-preview/assets/`. Intended tag is `v0.1.52-kolibri-preview.1`. No release/tag/upload
  has been created: no GitHub API authentication is available, and integrated T4 full-logit/runtime plus CUDA
  package qualification are still pending. Source/docs can be pushed over SSH; that is not a GitHub release.
- **Final review:** self-review; no independent reviewer was delegated. Actual CUDA 13.1 compile/link checks
  also cover the primary `strata` target. sm_75 and sm_86 images were verified with cuobjdump. NVIDIA execution,
  CUDA 12.8 project compilation, Windows and other GPU architectures remain unverified locally.
