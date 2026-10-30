# Kolibri-1 preview.1 — release notes (draft)

**Not published.** Intended tag: `v0.1.52-kolibri-preview.1`, repository `gboddaer/strateu`.
AMD qualification is complete for the scope below. Integrated T4 full-logit/runtime qualification and a CUDA
package remain pending. GitHub release upload needs an authenticated publishing session. Do not remove these
limits or publish a combined validated AMD/CUDA release until the missing checks are supplied.

## Result

Kolibri-1 Q4_K_M runs as a single stream without a drafter. The Linux R9700 candidate includes MMQ prefill,
host-assisted whole-stack decode replay and the CUDA build compatibility fixes from `5bbe6c4e`.

On the Radeon AI PRO R9700 32 GB + Ryzen 7 5700G, eight CPU lanes, 96 generated tokens, three rotating-order
pairs, the same packaged binary measured:

| Prompt ids | Explicit-off plain decode | Default graph decode |
|---|---:|---:|
| 512 (ctx 8192) | 16.00 tok/s | **23.94 tok/s** |
| 8191 (ctx 12288) | 11.69 tok/s | **17.75 tok/s** |

These are 33.2% and 34.2% reductions in median decode time. Load and prefill are excluded; capture encountered
in generation is included. Individual runs and variation are preserved. These are not NVIDIA forecasts.

## Install and scope

Read [the preview guide](../KOLIBRI_PREVIEW.md) before downloading. The engine archive is Linux x86_64,
**gfx1201 only**, built with ROCm 7.2.2 on Debian 13 / glibc 2.41. Older-glibc portability is not claimed.
It is unsigned and text-only. No model weights, tokenizer, Python server, Windows build or vision reader is
included. Keep all bundled libraries, kernel data, tuning and license files together.

Prepared assets:

- `Strateu-Kolibri-0.1.52-preview.1-linux-gfx1201.tar.gz`
- `Strateu-Kolibri-preview.1-evidence.tar.gz`
- `SHA256SUMS`

**No qualified CUDA binary asset is available yet.** CUDA 13.1 compile/link checks pass for sm_75 and sm_86,
including both engine entrypoints and decode tests. The operator's T4 report on earlier source plus local fixes
reports ~4.1% pool decode improvement and matching generated ids; it is not an integrated-revision full-logit
pass. No RTX 3090 runtime result exists. See [CUDA qualification](../KOLIBRI_CUDA_PREVIEW.md).

## Validation

Passed on AMD:

- Extracted-package manifest/dependency checks; default tuning discovery without library-path overrides.
- Full decode gate: finite bit-identical full logits at 16/511/512/513 ids, exact extending-prefix reuse and six
  repeats. Additional 8191/11000-id forced-history comparisons are bitwise.
- Both prefill profiles against unchanged expectation files.
- Fresh-source runtime installation, tokenizer round trips, HTTP chat, streaming, tools, disconnect cancellation
  and request recovery. Strict chat reuse missed the tested re-rendered boundary; opt-in prefix mode reused 768
  tokens. This opt-in does not change the default.
- 93 focused units; seven real-model graph tests; packaging/source-setup subset 175 passed, 3 skipped and 12
  passing subtests.

Failed/limited:

- Full HIP CTest: 170 passed, 12 skipped, **6 failed** of 188. The failures are `kolibri_load_test`, `vmm_test`,
  `ple_parity`, `platform_memory_test`, `conversation_copy_streams_test`, `isa_guard_check`, matching the six names
  previously reproduced on the baseline. The suite is not green.
- CUDA 12.8 compiler identification failed against the build host's glibc 2.41 before project compilation.
  CUDA 13.1 compilation is not CUDA 12.8 validation or GPU execution.
- The full Terminal suite is not claimed green; earlier attempts timed out. CUDA runtime/package, RTX 3090,
  other AMD architectures, Windows and GPU race-detector checks remain outside local qualification.

## Accuracy

Decode parity establishes that scheduling preserves the tested engine outputs. It is not proof of perfect
model fidelity or general task quality. A historical strict reference-KL target was not closed; later P4
finite-fixture screens passed their different stated thresholds using reused reference arrays. No fresh broad
quality evaluation or universal numerical tolerance is claimed. Follow the
[Kolibri accuracy policy](../KOLIBRI_ACCURACY_POLICY.md).

## Publishing handoff

Keep this release a draft until the CUDA scope is decided from actual results. On an authenticated machine,
after verifying the asset checksums, the draft can be created with:

```sh
# RELEASE_COMMIT is the reviewed documentation/source commit, not an uncommitted worktree.
gh release create v0.1.52-kolibri-preview.1 --repo gboddaer/strateu \
  --target "$RELEASE_COMMIT" --draft --prerelease \
  --title 'Kolibri-1 preview.1 — AMD and CUDA qualification' \
  --notes-file docs/releases/kolibri-preview.1.md \
  "$ASSET_DIR/Strateu-Kolibri-0.1.52-preview.1-linux-gfx1201.tar.gz" \
  "$ASSET_DIR/Strateu-Kolibri-preview.1-evidence.tar.gz" "$ASSET_DIR/SHA256SUMS"
```

Before making it public: integrate exact-revision T4 full-logit/lifecycle results and logs; qualify/checksum the
CUDA package on its target driver; update the notes and support matrix; confirm any untested platforms remain
labelled untested; verify every remote asset matches the local checksum. Never paste a publishing token into
chat or upload weights/credentials. Source pushes over SSH do not grant release-upload authentication.
