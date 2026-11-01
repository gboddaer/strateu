# Kolibri AMD and CUDA preview

**Status: release candidate, not yet a published GitHub release.** The Linux gfx1201 package has been qualified.
CUDA compilation passes locally, and the T4 contributor has recorded integrated-revision full-logit/runtime
qualification. A tested CUDA package is still pending. GitHub release creation/upload also needs an
authenticated publishing session.

This is **strateu's** Kolibri preview, not an upstream Strata release. It runs Kolibri-1 Q4_K_M as a single stream,
without a drafter. The engine package is text-only. Model weights, tokenizer files and the Python server are not
included in the engine archive; the tokenizer is generated from your own GGUF using the source tools.

## What is qualified

Engine source: `5bbe6c4e7327992d10888cc0c9478ea864b4f324`, engine version 0.1.52.

| Platform | Evidence | Limits |
|---|---|---|
| Linux gfx1201, R9700 32 GB | Fresh HIP build; extracted-package decode and both prefill gates; HTTP acceptance; paired speed runs | Ryzen 7 5700G, 93 GiB RAM, Debian 13, glibc 2.41, ROCm 7.2.2; card0 only |
| CUDA sm_75 / sm_86 | CUDA 13.1.115 compile/link of `strata`, `strata-kolibri`, rollover and wait test targets | No NVIDIA execution on this build host |
| Tesla T4 | Contributor-recorded `5bbe6c4e` build, full-logit/reuse/repeat/8191/11000 gates and wait test; see [CUDA qualification](KOLIBRI_CUDA_PREVIEW.md) | Source configuration qualified on that host; tested CUDA package and raw-log handoff pending |
| RTX 3090 | sm_86 code is present in the compiled CUDA binary | No hardware test; no performance claim |

Other AMD architectures, Windows, Metal, SYCL, vision, additional model quantizations and multi-GPU are not
qualified by this preview. Earlier CUDA results in [KOLIBRI1.md](KOLIBRI1.md) are historical, not validation of
this release candidate.

## Install the Linux engine archive

The prepared asset is `Strateu-Kolibri-0.1.52-preview.1-linux-gfx1201.tar.gz`. Once release assets are published,
download this archive and `SHA256SUMS` from **gboddaer/strateu**, not an unrelated upstream release. Before
publication, use the locally delivered files; this guide does not imply that a download is already live.

```sh
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf Strateu-Kolibri-0.1.52-preview.1-linux-gfx1201.tar.gz
export ENGINE_DIR="$PWD/0.1.52-linux-x86_64-hip-kolibri-preview.1"
export MODEL=/absolute/path/to/Kolibri-1-Q4_K_M.gguf
```

Keep the entire component folder together: runtime libraries, `rocblas/`, `hipblaslt/`, `tuning/`, licenses and
`engine-component.json`. The AMD kernel driver and the OS libraries in `SYSTEM-LIBS.txt` must be installed.
This build uses glibc 2.41; compatibility with older distributions is not claimed. Allow roughly 1 GB for the
compressed engine and 1.5 GB after extraction, plus the source/runtime environment. The model is 47.5 GB and
needs about 56 GB of system RAM; leave headroom for the OS and other programs.

From a source checkout with Python installed, verify every component hash and dependency:

```sh
python3 packaging/package_engine.py --verify "$ENGINE_DIR"
HIP_VISIBLE_DEVICES=0 "$ENGINE_DIR/strata-kolibri" --model "$MODEL" \
  --tokens-file "$PWD/tools/kolibri/oracle_ids512.txt" --ctx 8192 --n 16 --prefill --cpu-threads 8
```

For a source build instead, use the HIP setup path in [KOLIBRI1.md](KOLIBRI1.md), or the recorded Release
HIP/gfx1201/MMQ commands in the evidence archive. NVIDIA source-build and gate commands are in
[KOLIBRI_CUDA_PREVIEW.md](KOLIBRI_CUDA_PREVIEW.md); use `75` for T4 or `86` for RTX 3090, not the generic default.

## Serve chat and tool calls locally

Use the matching source checkout (`5bbe6c4e` for the qualified engine/server). Keep `ENGINE_DIR` and `MODEL`
set to absolute paths as above. Runtime installation was tested from a clean source archive and fresh venv:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python tools/strata_tokenizer.py --gguf "$MODEL" --out ./kolibri-pack --check

.venv/bin/python - <<'PY'
import json, os
from pathlib import Path
root = Path.cwd()
engine = Path(os.environ['ENGINE_DIR']).resolve()
json.dump({
    'exe': str(engine / 'strata-kolibri'),
    'model_name': 'kolibri-1', 'arch': 'kolibri1', 'gpu': 0,
    'cwd': str(engine), 'tokenizer': str(root / 'kolibri-pack/tokenizer'),
    'args': ['--model', os.environ['MODEL'], '--prefill', '--cpu-threads', '8', '--ctx', '12288'],
    'env': {'HIP_VISIBLE_DEVICES': '0', 'STRATA_GEMM_DETERMINISTIC': '1',
            'STRATA_KOLIBRI_EXPERT_F16': '0', 'STRATA_KOLIBRI_ATTN_MB': '48'},
    'log': str(root / 'kolibri-preview-engine.log'),
}, open('kolibri-preview.json', 'w'), indent=2)
PY
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m serve.server --engine strata \
  --config kolibri-preview.json --host 127.0.0.1 --port 8080 --offline
```

The API base is `http://127.0.0.1:8080/v1`. In another terminal:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kolibri-1","messages":[{"role":"user","content":"What is 27 times 43?"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"reasoning_effort":"none"}}'
```

HTTP tests passed for models listing, chat, SSE streaming, a JSON-argument tool call, disconnect cancellation
and the next request's recovery. Test processes were stopped after acceptance. There is no public bind or
API key in this localhost example; use authentication if you deliberately expose the server beyond localhost.

**Chat-prefix limit:** strict default reuse can miss a re-rendered chat boundary. The test conversation reported
zero cached tokens with the default, but 768 with existing opt-in `STRATA_KOLIBRI_REUSE_PREFIX=1`. Exact extending
id-prefix reuse passed the engine gates. Prefix mode is not enabled by this release; to try it, add that setting
to the config's `env`. Its cache-hit result is not a general quality or numerical-equivalence claim.

## Activation and rollback switches

- `MMQ active (direct shared-expert tuning verified)` confirms accelerated AMD prefill. The version-matched
  tuning table is discovered beside the engine; no tuning-path or library-path override was needed in package
  and HTTP tests. Missing prerequisites retain the pool with a notice.
- `DEC_TOKEN_GRAPH replay active` confirms whole-stack decode replay after warmup.
- Decode acceleration requires a CPU pool with no routed expert slots, no fp16 expert cache and no grouped
  expert override. Unsupported automatic configurations use the plain path and name the reason. An explicit
  `=1` request for an unsupported arm is an error rather than a silent fallback.
- `STRATA_KOLIBRI_PF_MMQ=0` selects pool prefill.
- To select plain decode, set **all four** of `STRATA_KOLIBRI_DEC_HANDOFF`, `STRATA_KOLIBRI_DEC_STREAM`,
  `STRATA_KOLIBRI_DEC_HANDOFF_GRAPH`, `STRATA_KOLIBRI_DEC_TOKEN_GRAPH` to `0`.

`GGML_CUDA_DISABLE_GRAPHS` is not an off switch for Kolibri's new decode graphs. The configuration above sets
`STRATA_GEMM_DETERMINISTIC=1` to match qualification. This is not a new engine default; its own performance cost
has not been separately measured. Changing thread count, context or residency changes the tested configuration.

## Packaged-binary decode measurements

Same extracted binary, three rotating-order pairs per workload, 96 generated tokens, eight CPU lanes,
`EXPERT_F16=0`, `ATTN_MB=48`, deterministic GEMMs. Both arms use the same prefill; the control explicitly turns
off all four decode arms. All six generated histories match within each workload. Timings are unprofiled `DONE`
decode time, excluding load/prefill and including capture encountered during generation.

| Prompt / allocated context | Plain median | Default median | Decode time reduction | Throughput |
|---|---:|---:|---:|---:|
| 512 / 8192 | 6001.1 ms / 96 tokens | 4009.2 ms / 96 tokens | 33.2% | 16.00 → **23.94 tok/s** |
| 8191 / 12288 | 8214.5 ms / 96 tokens | 5409.1 ms / 96 tokens | 34.2% | 11.69 → **17.75 tok/s** |

Every pair favoured the graph. Individual run variability is retained in the evidence; one short graph run was
4165.8 ms, compared with 4006.7 and 4009.2 ms. These are not NVIDIA predictions. Historical packaged P4 prefill
measurements are in [the P4 shipping record](KOLIBRI_P4_SHIPPING.md); they are not a new prefill measurement of
this candidate.

## Accuracy and known test limits

Follow [the accuracy policy](KOLIBRI_ACCURACY_POLICY.md). Finite, bit-identical **full logits** passed for the
packaged default versus explicit-off decode at 16/511/512/513 prompt ids, actual extending-prefix reuse and six
repeats. Additional nine-record comparisons passed at 8191 and 11000 ids. Their SHA256 values are respectively
`93274a15a769b28b4e4ea57e5ae5289fd624d01a172f9872aaa4ac27f1105c88` and
`daa679fd51ab69f33d0d48c9410eee3eea04b337074d7d05b0af34ffa96275cd`.

This proves regression parity for decode scheduling on the tested inputs, not perfect fidelity to a higher
precision model or general task quality. The historical AMD campaign did not close its strict reference-KL
target. Later P4 eight-prompt teacher-forced screens passed their stated, different thresholds (P4 mean KL
0.00236, 99% top-1 agreement); the reference arrays were reused, not freshly regenerated. Those are historical
results, not a fresh quality evaluation of this preview. No new universal error tolerance has been approved.

Fresh validation of this candidate:

- Extracted-package decode gate and both prefill profiles: PASS, unchanged expectations.
- Seven real-model graph/lifecycle tests: PASS. Focused decode/portability units: 93 PASS.
- Packaging/source-setup subset: 175 PASS, 3 skipped, 12 subtests PASS.
- Full HIP CTest: **170 passed, 12 skipped, 6 failed (188 total)**. Failures: `kolibri_load_test`, `vmm_test`,
  `ple_parity`, `platform_memory_test`, `conversation_copy_streams_test`, `isa_guard_check`. These are the same
  six names previously reproduced on the baseline; the full suite is **not green**.
- The full Terminal suite is not claimed green; prior attempts did not finish. The subset above is not a
  substitute for that result. RTX 3090, Windows and GPU race-detector coverage are not established by the local
  build checks. The contributed T4 runtime qualification is separate from those build checks.

The T4 contributor's later qualification is recorded in commit `9373ef88`, for engine source `5bbe6c4e`, with
CUDA 12.8.93 and driver 580.196.07. It reports full-logit parity through both long boundaries and the full short
reuse/repeat gate. Its suite is also not green: 189 passed, 7 baseline-reproduced failures, 14 skipped of 210.
These are contributor measurements, not runs performed on the AMD build host; raw T4 logs still need to accompany
its release artifact.

**Do not use the old T4 fp16 expert-cache speed configuration as an accuracy-qualified preset.** The contributor
measured mean KL 0.400 and a worst logit difference of 11.08 for `EXPERT_F16=24` versus the same build's plain path;
with `PF_PARTIAL=1`, mean KL was 0.168 and top-1 matched on only 7 of 9 forced steps. These are engine-vs-engine
screens, not llama.cpp-reference KL. The cause is not established by repeatability: rounding/accumulation changes
and deterministic implementation defects must still be distinguished. The preview recommendation is cache off,
slots 0 and decode arms on. See the CUDA guide for exact configuration and measurements.

## Evidence and publication

Prepared companion assets: the engine archive, `Strateu-Kolibri-preview.1-evidence.tar.gz` and `SHA256SUMS`.
The evidence contains sanitized commands/configuration, raw full-logit comparisons, gate logs, paired runs,
HTTP responses, build identities and clearly labelled historical records. It contains no weights or credentials.
Hashes detect corruption; this unsigned preview has no publisher-signature verification.

Publication must keep passed, failed and untested checks separate. The earlier T4 matching-id report has been
superseded by contributed integrated-revision full-logit results; preserve that provenance. CUDA still needs a
qualified package and its raw-log handoff before being listed as a validated binary release. Upload remains blocked until
GitHub API authentication is available; SSH source pushes alone cannot create/upload a GitHub release.
