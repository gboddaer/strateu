# Kolibri `EXPERT_F16` correctness on Tesla T4 — evidence ledger

This page is the evidence record for
[the plan](../plans/2026-10-30-kolibri-expert-f16-correctness.md). It holds what was run, on what, with what
result. It is not a summary written from memory: every line below names a file, a command or a measurement that
exists.

Read [the accuracy policy](../KOLIBRI_ACCURACY_POLICY.md) first. Accuracy comes before prefill, prefill before
decode, and a usable speedup is not promised.

## Status words used here

| word | meaning |
|---|---|
| **passed** | the command ran, exited 0 on this box, and its output satisfies the stated check |
| **failed** | it ran and did not satisfy the check; the failure is kept |
| **inconclusive** | it ran but cannot decide the question |
| **blocked** | it was not run because something outside this work is missing (a reference, an approval) |
| **not run** | deliberately out of scope at this point |

A primitive test that passes is not an end-to-end fidelity result. A repeated-identical run is not correctness.
Matching greedy tokens are not accuracy.

## Plan and tree identity (2026-10-30)

| item | value |
|---|---|
| branch | `kolibri-expert-f16-correctness` |
| HEAD | `1432941c469a184e458113c275d0a7024bef99df` — `[Docs] fix the EXPERT_F16 plan's blockers found by the code check` |
| plan actually executed | **the uncommitted working-tree revision**, not HEAD's committed copy |
| plan sha256 | `0f4ad114126342abe18433acf55addc8277b937ddbbcefc1c9dcd7f313fef9e9`; exact bytes kept as `proc/PLAN-executed-0f4ad114.md` in the archive |
| plan vs. HEAD | `114 insertions, 238 deletions` on `docs/plans/2026-10-30-kolibri-expert-f16-correctness.md` only |
| working-tree diff, `.pi/` excluded | sha256 `57d2ed5a1d7e7c50acf3d57229b130e57b75a2beb42eba4d41d339369283451b`; one changed file (the plan) |
| other changes | `.pi/` goal bookkeeping only (untracked goal file plus agent state); not part of this work |

The reviewed revision shortens the plan, drops the earlier "a microtest decides defect versus expected loss"
claim, requires a preserved `5bbe6c4e` control, and adds the Q6_K byte-208 fact. That revision is the one being
executed. It is left uncommitted so the review copy is not silently replaced; a copy of its exact bytes is kept
as `/data4/kolibri-f16-evidence/f16-t0/proc/PLAN-executed-0f4ad114.md` (see *Artifacts*).

## Host, toolchain and model

| item | value |
|---|---|
| GPU | Tesla T4, `sm_75`, 16384 MiB; `nvidia-smi` reports 14841 MiB used by a server before Task 0 stopped it, 0 MiB after |
| driver / CUDA | driver 580.196.07; `nvcc` release 12.8, V12.8.93 |
| CPU / RAM | 56 logical CPUs, 2 NUMA nodes (28 + 28); 755 GiB RAM, 497 GiB available at Task 0 |
| disk | `/` 24 G free, `/data2` 48 G free, `/data4` 3.9 T free |
| compiler, build tools | gcc 12.2.0 (Debian 12.2.0-14+deb12u1), CMake 3.25.1, Ninja 1.11.1, Python 3.11.2 |
| model | `/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`, 47,454,113,472 bytes, mtime 2026-10-16 21:26:33 +0200 |
| model sha256 | `c2ac1301424441ef210b6de50ce25e8ccf69f86494df53d6ba52ed558456062e` — computed here in full (`sha256sum`, exit 0) and equal to the shipped `Kolibri-1-Q4_K_M.gguf.done` file |
| inherited `STRATA_*` / `KOLIBRI_*` in the task shell | none. Only `CUDA_HOME=/usr/local/cuda` was set. Every run below sets its environment explicitly |

## The server that was stopped, and how to restore it

A server was already serving on this box, occupying the only usable GPU. The operator authorized stopping it
for exclusive use of the card.

| item | value |
|---|---|
| engine | PID 2519103, `build-kolibri-cuda/strata-kolibri --serve --model /data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf --prefill --cpu-threads 24 --ctx 131072 --expert-slots auto --expert-slots-hot --expert-slots-profile /tmp/kolibri_hot.prof --vram-reserve-mib 1024` |
| front end | PID 2519083, `/data2/develop/Strata/.venv/bin/python serve/server.py --engine strata --config /tmp/strateu_server.json --host 0.0.0.0 --port 8080 --no-api-key` |
| engine environment | `CUDA_VISIBLE_DEVICES=0`, `GGML_CUDA_DISABLE_GRAPHS=1` and the `STRATA_KOLIBRI_*` arms — full list in `proc/engine-environ.txt`; the effective arms were `EXPERT_F16=20`, `PF_PARTIAL=1`, `PF_STAGE=1`, `ROUTER_TILE=1`, `ROUTER_TJ=16`, `ROUTER_TK=3`, `ATTN_MB=256`, `ATTN_FLASH=1`, `RES_TOP=48`, `REUSE_PREFIX=1` |
| engine binary | sha256 `07e37adc5e4ddfae63e5c9a558f040bf051bcdec38a177fbbd5f3ba9a914844c` — byte-identical to the `1432941c` working build made in Task 0 |
| stop | `kill -TERM 2519083`, then 8 s; both PIDs gone; `nvidia-smi` reports 0 MiB used and no compute apps |
| restore | `/tmp/f16-t0/RESTORE-SERVER.sh` (copy in the archive). It re-runs the exact `serve/server.py` line against the unchanged `/tmp/strateu_server.json`, which still holds the engine argv, environment and sampling settings captured before the stop |

`/tmp/strateu_server.json` was not modified. `/tmp/kolibri_hot.prof` and the server's logs were copied, not moved.

## Builds

Both builds use the plan's recipe, so a difference between them is not a difference in configuration:

```sh
cmake -S . -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DSTRATA_ENABLE_HIP=OFF -DSTRATA_BUILD_DSV41=OFF -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_BUILD_TESTS=ON
cmake --build <dir> --target strata-kolibri dequant_bf16_test -j 8
```

| arm | tree | build dir | binary sha256 | build |
|---|---|---|---|---|
| **release control** | `5bbe6c4e7327992d10888cc0c9478ea864b4f324`, clean worktree at `/data2/develop/f16-control-5bbe6c4e` | `build-kolibri-cuda-control` | `74b0e2570d86231068a4193d9e2b226f77748c57bbf8ce89f72be6348ab7f418` | 168 targets, 5 m 13 s, exit 0, 0 `error:` lines |
| **working tree** | `1432941c` + the uncommitted plan | `build-kolibri-cuda` | `07e37adc5e4ddfae63e5c9a558f040bf051bcdec38a177fbbd5f3ba9a914844c` | nothing to rebuild, exit 0 — the tree is what produced the running server's binary |

The control binary is written only inside its own worktree build directory and is never overwritten. The
pre-Task-0 server binary was also copied to the archive before any build, as
`strata-kolibri.server-build-pre-task0.07e37adc…`.

`dequant_bf16_test` was built in both trees: control `16c58336a547f347e7de157f09ffb4971e46e1cdf3be96780c32128d69104c60`
(12,080,216 bytes), working `861bfc9471cb03068d81cab0945086868136276707a7df3f2ca5c0c949d32049`. Both are the
standalone GGUF-driven `main()` described by the plan, not gtest.

## Recovered historical recipe

Recovered from retained files, not from memory:

- **Four-length gate** — `tools/kolibri/decode_gate.sh` at this revision: prompt ids from
  `tools/kolibri/oracle_ids512.txt` repeated to 16 / 511 / 512 / 513 tokens; `--force 286,3515,310,262,14292,1343,1444,46`;
  `--n 8 --prefill --cpu-threads 8 --ctx 2048 --dump-full-logits`; control
  `STRATA_KOLIBRI_DEC_HANDOFF=0 STRATA_KOLIBRI_DEC_STREAM=0 STRATA_KOLIBRI_DEC_HANDOFF_GRAPH=0 STRATA_KOLIBRI_DEC_TOKEN_GRAPH=0`.
  The retained `/tmp/kolibri-t4-decode-gate/*.err` files show the pool banner `8 lane(s) = the host + 7 thread(s)`
  and the two-slice NUMA split, which matches that recipe.
- **Long context** — `/tmp/longctx.sh` (retained): `8191` ids with `--ctx 12288` and `11000` ids with `--ctx 14336`,
  `--force <same eight ids> --n 8 --prefill --cpu-threads 8 --dump-full-logits`, base
  `STRATA_GEMM_DETERMINISTIC=1 STRATA_KOLIBRI_EXPERT_F16=0` plus the four decode arms off.
- **512-token candidate/control pair in `/tmp/t4-cand/`** — no script survives. Their `.err` files show the
  `--cpu-threads`-8 pool banner and, in the `f16only512` case, the line
  "decodes acceleration off (the fp16 expert cache must be off); running the plain decode path". The exact argv
  that produced these three dumps is **unrecovered**; they are used as a consistency check on the recorded
  numbers only, never as fresh evidence.

**Unrecovered:** the shell-level environment of the earlier runs beyond what the retained scripts and `.err`
files show, and the exact argv of the `/tmp/t4-cand` trio. Nothing here claims an exact historical reproduction
from those.

## Inputs and hashes

All under `/tmp/f16-t0/inputs/`, hashes pinned in `SHA256SUMS`:

| file | bytes | sha256 | note |
|---|---:|---|---|
| `ids16.txt` | 76 | `6f2eefab…60b8` | byte-identical to the retained gate's `ids16.txt` |
| `ids511.txt` | 2253 | `eb01050e…90ff` | byte-identical to the retained gate's `ids511.txt` |
| `ids512.txt` | 2258 | `fecef9d5…1951` | byte-identical to the retained gate's `ids512.txt` |
| `ids513.txt` | 2265 | `51e17a45…65ae` | byte-identical to the retained gate's `ids513.txt` |
| `ids8191.txt` | — | `2a51bab0…c480` | retained long-context fixture |
| `ids11000.txt` | — | `72c175a3…fa9d` | retained long-context fixture |
| `forced_ids.txt` | — | `e3cee71e…87e8` | `286,3515,310,262,14292,1343,1444,46`, eight ids → nine records |

Expected geometry for one full-logit dump: `9 records × 128000 × 4 = 4,608,000` bytes.

## Configuration matrix

Three arms, all with `CUDA_VISIBLE_DEVICES=0` and `STRATA_GEMM_DETERMINISTIC=1`. Everything not listed is unset;
no `STRATA_*` value is inherited from the shell.

| arm | `STRATA_KOLIBRI_EXPERT_F16` | `STRATA_KOLIBRI_PF_PARTIAL` | decode arms |
|---|---|---|---|
| control | `0` | `0` | as the run under test requires (`0` for the cross-revision comparison; the gate's four explicit zeros for the gate) |
| cache-only | `24` | `0` | same as the control for that run |
| cache+partial | `24` | `1` | same as the control for that run |

`DEC_RES` and the unrelated opt-in knobs (`PF_STAGE`, `ROUTER_*`, `ATTN_*`, `RES_TOP`, `REUSE_PREFIX`) are set
explicitly per run or left unset, and the effective set is written into each run's `.env` file. A run whose
environment is not recorded is not evidence.

## Independent reference tooling

| item | state |
|---|---|
| dumper | `/data2/develop/strata-speed-wt/build/kolibri_llama_dump`, from `tools/kolibri/kolibri_llama_dump.cpp` at `strata-speed-wt` `023cbf2e`; it links the llama.cpp libraries in `/home/gbo/llama.cpp/build/bin` |
| llama.cpp under it | `/home/gbo/llama.cpp` at `a30fb4924` `[SYCL]: fix memory errors in mul_mat, split buffer, host pool (#29889)`, dirty in `conversion/` (an added `conversion/kolibri.py`) |
| load probe on this box | **passed** — 1,536,000-byte dump (`3 × 128000 × 4`) from `--tokens-file ids16.txt --force 286,3515 --ngl 99 --cpu-moe 999`, exit 0, on 2026-10-30 |
| offload arrangement feasible here | all 50 expert layers on the CPU (`--cpu-moe 999`), because the model is 47.45 GB and the card is 16 GB. An all-GPU llama.cpp arrangement is impossible on a T4. Recording this is required before any KL from it can be interpreted |
| limit | `tools/kolibri/kl.py` fixes `STRATA_DUMP_CTX=8192`, truncates with `min(len(P), len(Q))`, keys its reference cache by filename and deletes the Strata dumps. It is not a long-context qualification harness as it stands |
| `strata-kolibriref` | shares this engine's kernels; usable as a diagnostic control only, never as the independent reference |

## Task results

### Task 0 — freeze the control and make the experiment reproducible

| check | result |
|---|---|
| branch, status and dirty identity recorded | **passed** |
| running server recorded, stopped with permission, restore recipe written | **passed** |
| RAM, disk, NUMA and toolchain recorded | **passed** |
| control built separately at `5bbe6c4e` with `STRATA_Q6K_EXPERTS=ON`, hashed, never overwritten | **passed** |
| working build at `1432941c` built and hashed | **passed** |
| `/tmp/klcmp.py`, `/tmp/kolibri-t4-decode-gate/`, `/tmp/t4-cand/` preserved, plus `/tmp/t4-longctx/`, `/tmp/longctx.sh`, the ids fixtures, the server captures and both build logs | **passed** |
| historical argv/environment recovered where retained, gaps marked unrecovered | **passed, incomplete by nature** |
| input ids rebuilt and pinned; they reproduce the retained gate inputs byte for byte | **passed** |
| configuration matrix written down | **passed** |
| reference tool identified, its llama.cpp revision recorded, and a load probe run on this card | **passed**; not yet exercised at 8191 / 11000 tokens |
| no engine code changed in this task | **passed** — the diff for Task 0 is documentation only |

### Task 1 — reliable full-logit comparison and reproduction

**Instrument.** `tools/kolibri/klcmp.py` with `tools/kolibri/test_klcmp.py`, committed together with this entry.
Interface: `load(path, vocab)`, `summary(ref, cand)`, `main(argv=None)`;
CLI `--ref --cand --vocab [--json]`. Exit 0 = comparable and finite, 1 = non-finite input, 2 = invalid
shape or input. It reports record count, bytewise equality on float32 bits, differing records, max |a-b| and
where, per-record KL(P||Q), mean/median/p95/p99/max KL with the worst step, top-1 agreement, the reference's
winning margin, and the candidate's rank for the reference winner.

| check | result |
|---|---|
| tests written first, run before the implementation existed | **failed as expected**: `ModuleNotFoundError: No module named 'klcmp'` (`/tmp/f16-t1-tests-before-genuine-failure.txt`) |
| `pytest tools/kolibri/test_klcmp.py -q` | **passed**, 22 tests |
| hand-computed KL: P=[0,0,0] against Q=[ln 2,0,0] | **passed**, 0.0566330 against the closed form `1/3[ln(2/3)+2ln(4/3)]` |
| `-0.0` against `0.0`: bytewise different, value-equal | **passed**, reported not-identical with max |a-b| 0.0 |
| non-finite, record-count mismatch, empty, truncated, extra trailing bytes, vocab 0 / -1 | **passed**, rejected with the documented exit codes |
| do the tests detect wrong behaviour? | **passed**: reversing KL to `KL(Q||P)` fails the hand-computed and JSON tests; replacing the bitwise check with value equality fails the signed-zero and identity tests |
| comparison never truncates to the shorter dump | **passed**: a size that is not a whole number of records is `InvalidInput`, exit 2 |

A first version of `load()` used `np.fromfile` alone and **failed** the trailing-bytes test, because `numpy`
silently drops a partial element. The file size is now checked before reading. That failure is kept in the test
suite.

**Retained-pair check** (`tools/kolibri/klcmp.py --ref /tmp/kolibri-t4-decode-gate/base512.bin --cand
/tmp/t4-cand/f16only512.bin --vocab 128000`, exit 0):

```
records 9 x 128000 float32; bitwise identical: False
differing records 9/9; max |a-b| 11.0808 at record 1 index 1776
KL(P||Q)  mean 0.400402  median 0.000956086  p95 2.14169  p99 3.24207  max 3.51717 (step 1)
top-1 agreement 1.0000 (9/9)
reference margin at the worst step 0.638462; smallest margin 0.638462 at step 1
worst rank of the reference winner in the candidate 1
```

This reproduces the recorded table in [the T4 note](../KOLIBRI_CUDA_PREVIEW.md) to every printed digit (0.400 /
0.0010 / 3.24 / 11.08 / 9-of-9), so those two retained dumps are on the same forced history. The argv that made
the `/tmp/t4-cand` trio is still unrecovered; this is a consistency check, not fresh evidence.

**Three-arm matrix**, all with the preserved `5bbe6c4e` control binary
(`74b0e2570d86231068a4193d9e2b226f77748c57bbf8ce89f72be6348ab7f418`), `ids512.txt`
(`fecef9d5…1951`), forced `286,3515,310,262,14292,1343,1444,46`, `--ctx 2048 --prefill --cpu-threads 8`,
`--expert-slots` default, the four decode arms off, `STRATA_GEMM_DETERMINISTIC=1`, two runs per arm:

| arm | `EXPERT_F16` | `PF_PARTIAL` | runs | exit | dump bytes | wall |
|---|---:|---:|---|---|---:|---|
| control | 0 | 0 | 2 | 0 / 0 | 4,608,000 / 4,608,000 | 55 s / 55 s |
| cache-only | 24 | 0 | 2 | 0 / 0 | 4,608,000 / 4,608,000 | 49 s / 50 s |
| cache+partial | 24 | 1 | 2 | 0 / 0 | 4,608,000 / 4,608,000 | 48 s / 48 s |

Same-arm repeats first: **all three arms are bitwise identical run to run** (`check_decode.py` PASS, max |a-b| 0.0;
`klcmp.py` KL 0 everywhere). The divergence between arms is therefore deterministic arithmetic, not a race,
and the cross-arm differences below are not run-to-run noise. The wall times are incidental here and are **not**
a performance measurement.

Cross-arm, on the identical inputs and the same binary (P = first column):

| comparison | mean KL | median | p95 | p99 | max KL (step) | max abs |a-b| | top-1 | worst rank of ref winner |
|---|---:|---:|---:|---:|---|---:|---|---:|
| control → cache-only | 0.400402 | 0.000956 | 2.14169 | 3.24207 | 3.51717 (1) | **11.0808** (rec 1, idx 1776) | 9/9 | 1 |
| control → cache+partial | 0.167928 | 0.000518 | 0.87110 | 1.25618 | 1.35245 (4) | 5.4710 (rec 8, idx 19969) | 7/9 | 3 |
| cache-only → cache+partial | 0.465808 | 0.001796 | 2.29565 | 2.96170 | 3.12821 (1) | 10.3399 (rec 1, idx 1776) | 7/9 | 3 |

The second run of each pair gives the same numbers as the first, to every digit.

Shape of the error, recorded to point Task 3 and Task 4, **not** as an explanation: it is broad, not one bad
value. Per record (control → cache-only) the RMS logit difference is
0.22, 1.13, 0.82, 0.29, 0.59, 1.00, 0.65, 0.18, 0.46, and the count of logits differing by more than 1.0 is
0, 47,041, 30,949, 105, 11,546, 38,784, 15,357, 0, 1,421. The 45 largest-error slots across the nine records are
45 distinct vocabulary indices, so no single weight row explains the tail. At the worst slot (record 1, index
1776) the reference logit is 15.4681 against 4.3873, which moves that token from rank 2 to rank 103 in the
candidate.

**What Task 1 establishes:** the recorded discrepancy reproduces on a preserved control build with recorded
inputs and environment, it is deterministic, and it is not caused by the comparator. **What it does not establish:**
whether the difference is a defect or expected fp16 arithmetic. That is Task 4, and no number here is an
acceptance decision.

Artifacts: `/data4/kolibri-f16-evidence/f16-t1/` (matrix dumps, per-run `.env`, `.err`, `.out`, `.status`,
`SHA256SUMS`, `retained-pair-512.txt`, `repeats.txt`, `cross-arm.txt`, `error-shape.txt`), live copies under
`/tmp/f16-t1/`.

### Tasks 2–8

Not run yet. Each will be filled in with its commands, exit statuses, raw artifacts and a status word.

## Artifacts

Indispensable evidence is archived outside `/tmp`, which is not durable:

| path | contents |
|---|---|
| `/data4/kolibri-f16-evidence/f16-t0/proc/` | server identity, engine and front-end environments, `strateu_server.json`, GPU state before the stop, restore script |
| `/data4/kolibri-f16-evidence/f16-t0/preserved-from-tmp/` | `klcmp.py`, `kolibri-t4-decode-gate/`, `t4-cand/`, `t4-longctx/`, `longctx.sh`, `ids8191.txt`, `ids11000.txt` |
| `/data4/kolibri-f16-evidence/f16-t0/*.log`, `*.sha256` | both build logs, control and working binary hashes, the model hash |
| `/tmp/f16-t0/` | live working directory for this task: inputs, manifest, per-run logs |

The model files and the multi-megabyte logit dumps are never committed.
