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

### Task 2 — verify the fp16 conversion separately from fidelity

**What existed.** `dequant_bf16_test` was a GGUF-driven `main()` that compares the device dequantizers against the
artifact's CPU decoders for FP32 and BF16. It never called `dequant_f16`, which is the function the expert cache
uses. `dequant_f16` writes fp16 bits with `__float2half_rn` (`src/kernels/cuda/dequant_bf16.cu`); Q6_K's `d` is at
byte **208** (the kernel reads `h2f(b + 208)`), Q4_K's `d`/`dmin` at bytes 0/2.

**GGUF mode, before the change** (`dequant_bf16_test …Kolibri-1-Q4_K_M.gguf`, exit 0):
`type 14 output.weight 4 x 2560: f32 ok … bf16 ok`, `type 12 token_embd.weight 4 x 2560: f32 ok … bf16 ok`,
`2 types checked, OK`. Only two of the model's many tensors are 2-D and supported by that path.

**Registered, then run before implementing.** `add_test(NAME kolibri_dequant_f16_selftest COMMAND
dequant_bf16_test --selftest)` was added inside the existing `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)` guard and
the test was run first. It **failed** with

```
terminate called after throwing an instance of 'std::runtime_error'
  what():  cannot open --selftest
```

ctest exit 8 (subprocess aborted), not a usage exit 2 — exactly the trap the plan named: the unknown flag had
reached the GGUF-opening path.

**What the selftest does.** `--selftest` builds synthetic but valid Q4_K (144-byte) and Q6_K (210-byte) super-blocks
from a splitmix64 seed, evaluates them with the artifact's CPU decoder `strata::dequantize_q4_K/q6_K`, and compares
three things separately:

1. GPU `dequant_f32` against the CPU decoder, relative 1e-6 — catches a wrong offset, stride or scale unpack;
2. GPU `dequant_f16` bits against a **host round-to-nearest-even fp32→fp16 conversion written in the test**, applied
   to the GPU's own f32 values — this is the conversion check, and it does not share an implementation with
   `__float2half_rn`;
3. the same against the CPU f32 values, reported separately, so a CPU/GPU f32 difference is never blamed on the
   conversion.

The host converter is itself pinned to a 13-entry table of halfway, subnormal and overflow cases (`65519 → 0x7bff`,
`65520 → 0x7c00`, `2^-25 → 0`, `3·2^-25 → 0x0002`, `1+2^-11 → 0x3c00`, `1+3·2^-11 → 0x3c02`, …); the expected bits
were taken from an independent `numpy.float16` conversion, not from this code. Fixtures cover Q4_K and Q6_K
independently, exact zeros, normal values with a **non-power-of-two** scale (0.0001234 / 0.0000371, so the products
are inexact and the rounding path is taken), fp16-subnormal-range products (d = 2^-20), one and two super-blocks per
row, a row slice at `row0 = 1`, and an output stride `ld = cols + 8` whose padding must stay untouched. A type this
test claims to cover that is not registered is a **failure, not a skip**.

**Result** (`ctest --test-dir build-kolibri-cuda -R '^kolibri_dequant_f16_selftest$' -V`, exit 0):

```
selftest: host RNE conversion table ok (13 cases, halves and subnormals included)
selftest: Q4_K normal    4 x 256  f32 vs CPU ok (0 off, worst rel 0.00e+00);  f16 vs RNE(GPU f32) ok (0 off, worst 0 ulp) … rounded 980, subnormal 3
selftest: Q6_K normal    4 x 256  f32 vs CPU ok (0 off, worst rel 0.00e+00);  f16 vs RNE(GPU f32) ok (0 off, worst 0 ulp) … rounded 977
selftest: Q6_K           row slice row0=1 rows=2  ok (0 off)
selftest: Q6_K           ld=cols+8                ok (launched 1, 0 off, 0 padding value(s) written)
selftest: 7168 value(s) covered, 4935 rounded, 146 subnormal -> OK
```

On every covered value the GPU's f32 path is bit-identical to the CPU decoder, and the fp16 bits are the
round-to-nearest-even result of the GPU's own f32 values (0 ulp). 4,935 of the 7,168 values actually took the
rounding path and 146 landed in the fp16 subnormal range, so the pass is not a pass over exact values only.

**Do the assertions detect a wrong conversion or offset?** Three disposable mutations, all detected:

| mutation | result |
|---|---|
| `dequant_f16` replaced by `dequant_bf16` in the test (a bf16/f16 mix-up) | **failed**: 1,022 / 984 values off, worst 13,778 ulp |
| row slice called with `row0 = 0` instead of 1 | **failed**: 512 values off |
| kernel's Q6_K `d` offset `b + 208` changed to `b + 202`, rebuilt | **failed, exit 1**: f32 vs CPU off by up to 1.2e8 relative; the source was then `git checkout`-restored byte-identically and the rerun was exit 0 |

**GGUF mode after the change** is byte-identical to before it (`2 types checked, OK`, exit 0). The engine binary
`build-kolibri-cuda/strata-kolibri` is unchanged (`07e37adc…`), because only the test target and the kernel library
were rebuilt.

**What this establishes:** the fp16 *weight* conversion is round-to-nearest-even and its block offsets are right for
covered Q4_K/Q6_K inputs, so an 11-logit error is not explained by a broken converter. **What it does not
establish:** that the conversion is the whole fp16 story. The expert path also converts *activations* and the SwiGLU
hidden state to fp16 (`kolibri_engine.cu`), and it accumulates over a K of thousands rather than 256. Task 4 owns
those, and a primitive pass here is not an end-to-end fidelity result.

**Backends:** the test is registered under the CUDA/HIP guard and built and run here with CUDA on `sm_75`. A HIP
build was **not run** (no ROCm host in this work); SYCL is unaffected by this test.

Artifacts: `/data4/kolibri-f16-evidence/f16-t2/` — the pre-implementation failure, the implemented run, both GGUF
runs, the three mutation runs and both build logs.

### Task 3 — locate the first observable divergence

**The writer, read before writing the reader.** `src/kolibri/kolibri_engine.cu` (Kolibri's `run()`, which is what
`strata-kolibri` dispatches to at `generate.cpp` ~:2956) allocates
`std::vector<float> layers(n_layers * n_embd)` and, per processed position, calls `forward(e, s, tok, layers.data())`
and then writes `n_embd` floats per layer for `l = 0..n_layers-1`. Layout: **position-major, layer-minor, no input
slot** — `512,000` bytes per position at 50 x 2560 x 4, so the gate's nine positions are **4,608,000 bytes**, which is
what every run below produced. `forward()` fills it from `s.layer_trace_d` via `k_layer_snapshot` (one snapshot per
layer into slot `l`), i.e. the residual *after* layer `l`. `tools/kolibri/compare_dumps.py` reshapes to
`(-1, 50, 2560)`, which agrees with the writer.

**Instrument.** `tools/kolibri/layer_delta.py` and `tools/kolibri/test_layer_delta.py` (15 tests, written first:
the pre-implementation run failed with `ModuleNotFoundError: No module named 'layer_delta'`). It reports, per
(position, layer), `||a-b||/||a||` with `a` the reference, plus the median and the max over positions for every
layer *and* over layers for every position, so a single bad position cannot hide inside a median. It rejects empty,
misaligned, wrong-record-count and non-finite input explicitly (exit 2 / 1). Two tests exist because of mistakes the
first implementation made: the denominator was the candidate rather than the reference, and a zero reference norm
was flagged even when the candidate was also zero. Both are now pinned.

**Four runs**, preserved control binary, same ids/forced history as Task 1, `--ctx 2048 --prefill --cpu-threads 8`,
`--dump-layers` plus `--dump-full-logits` plus `--counters`, the four decode arms off:

| tag | `EXPERT_F16` | slots | exit | layer dump | logits |
|---|---|---:|---|---:|---:|
| `ctl_default` | 0 | default | 0 | 4,608,000 B | 4,608,000 B |
| `cache_default` | 24 | default | 0 | 4,608,000 B | 4,608,000 B |
| `ctl_slots0` | 0 | `--expert-slots 0` | 0 | 4,608,000 B | 4,608,000 B |
| `cache_slots0` | 24 | `--expert-slots 0` | 0 | 4,608,000 B | 4,608,000 B |

All four dumps are complete (size is a whole number of positions) and finite — `layer_delta.py` refuses either
otherwise and returned 0. The full-logit pair `ctl_default` vs `cache_default` reproduces Task 1 exactly (mean KL
0.400402, max |a-b| 11.0808 at record 1 index 1776, top-1 9/9), and both are **bitwise identical to Task 1's
`ctl_r1` / `cache_r1`**, so enabling `--dump-layers` did not change the arithmetic being compared.

**Resident activity, measured rather than assumed** (`--counters`, `print_counters` in `kolibri_engine.cu`):

| line | control | `EXPERT_F16=24` |
|---|---|---|
| `CNT vram free` after dense upload | 13.53 GiB | **4.74 GiB** |
| `CNT prefill pool` | `chunks=100 experts/chunk=117 bytes=28.09 GB pool=8347 ms` | `chunks=100 experts/chunk=118 bytes=22.44 GB pool=2439 ms **promotes=1551 fill=12.20 GB**` |
| `CNT run` slot hits/misses | `hits=0 miss=0` | `hits=0 miss=0` |
| `CNT prefill pool work` | `rows=153300` | `rows=43161` (the misses the fp16 cache absorbs) |

So the fp16 expert cache *is* allocated and used — during **prefill** — and the decode loop's slot map is untouched
in both arms.

**Where the numbers first differ** (`layer_delta.py --a ctl_default.layers --b cache_default.layers`):

```
positions 9 x layers 50 x embd 2560; relative L2 ||a-b||/||a|| (a = reference)
overall  median 0.103995  max 1.05077 at position 1 layer 33
per layer median over positions: layer0 0 | l1 0.000155 | l2 0.00062 | l3 0.00145 | l4 0.00195 | l5 0.00294 ...
per layer max   over positions: layer0 0 | l1 0.000267 | l2 0.000879 | l3 0.00455 ... l33 1.05 ... l49 0.704
positions/layers above 1e-3: 421; first (0, 3)
```

* Layer 0 is **bitwise identical in all nine positions**; the first non-zero difference is layer 1 (`1.5e-4`
  median), and the first crossing of the navigation threshold `1e-3` is at **position 0, layer 3**.
* The difference then grows with depth: median `0.156` by layer 30, max `1.05` at position 1 / layer 33, and it stays
  around `0.3-0.4` median to layer 49.
* `--expert-slots 0` versus the default is **bitwise identical in both arms** (`rel` exactly 0 at every position and
  layer), so the hot-slot count is not what this divergence depends on — the fp16 cache is a different cache from
  the slot map (`b.f16_slot_id` versus `slot_of`).

**The decode leg on its own is unaffected.** Two more runs with **no `--prefill`** (16 prompt ids fed through the
decode path, 24 positions dumped, `nopf_ctl` / `nopf_cache`): the layer dumps (24 x 50 x 2560) and the nine full-logit
records are **bitwise identical** between `EXPERT_F16=0` and `24`, and the counters show neither a `promotes=`/`fill=`
line nor any change in bytes — the fp16 cache was never used.

**What Task 3 establishes:** the divergence observed in the full logits is produced on the **prefill leg**, where the
fp16 expert cache is actually exercised (`promotes=1551`, `fill=12.20 GB`), and reaches the decode outputs only
through what prefill leaves behind (the KV values); the decode path's own expert arithmetic is identical between the
arms. That is a narrowing, not an attribution: it does **not** say the difference was made in layer 3, and the
layer-0/ layer-1 asymmetry is unexplained — a plausible reading is that prefill's layer-0 K/V stays bitwise
identical while later layers' does not, but that is a hypothesis for Task 4, not a result. Task 4 therefore
instruments the **prefill** expert path (first chunks, early layers) and the KV it writes, not the decode loop.

Artifacts: `/data4/kolibri-f16-evidence/f16-t3/` — the four runs (`.layers`, `.bin`, `.env`, `.err` with the CNT
lines, `.out`, `.status`), the two no-prefill runs, `layer-delta*.txt`, `logits-ctl-vs-cache.txt`,
`nopf-*.txt`, `SHA256SUMS`.

### Task 4 — isolate the arithmetic and the actual cache wiring

**The instrument.** `tests/kolibri/f16_expert_gemm_parity.cu`, registered as `f16_expert_gemm_parity` (synthetic
shapes by default, model shapes with `--embd 2560 --ff 512`). It separates the two things the FP16 expert path can
contribute, for the two expert projections the model actually runs — gate/up as **Q4_K** (`N = 512`, `K = 2560`) and
down as **Q6_K** (`N = 2560`, `K = 512`), the shapes read from the model's own metadata (`expert_feed_forward_length`
512, `embedding_length` 2560, 384 experts, 6 used):

* **(a)** float64 dot over the *unrounded* operands: CPU-dequantized f32 weights and the f32 activations;
* **(b)** float64 dot over the operands cuBLAS *actually receives*: the FP16 weights the device produced and the FP16
  activations, widened exactly;
* **(c)** the GPU's `Gemm::f16_batched` result.

The cuBLAS contract was read from `src/prefill/gemm.cu`: `cublasGemmBatchedEx`, A = W `CUDA_R_16F`, B = X
`CUDA_R_16F`, C = Y `CUDA_R_32F`, `CUBLAS_COMPUTE_32F`, `CUBLAS_GEMM_DEFAULT` — FP16 inputs, FP32 accumulate. The
gate is therefore **accumulation <= 4 · gamma_K · sum|w·x|** with `gamma_K = K·u/(1-K·u)`, `u = 2^-24`; the factor 4
is headroom for the order, not a fitted threshold. Operand rounding is reported, not gated: it is inherent to
storing weights and activations as FP16, and the plan asks for it to be kept separate.

A third, gating check was added after a mutation showed it was needed: **the FP16 weights the GPU used must be the
host's RNE conversion of the CPU-dequantized blocks**. Without it the test is blind to a shifted block offset or a
wrong-conversion path, because those move (a) and (b) together. Only 0 mismatches are accepted.

**Result** — 16 comparisons per shape set, `T` at the `kResBlk = 32` block boundary (1, 31, 32, 33), `batch` 1 and 3:

```
f16_expert_gemm_parity (toy, 256/256):      16 comparison(s), 0 failing value(s) -> PASS
f16_expert_gemm_parity --embd 2560 --ff 512: 16 comparison(s), 0 failing value(s) -> PASS
  gate N= 512 K=2560 T=33 batch=3  max|Y| 482.2  max sum|w*x| 4590.8  operand-rounding 4.98e-05  accumulation 2.85e-03 (0.01x of gamma_K*sum|w*x|)  weights not the reference's: 0
  down N=2560 K= 512 T=33 batch=3  max|Y| 413.3  max sum|w*x| 2083.0  operand-rounding 9.17e-05  accumulation 6.98e-04 (0.02x of gamma_K*sum|w*x|)  weights not the reference's: 0
```

Across every model-shape comparison: **operand rounding 3.0e-5 … 9.7e-5** relative to that element's own
`sum|w·x|`, and **accumulation at most 0.02x** of the fp32 ordering bound. So on the model's expert shapes the FP16
GEMM is fp32 accumulation over the rounded operands, and the rounding it introduces is ~1e-4 relative — the same
order as the first layer-to-layer residual difference Task 3 measured.

**The gate detects the wiring failures it exists for** (`--mutate`, three disposable runs):

| mutation | result |
|---|---|
| `swap` — two experts' weight pointers exchanged in the device array | **failed**: 99,287 values |
| `offset` — the Q6_K blob read one super-block late (pointer shifted, reference not) | **failed**: 2,095,858 values |
| `rows` — the reference covers `T-1` rows while the GPU computes `T` | **failed**: 8,192 values |

The `offset` mutation is worth recording twice: the first two attempts were **not detected**, because shifting the
fixture symmetrically moves both sides, and because a reference derived from the device's own weights cannot see a
wrong weight. Only the pointer-only shift plus the explicit weight-bits check caught it. That is the failure mode the
plan warned about, found by trying to break the test rather than by trusting it.

**Opt-in trace, default off** (`STRATA_KOLIBRI_F16TRACE=1`, `src/kolibri/kolibri_engine.cu`), two lines per
(chunk, layer), both placed where the loop already has the values on the host, so **no arithmetic change and no extra
synchronisation**:

* `F16TRACE chunk=… l=… picks=… uniq=… ids=<FNV-1a over the selected ids>` — did the two arms route the same experts?
* `F16SLOT chunk=… l=… residents=… n_cached=… res_eff=… dup=…` — the cache-ownership invariant of the promotion loop.

**What the trace shows** (512 ids, one prompt, `EXPERT_F16=0` against `24`, same inputs, 100 (chunk, layer) records):

| record | control digest | cache digest |
|---|---|---|
| chunk 0, layer 0 | `d5f5346b` | `d5f5346b` (identical) |
| chunk 0, layer 1 | `faf887c0` | `faf887c0` (identical) |
| chunk 0, layer 2 | `5db9db00` | `7c471334` (**differs**) |
| chunk 0, layer 3 | `4fdc34d7` | `210bb70a` (differs) |

**4 of 100** records route identically; 96 differ, while the pick count stays 1536 in every record. So the arms
agree completely through the first two layers and then select different experts — the router, a discrete function
of a slightly different hidden state, starts flipping once the first FP16-rounded expert output has propagated into
it. That is the amplification path from a ~1e-4 rounding difference to a ~1.0 relative residual difference and an
11-logit tail, and it is consistent with everything measured so far.

**The cache-ownership invariant**, measured in situ over the same 100 records: `residents=24`, `n_cached=24`,
`res_eff=24`, **`dup=0` in all 100** — every resident of a chunk owns its own slot, and residency is capped at the
slot count. This is the invariant whose violation the engine's own comments record as a previously fixed P1 bug
(duplicate slot indices, wrong expert weights). It holds on this configuration; the probe is default-off and the
run with it is bitwise identical to the run without it (`klcmp` reports bitwise identical, 0 differing records).

**What Task 4 establishes:** the FP16 expert GEMM converts and multiplies correctly on the model's shapes within an
fp32 accumulation bound, and the cache gives each resident in a chunk its own slot. The large full-logit difference
is explained by measured components — ~1e-4 per-GEMM rounding, then router flips from layer 2 onward, then
amplification — rather than by a demonstrated wiring defect. **What it does not establish:** that this explanation
covers the whole 11-logit tail, or that the path is accurate. The engine's SwiGLU and hidden-FP16 conversion inside
the resident path were **not** replayed against a float64 reference here, and no invariant was tested across
requests, resets or cancellations (the plan's Task 6 owns state transitions). The static reading of the promotion
loop (reuse by id, free slot, else least-recently-touched, `res_eff = min(res_top, n_cached)`) is **unconfirmed
static analysis** beyond the distinctness the probe measured.

Artifacts: `/data4/kolibri-f16-evidence/f16-t4/` — `toy.txt`, `model-shapes.txt`, the four `mutate-*.txt` runs,
`trace/` (both arms' `.err` with the F16TRACE lines, the normalized traces, the differing records, the slot-probe
run and its bitwise comparison).

### Task 5 — fix demonstrated defects, prove non-regression

**No defect was demonstrated, so no arithmetic changed.** Task 4 measured the FP16 expert GEMM inside an fp32
accumulation bound on the model's own expert shapes, found the cache gives each resident of a chunk its own slot,
and explained the divergence through measured components (rounding → router flips → amplification) rather than a
wiring fault. Writing a "fix" for an undemonstrated cause would be the change this plan forbids, so the only code
this task's evidence covers is the default-off trace added in Task 4. Under the accuracy policy, fixing nothing does
**not** qualify the changed arithmetic, and nothing here does.

**The cross-revision proof, which is the part that must hold.** The new build (`0bc772bf…`, `c1efc4a6` plus the
trace) against the preserved `5bbe6c4e` control (`74b0e257…`), identical forced histories, cache off
(`EXPERT_F16=0`, `PF_PARTIAL=0`, the four decode arms off), `--prefill --cpu-threads 8`:

| prompt length | ctx | control | new | result |
|---:|---:|---|---|---|
| 16 | 2048 | exit 0, 4,608,000 B | exit 0, 4,608,000 B | **PASS** 9 x 128,000 bitwise identical, max abs 0.0 |
| 511 | 2048 | exit 0 | exit 0 | **PASS** bitwise |
| 512 | 2048 | exit 0 | exit 0 | **PASS** bitwise |
| 513 | 2048 | exit 0 | exit 0 | **PASS** bitwise |
| 8,191 (KV wrap) | 12,288 | exit 0 | exit 0 | **PASS** bitwise |
| ~11,000 (attention transition) | 14,336 | exit 0 | exit 0 | **PASS** bitwise |

All twelve runs exited 0 with a complete 4,608,000-byte dump; every comparison is 9 records x 128,000 values with
`max |a-b| 0.0`. The 8,191 and 11,000 runs use the retained fixtures (`ids8191.txt`, `ids11000.txt`) and the recipe
recovered in Task 0. This is the check the plan calls for and the one an arms-only comparison cannot replace.

**`tools/kolibri/decode_gate.sh check`** with the new build and the cache off, `STRATA_GEMM_DETERMINISTIC=1`:

```
  1.16  PASS  PASS 9 record(s) x 128000 value(s) bitwise identical; max |a-b| 0.0
  1.511 PASS  PASS ... 1.512 PASS  PASS ... 1.513 PASS  PASS ...
  2.reuse PASS  2 request record(s) (cold 512 ids, then +64) bitwise identical, headers included
  3.repeat PASS  REPEAT PASS
DECODE GATE PASS
```

**Affected tests** (`ctest -R 'kolibri|dequant|f16_expert|q4k|gemm_dequant'`, exit 0):

| test | result |
|---|---|
| `kolibri_load_test`, `kolibri_window_test`, `kolibri_hot_policy_test` | passed |
| `kolibri_decode_wait_test` | passed |
| `q4k_grouped_vram_parity`, `f16_expert_gemm_parity` | passed |
| `dequant_s2_parity`, `dequant_wide_parity`, `gemm_dequant_side_parity`, `dequant_q5_1_test`, `ple_iq4nl_dequant_test` | passed |
| `kolibri_dequant_f16_selftest` | passed |
| `kolibri_router_parity`, `kolibri_pf_partial_guard_test`, `kolibri_pf_partial_guard_explicit_zero_test`, `kolibri_cpu_threads_floor_test` | passed |
| `kolibri_res_top_zero_test` | **skipped** by its own design |

16 passed, 0 failed, 1 skipped. `pytest tools/kolibri/test_klcmp.py tools/kolibri/test_layer_delta.py` → 37 passed;
`tools/kolibri/test_build_portability.py` → 2 passed.

**Backends.** CUDA (`sm_75`) built and executed: passed as above. **HIP: not run** — no `hipcc` on this host.
**SYCL: not run** — no `icpx` on this host. The touched files are the CUDA-only test registration, a host `.cpp`
test, `tests/kolibri/*.cu` for the same runtime via `_strata_gpu_runtime_target`, and two default-off `fprintf`
blocks in `kolibri_engine.cu`; that is a compilation claim for HIP/SYCL, not a tested one. The **full 210-test ctest
suite was not run** in this task; only the affected subset above.

Artifacts: `/data4/kolibri-f16-evidence/f16-t5/` — `cross/` (all twelve runs with `.env`, `.err`, `.out`, dumps,
`compare.txt`, `binaries.txt`), `gate/` and `gate-output.txt`, `ctest-affected.txt`.

### Task 6 — proposed qualification protocol, awaiting approval (no results collected)

**Nothing has been measured for qualification yet, and this section is written before any is.** The numbers quoted
anywhere above are diagnostic, not preregistered qualification, and will not be relabelled as such.

Proposed, for the maintainer to accept or amend:

1. **Corpus.** Three families of forced history, all run on the same ids for every arm: (a) 16-token and 512-token
   prompts from `tools/kolibri/oracle_ids512.txt`; (b) the 8,191-id KV-wrap fixture; (c) the ~11,000-id
   attention-transition fixture; (d) two additional prompt texts (one German, one code-shaped) of about 512 tokens,
   recorded with their ids. Nine forced decode steps each, as in every run above.
2. **Arms.** control (`EXPERT_F16=0`, `PF_PARTIAL=0`), cache-only (`24`, `0`), cache+partial (`24`, `1`), on the
   current build; the preserved `5bbe6c4e` control also dumped on the same histories so the reference comparison is
   not taken against a moving build.
3. **Reference.** The pinned llama.cpp dumper (`kolibri_llama_dump`, `/home/gbo/llama.cpp` `a30fb4924`) with
   `--ngl 99 --cpu-moe 999` — all experts on the CPU, which is the only arrangement that fits a 16 GB T4 with a
   47 GB model. Its offload/precision arrangement is recorded with every dump, and it is labelled as a CPU-expert
   reference, not an all-GPU one. `strata-kolibriref` is a diagnostic control only.
4. **Metrics.** Full-distribution KL (mean/median/p95/p99/worst) per arm against the reference, `|dlogit|`
   percentiles and worst, top-1 agreement, changes in the reference winner's rank and its margin, per prompt and
   per position, plus control-vs-reference reported separately so a defective control is visible.
5. **Proposed acceptance limit** (the maintainer must set or change this): mean KL <= 1e-3, p99 <= 5e-3, worst
   `|dlogit|` <= 0.05, top-1 agreement 100 % on every prompt, and no rank-1 flip where the reference margin exceeds
   0.05. Measured so far, the cache arms sit at mean KL 0.400 and 11.08 worst `|dlogit|` for a *diagnostic* 512-token
   case, so this limit is expected to **reject** both cache arms; the value of running the protocol is then to record
   that rejection on the record rather than to find a passing threshold. Any looser limit must be argued explicitly
   and is not proposed here.
6. **State checks.** Two repeats per arm plus first use, replay, prefix reuse, reset, cancellation and a clean
   request after cancellation, for cache-only and cache+partial, comparing like-for-like cold and reused histories.
   Non-finite values, stale outputs or unexplained same-arm variability reject regardless of the KL numbers.
7. **What this cannot decide.** Task quality (Task 7) and speed (Task 8) are separate. No default changes, no push.

**Pre-qualification work done while the criteria are unapproved** (no candidate has been measured against the
reference; nothing here is a qualification result):

* **A harness that fixes the four limits of `tools/kolibri/kl.py`.** `tools/kolibri/qualify.py` with
  `tools/kolibri/test_qualify.py` (13 tests, passing). It requires an **exact** record count (a shorter or longer dump
  is an error, never a prefix), takes the context and every other setting **explicitly per executable**, keys a
  reference dump by a SHA-256 of the configuration that produced it, and **never deletes a raw dump**. It refuses to
  run a child when the calling shell carries a setting that would change the numbers but is not in the explicit
  environment, and it writes argv, environment, exit status and byte count into a manifest beside the dump. It
  reports KL mean/median/p95/p99/worst, `|dlogit|` p50/p95/p99/worst, top-1 agreement, the worst rank of the
  reference winner, and the reference margin at every rank-1 flip. `--limit` compares a metric bundle with limits
  agreed in advance; it does not decide task quality.
* **The independent reference works at the required contexts.** `kolibri_llama_dump`
  (`/home/gbo/llama.cpp` `a30fb4924`), `--ngl 99 --cpu-moe 999`, with `STRATA_DUMP_CTX` set per run:
  | history | ctx | batch | exit | dump | records | finite |
  |---|---:|---:|---|---:|---|---|
  | 8,191 ids (KV wrap) | 12,288 | 2048 | 0 | 4,608,000 B | 9 | yes, range −13.82 … 13.98 |
  | ~11,000 ids (attention transition) | 14,336 | 2048 | 0 | 4,608,000 B | 9 | yes, range −13.52 … 18.29 |

  Recorded arrangement, from the dumper's own log: 51/51 layers offloaded to the T4, the expert tensors in
  CPU-mapped buffers (`--cpu-moe 999`), **K/V cache f16**, single sequence. So this is a **GPU-dense, CPU-expert**
  reference — not an all-GPU one — and every qualification table must say so. Reference dump identities:
  `85d2cfcf6f1612a2` (8,191 case) and `509ce249b7a00052` (11,000 case).
* **Not yet done, and needed before qualification means anything:** the reference's own repeat-to-repeat
  determinism at these contexts (two identical runs compared bitwise), and the same dumps for the extra German and
  code-shaped prompts in item 1. Both are cheap and will be run first if the criteria are approved.

**Status: blocked on approval.** No qualification data will be generated until the criteria in items 1-5 are agreed.

### Tasks 7–8

Not run. Task 7 requires its own approval of a preregistered protocol before any evaluation output is examined;
Task 8 requires the fidelity/quality decision first.

## Artifacts

Indispensable evidence is archived outside `/tmp`, which is not durable:

| path | contents |
|---|---|
| `/data4/kolibri-f16-evidence/f16-t0/proc/` | server identity, engine and front-end environments, `strateu_server.json`, GPU state before the stop, restore script |
| `/data4/kolibri-f16-evidence/f16-t0/preserved-from-tmp/` | `klcmp.py`, `kolibri-t4-decode-gate/`, `t4-cand/`, `t4-longctx/`, `longctx.sh`, `ids8191.txt`, `ids11000.txt` |
| `/data4/kolibri-f16-evidence/f16-t0/*.log`, `*.sha256` | both build logs, control and working binary hashes, the model hash |
| `/tmp/f16-t0/` | live working directory for this task: inputs, manifest, per-run logs |

The model files and the multi-megabyte logit dumps are never committed.
