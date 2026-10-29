# Kolibri decode: experts from VRAM, grouped Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the decode token's routed experts off the 32.4 GB/s host leg onto the card (grouped,
VRAM-resident, device-side grouping), then make the token host-free so it can be captured as one graph.
Target: 67.4 ms → ~33 ms per token (14.7 → ~30 tok/s) on the R9700, where the GPU is idle 64 % of a token.

**Architecture:** Kolibri's 42.90 GiB of expert blobs cannot fit in 32 GB, but **253 slots/layer
(28.38 GiB) do**, and 214 slots already served 96.6 % of picks on this box. Today the slot path costs
**3 launches per pick**, and the pool path costs a **blocking D2H + pool drain + H2D per layer**, which is
where the GPU sits idle. The plan reuses two pieces that already exist on either side of the gap: the
prefill's `native_expert_grouped` (one launch per layer, takes a **per-group pointer array**, so a table
mixing VRAM slots and the pinned host table needs no new kernel) and the prefill's device-side `k_group`.
Once the host no longer takes part in the token, the whole token becomes capturable, which removes the
1 344 launches and the 17.7 ms of dispatch gaps.

**Tech Stack:** C++17, HIP/CUDA, `src/kolibri/kolibri_engine.cu`, `src/kernels/cuda/native_moe*`,
CMake + Ninja, `rocprofv3`, `tools/kolibri/{decode_bench,kl,pf_gate.sh,check_repeat}.py`.

**Spec:** [docs/KOLIBRI_DECODE.md](../KOLIBRI_DECODE.md) — every number below is measured there.

## Global Constraints

- **GPU: card0 only.** Every command runs with `HIP_VISIBLE_DEVICES=0`. Never card1/card2.
- **The default path stays byte-identical.** Every new arm is behind `STRATA_KOLIBRI_DEC_SLOTS_GROUPED`
  (Task 3) and `STRATA_KOLIBRI_DEC_GRAPH` (Task 4), both default off.
- **Never `--record` over the pool's expectation files.** `pf_gate_expect_{smoke,hash}_gfx1201.txt` are the
  pool's; a numerical arm gets its own pair, as the P4 arm did.
- **A numerics-changing arm ships only with its own KL gate**: mean KL against the default arm, top-1
  agreement, and two identical runs. Bit-equality claims need `--dump-full-logits` (`--dump-logits` writes
  only the top-two records).
- **Every ledger row states** ctx, prompt length, decode length, threads, box load, and how the number was
  obtained (trace window, never a wall difference between two `--n` values).
- **A spike that shows no speed-up stops the plan.** Task 2 is a measurement, not a promise.
- Commit titles start with `kolibri(r9700):`.
- Model `/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf`; build tree `build-mmq/`; fixture
  `tools/kolibri/oracle_ids512.txt`.

## Review Focus

- **A pick that misses the slot table.** Roughly 3.4 % of picks at 214 slots. The grouped table must cover
  them from the pinned host table, or the layer must fall back as a whole — mixing the two inside one
  launch is the point of the change, and a mistake here silently computes a wrong expert.
- **`Q6_K` down projections.** Blobs are `[gate | up | down]`; the down half is Q6_K on 25 of 50 layers and
  the grouped kernel historically refused it. If it still refuses, those layers must keep a working path
  rather than drop their down contribution.
- **A fresh process with cold VRAM.** The slot set is filled after the prompt; the first decoded token must
  not read an unfilled slot.
- **The 42.90 GiB pinned table plus 28.38 GiB of slots at once.** `hiptrunc`/OOM on a 32 GB card must be a
  refusal with a message, not a crash.
- **Id equality is the wrong gate across arms.** Residency moves bytes between stores whose arithmetic
  differs (measured 0.855 mean |Δlogit|); judge by KL, and by two identical runs of the *same* arm.

---

### Task 1: An exact decode instrument (`tools/kolibri/decode_bench.py`)

**Files:**
- Create: `tools/kolibri/decode_bench.py`
- Test: `tools/kolibri/test_decode_bench.py`

**Interfaces:**
- Produces: `parse_token_periods(csv_path: Path) -> TokenStats` with fields `tokens`, `ms_per_token`,
  `tok_per_s`, `gpu_busy_ms`, `gpu_busy_share`, `kernels_per_token`, `layer_gaps`, `gap_ms`.
- Produces: CLI `decode_bench.py [--runs N] [--out DIR] -- <engine> <args...>`, printing that as JSON.

- [ ] **Step 1: Write the failing test**

```python
# tools/kolibri/test_decode_bench.py
import sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_bench

HDR = ("Kind,Agent_Id,Queue_Id,Stream_Id,Thread_Id,Dispatch_Id,Kernel_Id,Kernel_Name,Correlation_Id,"
       "Start_Timestamp,End_Timestamp,LDS_Block_Size,Scratch_Size,VGPR_Count,Accum_VGPR_Count,SGPR_Count,"
       "Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z,Grid_Size_X,Grid_Size_Y,Grid_Size_Z\n")

def row(start_us, dur_us, name):
    return (f"KERNEL_DISPATCH,Agent 1,2,1,1,1,1,{name},1,{start_us*1000},{start_us*1000+dur_us*1000},"
            "0,0,0,0,0,128,1,1,1,1,1\n")

def make_csv(path, tokens=6, per_token_us=60000):
    # The real trace prefixes the marker with "void (anonymous namespace)::". A parser that matches the
    # bare name finds nothing - that is exactly how the first attempt failed.
    lines = [HDR]; t = 0
    for _ in range(tokens):
        lines.append(row(t, 5, "void (anonymous namespace)::k_row<12>(unsigned char const*, unsigned long, int, int, float*)"))
        for _ in range(9):
            t += 6000
            lines.append(row(t, 5900, "void strata::kernels::(anonymous namespace)::native_q4_k_mmvq_kernel<false>(x, y)"))
        t += per_token_us
    Path(path).write_text("".join(lines))

class Parse(unittest.TestCase):
    def test_finds_the_token_period_through_the_void_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"; make_csv(p)
            s = decode_bench.parse_token_periods(p)
            self.assertEqual(s.tokens, 5)
            self.assertAlmostEqual(s.ms_per_token, 60.0, delta=1.0)
            self.assertAlmostEqual(s.tok_per_s, 1000 / 60.0, delta=0.5)
            self.assertGreater(s.kernels_per_token, 5)

    def test_refuses_a_trace_with_no_decode(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.csv"; make_csv(p, tokens=2)
            with self.assertRaises(SystemExit):
                decode_bench.parse_token_periods(p)

if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tools/kolibri/test_decode_bench.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'decode_bench'`.

- [ ] **Step 3: Implement `decode_bench.py`**

Stream the CSV (a real trace is ~180 MB) and keep only `KERNEL_DISPATCH` rows. A token boundary is a kernel
whose name contains `k_row<`. `ms_per_token` is the median of consecutive-mark periods over the last
`min(len(periods), 17)`; `gpu_busy_ms` is the union of kernel intervals inside the last full token;
`layer_gaps` counts intervals longer than 100 µs inside that token and `gap_ms` sums them. The CLI runs
`rocprofv3 --kernel-trace --output-format csv -d <out> -o dec -- <cmd>` and prints the JSON.

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tools/kolibri/test_decode_bench.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Measure the baseline twice and record both ledger rows**

```sh
HIP_VISIBLE_DEVICES=0 .venv/bin/python tools/kolibri/decode_bench.py --runs 2 --out /tmp/dec-base -- \
  ./build-mmq/strata-kolibri --model /fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 17 --ctx 8192 --cpu-threads 8
```

Expected: `ms_per_token` 60-70, `tok_per_s` 14-17, `gpu_busy_share` 0.3-0.4, `kernels_per_token` ≈ 1344,
`layer_gaps` = 50. Two runs within 5 %, or fix the instrument before changing any kernel.

- [ ] **Step 6: Commit**

```sh
git add tools/kolibri/decode_bench.py tools/kolibri/test_decode_bench.py
git commit -m "kolibri(r9700): an exact decode instrument - the trace's own token periods, not a wall difference"
```

---

### Task 2: Spike — can the grouped kernel serve a decode token from VRAM?

**This task changes nothing that ships. It answers one question before the build.**

**Files:**
- Modify: `src/kolibri/kolibri_engine.cu` (decode slot branch only, behind `STRATA_KOLIBRI_DEC_SLOTS_GROUPED`)
- Test: `tools/kolibri/decode_bench.py` (Task 1)

**Interfaces:**
- Consumes: `native_expert_grouped(L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok, cap_groups, n_ent, xq8, scratch, gout, st)`, already used at line ~2272 for prefill; `k_group(...)` at line ~1372.
- Produces: the answer, recorded in [docs/KOLIBRI_DECODE.md](../KOLIBRI_DECODE.md): whether grouped-from-VRAM
  beats **both** the pool (67.4 ms) and the 3-launch slot arm (42.6 ms), and by how much.

- [ ] **Step 1: Answer the two interface questions by reading, before writing code**

Report in the task's notes: (a) does `native_expert_grouped` accept a `grp_ptr` entry that points at
pinned host memory while another points at VRAM? (b) does it handle a Q6_K down blob, or refuse it as the
prefill comment says. If (b) refuses, the 25 Q4_K-down layers use the grouped path and the 25 Q6_K layers
keep the slot GEMV — say so in the ledger rather than averaging the two.

- [ ] **Step 2: Wire the arm behind the knob, decode only**

With `STRATA_KOLIBRI_DEC_SLOTS_GROUPED=1` and `--expert-slots auto --expert-slots-hot`: for each decode
layer, build `grp_ptr`/`grp_start` on the device from the slot map (as prefill's `k_group` does), call
`native_expert_grouped` once for the layer's slotted picks, and leave non-slotted picks on the existing
path. Unset, the binary is unchanged.

- [ ] **Step 3: Build, then measure with Task 1's instrument against both references**

```sh
# arm
HIP_VISIBLE_DEVICES=0 STRATA_KOLIBRI_DEC_SLOTS_GROUPED=1 .venv/bin/python tools/kolibri/decode_bench.py \
  --runs 2 --out /tmp/dec-grp -- ./build-mmq/strata-kolibri --model <model> \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 17 --ctx 8192 --cpu-threads 8 \
  --expert-slots auto --expert-slots-hot
```

Expected if the recipe transfers: `ms_per_token` well under 42.6. **If it is not under 42.6, stop here** and
report that grouped-from-VRAM does not beat the existing slot path on this card; do not proceed to Task 3.

- [ ] **Step 4: Record the KL against the default arm**

Run both arms with `--dump-full-logits` over the 8 parity prompts and `tools/kolibri/kl.py`, exactly as the
P4 arm did. Expected: finite logits, mean KL of the same order as the P4 arm's 0.00236, top-1 well above
90 %. Write the numbers into the ledger row.

- [ ] **Step 5: Check the two failure modes the Review Focus names, before any shipping claim**

(a) Ask for slots that cannot fit (`--vram-reserve-mib` large enough to leave no room) and confirm the
engine prints a refusal and exits non-zero instead of crashing or silently dropping layers. (b) Run a
fresh process and check the first decoded token after the prompt does not read an unfilled slot — the
repeated-run check in Task 3 covers this only if the first token's ids are compared, so compare them here.

- [ ] **Step 6: Commit (spike only, arm off by default)**

```sh
git add src/kolibri/kolibri_engine.cu docs/KOLIBRI_DECODE.md
git commit -m "kolibri(r9700): decode experts grouped from VRAM slots - the spike's measurement, arm off"
```

---

### Task 3: Ship the arm, with its own gate

**Files:**
- Modify: `src/kolibri/kolibri_engine.cu`, `tools/kolibri/pf_gate.sh`
- Test: `tools/kolibri/check_repeat.py`, `tools/kolibri/pf_gate.sh`

**Interfaces:**
- Consumes: Task 2's measured arm.
- Produces: a `KOLIBRI_GATE_PROFILE=dec-slots-gfx1201` profile with its own
  `pf_gate_expect_{smoke,hash}_dec_slots_gfx1201.txt` and an assertion that the arm activated.

- [ ] **Step 1: Add the profile to `pf_gate.sh`** with its own expectation files and the activation
  assertion, following the `mmq-gfx1201` profile's exact shape (it must refuse a silent fallback, and it must
  propagate a failed engine).

  **The misses must also leave the host.** A pick that is not in a slot has to be served from the pinned host
  table *inside the same grouped launch* (the mixed pointer array this plan's architecture relies on), with
  no host call. Verify it by measurement: with the arm on, Task 1's instrument must show the 50 per-layer
  `layer_gaps` collapsing — if they do not, the host is still in the token and Task 4 is illegal.
- [ ] **Step 2: `--record` into the arm's own files, then `check`** — `GATE PASS`, and the pool profile still
  passes unchanged:

  ```sh
  HIP_VISIBLE_DEVICES=0 STRATA_HIPBLASLT_TUNING=$PWD/tools/hip/gfx1201-hipblaslt-100202.txt \
    KOLIBRI_MODEL=/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf KOLIBRI_CM=$(command -v cmake) \
    KOLIBRI_BUILD=build-mmq KOLIBRI_BIN=$PWD/build-mmq/strata-kolibri \
    KOLIBRI_GATE_PROFILE=dec-slots-gfx1201 bash tools/kolibri/pf_gate.sh check
  ```
- [ ] **Step 3: `check_repeat.py --runs 6` on the arm** — bit-identical full logits across runs, ids stable.
- [ ] **Step 4: Full CTest on `build-mmq`** — record the count; the four known baseline failures are `vmm_test`, `ple_parity`, `platform_memory_test`, `conversation_copy_streams_test`.
- [ ] **Step 5: Commit.**

---

### Task 4: The whole-token graph (the multiplier)

**Files:**
- Modify: `src/kolibri/kolibri_engine.cu`
- Test: `tools/kolibri/decode_bench.py`, Task 3's gate

**Interfaces:**
- Consumes: Task 3's arm, which leaves no host call inside the token.
- Produces: `STRATA_KOLIBRI_DEC_GRAPH=1`, capturing the token once and replaying it; the position becomes a device scalar (`s.pos_d`) as `PF_PIPE` already does for prefill and as the Flash-Next engine does with `d_delta`.

- [ ] **Step 1: Confirm the precondition by measurement, not assumption** — with Task 3's arm on, Task 1's instrument must show `gap_ms` for the 50 layer gaps near zero. If the host is still in the token, stop: the graph is illegal and the answer is a different placement.
- [ ] **Step 2: Move the position to a device scalar and re-run Task 1's instrument unchanged** — same ids, same KL, and `gap_ms` should fall.
- [ ] **Step 3: Capture and replay the token behind `STRATA_KOLIBRI_DEC_GRAPH=1`** — `cudaStreamBeginCapture`/`EndCapture`/`GraphInstantiate`, one `cudaGraphLaunch` per token.
- [ ] **Step 4: Measure, and confirm `kernels_per_token` still counts what runs while the host launches one graph** — the expected end state is ~33 ms/token (~30 tok/s).
- [ ] **Step 5: Gate as in Task 3, and commit** — the arm becomes a default only with two identical runs, the KL bound, and the measured gain, exactly the P4 bar.

---

## What this plan deliberately does not do

- **No layer overlap for a single token.** The spec shows the per-layer chain forbids it; the prefill's
  `PF_PIPE` overlap works only because two *chunks* are independent. Overlap becomes available only with
  concurrent sequences (serve throughput) or batched verify (speculation).
- **No standalone per-layer graphs.** The spec lists B as its own step; here it is folded into Task 4,
  because per-layer capture still needs the position moved to the device and pays off mainly once the host
  has left the token. Doing it after Task 3 is the smaller total change, not the shorter path.
- **No attention work.** Measured: the layout fix is 1.01x (bitwise but worthless), and attention is 16 % of
  the token. It is worth its own plan after this one, for the 100K regime.
- **No new default** on the strength of one run.
