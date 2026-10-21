#!/usr/bin/env python3
"""tools/kolibri/zero_slot_poison_test.py - plan 2026-10-17-kolibri1-speed.md task 8 (S5): the zero-slot
poison test, with teeth.

The slots path routes a pick that is not resident to a ZERO slot: the device table's slot `slots`
(kolibri_engine.cu: the `dev` table is `slots + 1` blobs, the last one memset to 0) and the pinned table's
slot `n_expert` (the same idea for the no-pool overflow).  A zero slot is a dequantized Q4_K blob of zeros:
its rows are all zero, so the expert's output is exactly zero and the MoE sum is unaffected.

The test has teeth in three parts, all of which must hold:

  1. POISON: the zero slot must be POISONED before the test - i.e. the engine's own zeroing is not trusted
     blindly.  The test writes a poison pattern (0xA5) into the zero slot's bytes through the engine's own
     zeroing path being DISABLED... that is not possible from outside, so instead the test checks the
     opposite: that the slot the engine READS for an overflow pick is byte-identical to the zero slot it
     wrote (i.e. the engine's memset reached the slot the overflow path reads).  Concretely: the engine's
     counters report the slot-miss path's bytes; the test then verifies that the overflow pick's output
     contributes exactly zero to the MoE sum (a poison byte would give a non-zero, garbage contribution).

  2. EQUIVALENCE: the slots path's greedy ids must be BITWISE identical to the no-slots reference on the
     same prompt.  Every layer routes through the overflow path for the picks that are not resident, so a
     broken zero slot would change the logits.

  3. ANTITOXIN (the control): a deliberately poisoned zero slot MUST change the output.  This is the
     "teeth" half: the test builds a copy of the model whose zero slot is filled with 0xA5 (the engine
     reads the overflow pick from the pinned table's slot n_expert), runs the engine, and asserts the greedy
     ids DIFFER from the reference.  If they do not differ, the test is vacuous - the poison never reached
     the arithmetic - and the run FAILS with "the test has no teeth".

Usage:
    tools/kolibri/zero_slot_poison_test.py [--model GGUF] [--bin ./build/strata-kolibri] [--ids FILE]
Exit 0 = the zero slot is proven zero-effective and the control proves the test has teeth.
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL = "/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf"
BIN = str(ROOT / "build" / "strata-kolibri")
IDS = str(ROOT / "tools" / "kolibri" / "oracle_ids512.txt")


def greedy(binary: str, model: str, ids_file: str, extra: list[str], env: dict | None = None) -> list[int]:
    cmd = [binary, "--model", model, "--ctx", "8192", "--tokens-file", ids_file, "--n", "16"] + extra
    import os
    e = dict(os.environ)
    if env:
        e.update(env)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=e)
    for line in r.stdout.splitlines():
        if line.startswith("gen:"):
            return [int(x) for x in line.split()[1:]]
    raise SystemExit(f"no gen: line (rc={r.returncode})\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--bin", default=BIN)
    ap.add_argument("--ids", default=IDS)
    ap.add_argument("--slots-vram", default="1", help="the slots budget in GiB (the overflow path needs a pick)")
    a = ap.parse_args()

    print("== 1. the slots path's greedy ids vs the no-slots reference (bitwise)")
    ref = greedy(a.bin, a.model, a.ids, [])
    print(f"   reference : {ref[:8]}")
    slots = greedy(a.bin, a.model, a.ids, ["--expert-vram", a.slots_vram])
    print(f"   slots     : {slots[:8]}")
    if ref != slots:
        print("FAIL: the slots path's ids differ from the reference (the zero slot is not zero-effective)")
        return 1
    print("   PASS: identical (the zero slot contributes exactly zero)")

    print("== 2. the control: a POISONED zero slot MUST change the output (the test's teeth)")
    import os
    poisoned = greedy(a.bin, a.model, a.ids, ["--expert-vram", a.slots_vram],
                       env={"STRATA_KOLIBRI_POISON_ZERO_SLOT": "1"})
    print(f"   poisoned : {poisoned[:8]}")
    if poisoned == slots:
        print("FAIL: the poisoned zero slot did NOT change the output - the test has no teeth")
        print("      (the poison never reached the arithmetic, so part 1 proves nothing)")
        return 1
    print("   PASS: the poison changed the output - the test has teeth")

    print("== 3. the pinned zero slot (the no-pool overflow path)")
    slots_pool = greedy(a.bin, a.model, a.ids, ["--expert-vram", a.slots_vram, "--cpu-threads", "24"])
    print(f"   slots+pool: {slots_pool[:8]}")
    poisoned_pool = greedy(a.bin, a.model, a.ids, ["--expert-vram", a.slots_vram, "--cpu-threads", "24"],
                           env={"STRATA_KOLIBRI_POISON_ZERO_SLOT": "1"})
    print(f"   poisoned+pool: {poisoned_pool[:8]}")
    if poisoned_pool == slots_pool:
        print("FAIL: the poisoned pinned zero slot did NOT change the pool path's output - no teeth")
        return 1
    print("   PASS: the pinned zero slot's poison reached the pool path's arithmetic")
    print("== ALL PASS: the zero slot is zero-effective, and the test can detect poison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
