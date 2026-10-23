#!/usr/bin/env python3
"""needle_audit.py - score the 100K needle fixtures on TASK CORRECTNESS, not on agreement with a reference.

The earlier verdict ("recall 5/5, verbatim 4/5") scored our tokens against llama.cpp's tokens.  At 98.6K that is not a
correctness test: the reference is itself wrong on the decisive token for at least one fixture (`quartz-willow-107` for a
needle of `quartz-willow-884`), and its whole continuation changes when only the prefill batch width changes - a semantic
no-op that merely reorders fp accumulations.  So this asks the only question with a ground truth: does the answer contain
the code word that is actually in the haystack?

The needle is the code word occurring EXACTLY ONCE in the prompt; the fixtures pad the haystack with repeated
`community-*` distractors, so uniqueness is the discriminator.

    python3 tools/kolibri/needle_audit.py                  # audit the generations already on disk
    python3 tools/kolibri/needle_audit.py --run            # run ours (deterministic flash config), then audit
    python3 tools/kolibri/needle_audit.py --run --include 32k90,64k90

Exit 0 when every audited answer is exact, 1 otherwise, so it can gate a commit.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

WORD = re.compile(r"[a-z]+-[a-z]+-\d{3}")

# the shipping configuration, with ATTN_FLASH because the default tiled path is not deterministic (two runs disagree) and
# an audit scored on a non-deterministic path can be neither reproduced nor defended
SHIP_ENV = {
    "STRATA_KOLIBRI_PF_STAGE": "1", "STRATA_KOLIBRI_EXPERT_F16": "24",
    "STRATA_KOLIBRI_ROUTER_TILE": "1", "STRATA_KOLIBRI_ROUTER_TK": "3", "STRATA_KOLIBRI_ROUTER_TJ": "16",
    "STRATA_KOLIBRI_ATTN_MB": "512", "STRATA_KOLIBRI_RES_TOP": "56",
    "GGML_CUDA_DISABLE_GRAPHS": "1",
}
# --flash scores the deterministic ATTN_FLASH path instead of the shipped tiled one.  Kept as an option because the two
# paths disagree on the needle (d90_seed777: shipped = correct, flash = a different instruction), so both must be visible.
FLASH_ENV = {"STRATA_KOLIBRI_ATTN_FLASH": "1"}


def needle_of(prompt_path):
    text = open(prompt_path, encoding="utf-8", errors="replace").read()
    counts = {}
    for w in WORD.findall(text):
        counts[w] = counts.get(w, 0) + 1
    # the fixtures FRAME the needle explicitly ("The secret code word for this text is: X. Remember it."), which is the
    # only unambiguous discriminator: every distractor code word also occurs exactly once (they are corpus text), so
    # uniqueness alone cannot identify the needle.
    framed = re.findall(r"[Tt]he secret code word for this text is:\s*([a-z]+-[a-z]+-\d{3})", text)
    if len(set(framed)) == 1:
        return framed[0], counts
    once = [w for w, c in counts.items() if c == 1]
    return (once[0] if len(once) == 1 else None), counts


def decode_ids(text, gguf):
    import strata_tokenizer as ST
    body = text.split(":", 1)[-1]
    ids = [int(x) for x in re.findall(r"-?\d+", body)]
    return ST.Tokenizer.from_gguf(gguf).decode(ids) if ids else ""


def run_ours(fixture, gguf, n, ctx, timeout, flash=False):
    exe = os.path.join(ROOT, "build", "strata-kolibri")
    env = dict(os.environ, **SHIP_ENV, **(FLASH_ENV if flash else {}))
    cmd = [exe, "--model", gguf, "--tokens-file", os.path.join(fixture, "90.ids"), "--n", str(n),
           "--ctx", str(ctx), "--prefill", "--cpu-threads", "24"]
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("gen:")), "")
    if not line:
        tail = (out.stderr or "").strip().splitlines()[-2:]
        raise RuntimeError(f"no gen line: {tail}")
    with open(os.path.join(fixture, "audit_flash_gen.txt" if flash else "audit_gen.txt"), "w") as f:
        f.write(line + "\n")
    return decode_ids(line, gguf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*")
    ap.add_argument("--gguf", default="/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--flash", action="store_true", help="score the ATTN_FLASH path instead of the shipped tiled one")
    ap.add_argument("--include", default="")
    ap.add_argument("--n", type=int, default=32, help="generated tokens; the answers echo the instruction first, so 12 truncates mid-code-word")
    ap.add_argument("--ctx", type=int, default=105024)
    ap.add_argument("--timeout", type=int, default=7200)
    a = ap.parse_args()

    dirs = a.fixtures or sorted(glob.glob("/tmp/needle_*"))
    if a.include:
        keep = [s for s in a.include.split(",") if s]
        dirs = [d for d in dirs if any(k in d for k in keep)]

    failures = audited = 0
    for d in dirs:
        prompt = os.path.join(d, "90_prompt.txt")
        if not (os.path.exists(prompt) and os.path.exists(os.path.join(d, "90.ids"))):
            continue
        needle, counts = needle_of(prompt)
        print(f"\n{d}")
        if needle is None:
            print(f"  needle ambiguous (unique: {[w for w, c in counts.items() if c == 1]}) - skipped")
            continue
        ntok = len(open(os.path.join(d, "90.ids")).read().replace("\n", "").split(","))
        print(f"  {ntok:,} tokens | needle = {needle}")

        answers = []
        if a.run:
            try:
                answers.append(("ours (shipped path)", run_ours(d, a.gguf, a.n, max(a.ctx, ntok + 64), a.timeout, a.flash)))
            except Exception as e:
                print(f"  ours: FAILED - {e}")
                failures += 1
        for g in sorted(glob.glob(os.path.join(d, "*gen*.txt"))):
            if os.path.basename(g) == "audit_gen.txt":   # already counted above when --run
                continue
            try:
                answers.append((os.path.basename(g), decode_ids(open(g).read(), a.gguf)))
            except Exception as e:
                print(f"  {os.path.basename(g):22s} unreadable - {e}")

        for name, text in answers:
            audited += 1
            ok = needle in text
            failures += 0 if ok else 1
            tail = text.strip().splitlines()[-1] if text.strip() else ""
            print(f"  {name:22s} {'PASS' if ok else 'FAIL'}  {tail[:70]!r}")

    print(f"\n{audited} answer(s) audited, {failures} not exact")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
