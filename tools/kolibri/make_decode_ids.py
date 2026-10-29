#!/usr/bin/env python3
"""Build the decode benchmark's long prompt ids from a local text file.

Why a generated file and not the short fixture tiled: a tiled prompt repeats the same 512 ids 16 times, so
the router sees the same experts on a fixed schedule and the expert traffic per token is not the traffic a
document produces.  The decode benchmark needs a prompt that is *long* (KV length, attention cost) and
*ordinary* (mixed routing).  The text is tokenized once here and committed as ids, so the fixture is stable
even though the source document changes.

    .venv/bin/python tools/kolibri/make_decode_ids.py \
        --model /fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf \
        --text docs/DETAILS.md --n 8191 --out tools/kolibri/decode_ids8191.txt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import strata_tokenizer as ST  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("KOLIBRI_MODEL",
                                                      "/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf"))
    ap.add_argument("--text", required=True)
    ap.add_argument("--n", type=int, default=8191)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    text = open(a.text, encoding="utf-8").read()
    tok = ST.Tokenizer.from_gguf(a.model)
    ids = tok.encode(text)
    if len(ids) < a.n:
        print(f"make_decode_ids: {a.text} tokenizes to {len(ids)} id(s), need {a.n}", file=sys.stderr)
        return 2
    ids = ids[: a.n]
    with open(a.out, "w") as f:
        f.write(",".join(str(i) for i in ids) + "\n")
    print(f"make_decode_ids: {a.text} -> {len(ids)} id(s) in {a.out} "
          f"(first {ids[:5]}, last {ids[-5:]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
