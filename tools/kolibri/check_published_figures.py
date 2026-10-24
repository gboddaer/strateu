#!/usr/bin/env python3
"""check_published_figures.py - mechanical staleness check for published numbers (amendment 7).

Every speed/correctness figure quoted in the public docs must appear in docs/published-figures.md.  This scans the docs for
`<n> tok/s` occurrences and for `a / b tok/s` pairs, and reports any value that is neither in the manifest nor in the explicit
allowlist of numbers that are *not* rates (token counts, ms, percentages, model size).  Exit 1 if anything is unaccounted for.

    python3 tools/kolibri/check_published_figures.py            # check
    python3 tools/kolibri/check_published_figures.py --verbose  # also list the matched values
"""
import argparse, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs" / "published-figures.md"
DOCS = ["README.md", "STRATEU.md", "docs/strateu-campaign-report.md", "docs/KOLIBRI1.md"]
# numbers allowed to appear with "tok/s" without a manifest row: upstream's own marketing figures and historical
# comparisons in KOLIBRI1's llama.cpp columns, which are labelled as such in the text.
ALLOW = {"60", "723", "126.1", "157.5", "19.1", "18.0", "10.0", "9.3", "0.6", "1.3"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    manifest = MANIFEST.read_text()
    known = set(re.findall(r"(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?\s*tok/s", manifest))
    known_vals = {x for pair in known for x in pair if x} | ALLOW

    missing = []
    for rel in DOCS:
        p = ROOT / rel
        if not p.exists():
            print(f"  MISSING DOC {rel}")
            continue
        text = p.read_text()
        for m in re.finditer(r"(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?\s*(?:tok/s)", text):
            for v in (m.group(1), m.group(2)):
                if v and v not in known_vals:
                    line = text[: m.start()].count("\n") + 1
                    missing.append((rel, line, v, text.splitlines()[line - 1].strip()[:90]))
        if a.verbose:
            print(f"  {rel}: scanned")

    if missing:
        print(f"  {len(missing)} quoted rate(s) with no manifest row:")
        for rel, line, v, ctx in missing:
            print(f"    {rel}:{line}  {v} tok/s   {ctx}")
        return 1
    print(f"  all quoted rates are in {MANIFEST.relative_to(ROOT)} ({len(known_vals)} values known)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
