#!/usr/bin/env python3
"""Build the 100K needle fixtures for needle_audit.py on this box.

The campaign's driver (/tmp/needle_drive.py) was never committed, so this rebuilds the same shape:
a haystack of repeated `community-*` distractor lines with one framed needle sentence inserted at a
chosen depth.  The framing sentence is the audit's only unambiguous discriminator (its regex is
`[Tt]he secret code word for this text is:\\s*([a-z]+-[a-z]+-\\d{3})`), and the needle word itself
occurs exactly once in the whole prompt.

Writes /tmp/needle_<name>/{90_prompt.txt,90.ids} - the two files needle_audit.py looks for.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import strata_tokenizer as ST  # noqa: E402

GGUF = os.environ.get("KOLIBRI_MODEL", "/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf")
TARGET = int(os.environ.get("NEEDLE_TOKENS", "98600"))

# (name, needle, depth as a fraction of the haystack) - depths spread so a miss at one depth is
# distinguishable from a miss at another, which is what the campaign's table reported.
FIXTURES = [
    ("a05", "quartz-willow-884", 0.05),
    ("b25", "cobalt-orchard-311", 0.25),
    ("c50", "amber-glacier-742", 0.50),
    ("d75", "velvet-cedar-209", 0.75),
    ("e90", "quartz-willow-107", 0.90),
]

FRAME = "The secret code word for this text is: {needle}. Remember it.\n"
QUESTION = "\nQuestion: what is the secret code word for this text? Answer:"


def distractors(n):
    """n distinct `community-*-NNN` lines - the haystack filler the campaign used."""
    names = ["lantern", "harbour", "meadow", "cinder", "tessera", "bramble", "quarry",
             "ferry", "orchard", "willow"]
    out = []
    for i in range(n):
        name = names[i % len(names)]
        out.append(f"community-{name}-{i % 1000:03d} keeps its ledger in a drawer under the stairs.\n")
    return out


def build(tok, name, needle, depth):
    """Grow the haystack until the tokenized prompt reaches TARGET, then splice the needle in."""
    unit = "".join(distractors(60))
    body = unit * 40
    text = body + QUESTION
    n = len(tok.encode(text))
    while n < TARGET:
        body += unit
        text = body + QUESTION
        n = len(tok.encode(text))
    lines = body.splitlines(keepends=True)
    at = max(1, min(len(lines) - 1, int(len(lines) * depth)))
    framed = FRAME.format(needle=needle)
    # the needle word must occur exactly once: assert it is not already anywhere in the haystack
    assert needle not in body, f"{needle} collides with the haystack"
    prompt = "".join(lines[:at]) + framed + "".join(lines[at:]) + QUESTION
    ids = tok.encode(prompt)
    d = f"/tmp/needle_{name}"
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "90_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(os.path.join(d, "90.ids"), "w", encoding="utf-8") as f:
        f.write(",".join(str(i) for i in ids))
    once = prompt.count(needle)
    print(f"{d}: {len(ids):,} tokens, needle {needle} occurs {once}x at line {at}/{len(lines)}"
          f"{'  OK' if once == 1 else '  AMBIGUOUS'}")
    return once == 1


def main():
    tok = ST.Tokenizer.from_gguf(GGUF)
    ok = all(build(tok, *fx) for fx in FIXTURES)
    print("all fixtures unique-needle:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
