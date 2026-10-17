"""The Kolibri 1 tokenizer pack (plan docs/superpowers/plans/2026-10-16-kolibri1.md T11).

The GGUF declares `tokenizer.ggml.pre = "kolibri1"`, which llama.cpp maps to LLAMA_VOCAB_PRE_TYPE_QWEN2
(src/llama-vocab.cpp: `pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN2` for the kolibri1 pre).  The pack must ship that
pattern, not the qwen35 one: QWEN2 splits `\p{L}+` where qwen35 splits `[\p{L}\p{M}]+`, and the difference
shows on combining marks.  Round trip on a corpus with German umlauts, emoji, CJK, tool JSON and the chat
template; the oracle is the GGUF's own tokenizer.json when the HF `tokenizers` module is available, else the
13-string round-trip check.

    python -m unittest serve.test_tokenizer_kolibri     (skips without the GGUF)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402

GGUF = os.environ.get("STRATA_KOLIBRI_GGUF", "/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf")

ROCKET, GLOBE, FIRE, DE1, DE2 = (chr(c) for c in (0x1F680, 0x1F30D, 0x1F525, 0x1F1E9, 0x1F1EA))

# the 13 strings: the shapes the model's traffic actually holds
CORPUS = [
    "Kannst du mir bitte helfen, die Straße zu finden?",
    "Größe, Möwen, über allem — Ärger, Ölung, Ürinal.",
    "Einmal quer durch: " + ROCKET + GLOBE + FIRE + " und " + DE1 + DE2 + " als Flaggenfolge.",
    "你好世界，这是一个测试。德语：Straße。",
    '{"name": "get_weather", "arguments": {"city": "Berlin", "unit": "celsius"}}',
    "<|im_start|>system\nDu bist Kolibri.<|im_end|>\n<|im_start|>user\nHallo!<|im_end|>\n<|im_start|>assistant\n",
    "def add(a, b):\n    return a + b  # 2 + 2 * (3 - 1) / 4\n",
    "27 × 43 = 1161, und 1161 / 27 = 43.",
    "Contractions: don't, isn't, she'll, they've, I'm, you're, he'd.",
    "Combining: é, ä, o\u0303 and NFD names like café.",
    "Tabs\tand\ttwo\n\nnewlines\r\nkeep\tspace.",
    "Mixed: der Löffel + the spoon + 勺子 = 3 × Besteck.",
    "Ein sehr langer deutscher Satz ohne Leerzeichen,um zu zeigen,dass lange Wortketten rundreisen,und dazu Zahlen 1234567890 und Codes #hashtag @mention.",
]


def gguf_or_skip() -> str:
    if not Path(GGUF).exists():
        raise unittest.SkipTest("no kolibri GGUF at %s (set STRATA_KOLIBRI_GGUF)" % GGUF)
    return GGUF


class KolibriTokenizer(unittest.TestCase):
    def setUp(self):
        self.tk = ST.Tokenizer.from_gguf(gguf_or_skip())

    def test_pre_is_kolibri1(self):
        self.assertEqual(self.tk.pre, "kolibri1")

    def test_the_pattern_is_the_qwen2_one(self):
        pat, src = ST.PRE_PATTERNS["kolibri1"]
        self.assertIn(r"\p{L}+", pat)
        self.assertNotIn(r"[\p{L}\p{M}]+", pat)          # the qwen35 shape is NOT kolibri1's
        self.assertIn("QWEN2", src)

    def test_the_tokenizer_uses_the_kolibri1_pattern(self):
        pat, _ = ST.PRE_PATTERNS["kolibri1"]
        self.assertEqual(self.tk._re.pattern, pat)

    def test_round_trip_on_the_corpus(self):
        for s in CORPUS:
            self.assertEqual(self.tk.decode(self.tk.encode(s)), s, "round trip: %r" % s[:40])

    def test_chat_specials_map_to_their_ids(self):
        ids = self.tk.encode("<|im_start|>user\nHallo!<|im_end|>", parse_special=True)
        self.assertIn(127904, ids)                        # im_start
        self.assertIn(127906, ids)                        # im_end

    def test_the_pack_ships_the_kolibri1_pattern(self):
        with tempfile.TemporaryDirectory() as d:
            ST.extract(gguf_or_skip(), d)
            cfg = json.loads((Path(d) / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["pre"], "kolibri1")
        self.assertEqual(cfg["pre_pattern"], ST.PRE_PATTERNS["kolibri1"][0])
        self.assertIn("QWEN2", cfg["pre_pattern_source"])


if __name__ == "__main__":
    unittest.main()
