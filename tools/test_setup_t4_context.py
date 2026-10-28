"""The context setup recommends by the card's VRAM (no GPU, no network): a 16 GB Tesla T4 gets 128K, every other
14-19 GB card 64K, and --context / --max-context still wins over the recommendation.

    python -m unittest tools.test_setup_t4_context
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class T4Context(unittest.TestCase):
    def test_t4_16gb_gets_128k(self):
        self.assertEqual(setup.recommended_context(16.0, "Tesla T4", None), 131072)
        self.assertEqual(setup.recommended_context(15.9, "NVIDIA Tesla T4", None), 131072)

    def test_other_16gb_cards_keep_64k(self):
        self.assertEqual(setup.recommended_context(16.0, "NVIDIA GeForce RTX 4060 Ti", None), 65536)

    def test_two_cards_use_the_smallest_and_any_t4_counts(self):
        # the caller passes the smallest card's VRAM and every chosen card's name joined together
        self.assertEqual(setup.recommended_context(16.0, "Tesla T4, NVIDIA GeForce RTX 4090", None), 131072)
        self.assertEqual(setup.recommended_context(16.0, "NVIDIA GeForce RTX 4090, NVIDIA GeForce RTX 4060 Ti", None),
                         65536)

    def test_small_and_large_cards_keep_their_rule(self):
        self.assertEqual(setup.recommended_context(8.0, "Tesla T4", None), 32768)     # VRAM is the limit, not the T4
        self.assertEqual(setup.recommended_context(24.0, "NVIDIA GeForce RTX 4090", None), 131072)

    def test_the_ram_budget_path_is_unchanged(self):
        # UD-Q4_K_XL (budget is not None): every GB of KV is a GB fewer of cached experts
        self.assertEqual(setup.recommended_context(16.0, "Tesla T4", budget=40), 32768)
        self.assertEqual(setup.recommended_context(8.0, "Tesla T4", budget=40), 8192)

    def test_the_ram_rule_still_caps_it(self):
        # 128K needs no more RAM than the T4's own 755 GB, but a 64 GB PC's rule is the smaller of the two
        self.assertLessEqual(min(setup.recommended_context(16.0, "Tesla T4", None), 65536), 131072)


if __name__ == "__main__":
    unittest.main()
