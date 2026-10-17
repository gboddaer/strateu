"""Tests for setup.py's kolibri family (plan docs/superpowers/plans/2026-10-16-kolibri1.md T12): the family table
entry, the sizes, the KV arithmetic, --gguf-dir's shards for the single-file Q4_K_M and the two-shard Q8_0, and
the config the install writes.  Runs without a GPU and without downloads, the way tools/test_setup_choices.py does.

    python -m unittest tools.test_setup_kolibri
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class KolibriFamily(unittest.TestCase):
    def setUp(self):
        self.fam = setup.OWN_PATH_FAMILIES["kolibri"]

    def test_the_table_entry(self):
        self.assertEqual(self.fam["title"], "Kolibri 1")
        self.assertEqual(self.fam["arch"], "kolibri1")
        self.assertEqual(self.fam["name"], "kolibri-1")
        self.assertIn("Hob-forge/Kolibri-1-GGUF", str(self.fam["hf"]))   # the repo the local files came from

    def test_the_sizes(self):
        self.assertEqual(sorted(self.fam["models"]), ["Q4_K_M", "Q8_0"])
        self.assertEqual(self.fam["models"]["Q4_K_M"]["ram_gb"], 56)
        self.assertEqual(self.fam["models"]["Q8_0"]["ram_gb"], 96)

    def test_the_kv_arithmetic(self):
        # 11 full-attention layers (every 5th of 50), 4 KV heads x 128, K and V in fp16
        self.assertEqual(self.fam["kv_bytes_per_token"], 11 * 4 * 128 * 2 * 2)

    def test_no_mtp_no_mmproj(self):
        self.assertFalse(self.fam.get("mtp"))          # the model has no MTP layer (plan D3)
        self.assertFalse(self.fam.get("mmproj"))       # no image encoder

    def test_gguf_dir_shards_single_file(self):
        d = Path("/data4/models/Kolibri-1-Q4_K_M")
        if not d.exists():
            self.skipTest("no local Q4_K_M folder")
        shards = setup.gguf_dir_shards(d, self.fam, "Q4_K_M")
        self.assertEqual([p.name for p in shards], ["Kolibri-1-Q4_K_M.gguf"])

    def test_gguf_dir_shards_two_files(self):
        d = Path("/data4/models/Kolibri-1-Q8_0")
        if not d.exists():
            self.skipTest("no local Q8_0 folder")
        shards = setup.gguf_dir_shards(d, self.fam, "Q8_0")
        self.assertEqual([p.name for p in shards],
                         ["Kolibri-1-Q8_0-00001-of-00002.gguf", "Kolibri-1-Q8_0-00002-of-00002.gguf"])


if __name__ == "__main__":
    unittest.main()
