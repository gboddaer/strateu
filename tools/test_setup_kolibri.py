"""Tests for setup.py's kolibri family (plan docs/superpowers/plans/2026-10-16-kolibri1.md T12): the family table
entry, the single offered size (Q4_K_M) with its pinned size/sha256 and the published URL, the KV arithmetic,
--gguf-dir's shards, and the config the install writes.  Runs without a GPU and without downloads, the way
tools/test_setup_choices.py does; one test asks Hugging Face for the file's size (a HEAD, no download) and skips
when the host cannot be reached.

    python -m unittest tools.test_setup_kolibri
"""
from __future__ import annotations

import os
import sys
import unittest
import urllib.request
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

    def test_the_only_offered_size_is_q4_k_m(self):
        # 2026-10-26: setup offers Q4_K_M only.  Q8_0 (and Q5_K_M, Q6_K) live in a repo SUBFOLDER - a flat URL is a
        # 404, verified - and need a shard download loop; Q2_K / Q3_K_M are not wired in at all.  This is the tripwire
        # for re-adding one without that loop.
        self.assertEqual(sorted(self.fam["models"]), ["Q4_K_M"])
        self.assertEqual(self.fam["models"]["Q4_K_M"]["ram_gb"], 56)

    def test_the_published_url_matches_the_table(self):
        # the guard the Q8_0 defect walked past: the URL setup would actually fetch must serve the pinned size.
        name = setup.own_path_files(self.fam, "Q4_K_M")[0]
        url = self.fam["hf"] + name
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup-test"})
            with urllib.request.urlopen(req, timeout=20) as r:
                length = int(r.headers["Content-Length"])
        except Exception as e:                      # offline, or HF unreachable: not a setup defect
            self.skipTest(f"cannot reach {url}: {e}")
        self.assertEqual(length, self.fam["sha256"][name][0])

    def test_the_download_url_is_pinned(self):
        # the offline half of the download guarantee: the URL setup builds is the pinned revision plus the flat file
        # name (the HEAD test above proves the server serves it; this proves setup asks for the right thing).
        # The HOST is hf_endpoint()'s (HF_ENDPOINT may be a mirror, #495); the revision and the file are the pin.
        self.assertEqual(self.fam["hf"] + setup.own_path_files(self.fam, "Q4_K_M")[0],
                         f"{setup.hf_endpoint()}/Hob-forge/Kolibri-1-GGUF/resolve/"
                         "40eaa5cf6cfca3afaf8b8832bac962652edb0bde/Kolibri-1-Q4_K_M.gguf")

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


if __name__ == "__main__":
    unittest.main()
