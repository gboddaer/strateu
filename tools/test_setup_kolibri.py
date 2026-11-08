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


class KolibriSourceTuning(unittest.TestCase):
    """The real config writer, with the existing offline install harness (no GPU/downloads)."""
    def setUp(self):
        from tools.test_setup_qwen36 import InstallOwnPath
        InstallOwnPath.setUp(self)
        self.families = {"kolibri1"}
        from unittest import mock
        for patch in (mock.patch.object(setup, "get_llama_cpp", return_value=self.root / "llama"),
                      mock.patch.object(setup, "find_tool", side_effect=lambda name: "/tools/" + name),
                      mock.patch.object(setup, "rocm_root", return_value=(self.root / "rocm", ["/rocm/lib"]))):
            patch.start()
            self.addCleanup(patch.stop)

    def install(self, hip=True, gpu=None, ram=96, model="Q4_K_M"):
        from tools.test_setup_qwen36 import args, run
        return run(setup.install_own_path, args(models_dir=str(self.root / "models"), model=model, vision="no"),
                   "kolibri", gpu or {"index": 0, "count": 1, "vram_gb": 32, "arch": "gfx1201", "name": "R9700"},
                   hip, [], 13, [], 8080, ram, None)

    def test_source_install_selects_compatible_shared_expert_tuning(self):
        from unittest import mock
        table = self.root / "gfx1201-hipblaslt-100202.txt"
        with mock.patch.object(setup, "hipblaslt_table", return_value=table):
            rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["env"]["STRATA_HIPBLASLT_TUNING"], str(table))

    def test_fresh_amd_source_build_enables_hip_mmq_not_cuda(self):
        from unittest import mock
        with mock.patch.object(setup, "hipblaslt_table", return_value=None):
            rc, out = self.install()
        self.assertEqual(rc, 0, out)
        conf = next(c[1] for c in self.calls if c[0] == "run" and "-S" in c[1])
        self.assertIn("-DSTRATA_ENABLE_HIP=ON", conf)
        self.assertIn("-DSTRATA_ENABLE_CUDA=OFF", conf)
        self.assertIn("-DSTRATA_PREFILL_MMQ=ON", conf)
        self.assertIn("-DCMAKE_HIP_ARCHITECTURES=gfx1201", conf)

    def test_missing_table_keeps_the_runtime_pool_fallback(self):
        from unittest import mock
        with mock.patch.object(setup, "hipblaslt_table", return_value=None):
            rc, out = self.install()
        self.assertEqual(rc, 0, out)
        self.assertNotIn("env", self.cfg)

    def test_cuda_config_persists_model_specific_defaults_without_hip_tuning(self):
        from unittest import mock
        with mock.patch.object(setup, "hipblaslt_table") as find:
            rc, out = self.install(hip=False)
        self.assertEqual(rc, 0, out)
        find.assert_not_called()
        self.assertEqual(self.cfg["env"]["STRATA_KOLIBRI_EXPERT_F16"], "0")
        self.assertEqual(self.cfg["env"]["STRATA_KOLIBRI_STREAM_DEV_ACT"], "1")
        self.assertEqual(self.cfg["env"]["STRATA_KOLIBRI_PREFIX_CACHE_MB"], "128")
        self.assertEqual(self.cfg["sampling"]["temperature"], 1.0)

    def test_t4_defaults_keep_exact_streaming_not_rejected_fp16_cache(self):
        from unittest import mock
        gpu = {"index": 0, "count": 1, "vram_gb": 16, "arch": 75, "name": "Tesla T4"}
        with mock.patch.object(setup, "expert_worker_count", return_value=14):
            rc, out = self.install(hip=False, gpu=gpu)
        self.assertEqual(rc, 0, out)
        env, argv = self.cfg["env"], self.cfg["args"]
        self.assertEqual(env["STRATA_KOLIBRI_EXPERT_STREAM"], "1")
        self.assertEqual(env["STRATA_KOLIBRI_EXPERT_F16"], "0")
        self.assertEqual(env["STRATA_KOLIBRI_POOL_ALL"], "1")
        self.assertEqual(env["STRATA_KOLIBRI_REUSE_PREFIX"], "1")
        self.assertEqual(env["STRATA_KOLIBRI_REUSE_ALIGN"], "1")
        self.assertEqual(env["STRATA_KOLIBRI_PF_CHUNK"], "512")
        self.assertEqual(argv[argv.index("--cpu-threads") + 1], "14")
        self.assertEqual(argv[argv.index("--expert-slots") + 1], "0")
        self.assertNotIn("--expert-slots-hot", argv)
        self.assertNotIn("STRATA_KOLIBRI_PF_PARTIAL", env)
        self.assertEqual(self.cfg["sampling"], {"temperature": 1.0})


    CUDA = {"index": 0, "count": 1, "vram_gb": 16, "arch": 75, "name": "Tesla T4"}
    PREFIX = ["numactl", "--cpunodebind=0", "--membind=0"]
    TUNING = Path("/data/strata/gfx1201-hipblaslt-100202.txt")
    # the CUDA profile the approved default persists: five table-leg entries plus five that stay in every placement
    STREAM_ENV = {"STRATA_KOLIBRI_EXPERT_STREAM": "1", "STRATA_KOLIBRI_STREAM_DEV_ACT": "1",
                  "STRATA_KOLIBRI_STREAM_GU": "1", "STRATA_KOLIBRI_POOL_ALL": "1",
                  "STRATA_KOLIBRI_PF_CHUNK": "512"}
    COMMON_ENV = {"STRATA_KOLIBRI_EXPERT_F16": "0", "STRATA_KOLIBRI_REUSE": "1",
                  "STRATA_KOLIBRI_REUSE_PREFIX": "1", "STRATA_KOLIBRI_REUSE_ALIGN": "1",
                  "STRATA_KOLIBRI_PREFIX_CACHE_MB": "128"}
    CUDA_ENV = {**STREAM_ENV, **COMMON_ENV}
    DEMAND = {"STRATA_KOLIBRI_NOPIN_PROFILE": "demand"}

    def gguf(self):
        return self.root / "models" / "kolibri-Q4_K_M" / "Kolibri-1-Q4_K_M.gguf"

    def profile(self):
        return self.root / "data" / "hot-kolibri-q4_k_m.prof"

    def mmap_args(self):
        return ["--expert-no-pin", "--expert-slots", "auto", "--expert-slots-hot",
                "--expert-slots-profile", str(self.profile()), "--vram-reserve-mib", "1024"]

    def slot_args(self):
        return ["--expert-slots", "auto", "--expert-slots-hot",
                "--expert-slots-profile", str(self.profile()), "--vram-reserve-mib", "1024"]

    def placement_install(self, hip, ram, **kw):
        """The same install, with the placement's inputs fixed so the fixtures are deterministic."""
        from unittest import mock
        def numa(nvidia=True, **kw):
            return [] if not nvidia else self.PREFIX          # the real one binds only on NVIDIA

        with mock.patch.object(setup, "expert_worker_count", return_value=14), \
             mock.patch.object(setup, "expert_numa_prefix", side_effect=numa), \
             mock.patch.object(setup, "hipblaslt_table", return_value=self.TUNING if hip else None):
            return self.install(hip=hip, gpu=None if hip else self.CUDA, ram=ram, **kw)

    def test_the_high_ram_cuda_config_is_unchanged(self):
        for ram in (64, 128):
            rc, out = self.placement_install(hip=False, ram=ram)
            self.assertEqual(rc, 0, out)
            self.assertEqual(self.cfg["args"], ["--model", str(self.gguf()), "--ctx", "32768", "--prefill",
                                                "--cpu-threads", "14", "--expert-slots", "0"], ram)
            self.assertEqual(self.cfg["env"], self.CUDA_ENV, ram)
            self.assertEqual(self.cfg["launch_prefix"], self.PREFIX, ram)

    def test_the_high_ram_hip_config_is_unchanged(self):
        rc, out = self.placement_install(hip=True, ram=128)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"], ["--model", str(self.gguf()), "--ctx", "131072", "--prefill",
                                            "--cpu-threads", "14"] + self.slot_args())
        self.assertEqual(self.cfg["env"], {"STRATA_HIPBLASLT_TUNING": str(self.TUNING)})
        self.assertNotIn("launch_prefix", self.cfg)      # the real prefix is NVIDIA-only

    def test_low_ram_cuda_writes_the_mmap_placement(self):
        rc, out = self.placement_install(hip=False, ram=32)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"], ["--model", str(self.gguf()), "--ctx", "32768", "--prefill"]
                         + self.mmap_args())
        self.assertNotIn("--cpu-threads", self.cfg["args"])
        self.assertEqual(self.cfg["env"], {**self.COMMON_ENV, **self.DEMAND})
        self.assertIn("table-placement threshold", out)
        self.assertNotIn("may not start", out)

    def test_low_ram_cuda_keeps_no_table_leg_environment(self):
        rc, out = self.placement_install(hip=False, ram=32)
        self.assertEqual(rc, 0, out)
        for key in self.STREAM_ENV:
            self.assertNotIn(key, self.cfg["env"])

    def test_low_ram_hip_writes_the_mmap_placement_and_keeps_its_tuning(self):
        rc, out = self.placement_install(hip=True, ram=32)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["args"], ["--model", str(self.gguf()), "--ctx", "131072", "--prefill"]
                         + self.mmap_args())
        self.assertNotIn("--cpu-threads", self.cfg["args"])
        self.assertEqual(self.cfg["env"], {**self.DEMAND,
                                           "STRATA_HIPBLASLT_TUNING": str(self.TUNING)})

    def test_low_ram_hip_without_a_tuning_table_has_exactly_the_demand_environment(self):
        from unittest import mock
        with mock.patch.object(setup, "expert_worker_count", return_value=14), \
             mock.patch.object(setup, "expert_numa_prefix", return_value=[]), \
             mock.patch.object(setup, "hipblaslt_table", return_value=None):
            rc, out = self.install(hip=True, ram=32)
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.cfg["env"], self.DEMAND)

    def test_the_saved_config_follows_the_threshold_on_both_backends(self):
        for ram, mmap_expected in ((51.9, True), (52, False)):
            for hip in (False, True):
                rc, out = self.placement_install(hip=hip, ram=ram)
                self.assertEqual(rc, 0, out)
                self.assertEqual("--expert-no-pin" in self.cfg["args"], mmap_expected, (ram, hip))

    def test_a_low_ram_unattended_install_needs_no_table_question(self):
        rc, out = self.placement_install(hip=False, ram=32, model=None)
        self.assertEqual(rc, 0, out)
        self.assertIn("--expert-no-pin", self.cfg["args"])
        self.assertIn("table-placement threshold", out)

class KolibriPlacement(unittest.TestCase):
    """The RAM rule that picks Kolibri's expert placement (plan 2026-11-07, Task 1).

    ram_gb() and the size table are GiB despite the historical _gb suffix; the threshold is setup's
    recommendation, so the rows are the published 56 GiB need less the 4 GiB shave the repo already uses.
    """

    PROFILE = Path("/data/strata/hot-kolibri-q4_k_m.prof")
    TABLE = ["--expert-slots", "auto", "--expert-slots-hot",
             "--expert-slots-profile", str(PROFILE), "--vram-reserve-mib", "1024"]
    MMAP = ["--expert-no-pin"] + TABLE
    DEMAND = {"STRATA_KOLIBRI_NOPIN_PROFILE": "demand"}

    def setUp(self):
        self.fam = setup.OWN_PATH_FAMILIES["kolibri"]

    def placement(self, ram, on_hip=False, reserve=1024):
        return setup.kolibri_expert_placement(self.fam, "Q4_K_M", ram, on_hip, self.PROFILE, reserve)

    def test_the_table_metadata_and_the_published_need(self):
        d = self.fam["models"]["Q4_K_M"]
        self.assertEqual(d["ram_gb"], 56)                 # the published need does not move
        self.assertEqual(d["expert_table_gib"], 42.9)     # the engine's "42.90 GiB pinned" load line

    def test_cuda_holds_the_table_from_the_threshold_up(self):
        for ram in (52, 56, 64, 128):
            self.assertEqual(self.placement(ram), (["--expert-slots", "0"], {}, ""), ram)

    def test_cuda_takes_the_mmap_cache_below_the_threshold(self):
        for ram in (24, 30, 32, 48, 51, 51.9):
            args, env, note = self.placement(ram)
            self.assertEqual(args, self.MMAP, ram)
            self.assertEqual(env, self.DEMAND, ram)
            self.assertTrue(note, ram)

    def test_hip_holds_the_slot_cache_from_the_threshold_up(self):
        for ram in (52, 56, 64, 128):
            self.assertEqual(self.placement(ram, on_hip=True), (self.TABLE, {}, ""), ram)

    def test_hip_takes_the_mmap_cache_below_the_threshold(self):
        for ram in (24, 32, 51.9):
            args, env, note = self.placement(ram, on_hip=True)
            self.assertEqual(args, self.MMAP, ram)
            self.assertEqual(env, self.DEMAND, ram)
            self.assertTrue(note, ram)

    def test_the_threshold_is_the_published_need_minus_the_shave(self):
        self.assertEqual(setup.KOLIBRI_TABLE_RAM_SHAVE_GB, 4)
        self.assertEqual(self.placement(52)[2], "")
        self.assertNotEqual(self.placement(51.9)[2], "")

    def test_an_explicit_reserve_is_carried_through(self):
        self.assertEqual(self.placement(32, reserve=1536)[0][-1], "1536")
        self.assertEqual(self.placement(64, on_hip=True, reserve=1536)[0][-1], "1536")

    def test_the_note_states_its_numbers_in_gib(self):
        note = self.placement(32)[2]
        self.assertIn("32 GiB", note)
        self.assertIn("52 GiB", note)
        self.assertIn("43 GiB", note)

    def test_the_note_promises_neither_startup_nor_speed(self):
        note = self.placement(32)[2].lower()
        for claim in ("guarantee", "will start", "always start", "faster", "slower"):
            self.assertNotIn(claim, note)

    def test_unreadable_ram_is_conservative_and_says_so(self):
        args, env, note = self.placement(0)
        self.assertIn("--expert-no-pin", args)
        self.assertIn("unknown", note.lower())
        self.assertNotIn("0 GiB", note)



if __name__ == "__main__":
    unittest.main()
