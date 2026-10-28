"""Kolibri 1 in the auto-tuner: its own family, its own machine class, and the knobs its engine path actually has.

Kolibri 1 is host-streaming on every card - the 42.90 GiB expert table is pinned in system RAM and every expert layer is
served by the CPU pool - so a flash-next knob (`--expert-cache`, `--kv-resident`, `--spec`) is not one of its levers:
its engine path takes `--expert-slots*`, `--cpu-threads`, `--prefill` and a `STRATA_KOLIBRI_*` environment.  A config
names its family in "arch"; without that the arguments look like a single-GGUF flash-next run.
"""
from __future__ import annotations

import unittest

from strata_control import autotune, tune_knobs

KOLIBRI_ARGS = ["--model", "/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf", "--prefill", "--cpu-threads", "8",
                "--ctx", "131072"]


def kolibri_ctx(**kw) -> tune_knobs.Context:
    """This box: R9700 32 GB (gfx1201), 93 GiB RAM, a 47.5 GB model, 128K context, 8 physical cores of 16 threads."""
    fields = dict(backend="hip", family="kolibri1", gpu_count=1, vram_mib=32624, ram_gib=93, model_gib=47.5,
                  context_tokens=131072, base_args=list(KOLIBRI_ARGS), logical_threads=16, arch="gfx1201")
    fields.update(kw)
    ctx = tune_knobs.Context(**fields)
    ctx.facts = {"workers": 8}
    return ctx


def ids(planned) -> list[str]:
    return [p.arm.id for p in planned]


class WorkerCount(unittest.TestCase):
    """The engine's own warning: "8 CPU expert workers are configured, but this CPU has 8 physical cores" - two
    workers on one core share its memory bandwidth."""

    def test_one_worker_fewer_is_offered_from_the_configs_own_number(self):
        arms = {p.arm.id: p for p in tune_knobs.plan_arms(kolibri_ctx(), mode="thorough")[0]}
        self.assertEqual(arms["k_threads_minus1"].args["--cpu-threads"], 7)
        self.assertEqual(arms["k_threads_minus1"].arm.answer, "exact")

    def test_a_configuration_without_the_lane_count_is_left_alone(self):
        ctx = kolibri_ctx(base_args=["--model", "m.gguf", "--ctx", "131072"])
        self.assertNotIn("k_threads_minus1", ids(tune_knobs.plan_arms(ctx, mode="thorough")[0]))


class Family(unittest.TestCase):
    SYSTEM = {"memory": {"total_mib": 93 * 1024},
              "gpus": [{"index": 0, "vendor": "amd", "vram_total_mib": 32624},
                       {"index": 1, "vendor": "amd", "vram_total_mib": 49152},
                       {"index": 2, "vendor": "amd", "vram_total_mib": 512}],      # an iGPU is a card too
              "cpu": {"logical": 16, "hybrid": False}}

    def test_the_config_names_the_family(self):
        cfg = {"arch": "kolibri1", "args": KOLIBRI_ARGS}
        ctx = autotune.context_from_config(cfg, engine_backend="hip", opts=autotune.Options(), system=self.SYSTEM)
        self.assertEqual(ctx.family, "kolibri1")

    def test_a_config_without_the_family_stays_flash_next(self):
        ctx = autotune.context_from_config({"args": KOLIBRI_ARGS}, engine_backend="hip", opts=autotune.Options(),
                                           system=self.SYSTEM)
        self.assertEqual(ctx.family, "flash-next")          # nothing changes for the other families

    def test_the_card_the_config_names_is_the_one_measured(self):
        """A box with an iGPU: without "gpu" the tuner takes the smallest card, and an expert-cache arm is not offered."""
        named = autotune.context_from_config({"arch": "kolibri1", "gpu": 0, "args": KOLIBRI_ARGS},
                                             engine_backend="hip", opts=autotune.Options(), system=self.SYSTEM)
        self.assertEqual(named.vram_mib, 32624)
        unnamed = autotune.context_from_config({"arch": "kolibri1", "args": KOLIBRI_ARGS},
                                               engine_backend="hip", opts=autotune.Options(), system=self.SYSTEM)
        self.assertEqual(unnamed.vram_mib, 512)

    def test_the_kolibri_engine_spelling_of_the_context_is_read(self):
        """`--ctx` (this engine path) is `--max-context` elsewhere: unread, 128K looked like 32K."""
        self.assertEqual(KOLIBRI_ARGS[KOLIBRI_ARGS.index("--ctx") + 1], "131072")
        ctx = autotune.context_from_config({"arch": "kolibri1", "args": KOLIBRI_ARGS}, engine_backend="hip",
                                           opts=autotune.Options(), system=self.SYSTEM)
        self.assertEqual(ctx.context_tokens, 131072)
        long_front_end = autotune.context_from_config({"arch": "kolibri1", "args": ["--max-context", "65536"]},
                                                      engine_backend="hip", opts=autotune.Options(), system=self.SYSTEM)
        self.assertEqual(long_front_end.context_tokens, 65536)


class MachineClass(unittest.TestCase):
    def test_kolibri_is_its_own_class(self):
        self.assertEqual(tune_knobs.machine_classes(kolibri_ctx()), ["K"])

    def test_flash_next_never_sees_the_kolibri_class(self):
        flash = tune_knobs.Context(backend="hip", family="flash-next", gpu_count=1, vram_mib=32624, ram_gib=93,
                                   model_gib=22.0, context_tokens=131072, base_args=["--pack", "P", "--native", "m.gguf"])
        self.assertNotIn("K", tune_knobs.machine_classes(flash))


class Arms(unittest.TestCase):
    def test_the_candidates_are_the_engine_paths_own_knobs(self):
        arms, skipped = tune_knobs.plan_arms(kolibri_ctx(allow_placement=True), mode="thorough")
        offered = ids(arms)
        for expected in ("k_slots_auto", "k_slots_hot", "k_f16_off", "k_attn_mb_256", "k_pf_partial",
                         "k_threads_logical"):
            self.assertIn(expected, offered)
        self.assertFalse([s for s in skipped if s["id"] in offered])       # each one is inside the rules too
        for p in arms:
            self.assertNotIn("--expert-cache", p.args)                      # not a knob on this engine path
            self.assertNotIn("--kv-resident", p.args)
            self.assertNotIn("--spec", p.args)

    def test_a_configuration_that_already_reads_prompts_in_chunks_is_not_offered_it(self):
        no_prefill = kolibri_ctx(base_args=["--model", "m.gguf", "--cpu-threads", "8"])
        self.assertIn("k_prefill", ids(tune_knobs.plan_arms(no_prefill, mode="thorough")[0]))
        self.assertNotIn("k_prefill", ids(tune_knobs.plan_arms(kolibri_ctx(), mode="thorough")[0]))

    def test_the_other_families_do_not_get_kolibri_arms(self):
        flash = kolibri_ctx(family="flash-next", allow_placement=True)
        flash.base_args = ["--pack", "P", "--native", "m.gguf", "--max-context", "131072"]
        self.assertEqual([i for i in ids(tune_knobs.plan_arms(flash, mode="thorough")[0]) if i.startswith("k_")], [])
        self.assertEqual([i for i in ids(tune_knobs.plan_arms(kolibri_ctx(allow_placement=True), mode="thorough")[0])
                           if i in ("kv_resident_64k", "prefill_32k", "spec3")], [])

    def test_a_card_without_room_is_not_offered_an_expert_cache(self):
        small = kolibri_ctx(vram_mib=6144)
        offered = ids(tune_knobs.plan_arms(small, mode="thorough")[0])
        self.assertNotIn("k_slots_auto", offered)
        self.assertIn("k_f16_off", offered)                                 # costs no VRAM, still worth trying

    def test_what_already_is_off_is_not_tried(self):
        base = kolibri_ctx(base_args=list(KOLIBRI_ARGS) + ["--expert-slots", "auto"])
        self.assertNotIn("k_slots_auto", ids(tune_knobs.plan_arms(base, mode="thorough")[0]))

    def test_decode_residency_is_refused(self):
        ctx = kolibri_ctx()
        self.assertTrue(tune_knobs.check_setting({}, {"STRATA_KOLIBRI_DEC_RES": "1"}, ctx))
        self.assertEqual(tune_knobs.check_setting({"--expert-slots": "auto", "--vram-reserve-mib": 1024},
                                                  {}, ctx), [])            # inside the rules: no bounds on these

    def test_the_hyperthread_arm_reads_the_machine(self):
        ctx = kolibri_ctx(allow_placement=True)
        arms = {p.arm.id: p for p in tune_knobs.plan_arms(ctx, mode="thorough")[0]}
        self.assertEqual(arms["k_threads_logical"].args["--cpu-threads"], 16)
        one_per_core = kolibri_ctx(logical_threads=8, allow_placement=True)   # no hyperthreads: same as the base
        self.assertNotIn("k_threads_logical", ids(tune_knobs.plan_arms(one_per_core, mode="thorough")[0]))
        self.assertNotIn("k_threads_logical", ids(tune_knobs.plan_arms(kolibri_ctx(), mode="thorough")[0]))  # no tick


class RequestSet(unittest.TestCase):
    """What a run costs: a host-streaming model reads a prompt at ~60-100 tok/s, so the standard 96K prompt would make
    one run last 16 minutes and a whole tune a couple of hours."""

    def test_the_long_prompt_is_capped_for_this_family(self):
        spec = autotune.request_set(kolibri_ctx())
        self.assertEqual(spec.long_tokens, 8192)
        self.assertEqual(spec.pp_tokens, 2048)

    def test_the_other_families_keep_their_sizes(self):
        flash = kolibri_ctx(family="flash-next")
        spec = autotune.request_set(flash)
        self.assertEqual((spec.long_tokens, spec.pp_tokens), (98304, 4096))

    def test_a_short_context_gets_no_long_prompt(self):
        self.assertEqual(autotune.request_set(kolibri_ctx(context_tokens=32768)).long_tokens, 0)


if __name__ == "__main__":
    unittest.main()
