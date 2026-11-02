"""Node-local worker sizing, safe topology fallbacks, and prefix preservation across restarts."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup as setup_mod


def fake_tree(root: Path, node_cpus: dict[int, list[int]], cores: list[tuple[int, int]], gpu_node: int | None) -> None:
    """node_cpus: node -> logical cpu list. cores: (package, core_id) per logical cpu, in cpu order."""
    for node, cpus in node_cpus.items():
        d = root / "devices" / "system" / "node" / f"node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(",".join(str(c) for c in cpus) + "\n")
    for cpu, (pkg, core) in enumerate(cores):
        d = root / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
        d.mkdir(parents=True)
        (d / "physical_package_id").write_text(f"{pkg}\n")
        (d / "core_id").write_text(f"{core}\n")
    bus = "0000-82-00.0"
    d = root / "bus" / "pci" / "devices" / bus
    d.mkdir(parents=True)
    (d / "numa_node").write_text(("-1" if gpu_node is None else str(gpu_node)) + "\n")


class ExpertWorkerCount(unittest.TestCase):
    def count(self, root, smi="00000000-82-00.0\n", cpus=56):
        with mock.patch.object(setup_mod.subprocess, "run",
                               return_value=mock.Mock(stdout=smi, returncode=0)), \
             mock.patch.object(setup_mod.os, "cpu_count", return_value=cpus):
            return setup_mod.expert_worker_count(str(root))

    def test_gpu_node_physical_cores_not_logical(self):
        """4 logical CPUs on node 1 that are 2 physical cores (SMT pairs) -> 2 workers."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (0, 0), (0, 1)], gpu_node=1)
            self.assertEqual(self.count(root), 2)

    def test_two_sockets_and_gpu_on_the_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.count(root), 2)

    def test_single_node_falls_back_to_the_old_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1, 2, 3]}, [(0, 0), (0, 1), (0, 2), (0, 3)], gpu_node=0)
            self.assertEqual(self.count(root, cpus=4), 4)

    def test_unknown_gpu_node_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=None)
            self.assertEqual(self.count(root, cpus=8), 8)

    def test_no_nvidia_smi_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            with mock.patch.object(setup_mod.subprocess, "run", side_effect=OSError("no nvidia-smi")), \
                 mock.patch.object(setup_mod.os, "cpu_count", return_value=8):
                self.assertEqual(setup_mod.expert_worker_count(str(root)), 8)

    def test_sparse_node_ids_use_the_actual_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 2: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=2)
            self.assertEqual(self.count(root, cpus=8), 2)

    def test_missing_node_falls_back_instead_of_using_another_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 2: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.count(root, cpus=8), 8)

    def test_multiple_gpus_keep_the_existing_worker_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.count(root, smi="00000000-82-00.0\n00000000-83-00.0\n", cpus=8), 8)

    def test_non_nvidia_setup_keeps_existing_count_without_probing(self):
        with mock.patch.object(setup_mod.os, "cpu_count", return_value=8), \
             mock.patch.object(setup_mod, "_gpu_numa_node", side_effect=AssertionError("must not probe NVIDIA")):
            self.assertEqual(setup_mod.expert_worker_count(nvidia=False), 8)

    def test_cap_is_24(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpus = list(range(64))
            fake_tree(root, {0: cpus[:32], 1: cpus[32:]}, [(1, c) for c in range(32)] + [(1, c) for c in range(32)],
                      gpu_node=1)
            self.assertEqual(self.count(root, cpus=64), 24)


class NumaPrefix(unittest.TestCase):
    """setup.py expert_numa_prefix: bind the engine's memory to the GPU's node, or nothing at all."""

    def prefix(self, root, numactl="/usr/bin/numactl", win=False):
        with mock.patch.object(setup_mod, "WIN", win), \
             mock.patch.object(setup_mod.subprocess, "run",
                               return_value=mock.Mock(stdout="00000000-82-00.0\n", returncode=0)):
            return setup_mod.expert_numa_prefix(str(root), numactl=numactl)

    def test_two_nodes_binds_the_gpu_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.prefix(root), ["/usr/bin/numactl", "--cpunodebind=1", "--membind=1"])

    def test_single_node_needs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1]}, [(0, 0), (0, 1)], gpu_node=0)
            self.assertEqual(self.prefix(root), [])

    def test_no_numactl_starts_the_engine_as_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.prefix(root, numactl=""), [])

    def test_unknown_gpu_node_needs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=None)
            self.assertEqual(self.prefix(root), [])

    def test_sparse_node_ids_bind_the_actual_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 2: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=2)
            self.assertEqual(self.prefix(root), ["/usr/bin/numactl", "--cpunodebind=2", "--membind=2"])

    def test_missing_node_does_not_emit_an_invalid_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 2: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.prefix(root), [])

    def test_multiple_gpus_do_not_bind_all_engines_to_the_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            with mock.patch.object(setup_mod.subprocess, "run", return_value=mock.Mock(
                    stdout="00000000-82-00.0\n00000000-83-00.0\n", returncode=0)):
                self.assertEqual(setup_mod.expert_numa_prefix(str(root), numactl="/usr/bin/numactl"), [])

    def test_non_nvidia_setup_has_no_prefix(self):
        with mock.patch.object(setup_mod, "_gpu_numa_node", side_effect=AssertionError("must not probe NVIDIA")):
            self.assertEqual(setup_mod.expert_numa_prefix(nvidia=False), [])

    def test_denied_memory_binding_has_no_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            with mock.patch.object(setup_mod.subprocess, "run", side_effect=[
                    mock.Mock(stdout="00000000-82-00.0\n", returncode=0), mock.Mock(returncode=1)]):
                self.assertEqual(setup_mod.expert_numa_prefix(str(root), numactl="/usr/bin/numactl"), [])

    def test_binding_probe_timeout_has_no_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            with mock.patch.object(setup_mod.subprocess, "run", side_effect=[
                    mock.Mock(stdout="00000000-82-00.0\n", returncode=0),
                    setup_mod.subprocess.TimeoutExpired("numactl", 5)]):
                self.assertEqual(setup_mod.expert_numa_prefix(str(root), numactl="/usr/bin/numactl"), [])

    def test_windows_needs_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_tree(root, {0: [0, 1], 1: [2, 3]}, [(0, 0), (0, 1), (1, 0), (1, 1)], gpu_node=1)
            self.assertEqual(self.prefix(root, win=True), [])


class EngineCommand(unittest.TestCase):
    """serve/server.py engine_cmd: the prefix goes in front, and an empty prefix changes nothing."""

    def test_prefix_is_prepended(self):
        from serve.server import engine_cmd
        self.assertEqual(engine_cmd("/e/strata", ["--model", "m"], ["/usr/bin/numactl", "--membind=1"]),
                         ["/usr/bin/numactl", "--membind=1", "/e/strata", "--serve", "--model", "m"])

    def test_no_prefix_is_the_old_argv_exactly(self):
        from serve.server import engine_cmd
        self.assertEqual(engine_cmd("/e/strata", ["--model", "m"], None), ["/e/strata", "--serve", "--model", "m"])

    def test_restart_prefix_is_a_snapshot_not_the_callers_mutable_list(self):
        from serve.server import StrataEngine
        prefix = ["/usr/bin/numactl", "--membind=1"]
        engine = StrataEngine("/nonexistent/strata", [], lazy=True, prefix=prefix)
        prefix.append("unexpected")
        self.assertEqual(engine.launch_prefix, ["/usr/bin/numactl", "--membind=1"])

    def test_restart_replays_prefix_by_name(self):
        from serve.server import StrataEngine
        prefix = ["/usr/bin/numactl", "--membind=1"]
        engine = StrataEngine("/nonexistent/strata", ["--model", "fixture"], lazy=True, prefix=prefix)
        with mock.patch.object(engine, "close"), mock.patch.object(engine, "__init__", return_value=None) as restart:
            engine.restart()
        self.assertEqual(restart.call_args.kwargs["prefix"], prefix)
        self.assertEqual(restart.call_args.args[1], ["--model", "fixture"])
        self.assertFalse(restart.call_args.args[5])

    def test_conversation_cache_can_still_read_the_spawn_arguments(self):
        from serve.server import StrataEngine, conversation_slots
        engine = StrataEngine("/nonexistent/strata", ["--conversation-cache-slots", "7"], lazy=True,
                              prefix=["/usr/bin/numactl", "--membind=1"])
        self.assertEqual(conversation_slots(engine), 7)

    def test_every_engine_construction_passes_the_launch_prefix(self):
        """All three paths - single engine, replicas, split - must hand the config's prefix on.
        A missing keyword here is silent: the engine starts unbound and looks healthy."""
        import ast
        from serve import server
        tree = ast.parse((Path(server.__file__)).read_text())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "StrataEngine"]
        self.assertEqual(len(calls), 3, "expected the single, replica and split construction sites")
        for call in calls:
            names = [k.arg for k in call.keywords]
            self.assertIn("prefix", names, f"StrataEngine call at line {call.lineno} drops the launch prefix")

    def test_config_key_is_what_the_launcher_reads(self):
        """setup.py writes "launch_prefix"; the server reads rc.get("launch_prefix")."""
        import inspect
        from serve import server
        src = inspect.getsource(server)
        self.assertIn('rc.get("launch_prefix")', src)
        self.assertIn("engine_cmd(exe, args, self.launch_prefix)", src)


if __name__ == "__main__":
    unittest.main()
