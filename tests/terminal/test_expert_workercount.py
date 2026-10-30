"""setup.py expert_worker_count: the CPU expert pool gets one node's physical cores, not every logical CPU.

Measured on the T4 box (docs/experiments/kolibri-perf-gap.md, Phase 1): 8 workers spread over two NUMA nodes = 10,398 ms
for a 511-token prefill; 13 workers on the GPU node's physical cores = 8,128 ms; with the memory bound to that node as
well, 7,922 ms and half the decode expert time. The rule below is the first of those three steps, and it must degrade
to the previous behaviour whenever the topology cannot be described - a wrong guess here would silently slow every
machine down.
"""
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
    bus = "0000:82:00.0"
    d = root / "bus" / "pci" / "devices" / bus
    d.mkdir(parents=True)
    (d / "numa_node").write_text(("-1" if gpu_node is None else str(gpu_node)) + "\n")


class ExpertWorkerCount(unittest.TestCase):
    def count(self, root, smi="00000000:82:00.0\n", cpus=56):
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

    def test_cap_is_24(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cpus = list(range(64))
            fake_tree(root, {0: cpus[:32], 1: cpus[32:]}, [(1, c) for c in range(32)] + [(1, c) for c in range(32)],
                      gpu_node=1)
            self.assertEqual(self.count(root, cpus=64), 24)


if __name__ == "__main__":
    unittest.main()


class NumaPrefix(unittest.TestCase):
    """setup.py expert_numa_prefix: bind the engine's memory to the GPU's node, or nothing at all."""

    def prefix(self, root, numactl="/usr/bin/numactl", win=False):
        with mock.patch.object(setup_mod, "WIN", win), \
             mock.patch.object(setup_mod.subprocess, "run",
                               return_value=mock.Mock(stdout="00000000:82:00.0\n", returncode=0)):
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

    def test_config_key_is_what_the_launcher_reads(self):
        """setup.py writes "launch_prefix"; the server reads rc.get("launch_prefix")."""
        import inspect
        from serve import server
        src = inspect.getsource(server)
        self.assertIn('rc.get("launch_prefix")', src)
        self.assertIn("engine_cmd(exe, args, self.launch_prefix)", src)
