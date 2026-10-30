"""The broad regression suites must actually run on PRs, with their real prerequisites."""

import ast
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "broad-tests.yml"


class BroadCI(unittest.TestCase):
    def workflow(self):
        self.assertTrue(WORKFLOW.is_file(), "the broad suite needs its own automatic CI workflow")
        return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))

    def test_runs_on_every_pr_and_main_push(self):
        workflow = self.workflow()
        triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 treats `on` as a boolean
        self.assertIsNone(triggers["pull_request"])
        self.assertEqual(triggers["push"]["branches"], ["main"])
        self.assertEqual(workflow["permissions"], {"contents": "read"})

    def test_covers_terminal_setup_and_mcp_on_linux_and_windows(self):
        job = self.workflow()["jobs"]["broad"]
        matrix = job["strategy"]["matrix"]
        self.assertEqual(set(matrix["os"]), {"ubuntu-latest", "windows-latest"})
        self.assertEqual(set(matrix["suite"]), {"terminal", "setup"})
        self.assertFalse(job["strategy"]["fail-fast"])
        self.assertLessEqual(job["timeout-minutes"], 60)
        self.assertNotIn("continue-on-error", job)
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn("python -m pytest tests/terminal", commands)
        self.assertIn("tools/test_setup_*.py tools/test_strata_mcp.py", commands)
        self.assertNotIn("continue-on-error", " ".join(str(s) for s in job["steps"]))

    def test_uses_the_installers_pinned_gguf_not_a_mock_or_unrelated_pypi_package(self):
        job = self.workflow()["jobs"]["broad"]
        deps = next(s for s in job["steps"] if s.get("with", {}).get("repository") == "ggml-org/llama.cpp")
        tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
        pin = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "LLAMA_CPP_COMMIT" for t in n.targets))
        self.assertEqual(deps["with"]["ref"], pin)
        self.assertEqual(deps["with"]["sparse-checkout"], "gguf-py")
        self.assertFalse(deps["with"]["persist-credentials"])
        self.assertIn("gguf-py", job["env"]["STRATA_GGUF_PY"])

    def test_desktop_patch_is_checked_on_native_hosts(self):
        job = self.workflow()["jobs"]["desktop"]
        self.assertEqual(set(job["strategy"]["matrix"]["os"]),
                         {"ubuntu-latest", "windows-latest", "macos-15"})
        node = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-node@"))
        self.assertEqual(node["with"]["node-version"], "24")
        commands = "\n".join(s.get("run", "") for s in job["steps"])
        self.assertIn("cargo +1.90.0 check --locked", commands)
        self.assertIn("npm --prefix apps/host test", commands)
        self.assertIn("npm --prefix apps/host run build", commands)
        self.assertNotIn("continue-on-error", " ".join(str(s) for s in job["steps"]))

    def test_actions_are_pinned_and_failure_reports_are_retained(self):
        steps = self.workflow()["jobs"]["broad"]["steps"]
        for step in steps:
            if "uses" in step:
                self.assertRegex(step["uses"], r"@[0-9a-f]{40}$")
        upload = next(s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@"))
        self.assertEqual(upload["if"], "always()")
        self.assertIn("xml", upload["with"]["path"])
        self.assertNotIn("**", upload["with"]["path"], "do not upload user state or arbitrary files")


if __name__ == "__main__":
    unittest.main()
