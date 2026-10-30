"""Reject unsupported server/setup interpreters before doing work; keep old venvs intact."""
import contextlib
import io
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import setup
import strata
from tools import strata_mcp

ROOT = Path(__file__).resolve().parents[2]


class RuntimeMinimum(unittest.TestCase):
    def test_setup_rejects_python310_before_setup_writes(self):
        from strata_control import private_files
        output = io.StringIO()
        with mock.patch.object(sys, "version_info", (3, 10, 99)), \
                mock.patch.object(sys, "argv", ["setup.py", "--help"]), \
                mock.patch.object(private_files, "adopt_user_defaults") as adopt, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                code = setup.main()
            except SystemExit as exc:
                code = exc.code
        self.assertEqual(code, 2)
        self.assertIn("3.11", output.getvalue())
        adopt.assert_not_called()

    def test_terminal_rejects_python310_before_checking_packages(self):
        output = io.StringIO()
        with mock.patch.object(sys, "version_info", (3, 10, 99)), \
                mock.patch.object(strata.importlib.util, "find_spec") as find, \
                contextlib.redirect_stderr(output):
            code = strata._preflight(["serve"])
        self.assertEqual(code, 5)
        self.assertIn("3.11", output.getvalue())
        find.assert_not_called()

    def test_terminal_help_on_python311_needs_no_packages(self):
        with mock.patch.object(sys, "version_info", (3, 11, 0)), \
                mock.patch.object(strata.importlib.util, "find_spec") as find:
            self.assertEqual(strata._preflight(["--help"]), 0)
        find.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("jinja2"), "server dependency suite exercises this guard")
    def test_server_rejects_python310_before_creating_user_files(self):
        from serve import server
        from strata_control import private_files
        output = io.StringIO()
        with mock.patch.object(sys, "version_info", (3, 10, 99)), \
                mock.patch.object(private_files, "adopt_user_defaults") as adopt, \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                code = server.main(["--help"])
            except SystemExit as exc:
                code = exc.code
        self.assertEqual(code, 2)
        self.assertIn("3.11", output.getvalue())
        adopt.assert_not_called()

    def test_direct_server_rejects_python310_without_optional_packages(self):
        command = ("import runpy, sys; sys.version_info = (3, 10, 99); "
                   f"runpy.run_path({str(ROOT / 'serve/server.py')!r}, run_name='__main__')")
        result = subprocess.run([sys.executable, "-S", "-c", command], cwd=ROOT,
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("3.11", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_mcp_rejects_python310_before_running_an_install_job(self):
        output = io.StringIO()
        with mock.patch.object(sys, "version_info", (3, 10, 99)), \
                mock.patch.object(strata_mcp, "install_job", return_value=0) as install, \
                contextlib.redirect_stderr(output):
            code = strata_mcp.main(["--install-job", "unused.json"])
        self.assertEqual(code, 2)
        self.assertIn("3.11", output.getvalue())
        install.assert_not_called()

    def test_mcp_install_plan_rejects_python310_before_probing_hardware(self):
        state = mock.Mock()
        state.tables.return_value = ({}, {}, [], "fixture")
        state.hardware.side_effect = AssertionError("unsupported Python reached hardware probing")
        mcp = strata_mcp.Tools(state)
        with mock.patch.object(sys, "version_info", (3, 10, 99)):
            with self.assertRaisesRegex(strata_mcp.ToolError, "3.11"):
                mcp.install_plan(None, None, None, None, None, None, None, None, None, None, None)
        state.hardware.assert_not_called()


@unittest.skipUnless(os.name == "nt", "native Windows setup wrapper")
class WindowsSetupEnvironment(unittest.TestCase):
    def test_existing_environment_requires_python311_and_is_never_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(ROOT / "START-HERE.bat", root / "START-HERE.bat")
            (root / "setup.py").write_text("from pathlib import Path\nPath('setup-ran').touch()\n")
            subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")], check=True,
                           capture_output=True, timeout=60)
            sentinel = root / ".venv" / "keep.txt"
            sentinel.write_text("keep this environment")
            result = subprocess.run(["cmd", "/d", "/c", "START-HERE.bat", "--help"], cwd=root,
                                    input="", capture_output=True, text=True, timeout=30)
            self.assertTrue(sentinel.exists())
            if sys.version_info < (3, 11):
                self.assertEqual(result.returncode, 2)
                self.assertIn("3.11", result.stdout + result.stderr)
                self.assertFalse((root / "setup-ran").exists())
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((root / "setup-ran").exists())


@unittest.skipIf(os.name == "nt", "POSIX setup wrapper; Windows guards are exercised on Windows")
class ExistingSetupEnvironment(unittest.TestCase):
    def run_wrapper(self, version):
        """The stand-in interpreter delegates to real Python with the requested version probe."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        shutil.copy2(ROOT / "setup.sh", root / "setup.sh")
        (root / "setup.py").write_text("from pathlib import Path\nPath('setup-ran').touch()\n")
        python = root / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            "if len(sys.argv) > 1 and sys.argv[1] == '-c':\n"
            f"    sys.version_info = {version!r}\n"
            "    exec(sys.argv[2])\n"
            "elif sys.argv[1:4] == ['-m', 'pip', '--version']:\n"
            "    print('pip fixture')\n"
            "else:\n"
            "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
        )
        python.chmod(0o755)
        sentinel = root / ".venv" / "keep.txt"
        sentinel.write_text("existing environment must not be removed")
        result = subprocess.run(["sh", str(root / "setup.sh"), "--help"], cwd=root,
                                capture_output=True, text=True, timeout=10)
        self.assertTrue(sentinel.exists())
        return root, result

    def test_existing_python310_environment_is_preserved_but_not_used(self):
        root, result = self.run_wrapper((3, 10, 99))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("3.11", result.stdout + result.stderr)
        self.assertFalse((root / "setup-ran").exists())

    def test_existing_python311_environment_is_reused(self):
        root, result = self.run_wrapper((3, 11, 0))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((root / "setup-ran").exists())

    def test_terminal_rejects_an_old_explicit_override_before_package_probing(self):
        root, _ = self.run_wrapper((3, 10, 99))
        (root / "bin").mkdir()
        shutil.copy2(ROOT / "bin/strata", root / "bin/strata")
        (root / "strata.py").write_text("raise SystemExit('the CLI must not run')\n")
        env = {**os.environ, "STRATA_PYTHON": str(root / ".venv/bin/python")}
        result = subprocess.run(["sh", str(root / "bin/strata"), "serve"], cwd=root,
                                env=env, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 5)
        self.assertIn("3.11", result.stderr)
        self.assertNotIn("missing packages", result.stderr)
        self.assertNotIn("the CLI must not run", result.stderr)


if __name__ == "__main__":
    unittest.main()
