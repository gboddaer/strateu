"""Cross-target locks must resolve direct pins for the target, not the build host."""
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packaging"))
import lock_wheels  # noqa: E402


class TargetMarkers(unittest.TestCase):
    def test_download_uses_the_target_python_and_os_requirements(self):
        for target, expected in (
            ("linux-x86_64", {"numpy": "2.5.3"}),
            ("darwin-arm64", {"numpy": "2.5.3"}),
            ("windows-x86_64", {"colorama": "0.4.6", "numpy": "2.5.3"}),
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "requirements.txt").write_text(
                    'numpy==2.5.3; python_version >= "3.12"\n'
                    'numpy==2.4.6; python_version == "3.11"\n'
                    'colorama==0.4.6; sys_platform == "win32"\n'
                )
                (root / "requirements-monitor.txt").write_text("# No other direct pins in this fixture.\n")

                def download(command, *, check):
                    self.assertTrue(check)
                    direct = []
                    for i, argument in enumerate(command):
                        if argument == "-r":
                            direct.extend(line for line in Path(command[i + 1]).read_text().splitlines()
                                          if line.strip() and not line.startswith("#"))
                    self.assertEqual(sorted(direct), [f"{name}=={version}" for name, version in sorted(expected.items())])
                    destination = Path(command[command.index("-d") + 1])
                    for name, version in expected.items():
                        (destination / f"{name}-{version}-py3-none-any.whl").write_bytes(b"wheel fixture")

                with mock.patch.object(lock_wheels, "ROOT", root), \
                        mock.patch.object(lock_wheels.subprocess, "run", side_effect=download):
                    text = lock_wheels.build_lock(target)
                actual = dict(re.findall(r"^([a-z0-9-]+)==([^\s]+)", text, re.MULTILINE))
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
