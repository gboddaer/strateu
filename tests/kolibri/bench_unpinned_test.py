"""CPU-only argument/data-safety checks for the offline measurement tool."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

TOOL = Path(__file__).resolve().parents[2] / 'tools/kolibri/bench_unpinned.py'


class BenchmarkArguments(unittest.TestCase):
    def run_tool(self, directory, *extra):
        return subprocess.run([sys.executable, str(TOOL), '--bin', '/does-not-exist',
                               '--model', 'unused', '--out', directory, '--label', 'case', *extra],
                              capture_output=True, text=True)

    def test_rejects_empty_history_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_tool(d, '--n', '0')
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('positive', r.stderr)

    def test_never_overwrites_its_input_profile(self):
        with tempfile.TemporaryDirectory() as d:
            profile = Path(d) / 'case.profile'
            profile.write_text('original input\n')
            r = self.run_tool(d, '--profile-input', str(profile))
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertEqual(profile.read_text(), 'original input\n')


if __name__ == '__main__':
    unittest.main()
