"""CPU-only argument/data-safety checks for the offline measurement tool."""
from pathlib import Path
import importlib.util
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest

TOOL = Path(__file__).resolve().parents[2] / 'tools/kolibri/bench_unpinned.py'
SPEC = importlib.util.spec_from_file_location('bench_unpinned', TOOL)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkArguments(unittest.TestCase):
    def run_tool(self, directory, *extra, timeout=None):
        return subprocess.run([sys.executable, str(TOOL), '--bin', '/does-not-exist',
                               '--model', 'unused', '--out', directory, '--label', 'case', *extra],
                              capture_output=True, text=True, timeout=timeout)

    def fake_bin(self, directory, exit_code=7):
        """Launcher stub: records argv and exits without producing logits."""
        marker = Path(directory) / 'launched.txt'
        binary = Path(directory) / 'fake-engine'
        binary.write_text('#!/bin/sh\nprintf "%%s\\n" "$@" > "%s"\nexit %d\n'
                          % (marker, exit_code))
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary, marker

    def test_rejects_empty_history_before_launch(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_tool(d, '--n', '0')
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('positive', r.stderr)

    def test_rejects_nonpositive_budget_limits(self):
        with tempfile.TemporaryDirectory() as d:
            for extra in (['--timeout', '0'], ['--timeout', '-5'], ['--ctx', '0'], ['--ctx', '-1']):
                with self.subTest(extra=extra):
                    r = self.run_tool(d, *extra)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn('positive', r.stderr)

    def test_rejects_nonfinite_timeout_and_unrepresentable_context(self):
        with tempfile.TemporaryDirectory() as d:
            for extra in (['--timeout', 'nan'], ['--timeout', 'inf'], ['--ctx', '2147483648']):
                with self.subTest(extra=extra):
                    r = self.run_tool(d, *extra)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn('--', r.stderr)

    def test_rejects_decode_that_exceeds_context(self):
        with tempfile.TemporaryDirectory() as d:
            r = self.run_tool(d, '--prompt', '131065', '--n', '8', '--ctx', '131072')
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('context', r.stderr.lower())
            self.assertFalse((Path(d) / 'case.ids').exists())

    def test_accepts_exact_context_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            binary, marker = self.fake_bin(d)
            r = self.run_tool(d, '--bin', str(binary), '--prompt', '131064', '--n', '8',
                              '--ctx', '131072')
            self.assertEqual(r.returncode, 1, r.stderr)  # launch happened, logits are absent
            self.assertTrue(marker.exists())
            argv = marker.read_text().splitlines()
            self.assertEqual(argv[argv.index('--ctx') + 1], '131072')

    def test_rejects_invalid_token_fixtures(self):
        cases = {
            'empty': '',
            'non-integer': '1, 2, nope, 3',
            'negative': '1,2,-3',
            'out-of-range': '1,2,128000',
        }
        with tempfile.TemporaryDirectory() as d:
            for name, text in cases.items():
                path = Path(d) / f'{name}.txt'
                path.write_text(text)
                with self.subTest(name=name):
                    r = self.run_tool(d, '--ids-file', str(path), '--n', '8')
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertEqual(path.read_text(), text)

    def test_ids_file_defines_exact_input_and_is_copied(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'external.txt'
            source.write_text('5 6 7\n8,9')
            binary, marker = self.fake_bin(d)
            r = self.run_tool(d, '--bin', str(binary), '--ids-file', str(source), '--n', '2',
                              '--ctx', '16')
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertTrue(marker.exists())
            self.assertEqual(source.read_text(), '5 6 7\n8,9')
            self.assertEqual((Path(d) / 'case.ids').read_text(), '5,6,7,8,9')
            self.assertEqual((Path(d) / 'case.ids-source').read_text(), '5 6 7\n8,9')

    def test_rejects_combining_explicit_length_with_ids_file(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'external.txt'
            source.write_text('1,2,3')
            r = self.run_tool(d, '--ids-file', str(source), '--prompt', '3')
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('--prompt', r.stderr)

    def test_never_overwrites_its_input_profile(self):
        with tempfile.TemporaryDirectory() as d:
            profile = Path(d) / 'case.profile'
            profile.write_text('original input\n')
            binary, marker = self.fake_bin(d)
            r = self.run_tool(d, '--profile-input', str(profile), '--bin', str(binary))
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('must differ', r.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(profile.read_text(), 'original input\n')

    def test_never_overwrites_the_ids_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'case.ids'
            target.write_text('1,2,3')
            binary, marker = self.fake_bin(d)
            r = self.run_tool(d, '--ids-file', str(target), '--n', '1', '--bin', str(binary))
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('must differ', r.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(target.read_text(), '1,2,3')

    def test_inputs_cannot_alias_any_output_artifact(self):
        cases = [('--ids-file', 'case.profile', '1,2,3'), ('--ids-file', 'case.log', '1,2,3'),
                 ('--profile-input', 'case.ids', 'keep me\n'),
                 ('--profile-input', 'case.ids-source', 'keep me\n'),
                 ('--compare', 'case.bin', 'keep me\n'), ('--model', 'case.bin', 'keep me\n')]
        with tempfile.TemporaryDirectory() as d:
            for flag, name, content in cases:
                target = Path(d) / name
                target.write_text(content)
                binary, marker = self.fake_bin(d)
                with self.subTest(flag=flag, name=name):
                    r = self.run_tool(d, flag, str(target), '--n', '1', '--bin', str(binary))
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn('must differ', r.stderr)
                    self.assertFalse(marker.exists())
                    self.assertEqual(target.read_text(), content)
            target = Path(d) / 'case.bin'
            target.write_text('#!/bin/sh\nexit 0\n')
            target.chmod(target.stat().st_mode | stat.S_IXUSR)
            r = self.run_tool(d, '--bin', str(target), '--n', '1')
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('--bin must differ', r.stderr)
            self.assertEqual(target.read_text(), '#!/bin/sh\nexit 0\n')

    def test_hardlinked_input_cannot_alias_an_output(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'case.ids'
            target.write_text('1,2,3')
            source = Path(d) / 'hardlink.txt'
            source.hardlink_to(target)
            binary, marker = self.fake_bin(d)
            r = self.run_tool(d, '--ids-file', str(source), '--n', '1', '--bin', str(binary))
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('must differ', r.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(target.read_text(), '1,2,3')

    def test_symlinked_input_cannot_alias_an_output(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'case.ids'
            target.write_text('1,2,3')
            source = Path(d) / 'symlink.txt'
            source.symlink_to(target)
            binary, marker = self.fake_bin(d)
            r = self.run_tool(d, '--ids-file', str(source), '--n', '1', '--bin', str(binary))
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn('must differ', r.stderr)
            self.assertFalse(marker.exists())
            self.assertEqual(target.read_text(), '1,2,3')

    def test_preflight_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            for extra in (['--profile-input', str(Path(d) / 'missing.profile')],
                          ['--env', 'not-an-assignment']):
                with self.subTest(extra=extra):
                    r = self.run_tool(d, *extra)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertFalse((Path(d) / 'case.ids').exists())
            missing_out = Path(d) / 'new-output'
            r = subprocess.run([sys.executable, str(TOOL), '--bin', '/does-not-exist',
                                '--model', 'unused', '--out', str(missing_out), '--label', 'case',
                                '--profile-input', str(Path(d) / 'missing.profile')],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertFalse(missing_out.exists())

    def test_evidence_environment_uses_an_allowlist(self):
        env = {'STRATA_KOLIBRI_BENCH': '1', 'STRATA_KOLIBRI_UNUSED_SECRET': 'sentinel',
               'PATH': '/usr/bin'}
        self.assertEqual(MODULE.evidence_env(env), {'STRATA_KOLIBRI_BENCH': '1'})
        self.assertEqual(MODULE.recorded_env([('STRATA_KOLIBRI_BENCH', '1'), ('SECRET', 'sentinel')]),
                         ['STRATA_KOLIBRI_BENCH=1', 'SECRET=<redacted>'])

    def test_watchdog_stops_a_hung_engine(self):
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / 'fake-engine'
            binary.write_text('#!/bin/sh\nsleep 30\n')
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            r = self.run_tool(d, '--bin', str(binary), '--prompt', '4', '--n', '1',
                              '--timeout', '1', timeout=15)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertIn('timed out', r.stderr)
            self.assertIn('1', r.stderr)

    def test_watchdog_kills_descendants(self):
        with tempfile.TemporaryDirectory() as d:
            pidfile = Path(d) / 'child.pid'
            binary = Path(d) / 'fake-wrapper'
            binary.write_text('#!/bin/sh\nsleep 30 &\necho $! > "%s"\nwait\n' % pidfile)
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            r = self.run_tool(d, '--bin', str(binary), '--prompt', '4', '--n', '1',
                              '--timeout', '2', timeout=20)
            self.assertEqual(r.returncode, 1, r.stderr)
            self.assertTrue(pidfile.exists())
            pid = int(pidfile.read_text().strip())
            for _ in range(50):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail(f'descendant {pid} survived the watchdog')


if __name__ == '__main__':
    unittest.main()
