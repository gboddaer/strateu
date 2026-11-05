"""Always-run regression for the former retention RED tests (no model/GPU)."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which('g++') is None, reason='C++17 compiler required')
def test_shorter_prefix_and_mid_history_edit_keep_anchors(tmp_path):
    binary = tmp_path / 'prefix-cache-gaps'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'src'), str(ROOT / 'tests/kolibri/prefix_cache_gaps_test.cpp'),
                    '-o', str(binary)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr


def test_engine_gate_requires_explicit_build_before_loading_model(monkeypatch, tmp_path):
    import known_gaps

    def forbidden_start(*args):
        pytest.fail('invalid diagnostic invocation tried to load a model')

    monkeypatch.setattr(known_gaps, 'run_engine', forbidden_start)
    monkeypatch.setattr('sys.argv', ['known_gaps.py', '--engine', '--output', str(tmp_path / 'out')])
    with pytest.raises(SystemExit) as error:
        known_gaps.main()
    assert error.value.code == 2
    assert not (tmp_path / 'out').exists()
