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
