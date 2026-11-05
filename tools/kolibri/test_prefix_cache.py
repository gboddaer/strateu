"""CPU ownership/budget regression for the exact ring checkpoint store."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which('g++') is None, reason='C++17 compiler required')
def test_prefix_cache_owns_bytes_and_prunes_forks(tmp_path):
    binary = tmp_path / 'prefix-cache-test'
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'src'), str(ROOT / 'tests/kolibri/prefix_cache_test.cpp'),
                    '-o', str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, timeout=10)
