"""CPU-only regressions for Kolibri's CUDA/HIP build boundaries."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_hip_grid_bench_is_absent_when_hip_is_disabled(tmp_path):
    """A HIP-only diagnostic must not make a CUDA configure require HIP."""
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("cmake not installed")
    text = (ROOT / "CMakeLists.txt").read_text()
    start = text.index("  # T2 (plan 2026-10-28-kolibri-r9700-speed.md)")
    end = text.index("  # #783 PR-c:", start)
    source = tmp_path / "tools/kolibri/grid_shape_bench.cu"
    source.parent.mkdir(parents=True)
    source.write_text("// The HIP source must not be requested in this configuration.\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(kolibri_backend_boundary LANGUAGES NONE)\n"
        "set(STRATA_ENABLE_HIP OFF)\n" + text[start:end] +
        "\nif(TARGET grid_shape_bench)\n"
        '  message(FATAL_ERROR "HIP-only benchmark leaked into a non-HIP build")\n'
        "endif()\n"
    )
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build")],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_decode_control_publication_compiles_and_runs_as_cxx17(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler not installed")
    executable = tmp_path / "decode_control_host_test"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-pthread", "-I", str(ROOT / "include"),
         str(ROOT / "tests/kolibri/decode_control_host_test.cpp"), "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=30)
