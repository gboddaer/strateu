"""Static backend-boundary checks; these do not replace CUDA/HIP compilation."""
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]


def target_guards():
    """Track enclosing CMake conditions at each target declaration."""
    guards = []
    targets = {}
    for line in (ROOT / 'CMakeLists.txt').read_text().splitlines():
        code = line.strip()
        if re.match(r'if\s*\(', code):
            guards.append(code)
        elif re.match(r'endif\s*\(', code):
            guards.pop()
        match = re.match(r'add_executable\(\s*(\S+)', code)
        if match:
            targets[match[1]] = tuple(guards)
    return targets


@pytest.mark.parametrize('target', [
    'kolibri_expert_stream_pipeline_test',
    'kolibri_expert_stream_kernel_test',
    'expert_stream_replay',
    'f16_expert_gemm_parity',
])
def test_cuda_only_targets_have_a_cuda_only_guard(target):
    # A CUDA-or-HIP guard alone would feed CUDA-only TUs to HIP builds.
    assert 'if(STRATA_ENABLE_CUDA)' in target_guards()[target]


def test_host_plan_test_is_available_without_a_gpu_backend():
    guards = target_guards()['kolibri_expert_stream_plan_test']
    assert 'if(STRATA_BUILD_TESTS)' in guards
    assert not any('STRATA_ENABLE_' in guard for guard in guards)
