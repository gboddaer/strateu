"""Engine guard tests: invalid arm combinations must fail before model loading or GPU allocation.

Run after building strata-kolibri. No model or GPU is used by these negative cases.
"""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get('KOLIBRI_BIN', ROOT / 'build-mmq/strata-kolibri'))


@pytest.mark.skipif(not BIN.exists(), reason='build strata-kolibri first')
@pytest.mark.parametrize('arm', ['STRATA_KOLIBRI_DEC_HANDOFF', 'STRATA_KOLIBRI_DEC_STREAM'])
@pytest.mark.parametrize('args,reason', [(['--cpu-threads', '0'], '--cpu-threads'),
                                       (['--cpu-threads', '8', '--grouped'], '--grouped'),
                                       (['--cpu-threads', '8', '--expert-slots', '1'], '--expert-slots')])
def test_unsupported_arm_fails_before_model_read(arm, args, reason):
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_DEC_')}
    env.update(HIP_VISIBLE_DEVICES='0', **{arm: '1'})
    result = subprocess.run([str(BIN), '--model', '/no-such-kolibri-test-model.gguf', *args],
                            env=env, text=True, capture_output=True, timeout=10)
    assert result.returncode == 2
    assert arm in result.stderr and reason in result.stderr
    assert 'cannot open' not in result.stderr
