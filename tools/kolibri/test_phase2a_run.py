"""Guard the experimental matrix against silently testing the wrong arm."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase2a_run as run


def test_environment_removes_unrequested_numerical_arms(monkeypatch):
    monkeypatch.setenv('STRATA_KOLIBRI_DENSE_F16','1')
    monkeypatch.setenv('STRATA_KOLIBRI_KV_INT8','1')
    monkeypatch.setenv('STRATA_KOLIBRI_PF_PARTIAL','1')
    monkeypatch.setenv('KOLIBRI_UNREQUESTED','1')
    for arm,want in [('f16','0'),('int8','1')]:
        env=run.environment(arm)
        assert env['STRATA_KOLIBRI_KV_INT8']==want
        assert env['STRATA_KOLIBRI_PF_PARTIAL']=='0'
        assert 'STRATA_KOLIBRI_DENSE_F16' not in env
        assert 'KOLIBRI_UNREQUESTED' not in env


def test_constant_logit_shift_cannot_pass_on_zero_kl(tmp_path):
    a=np.zeros((9,128000),dtype=np.float32)
    first=tmp_path/'a.bin'; second=tmp_path/'b.bin'
    first.write_bytes(a.tobytes());second.write_bytes((a+np.float32(.125)).tobytes())
    result=run.comparison(first,second)
    assert abs(result['metrics']['kl_mean'])<1e-12
    assert result['metrics']['dlogit_worst']==.125
    assert not result['pass_limits']


def test_nonfinite_candidate_is_not_a_tolerance_failure(tmp_path):
    a=np.zeros((9,128000),dtype=np.float32)
    first=tmp_path/'a.bin'; second=tmp_path/'b.bin'
    first.write_bytes(a.tobytes());a[0,0]=np.nan;second.write_bytes(a.tobytes())
    with pytest.raises(run.qualify.NonFiniteInput): run.comparison(first,second)


def test_corpus_and_limits_are_fixed():
    assert [x[0] for x in run.CORPUS]==['16','512','de512','code512','8191','11000']
    assert run.LIMITS==dict(kl_mean=.001,kl_p99=.005,dlogit_worst=.05,
                           top1_agreement=1.,margin_floor=.05)
