"""Guard the experimental matrix against silently testing the wrong arm."""
import sys
import json
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase2a_run as run
import phase2a_speed as speed


def test_environment_removes_unrequested_numerical_arms(monkeypatch):
    monkeypatch.setenv('STRATA_KOLIBRI_DENSE_F16','1')
    monkeypatch.setenv('STRATA_KOLIBRI_KV_INT8','1')
    monkeypatch.setenv('STRATA_KOLIBRI_PF_PARTIAL','1')
    monkeypatch.setenv('KOLIBRI_UNREQUESTED','1')
    for arm,want in [('f16','0'),('int8','1')]:
        env=run.environment(arm)
        assert env['STRATA_KOLIBRI_KV_INT8']==want
        assert env['STRATA_KOLIBRI_PF_PARTIAL']=='0'
        assert env['STRATA_KOLIBRI_ATTN_KT']=='1'
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


def test_recorded_environment_redacts_credentials_without_mutating_child():
    env={'OPENAI_API_KEY':'private','HF_TOKEN':'private','PI_SESSION_TOKEN':'private',
         'PATH':'/usr/bin','STRATA_KOLIBRI_DEC_TOKEN_GRAPH':'0'}
    public=run.public_env(env)
    assert public['OPENAI_API_KEY']=='<redacted>'
    assert public['HF_TOKEN']=='<redacted>'
    assert public['PI_SESSION_TOKEN']=='<redacted>'
    assert public['STRATA_KOLIBRI_DEC_TOKEN_GRAPH']=='0'
    assert public['PATH']=='/usr/bin'
    assert env['OPENAI_API_KEY']=='private'


def test_speed_gate_rejects_failed_accuracy(tmp_path):
    metrics={fam:dict(pass_limits=True,repeat_bitwise=True) for fam,_,_ in run.CORPUS}
    metrics['16']['pass_limits']=False
    (tmp_path/'candidate-metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError,match='rejected accuracy'):
        speed.require_accuracy(tmp_path)


def test_speed_gate_requires_complete_evidence(tmp_path):
    with pytest.raises(RuntimeError): speed.require_accuracy(tmp_path)
    metrics={fam:dict(pass_limits=True,repeat_bitwise=True) for fam,_,_ in run.CORPUS}
    (tmp_path/'candidate-metrics.json').write_text(json.dumps(metrics))
    with pytest.raises(RuntimeError): speed.require_accuracy(tmp_path)


def test_speed_gate_accepts_only_numerical_and_state_success(tmp_path):
    metrics={fam:dict(pass_limits=True,repeat_bitwise=True) for fam,_,_ in run.CORPUS}
    (tmp_path/'candidate-metrics.json').write_text(json.dumps(metrics))
    for arm in ['f16','int8']:
        (tmp_path/f'state-{arm}.json').write_text(json.dumps(dict(result='PASS',prefix_cold_comparison=dict(pass_check=True))))
    speed.require_accuracy(tmp_path)


def test_corpus_and_limits_are_fixed():
    assert [x[0] for x in run.CORPUS]==['16','512','de512','code512','8191','11000']
    assert run.LIMITS==dict(kl_mean=.001,kl_p99=.005,dlogit_worst=.05,
                           top1_agreement=1.,margin_floor=.05)
