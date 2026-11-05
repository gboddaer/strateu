"""CPU-only safety checks: diagnostics must not claim another server's port."""
from argparse import Namespace
from pathlib import Path
import json
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefix_cache_http_gate as gate


@pytest.mark.parametrize('flag,value', [('--repeats', '0'), ('--port', '0'), ('--port', '65536')])
def test_invalid_options_do_not_start_a_model(tmp_path, monkeypatch, flag, value):
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['probe', '--config', 'unused', '--output', str(out), flag, value])
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code == 2
    assert not out.exists()


def test_occupied_port_refuses_before_engine_start(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('must not launch an engine on an occupied diagnostic port')
    monkeypatch.setattr(gate.subprocess, 'Popen', forbidden)
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        args = Namespace(output=tmp_path, port=listener.getsockname()[1])
        with pytest.raises(OSError):
            gate.run(args, {}, 'on')
    assert not (tmp_path / 'http-on').exists()


def test_exported_config_omits_service_secrets_and_remote_tools(tmp_path, monkeypatch):
    class Stopped(Exception):
        pass
    def stop(*args, **kwargs):
        raise Stopped()
    monkeypatch.setattr(gate.subprocess, 'Popen', stop)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    original = {'exe': 'engine', 'args': [], 'api_key': 'DO_NOT_EXPORT',
                'env': {'EXTERNAL_SECRET': 'DO_NOT_EXPORT'}, 'mcp_servers': {'remote': 'DO_NOT_EXPORT'}}
    with pytest.raises(Stopped):
        gate.run(Namespace(output=tmp_path, port=port), original, 'on')
    text = (tmp_path / 'http-on/config.json').read_text()
    assert 'DO_NOT_EXPORT' not in text
    cfg = json.loads(text)
    assert cfg['host'] == '127.0.0.1'
    assert cfg['env']['STRATA_KOLIBRI_EXPERT_F16'] == '0'
