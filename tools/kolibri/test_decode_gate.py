"""CPU-only regression tests for the persistent decode-gate child and headered dumps."""
import os
from pathlib import Path
import signal
import struct
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_decode as cd
import decode_gate_serve as dg


def test_serve_dump_shape_is_independent_of_file_length(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    raw = struct.pack('<iii2f', 0, 512, 2, 1., 2.) + struct.pack('<iii2f', 512, 64, 2, 3., 4.)
    a.write_bytes(raw)
    b.write_bytes(raw)
    assert cd.compare_serve(a, b, vocab=2, records=2)['records'] == 2
    for invalid in [b'', raw[:-4], raw + raw]:
        a.write_bytes(invalid)
        b.write_bytes(invalid)
        with pytest.raises(cd.Mismatch):
            cd.compare_serve(a, b, vocab=2, records=2)


def test_serve_dump_rejects_nonfinite_even_when_bytes_match(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    raw = struct.pack('<iii2f', 0, 512, 2, float('nan'), 2.)
    a.write_bytes(raw)
    b.write_bytes(raw)
    with pytest.raises(cd.Mismatch, match='finite'):
        cd.compare_serve(a, b, vocab=2, records=1)


def test_serve_dump_requires_declared_vocab(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    raw = struct.pack('<iii2f', 0, 512, 3, 1., 2.)
    a.write_bytes(raw)
    b.write_bytes(raw)
    with pytest.raises(cd.Mismatch, match='vocab'):
        cd.compare_serve(a, b, vocab=2, records=1)


def fake_engine(tmp_path, *, reuse=512, exit_code=0, hang=False, require_live=False):
    p = tmp_path / 'engine'
    pid_file = tmp_path / 'engine.pid'
    p.write_text(f'''#!{sys.executable}
import os, sys, time
open({str(pid_file)!r}, 'w').write(str(os.getpid()))
print('READY 1024 stop', flush=True)
for i in range(2):
    line = sys.stdin.readline()
    if {hang!r}: time.sleep(60)
    toks = line.split()[-1].split(',')
    if i == 1 and {require_live!r} and toks[:516] != [str(j) for j in range(512)] + ['10'] * 4:
        print('ERR next request omitted generated prefix', flush=True)
        sys.exit(2)
    for j in range(4): print('T 10', flush=True)
    reuse = 0 if i == 0 else {reuse}
    print('DONE 4 %d 10.0 20.0 length 0 0 %d 0 0 0 0 0.0 %d 0' % (len(toks), reuse, len(toks)-reuse), flush=True)
sys.stdin.read()
sys.exit({exit_code})
''')
    p.chmod(0o755)
    ids = tmp_path / 'ids'
    ids.write_text(','.join(str(i) for i in range(512)))
    return p, ids, pid_file


def test_serve_driver_checks_real_reuse(tmp_path):
    engine, ids, _ = fake_engine(tmp_path, reuse=0)
    assert dg.main(['gate', str(engine), 'unused', str(ids)]) != 0


def test_serve_driver_propagates_failure_after_done(tmp_path):
    engine, ids, _ = fake_engine(tmp_path, exit_code=7)
    assert dg.main(['gate', str(engine), 'unused', str(ids)]) != 0


def test_serve_driver_deadline_reaps_engine(tmp_path, monkeypatch):
    engine, ids, pid_file = fake_engine(tmp_path, hang=True)
    # The old driver uses blocking readline; do not launch its known 60-second hang in the unit suite.
    assert hasattr(dg, 'drive'), 'serve driver needs a deadline-aware drive() implementation'
    monkeypatch.setattr(dg, 'TIMEOUT_S', 0.2)
    start = time.monotonic()
    assert dg.main(['gate', str(engine), 'unused', str(ids)]) != 0
    assert time.monotonic() - start < 5
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_serve_driver_accepts_two_complete_requests(tmp_path):
    engine, ids, _ = fake_engine(tmp_path)
    assert dg.main(['gate', str(engine), 'unused', str(ids)]) == 0


def test_serve_driver_extends_generated_history_not_only_prompt(tmp_path):
    engine, ids, _ = fake_engine(tmp_path, require_live=True)
    assert dg.main(['gate', str(engine), 'unused', str(ids)]) == 0
