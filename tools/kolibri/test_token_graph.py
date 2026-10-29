"""Opt-in real-model integration tests. KOLIBRI_GPU_TEST=1 HIP_VISIBLE_DEVICES=0 python -m pytest ..."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_decode

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get('KOLIBRI_GPU_TEST') != '1', reason='explicit real-model GPU test')

# The decode arms are ON by default, so a control run must switch them off explicitly; an unset environment is the
# candidate, not the baseline.
PLAIN = {f'STRATA_KOLIBRI_DEC_{name}': '0' for name in ('HANDOFF', 'STREAM', 'HANDOFF_GRAPH', 'TOKEN_GRAPH')}


@pytest.fixture(autouse=True)
def deterministic_prefill(monkeypatch):
    # Paired decode gates require identical incoming state. The default tiled prefill can use atomic GEMMs.
    # This is an existing test configuration applied to BOTH arms, never a production default change.
    monkeypatch.setenv('STRATA_GEMM_DETERMINISTIC', '1')


@pytest.mark.parametrize('rollover', [False, True])
def test_whole_stack_actually_replays_and_matches_full_logits(tmp_path, rollover):
    ids = (ROOT / 'tools/kolibri/oracle_ids512.txt').read_text().strip().split(',')[:16]
    prompt = tmp_path / 'ids'
    prompt.write_text(','.join(ids))
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_DEC_')}
    env.update(HIP_VISIBLE_DEVICES='0', STRATA_HIPBLASLT_TUNING=str(ROOT / 'tools/hip/gfx1201-hipblaslt-100202.txt'))
    cmd = [str(ROOT / 'build-mmq/strata-kolibri'), '--model',
           os.environ.get('KOLIBRI_MODEL', '/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf'),
           '--tokens-file', str(prompt), '--force', '286,3515,310,262,14292,1343,1444,46',
           '--ctx', '2048', '--prefill', '--cpu-threads', '8']
    for arm in ('base', 'token'):
        flags = dict(PLAIN) if arm == 'base' else {name: '1' for name in (
            'STRATA_KOLIBRI_DEC_HANDOFF', 'STRATA_KOLIBRI_DEC_STREAM',
            'STRATA_KOLIBRI_DEC_HANDOFF_GRAPH', 'STRATA_KOLIBRI_DEC_TOKEN_GRAPH')}
        run_cmd = list(cmd)
        if rollover and arm == 'token':
            run_cmd[0] = str(ROOT / 'build-mmq/kolibri_decode_rollover')
        with (tmp_path / f'{arm}.out').open('w') as out, (tmp_path / f'{arm}.err').open('w') as err:
            child = subprocess.Popen(run_cmd + ['--dump-full-logits', str(tmp_path / f'{arm}.bin')],
                                     cwd=ROOT, env=env | flags, stdout=out, stderr=err, start_new_session=True)
            try:
                rc = child.wait(timeout=120)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            assert rc == 0, (tmp_path / f'{arm}.err').read_text()
        marker = 'DEC_TOKEN_GRAPH replay active'
        assert (marker in (tmp_path / f'{arm}.err').read_text()) == (arm == 'token')
        if rollover and arm == 'token':
            assert 'PASS cold sequence rollover and generation rollover' in (tmp_path / f'{arm}.err').read_text()
    check_decode.compare(tmp_path / 'base.bin', tmp_path / 'token.bin', vocab=128000, records=9)


@pytest.mark.parametrize('failure', ['mapped', 'parent', 'producer'])
def test_whole_stack_failure_paths_never_emit_an_invalid_record(tmp_path, failure):
    prompt = tmp_path / 'ids'
    prompt.write_text(','.join((ROOT / 'tools/kolibri/oracle_ids512.txt').read_text().strip().split(',')[:16]))
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_DEC_')}
    env.update(HIP_VISIBLE_DEVICES='0', STRATA_HIPBLASLT_TUNING=str(ROOT / 'tools/hip/gfx1201-hipblaslt-100202.txt'))
    common = ['--model', os.environ.get('KOLIBRI_MODEL', '/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf'),
              '--tokens-file', str(prompt), '--force', '286,3515,310,262,14292,1343,1444,46',
              '--ctx', '2048', '--prefill', '--cpu-threads', '8']
    baseline = tmp_path / 'baseline.bin'
    if failure != 'mapped':
        result = subprocess.run([str(ROOT / 'build-mmq/strata-kolibri'), *common,
                                 '--dump-full-logits', str(baseline)], cwd=ROOT, env=env | PLAIN,
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
    for flag in ('HANDOFF', 'STREAM', 'HANDOFF_GRAPH', 'TOKEN_GRAPH'):
        env['STRATA_KOLIBRI_DEC_' + flag] = '1'
    env['KOLIBRI_TEST_FAILURE'] = failure
    dump = tmp_path / 'failure.bin'
    child = subprocess.Popen([str(ROOT / 'build-mmq/kolibri_decode_rollover'), *common,
                              '--dump-full-logits', str(dump)], cwd=ROOT, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    (tmp_path / 'failure.err').write_text(stderr)
    assert child.returncode == 1, stderr
    if failure == 'mapped':
        assert 'TEST mapped allocation failure' in stderr and 'cannot allocate' in stderr
        assert not dump.exists() and stdout == ''
    else:
        marker = 'TEST parent executable allocation failure' if failure == 'parent' else \
                 'PASS CPU producer failure graph already quiescent'
        assert marker in stderr
        raw = dump.read_bytes()
        assert 0 < len(raw) < 9 * 128000 * 4 and len(raw) % (128000 * 4) == 0
        expected = tmp_path / 'completed-prefix.bin'
        expected.write_bytes(baseline.read_bytes()[:len(raw)])
        check_decode.compare(expected, dump, vocab=128000, records=len(raw) // (128000 * 4))


def test_whole_stack_timeout_emits_no_invalid_logit_record(tmp_path):
    prompt = tmp_path / 'ids'
    prompt.write_text(','.join((ROOT / 'tools/kolibri/oracle_ids512.txt').read_text().strip().split(',')[:16]))
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_DEC_')}
    env.update(HIP_VISIBLE_DEVICES='0', STRATA_HIPBLASLT_TUNING=str(ROOT / 'tools/hip/gfx1201-hipblaslt-100202.txt'))
    for flag in ('HANDOFF', 'STREAM', 'HANDOFF_GRAPH', 'TOKEN_GRAPH'):
        env['STRATA_KOLIBRI_DEC_' + flag] = '1'
    env['STRATA_KOLIBRI_DEC_WAIT_TICKS'] = '0'
    dump = tmp_path / 'timeout.bin'
    child = subprocess.Popen([str(ROOT / 'build-mmq/strata-kolibri'), '--model',
                              os.environ.get('KOLIBRI_MODEL', '/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf'),
                              '--tokens-file', str(prompt), '--force', '286,3515,310,262,14292,1343,1444,46',
                              '--ctx', '2048', '--prefill', '--cpu-threads', '8', '--dump-full-logits', str(dump)],
                             cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(timeout=120)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    (tmp_path / 'timeout.err').write_text(stderr)
    assert child.returncode != 0
    assert 'DEC_TOKEN_GRAPH failed' in stderr and 'no token emitted' in stderr
    assert 0 < dump.stat().st_size < 9 * 128000 * 4
    assert dump.stat().st_size % (128000 * 4) == 0


def test_whole_stack_cancel_drains_and_next_request_is_cold_bitwise(tmp_path):
    import queue
    import threading
    import time
    from decode_bench import parse_done

    ids = (ROOT / 'tools/kolibri/oracle_ids512.txt').read_text().strip().split(',')
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_DEC_')}
    env.update(HIP_VISIBLE_DEVICES='0', STRATA_HIPBLASLT_TUNING=str(ROOT / 'tools/hip/gfx1201-hipblaslt-100202.txt'),
               STRATA_KOLIBRI_LOGITS_DUMP=str(tmp_path / 'requests.bin'))
    for flag in ('HANDOFF', 'STREAM', 'HANDOFF_GRAPH', 'TOKEN_GRAPH', 'TOKEN_TRACE'):
        env['STRATA_KOLIBRI_DEC_' + flag] = '1'
    child = subprocess.Popen([str(ROOT / 'build-mmq/strata-kolibri'), '--model',
                              os.environ.get('KOLIBRI_MODEL', '/fast/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf'),
                              '--serve', '--ctx', '2048', '--prefill', '--cpu-threads', '8'],
                             cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, bufsize=1, start_new_session=True)
    events = queue.Queue()
    logs = {'out': [], 'err': []}

    def reader(pipe, kind):
        for line in pipe:
            logs[kind].append(line)
            events.put((kind, line.strip()))
        events.put((kind, None))

    threads = [threading.Thread(target=reader, args=(child.stdout, 'out'), daemon=True),
               threading.Thread(target=reader, args=(child.stderr, 'err'), daemon=True)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 120

    def event():
        remaining = deadline - time.monotonic()
        assert remaining > 0, 'engine deadline'
        return events.get(timeout=remaining)

    def request(tokens, n, cancel_at_publication=False):
        child.stdin.write(f'GEN {n} ' + ','.join(tokens) + '\n')
        child.stdin.flush()
        generated = []
        sent_stop = False
        while True:
            kind, line = event()
            assert line is not None, 'engine EOF'
            if cancel_at_publication and not sent_stop and kind == 'err' and \
                    line.startswith(f'DEC_TOKEN_PUBLICATION pos={len(tokens)} layer=0 '):
                child.stdin.write('STOP\n')
                child.stdin.flush()
                sent_stop = True
            if kind == 'out' and line.startswith('T '):
                generated.append(line.split()[1])
            if kind == 'out' and line.startswith('DONE '):
                return generated, parse_done(line), sent_stop

    try:
        while True:
            kind, line = event()
            assert line is not None
            if kind == 'out' and line.startswith('READY '):
                break
        warm, first, _ = request(ids, 8)
        assert len(warm) == 8
        held = ids + warm
        assert len(held) == 520
        cancelled, done, sent_stop = request(held, 32, cancel_at_publication=True)
        assert sent_stop and done['finish'] == 'cancel' and len(cancelled) < 32
        again, last, _ = request(ids, 8)
        assert again == warm and last['reused'] == 0
        child.stdin.close()
        assert child.wait(timeout=10) == 0
    finally:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=1)
        for kind, lines in logs.items():
            (tmp_path / f'cancel.{kind}').write_text(''.join(lines))
    assert 'DEC_TOKEN_GRAPH cancelled: graph drained; token discarded' in ''.join(logs['err'])
    raw = (tmp_path / 'requests.bin').read_bytes()
    stride = 12 + 128000 * 4
    assert len(raw) == 3 * stride
    assert raw[:stride] == raw[2 * stride:]
    check_decode.compare_serve(tmp_path / 'requests.bin', tmp_path / 'requests.bin', vocab=128000, records=3)
