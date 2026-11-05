#!/usr/bin/env python3
"""Measure cold/cached HTTP prefill and compare full cache-on/off logits.

CUDA Kolibri-only diagnostic. Starts/stops its own loopback server, never changes
the input config. Uses synthetic transcripts; does not execute generated tools.
Do not run concurrently with another GPU workload when measuring speed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from degeneration_gate import read_records

ROOT = Path(__file__).resolve().parents[2]


def run(args, original, mode):
    # Refuse an occupied port before starting an engine or sending any request.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    folder = args.output / ('http-' + mode)
    folder.mkdir(exist_ok=False)
    dump = folder / 'logits.bin'
    # Copy engine facts only: no credentials, remote tools or unrelated service options.
    cfg = {k: original[k] for k in ('exe', 'args', 'cwd', 'tokenizer', 'arch', 'model_name',
                                   'engine_ready_s', 'lib_dirs', 'cuda', 'backend', 'gpu', 'launch_prefix')
           if k in original}
    cfg.update(host='127.0.0.1', port=args.port, arch='kolibri1')
    cfg['env'] = {'STRATA_KOLIBRI_EXPERT_STREAM': '1', 'STRATA_KOLIBRI_STREAM_DEV_ACT': '1',
                  'STRATA_KOLIBRI_STREAM_GU': '1', 'STRATA_KOLIBRI_EXPERT_F16': '0',
                  'STRATA_KOLIBRI_POOL_ALL': '1', 'STRATA_KOLIBRI_REUSE': '1',
                  'STRATA_KOLIBRI_REUSE_PREFIX': '1', 'STRATA_KOLIBRI_REUSE_ALIGN': '1',
                  'STRATA_KOLIBRI_PF_CHUNK': '512',
                  'STRATA_KOLIBRI_PREFIX_CACHE_MB': '128' if mode == 'on' else '0',
                  'STRATA_KOLIBRI_LOGITS_DUMP': str(dump)}
    (folder / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    url = f'http://127.0.0.1:{args.port}'
    ready = folder / 'ready.json'
    cmd = [sys.executable, str(ROOT / 'serve/server.py'), '--engine', 'strata', '--config',
           str(folder / 'config.json'), '--host', '127.0.0.1', '--port', str(args.port), '--no-api-key',
           '--ready-file', str(ready)]
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_')}
    env.update(TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
               STRATEU_DUMP_PROMPT=str(folder / 'prompt-ids.jsonl'))
    env.pop('STRATA_API_KEY', None)
    with (folder / 'server.log').open('w') as log:
        server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, stdin=subprocess.DEVNULL)
    results = []

    def call(tag, messages, **kwargs):
        request = {'model': cfg.get('model_name', 'kolibri-1'), 'messages': messages,
                   'max_tokens': 24, 'temperature': 0, 'chat_template_kwargs': {
                       'enable_thinking': False, 'reasoning_effort': 'none'}, 'stream': False, **kwargs}
        start = time.monotonic()
        req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(request).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as response:
            answer = json.load(response)
        wall = time.monotonic() - start
        with urllib.request.urlopen(url + '/v1/status', timeout=10) as response:
            status = json.load(response)
        records = read_records(dump)
        data = {'tag': tag, 'request': request, 'answer': answer, 'wall_s': wall,
                'timings': status.get('last_timings'), 'logit_record_index': len(records) - 1}
        if not records or len(records[-1][2]) != 128000 * 4:
            raise ValueError('missing complete Kolibri logits')
        results.append(data)
        (folder / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(mode, tag, data['timings'], flush=True)
        return answer['choices'][0]['message'], records[-1][2]

    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError('diagnostic server exited')
            # This unique marker is written only after our server binds. Never
            # mistake another listener that won the startup race for our model.
            if not ready.exists():
                time.sleep(2)
                continue
            try:
                with urllib.request.urlopen(url + '/v1/status', timeout=2) as response:
                    status = json.load(response)
                if status.get('loaded'):
                    break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('readiness timeout')
        for repeat in range(args.repeats):
            text = 'Coding-session reference ' + str(repeat) + ':\n' + '\n'.join(
                f'fact {i:04d}: object {chr(65+i%26)}{i} index {i*17%1009:04d} group {i%37:02d}.'
                for i in range(120))
            first = [{'role': 'system', 'content': text},
                     {'role': 'user', 'content': 'What is the index of fact 0077?'}]
            message, _ = call('cold-' + str(repeat), first)
            second = first + [message, {'role': 'user', 'content': 'And its group?'}]
            _, warm = call('cached-turn-' + str(repeat), second)
            if mode == 'on' and results[-1]['timings']['cache_n'] < 1024:
                raise AssertionError('growing conversation did not reuse a long prefix')
            call('reset-' + str(repeat), [{'role': 'system', 'content': 'Unrelated reset session.'},
                                        {'role': 'user', 'content': 'Say reset.'}])
            _, cold = call('cold-turn-control-' + str(repeat), second)
            assert cold == warm, 'HTTP cached logits differ from identical cold request'
        tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a source file.',
                  'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                                 'required': ['path']}}}]
        first = [{'role': 'system', 'content': text},
                 {'role': 'user', 'content': 'Read src/demo.cpp using read_file.'}]
        call('tool-primer', first, tools=tools)
        second = first + [{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_bench', 'type': 'function', 'function': {
                'name': 'read_file', 'arguments': '{"path":"src/demo.cpp"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_bench', 'content': 'int value = 7;'},
            {'role': 'user', 'content': 'What is value?'}]
        _, warm = call('tool-result-cached', second, tools=tools)
        if mode == 'on' and results[-1]['timings']['cache_n'] < 1024:
            raise AssertionError('tool-result transcript did not reuse a long prefix')
        call('tool-reset', [{'role': 'system', 'content': 'Reset separate owner.'},
                            {'role': 'user', 'content': 'Say reset.'}])
        _, cold = call('tool-result-cold', second, tools=tools)
        assert warm == cold, 'tool transcript cached logits differ from cold'
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    return results, read_records(dump)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8212)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1,65535]')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.config.read_text())
    off, a = run(args, original, 'off')
    on, b = run(args, original, 'on')
    pairs = []
    for x, y in zip(off, on):
        assert x['tag'] == y['tag'] and x['request'] == y['request'], 'HTTP controls changed requests'
        equal = a[x['logit_record_index']][2] == b[y['logit_record_index']][2]
        pairs.append({'tag': x['tag'], 'finite_bitwise_equal': equal})
    (args.output / 'cross-mode-results.json').write_text(json.dumps(pairs, indent=2) + '\n')
    ids_off = [json.loads(line)['ids'] for line in (args.output / 'http-off/prompt-ids.jsonl').read_text().splitlines()]
    ids_on = [json.loads(line)['ids'] for line in (args.output / 'http-on/prompt-ids.jsonl').read_text().splitlines()]
    assert ids_off and ids_off == ids_on, 'rendered HTTP prompt IDs differ between arms'
    (args.output / 'prompt-id-parity.json').write_text(json.dumps(
        {'rendered_prompts': len(ids_off), 'exact_id_parity': True}, indent=2) + '\n')
    assert len(off) == len(on) and all(x['finite_bitwise_equal'] for x in pairs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
