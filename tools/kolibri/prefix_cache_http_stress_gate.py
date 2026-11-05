#!/usr/bin/env python3
"""HTTP non-happy-path gate for exact Kolibri prefix checkpoints.

Runs one deterministic sequence of OpenAI-shaped calls twice, cache off then on,
and requires every completed call to have byte-identical finite logits across the
two arms. Calls that the design promises to accelerate must report a long reuse in
the cache-on arm. Client aborts are real socket closes mid-prefill and mid-stream;
their effect is judged by the next request, which must stay exact.

Synthetic transcripts only: generated tool calls are never executed. Starts and
stops its own loopback servers. The input config is read, never modified.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from degeneration_gate import read_records

ROOT = Path(__file__).resolve().parents[2]
LONG = ('Reference material for the session:\n'
        + '\n'.join('fact %04d: object %s%d index %04d group %02d.'
                    % (i, chr(65 + i % 26), i, i * 17 % 1009, i % 37) for i in range(120)))
SHORT = ('Repository notes:\n'
         + '\n'.join('module %03d: owner team%d, revision %d, tests %d.'
                     % (i, i % 9, 1000 + i, i % 5) for i in range(52)))

TOOLS = [{'type': 'function', 'function': {
    'name': 'read_file', 'description': 'Read a source file.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                   'required': ['path']}}}]
MORE_TOOLS = TOOLS + [{'type': 'function', 'function': {
    'name': 'write_file', 'description': 'Write a source file.',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'},
                                                    'content': {'type': 'string'}},
                   'required': ['path', 'content']}}}]


def call_raw(url, payload, stream, read_timeout=600):
    """POST once. Returns (response object or None, elapsed seconds)."""
    connection = http.client.HTTPConnection('127.0.0.1', payload['port'], timeout=read_timeout)
    body = json.dumps({k: v for k, v in payload.items() if k != 'port'}).encode()
    start = time.monotonic()
    connection.request('POST', '/v1/chat/completions', body,
                       {'Content-Type': 'application/json'})
    response = connection.getresponse()
    return connection, response, time.monotonic() - start


def run_arm(args, original, mode, port):
    folder = args.output / ('http-' + mode)
    folder.mkdir(exist_ok=False)
    dump = folder / 'logits.bin'
    cfg = {k: original[k] for k in ('exe', 'args', 'cwd', 'tokenizer', 'arch', 'model_name',
                                    'engine_ready_s', 'lib_dirs', 'cuda', 'backend', 'gpu',
                                    'launch_prefix')
           if k in original}
    cfg.update(host='127.0.0.1', port=port, arch='kolibri1')
    cfg['env'] = {'STRATA_KOLIBRI_EXPERT_STREAM': '1', 'STRATA_KOLIBRI_STREAM_DEV_ACT': '1',
                  'STRATA_KOLIBRI_STREAM_GU': '1', 'STRATA_KOLIBRI_EXPERT_F16': '0',
                  'STRATA_KOLIBRI_POOL_ALL': '1', 'STRATA_KOLIBRI_REUSE': '1',
                  'STRATA_KOLIBRI_REUSE_PREFIX': '1', 'STRATA_KOLIBRI_REUSE_ALIGN': '1',
                  'STRATA_KOLIBRI_PF_CHUNK': '512',
                  'STRATA_KOLIBRI_PREFIX_CACHE_MB': '128' if mode == 'on' else '0',
                  'STRATA_KOLIBRI_LOGITS_DUMP': str(dump)}
    (folder / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    ready = folder / 'ready.json'
    cmd = [sys.executable, str(ROOT / 'serve/server.py'), '--engine', 'strata', '--config',
           str(folder / 'config.json'), '--host', '127.0.0.1', '--port', str(port),
           '--no-api-key', '--ready-file', str(ready)]
    env = {k: v for k, v in os.environ.items() if not k.startswith('STRATA_KOLIBRI_')}
    env.update(TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp',
               STRATEU_DUMP_PROMPT=str(folder / 'prompt-ids.jsonl'))
    env.pop('STRATA_API_KEY', None)
    with (folder / 'server.log').open('w') as log:
        server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, stdin=subprocess.DEVNULL)
    url = 'http://127.0.0.1:%d' % port
    rows = []

    def record(tag, request, answer, elapsed, prompt_ids=None):
        # Every completed call contributes its rendered IDs, so the arms are
        # compared on the exact prompt each one ran, not just on the aborts.
        prompt_ids = prompt_ids if prompt_ids is not None else last_ids(tag)
        with urllib.request.urlopen(url + '/v1/status', timeout=10) as response:
            status = json.load(response)
        records = read_records(dump)
        if not records or len(records[-1][2]) != 128000 * 4:
            raise ValueError('missing complete Kolibri logits')
        row = {'tag': tag, 'request': request, 'answer': answer, 'wall_s': elapsed,
               'timings': status.get('last_timings'), 'logit_record_index': len(records) - 1,
               'rendered_ids': prompt_ids, 'aborted': None}
        rows.append(row)
        (folder / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(mode, tag, row['timings'], flush=True)
        return records[-1][2]

    def call(tag, messages, **kwargs):
        request = {'model': cfg.get('model_name', 'kolibri-1'), 'messages': messages,
                   'max_tokens': 24, 'temperature': 0, 'stream': False,
                   'chat_template_kwargs': {'enable_thinking': False,
                                            'reasoning_effort': 'none'}, **kwargs}
        connection, response, elapsed = call_raw(url, dict(request, port=port), False)
        try:
            answer = json.load(response)
        finally:
            connection.close()
        return record(tag, request, answer, elapsed)

    def last_ids(tag=None):
        path = folder / 'prompt-ids.jsonl'
        if not path.exists():
            return None
        entries = [json.loads(line) for line in path.read_text().splitlines()]
        # Both clients may have rendered before either finishes. The last entry
        # is not necessarily this worker's prompt; match its unique synthetic text.
        needle = {'concurrent-a': 'Name fact 0001.',
                  'concurrent-b': 'Name module 003.'}.get(tag)
        if needle:
            matches = [entry for entry in entries if needle in entry['text']]
            if not matches:
                raise AssertionError('missing rendered prompt for ' + tag)
            return matches[-1]['ids']
        return entries[-1]['ids'] if entries else None

    def abort(tag, messages, seconds=None, after_delta=False, **kwargs):
        """Close the socket during the prompt read or right after the first delta."""
        request = {'model': cfg.get('model_name', 'kolibri-1'), 'messages': messages,
                   'max_tokens': 48, 'temperature': 0, 'stream': True,
                   'chat_template_kwargs': {'enable_thinking': False,
                                            'reasoning_effort': 'none'}, **kwargs}
        started = time.monotonic()
        connection, response, elapsed = call_raw(url, dict(request, port=port), True)
        received, first_delta = 0, None
        try:
            deadline = time.monotonic() + (seconds if seconds is not None else 600)
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                ready_read, _, _ = select.select([response.fp], [], [], min(0.2, left))
                if not ready_read:
                    continue
                line = response.readline()
                if not line:
                    break
                received += len(line)
                if after_delta and first_delta is None and line.startswith(b'data: '):
                    # A real generated token, not the role frame: this is the first
                    # point at which the engine is decoding rather than reading.
                    try:
                        frame = json.loads(line[6:])
                        delta = (frame.get('choices') or [{}])[0].get('delta') or {}
                    except (ValueError, IndexError, AttributeError):
                        delta = {}
                    if delta.get('content'):
                        first_delta = time.monotonic() - started
                        break
        finally:
            response.close()
            connection.close()                     # a real client hang-up
        entry = {'tag': tag, 'request': request, 'answer': None, 'wall_s': elapsed,
                 'timings': None, 'logit_record_index': None,
                 'rendered_ids': last_ids(), 'aborted': {'bytes': received,
                                                        'first_delta_s': first_delta}}
        rows.append(entry)
        (folder / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(mode, tag, 'aborted', entry['aborted'], flush=True)
        return entry

    def settle(seconds=6.0):
        # Give the disconnect watcher and the engine's STOP a moment before the
        # next request, so this measures recovery rather than queueing.
        time.sleep(seconds)

    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError('diagnostic server exited')
            if not ready.exists():
                time.sleep(2)
                continue
            try:
                with urllib.request.urlopen(url + '/v1/status', timeout=2) as response:
                    if json.load(response).get('loaded'):
                        break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('readiness timeout')

        # --- multi-round tool calling on one growing transcript -------------------
        first = [{'role': 'system', 'content': SHORT},
                 {'role': 'user', 'content': 'Read src/demo.cpp using read_file.'}]
        call('tools-primer', first, tools=TOOLS)
        conversation = first + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_1', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"src/demo.cpp"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': 'int value = 7;'}]
        call('tools-round-1', conversation, tools=TOOLS)
        conversation = conversation + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_2', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"src/util.cpp"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_2',
             'content': 'const char* kName = "demo";'},
            {'role': 'user', 'content': 'Summarise the two files.'}]
        call('tools-round-2', conversation, tools=TOOLS)
        # Parallel calls in ONE assistant message, two results.
        parallel = conversation + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_a', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"a.cpp"}'}},
                {'id': 'call_b', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"b.cpp"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_a', 'content': 'static int a = 1;'},
            {'role': 'tool', 'tool_call_id': 'call_b', 'content': 'static int b = 2;'},
            {'role': 'user', 'content': 'Which is larger?'}]
        call('tools-parallel', parallel, tools=TOOLS)
        # Malformed assistant arguments kept in history, plus an error result.
        malformed = parallel + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_bad', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":'}}]},
            {'role': 'tool', 'tool_call_id': 'call_bad',
             'content': 'error: the tool arguments were not valid JSON'},
            {'role': 'user', 'content': 'Retry with a valid path.'}]
        call('tools-malformed', malformed, tools=TOOLS)
        # The tool list itself changes: the rendered prompt diverges at the top.
        call('tools-list-changed', malformed, tools=MORE_TOOLS)
        # An empty tool result and an empty assistant message.
        empties = malformed + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_empty', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"empty.cpp"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_empty', 'content': ''},
            {'role': 'assistant', 'content': ''},
            {'role': 'user', 'content': 'Continue.'}]
        call('tools-empty', empties, tools=TOOLS)
        # A very large tool result.
        big = malformed + [
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_big', 'type': 'function',
                 'function': {'name': 'read_file', 'arguments': '{"path":"big.txt"}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_big',
             'content': '\n'.join('line %04d: %s' % (i, 'x' * 24) for i in range(90))},
            {'role': 'user', 'content': 'How many lines?'}]
        call('tools-big-result', big, tools=TOOLS)

        # --- an agent loop of 1-token turns ---------------------------------------
        loop = [{'role': 'system', 'content': SHORT},
                {'role': 'user', 'content': 'Count slowly.'}]
        for step in range(6):
            answer = call('loop-' + str(step), loop, max_tokens=1)
            loop = loop + [{'role': 'assistant', 'content': 'step %d,' % step},
                           {'role': 'user', 'content': 'next'}]

        # --- a truncated assistant turn (finish_reason=length) then a new turn ---
        trunc = [{'role': 'system', 'content': SHORT},
                 {'role': 'user', 'content': 'Write a long function.'}]
        call('truncated-first', trunc, max_tokens=8)
        trunc = trunc + [{'role': 'assistant', 'content': 'int f(void) {'},
                          {'role': 'user', 'content': 'Finish that function.'}]
        call('truncated-continue', trunc)

        # --- a refused request must not disturb the held prefix -------------------
        bad = {'messages': 'not a list'}
        connection, response, elapsed = call_raw(url, dict(bad, port=port), False)
        status = response.status
        connection.close()
        rows.append({'tag': 'refusal', 'request': bad, 'answer': {'status': status},
                     'wall_s': elapsed, 'timings': None, 'logit_record_index': None,
                     'rendered_ids': None, 'aborted': None})
        (folder / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(mode, 'refusal', status, flush=True)
        call('after-refusal', trunc)

        # --- abort during the prompt read, then the very same prompt -------------
        long_first = [{'role': 'system', 'content': LONG},
                      {'role': 'user', 'content': 'What is the index of fact 0077?'}]
        abort('abort-prefill', long_first, seconds=3)
        settle()
        call('abort-prefill-repeat', long_first)

        # --- abort mid-generation, then replay and then a diverging turn ---------
        abort('abort-token', long_first, after_delta=True)
        settle()
        call('abort-token-replay', long_first)
        diverge = long_first + [{'role': 'assistant', 'content': 'A discarded partial reply.'},
                                {'role': 'user', 'content': 'Answer again.'}]
        call('abort-token-diverge', diverge)

        # --- two clients at once --------------------------------------------------
        prompts = {'concurrent-a': [{'role': 'system', 'content': LONG},
                                    {'role': 'user', 'content': 'Name fact 0001.'}],
                   'concurrent-b': [{'role': 'system', 'content': SHORT},
                                    {'role': 'user', 'content': 'Name module 003.'}]}
        errors = []

        def worker(tag):
            try:
                call(tag, prompts[tag])
            except Exception as exc:               # surfaced below, not swallowed
                errors.append('%s: %r' % (tag, exc))

        threads = [threading.Thread(target=worker, args=(tag,)) for tag in prompts]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError('concurrent arm failed: ' + '; '.join(errors))

        # Explicitly establish the owner; concurrent admission order is not a
        # promise, and a first request of a different session must go cold.
        call('final-long-primer', diverge)
        # --- after all of that, a long exact extension still has to work ---------
        call('final-long-turn',
             diverge + [{'role': 'assistant', 'content': 'It is 1309.'},
                        {'role': 'user', 'content': 'And its group?'}])
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
    return rows, read_records(dump)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8215)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be in [1,65535]')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    original = json.loads(args.config.read_text())
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', args.port))
    off, a = run_arm(args, original, 'off', args.port)
    on, b = run_arm(args, original, 'on', args.port + 1)
    # Concurrent clients need not complete in the same order across arms.
    off_by_tag, on_by_tag = ({r['tag']: r for r in arm} for arm in (off, on))
    if off_by_tag.keys() != on_by_tag.keys() or len(off_by_tag) != len(off):
        raise AssertionError('the two arms ran different scenarios or duplicate tags')
    pairs, failures = [], []
    for tag, x in off_by_tag.items():
        y = on_by_tag[tag]
        if x['request'] != y['request']:
            raise AssertionError('request %s differs between arms' % x['tag'])
        if x['aborted'] or y['aborted'] or x['logit_record_index'] is None:
            pairs.append({'tag': x['tag'], 'aborted': bool(x['aborted'] or y['aborted']),
                          'comparable': False,
                          'off_answer': x['answer'], 'on_answer': y['answer']})
            if x['answer'] != y['answer']:
                failures.append(x['tag'] + ': refusal/outcome differs between arms')
            continue
        equal = a[x['logit_record_index']][2] == b[y['logit_record_index']][2]
        pairs.append({'tag': x['tag'], 'finite_bitwise_equal': equal,
                      'off_timings': x['timings'], 'on_timings': y['timings']})
        if not equal:
            failures.append(x['tag'] + ': cache-on logits differ from cache-off')
    # Same scenario, same transcript: the rendered token IDs must agree too.
    off_ids = {r['tag']: r['rendered_ids'] for r in off if r['rendered_ids']}
    on_ids = {r['tag']: r['rendered_ids'] for r in on if r['rendered_ids']}
    if off_ids != on_ids:
        failures.append('rendered prompt IDs differ between arms')
    for mode, arm in (('off', off_by_tag), ('on', on_by_tag)):
        message = arm['tools-primer']['answer']['choices'][0]['message']
        calls = message.get('tool_calls') or []
        try:
            function = calls[0]['function']
            valid = (function['name'] == 'read_file' and
                     json.loads(function['arguments']) == {'path': 'src/demo.cpp'})
        except (IndexError, KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            failures.append(mode + ': model did not return the requested valid read_file call')
    # Requests that grow a matching transcript must reuse a long prefix.
    promised = [t for t in ('tools-round-1', 'tools-round-2', 'tools-parallel',
                            'tools-malformed', 'tools-empty', 'tools-big-result',
                            'truncated-continue', 'after-refusal',
                            'abort-prefill-repeat',
                            'abort-token-replay', 'abort-token-diverge',
                            'final-long-turn')
                if t in {r['tag'] for r in on}]
    hits = {}
    for row in on:
        if row['tag'] in promised:
            reuse = (row['timings'] or {}).get('cache_n') or 0
            hits[row['tag']] = reuse
            # A promptly cancelled prefill may have completed only one chunk.
            # Require that real checkpoint, not work the client asked to abort.
            minimum = 512 if row['tag'] == 'abort-prefill-repeat' else 1024
            if reuse < minimum:
                failures.append('%s: only %d reused tokens (minimum %d)'
                                % (row['tag'], reuse, minimum))
    # The 1-token loop grows by one short message per turn: it must not regress
    # into a full re-read of the transcript, but one token of prefix is not enough
    # to demand a long hit, so it only has to stay exact.
    (args.output / 'cross-mode-results.json').write_text(json.dumps(pairs, indent=2) + '\n')
    (args.output / 'prompt-id-parity.json').write_text(json.dumps(
        {'rendered_prompts': len(off_ids), 'exact_id_parity': bool(off_ids == on_ids)}, indent=2) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(
        {'steps': len(pairs), 'failures': failures, 'required_hits': hits,
         'promised_requests_with_reuse': len(hits)}, indent=2) + '\n')
    print(json.dumps({'failures': failures, 'hits': hits}, indent=2))
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
