#!/usr/bin/env python3
"""Require real long-prefix hits AND complete finite logits equal to cold reads.

Starts an isolated model/GPU process. Output directory must be new. Every cold
reference uses the candidate's exact forced IDs, never independent generation.
"""
import argparse
import json
import os
from pathlib import Path
from degeneration_gate import read_records
from reuse_gate import Engine, load


class ProbeEngine(Engine):
    def __init__(self, *args, **kwargs):
        self.info = {}
        super().__init__(*args, **kwargs)

    def _read(self, *args, **kwargs):
        line = super()._read(*args, **kwargs)
        if line.startswith('INFO '):
            for field in line.split()[1:]:
                key, _, value = field.partition('=')
                self.info[key] = value
        return line

    def cancel_prefill(self, ids):
        self.p.stdin.write('GEN 1024 ' + ','.join(map(str, ids)) + '\n')
        self.p.stdin.flush()
        sent = False
        result = {'gen': [], 'done': None}
        while True:
            line = self._read()
            if line.startswith('PP ') and not sent:
                self.p.stdin.write('STOP\n')
                self.p.stdin.flush()
                sent = True
            elif line.startswith('T '):
                result['gen'].append(int(line.split()[1]))
            elif line.startswith('DONE'):
                result['done'] = line.split()
                return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--cpu-threads', type=int, default=14)
    ap.add_argument('--cache-mb', type=int, default=128)
    ap.add_argument('--stream', choices=['on', 'off'], default='on')
    ap.add_argument('--pool-mode', type=int, default=1)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--long', action='store_true', help='also qualify 11000 and 32768 ID histories')
    ap.add_argument('--cancel-only', action='store_true', help='qualify the two cancellation paths only')
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump = out / 'logits.bin'
    fixture = load(Path(__file__).with_name('reuse_ids3k.txt'))
    tail = load(Path(__file__).with_name('reuse_ids_tail64.txt'))
    expanded = fixture * 12
    long_ids = expanded[:8191]
    # Isolate this process from unrelated experimental parent overrides.
    for key in list(os.environ):
        if key.startswith('STRATA_KOLIBRI_'):
            del os.environ[key]
    env = {'STRATA_KOLIBRI_LOGITS_DUMP': str(dump),
           'STRATA_KOLIBRI_PREFIX_CACHE_MB': str(args.cache_mb),
           'STRATA_KOLIBRI_EXPERT_STREAM': '1' if args.stream == 'on' else '0',
           'STRATA_KOLIBRI_STREAM_DEV_ACT': '1',
           'STRATA_KOLIBRI_EXPERT_F16': '0',
           'STRATA_KOLIBRI_POOL_ALL': str(args.pool_mode),
           'STRATA_KOLIBRI_REUSE': '1',
           'STRATA_KOLIBRI_REUSE_PREFIX': '1',
           'STRATA_KOLIBRI_REUSE_ALIGN': '1'}
    engine = ProbeEngine(args.bin, args.model, 65536 if args.long else 16384, args.cpu_threads, env)
    (out / 'inputs.json').write_text(json.dumps(
        {'environment': env, 'engine_info': engine.info, 'fixture': fixture, 'tail': tail}, indent=2) + '\n')
    rows = []
    expect_cache = args.cache_mb >= 41 and args.pool_mode == 1

    def request(ids):
        response = engine.send(ids, 1)
        records = read_records(dump)
        if not records or len(records[-1][2]) != 128000 * 4:
            raise ValueError('missing complete Kolibri logits')
        return response, records[-1][2]

    def record(name, cold_logits, warm, warm_logits, require_hit=True, cancelled=None, minimum_hit=1024):
        hit = warm['resume'] or 0
        row = {'case': name, 'reuse': hit, 'finite_bitwise_equal': cold_logits == warm_logits,
               'required_long_hit': require_hit and expect_cache,
               'hit_ok': not (require_hit and expect_cache) or hit >= minimum_hit}
        if not require_hit and cancelled is None:
            row['expected_cold_fallback'] = True
            row['hit_ok'] = hit == 0
        if cancelled is not None:
            row['cancel_finish'] = cancelled['done'][5]
            row['produced_ids_before_cancel'] = cancelled['gen']
            row['cancel_ok'] = cancelled['done'][5] == 'cancel'
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)

    def check(name, previous, target, require_hit=True, minimum_hit=1024):
        engine.resetter(1)
        _, cold_logits = request(target)
        engine.resetter(1)
        request(previous)
        warm, warm_logits = request(target)
        record(name, cold_logits, warm, warm_logits, require_hit, minimum_hit=minimum_hit)

    def chain(name, previous, targets, required):
        refs, seen = [], {}
        for target in targets:
            key = tuple(target)
            if key not in seen:
                engine.resetter(1)
                seen[key] = request(target)[1]
            refs.append(seen[key])
        engine.resetter(1)
        request(previous)
        for i, (target, reference, hit) in enumerate(zip(targets, refs, required)):
            warm, logits = request(target)
            record(name + '-' + str(i), reference, warm, logits, hit)

    try:
        if not args.cancel_only:
            check('long-prefix-extension', fixture, fixture + tail)
        if not args.quick and not args.cancel_only:
            check('one-token-suffix', fixture[:513], fixture[:514], minimum_hit=512)
            check('long-prefix-edit', fixture, fixture[:2800] + tail)
            check('long-replay', fixture, fixture)
            check('swa-wrap-8191', long_ids, long_ids + tail[:8])
            check('early-edit-eviction-fallback', long_ids, fixture[:45] + tail, False)
            b = fixture[:2800] + tail
            unrelated = [2] + fixture[:700]
            chain('branch-ownership', fixture,
                  [b, b + tail[:16], fixture + tail[:8], unrelated, fixture + tail],
                  [True, True, True, False, False])
            chain('forced-continuation', fixture,
                  [fixture + tail + tail[:step] for step in range(3)], [True] * 3)
            chain('repeat-six', fixture, [fixture + tail] * 6, [True] * 6)
            if args.long:
                for length in (11000, 32768):
                    previous = expanded[:length]
                    check('long-' + str(length), previous, previous + tail[:8])
        if not args.quick or args.cancel_only:
            for phase in ('token', 'prefill'):
                engine.resetter(1)
                cancelled = (engine.send_cancel(fixture, 1024) if phase == 'token'
                             else engine.cancel_prefill(fixture))
                target = fixture + cancelled['gen'] + tail[:8]
                warm, warm_logits = request(target)
                engine.resetter(1)
                _, cold_logits = request(target)
                record('cancel-' + phase, cold_logits, warm, warm_logits, False, cancelled)
    finally:
        engine.close()
    return 0 if all(r['finite_bitwise_equal'] and r['hit_ok'] and r.get('cancel_ok', True)
                    for r in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
