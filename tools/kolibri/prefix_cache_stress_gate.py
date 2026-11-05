#!/usr/bin/env python3
"""Non-happy-path gate for exact Kolibri prefix checkpoints.

Every reference is the candidate's own cold read of the *exact* forced IDs. A
comparison passes only when all 128000 finite FP32 logits are byte-identical.
Positive reuse is required where the design promises it; where a scenario must
legitimately fall back cold, zero reuse is required instead; scenarios where either
outcome can be correct require exactness and record the reuse they got. Nothing
here scores generated text or claims a faster kernel.

Groups:
  basic    boundaries, truncation, edits, idempotence, 1-token chains, ring wrap,
           short prompts, engine refusals followed by a valid request
  prefill  STOP requested during the prompt read, repeated stops, replay, extend
  token    STOP after N decoded tokens, branch from an aborted reply, stop-after-hit
  budget   a one-checkpoint budget (needs its own process, --cache-mb 48)

One engine process serves the whole run. The output directory must not exist yet.
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

    def run(self, ids, n, stop_after_pp=None, stop_after_tokens=None):
        """One request, optionally STOPped after N PP lines or after N tokens.

        Records PP lines before and after the STOP, so a case can show whether the
        prompt read was interrupted or only followed by a cancel. An `ERR` is the
        engine refusing the request outright: no DONE follows.
        """
        self.p.stdin.write('GEN ' + str(n) + ' ' + ','.join(map(str, ids)) + '\n')
        self.p.stdin.flush()
        result = {'gen': [], 'resume': None, 'done': None, 'pp': 0, 'error': None,
                  'pp_at_stop': None, 'tokens_at_stop': None}
        stopped = False
        while True:
            line = self._read()
            if line.startswith('T '):
                result['gen'].append(int(line.split()[1]))
                if not stopped and stop_after_tokens is not None and len(result['gen']) >= stop_after_tokens:
                    self.p.stdin.write('STOP\n')
                    self.p.stdin.flush()
                    stopped = True
                    result['tokens_at_stop'] = len(result['gen'])
            elif line.startswith('PP '):
                result['pp'] += 1
                if not stopped and stop_after_pp is not None and result['pp'] >= stop_after_pp:
                    self.p.stdin.write('STOP\n')
                    self.p.stdin.flush()
                    stopped = True
                    result['pp_at_stop'] = result['pp']
            elif line.startswith('RESUME '):
                result['resume'] = int(line.split()[1])
            elif line.startswith('ERR'):
                result['error'] = line
                return result
            elif line.startswith('DONE'):
                result['done'] = line.split()
                return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--group', choices=['basic', 'prefill', 'token', 'budget', 'all'], default='all')
    ap.add_argument('--cpu-threads', type=int, default=14)
    ap.add_argument('--cache-mb', type=int, default=128)
    args = ap.parse_args()
    if args.group == 'budget' and args.cache_mb > 64:
        ap.error('--group budget needs a one-checkpoint budget, e.g. --cache-mb 48')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump = out / 'logits.bin'
    ids3k = load(Path(__file__).with_name('reuse_ids3k.txt'))
    tail = load(Path(__file__).with_name('reuse_ids_tail64.txt'))
    expanded = ids3k * 12
    # A 9,216-id prompt: enough chunks that a STOP lands after several checkpoints.
    prefill_ids = expanded[:9216]
    ctx = 32768

    for key in list(os.environ):
        if key.startswith('STRATA_KOLIBRI_'):
            del os.environ[key]
    env = {'STRATA_KOLIBRI_LOGITS_DUMP': str(dump),
           'STRATA_KOLIBRI_PREFIX_CACHE_MB': str(args.cache_mb),
           'STRATA_KOLIBRI_EXPERT_STREAM': '1',
           'STRATA_KOLIBRI_STREAM_DEV_ACT': '1',
           'STRATA_KOLIBRI_EXPERT_F16': '0',
           'STRATA_KOLIBRI_POOL_ALL': '1',
           'STRATA_KOLIBRI_REUSE': '1',
           'STRATA_KOLIBRI_REUSE_PREFIX': '1',
           'STRATA_KOLIBRI_REUSE_ALIGN': '1'}
    engine = ProbeEngine(args.bin, args.model, ctx, args.cpu_threads, env)
    (out / 'inputs.json').write_text(json.dumps(
        {'environment': env, 'engine_info': engine.info, 'group': args.group,
         'cache_mb': args.cache_mb, 'fixture_len': len(ids3k), 'ctx': ctx}, indent=2) + '\n')
    rows, failures = [], []

    def request(ids, n=1, **kwargs):
        response = engine.run(ids, n, **kwargs)
        if response['error']:
            return response, None
        records = read_records(dump)
        if not records or len(records[-1][2]) != 128000 * 4:
            raise ValueError('missing complete Kolibri logits')
        return response, records[-1][2]

    def record(name, equal, reuse, require_hit, stopped=None, extra=None,
               minimum_reuse=1024, prefix_limit=None):
        """require_hit: True = a long hit is promised, False = a cold fallback is
        promised, None = either may happen, exactness alone decides. `prefix_limit`
        is the longest prefix the request could possibly share with what was held:
        reusing more than that would be a wrong claim even if the logits happened
        to match."""
        if require_hit is True:
            hit_ok = reuse >= minimum_reuse
        elif require_hit is False:
            hit_ok = reuse == 0
        else:
            hit_ok = True
        over_claim = prefix_limit is not None and reuse > prefix_limit
        row = {'case': name, 'reuse': reuse, 'finite_bitwise_equal': equal,
               'hit_expectation': {True: 'long-hit', False: 'cold', None: 'either'}[require_hit],
               'hit_ok': hit_ok and not over_claim}
        if stopped is not None:
            row['finish'] = stopped['done'][5]
            row['produced_before_cancel'] = stopped['gen']
            row['pp_lines'] = stopped['pp']
            row['pp_at_stop'] = stopped['pp_at_stop']
            row['cancel_ok'] = stopped['done'][5] == 'cancel'
            if stopped['pp_at_stop'] is not None:
                row['cancel_ok'] = (row['cancel_ok'] and not stopped['gen'] and
                                    stopped['pp'] <= stopped['pp_at_stop'] + 1)
        if extra:
            row.update(extra)
        rows.append(row)
        (out / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(json.dumps(row), flush=True)
        if not equal:
            failures.append(name + ': finite logits differ from the cold reference')
        if over_claim:
            failures.append('%s: reused %d tokens but could share at most %d'
                            % (name, reuse, prefix_limit))
        elif not hit_ok:
            failures.append(name + ': reuse %d contradicts the %s expectation'
                            % (reuse, row['hit_expectation']))

    def cold(ids, n=1):
        engine.resetter(1)
        return request(ids, n)[1]

    def check(name, previous, target, require_hit=True, minimum_reuse=1024):
        reference = cold(target)
        engine.resetter(1)
        request(previous)
        warm, logits = request(target)
        reuse = warm['resume'] or 0
        record(name, reference == logits, reuse, require_hit, minimum_reuse=minimum_reuse,
               prefix_limit=min(len(previous), len(target)))

    try:
        group = args.group

        if group in ('basic', 'all'):
            # A checkpoint exists only at a completed canonical chunk: a 513-id
            # request reads 512 rows and can be reused at 512, a 511-id one cannot.
            check('boundary-512', ids3k[:513], ids3k[:513] + tail[:8], minimum_reuse=512)
            check('below-boundary-511', ids3k[:511], ids3k[:511] + tail[:8], require_hit=False)
            check('boundary-1024', ids3k[:1025], ids3k[:1025] + tail[:8], minimum_reuse=1024)
            # Former gaps F1/F2: early anchors must survive newer captures.
            check('truncated-target', ids3k, ids3k[:600], minimum_reuse=512)
            check('middle-edit', ids3k, ids3k[:1500] + [ids3k[1500] ^ 3] + ids3k[1501:] + tail[:8])
            check('suffix-over-ring', ids3k, ids3k + ids3k[:600])
            engine.resetter(1)
            repeated = [request(ids3k + tail[:8]) for _ in range(3)]
            record('idempotent-repeat', all(r[1] == repeated[0][1] for r in repeated),
                   min(r[0]['resume'] or 0 for r in repeated[1:]), True)
            references = [cold(ids3k + tail[:step + 1]) for step in range(6)]
            engine.resetter(1)
            request(ids3k)
            for step in range(6):
                warm, logits = request(ids3k + tail[:step + 1])
                record('one-token-chain-' + str(step), references[step] == logits,
                       warm['resume'] or 0, True)
            engine.resetter(1)
            request([2, 3, 4])
            warm, logits = request(ids3k + tail[:8])
            record('short-then-long', logits == cold(ids3k + tail[:8]), warm['resume'] or 0, False)
            # An engine refusal must leave the held prefix alone.
            engine.resetter(1)
            request(ids3k)
            refused, _ = request(ids3k + tail, ctx)
            warm, logits = request(ids3k + tail[:8])
            record('refusal-overflow', refused['error'] is not None and logits == cold(ids3k + tail[:8]),
                   warm['resume'] or 0, True,
                   extra={'refusal': (refused['error'] or '')[:120]})
            engine.resetter(1)
            request(ids3k)
            refused, _ = request(ids3k[:-1] + [128000], 1)
            warm, logits = request(ids3k + tail[:8])
            record('refusal-bad-token', refused['error'] is not None and logits == cold(ids3k + tail[:8]),
                   warm['resume'] or 0, True,
                   extra={'refusal': (refused['error'] or '')[:120]})

        if group in ('prefill', 'all'):
            # STOP is requested one PP line into a long read. The engine has no
            # mid-chunk cancellation check, so this measures what actually happens
            # instead of assuming the read was truncated.
            for label, pp_after in (('first-chunk', 1), ('third-chunk', 3)):
                engine.resetter(1)
                stopped = engine.run(prefill_ids, 8, stop_after_pp=pp_after)
                target = prefill_ids + tail[:8]
                warm, logits = request(target)
                reuse = warm['resume'] or 0
                record('prefill-stop-' + label, logits == cold(target), reuse, None, stopped,
                       {'prompt_read_continued': stopped['pp'] > (stopped['pp_at_stop'] or 0)})
            engine.resetter(1)
            first = engine.run(prefill_ids, 8, stop_after_pp=1)
            second = engine.run(prefill_ids, 8, stop_after_pp=1)
            warm, logits = request(prefill_ids + tail[:8])
            record('prefill-double-stop', logits == cold(prefill_ids + tail[:8]),
                   warm['resume'] or 0, None, second,
                   {'first_finish': first['done'][5],
                    'produced_total': len(first['gen']) + len(second['gen'])})
            engine.resetter(1)
            stopped = engine.run(prefill_ids, 8, stop_after_pp=1)
            replay, replay_logits = request(prefill_ids, 1)
            replay2, replay2_logits = request(prefill_ids, 1)
            record('prefill-stop-then-replay',
                   replay_logits == cold(prefill_ids) and replay2_logits == replay_logits,
                   replay2['resume'] or 0, True, stopped)

        if group in ('token', 'all'):
            engine.resetter(1)
            request(ids3k)
            stopped = engine.run(ids3k + tail[:8], 64, stop_after_tokens=3)
            produced = stopped['gen']
            # A later turn is the prefix plus what the aborted request really wrote.
            target = ids3k + tail[:8] + produced + tail[8:16]
            warm, logits = request(target)
            record('token-stop-then-continue', logits == cold(target), warm['resume'] or 0, True, stopped,
                   {'continuation_len': len(target)})
            # A branch that diverges inside the aborted reply must not borrow its rows.
            branch = ids3k + tail[:8] + produced[:-1] + [produced[-1] ^ 5] + tail[8:16]
            warm, logits = request(branch)
            record('token-stop-branch', logits == cold(branch), warm['resume'] or 0, None,
                   prefix_limit=len(ids3k) + 8 + len(produced) - 1)
            # Abort during a request that itself started from a checkpoint.
            engine.resetter(1)
            request(ids3k)
            request(ids3k + tail[:8])
            hit = engine.run(ids3k + tail[:8] + tail[8:16], 64, stop_after_tokens=2)
            aborted_reuse = hit['resume'] or 0
            target = ids3k + tail[:16] + hit['gen'] + tail[16:24]
            warm, logits = request(target)
            record('token-stop-after-hit', aborted_reuse >= 1024 and logits == cold(target),
                   warm['resume'] or 0, True, hit,
                   {'aborted_request_reuse': aborted_reuse})
            # Repeated aborts must not leave a stale claim behind.
            engine.resetter(1)
            request(ids3k)
            aborts = [engine.run(ids3k + tail[:8], 64, stop_after_tokens=2) for _ in range(3)]
            same = all(a['gen'] == aborts[0]['gen'] for a in aborts)
            target = ids3k + tail[:8] + aborts[-1]['gen'] + tail[8:16]
            warm, logits = request(target)
            record('token-stop-repeated', same and logits == cold(target), warm['resume'] or 0, True,
                   aborts[-1], {'all_aborts_identical': same})

        if group == 'budget':
            # One checkpoint fits. Every append must stay exact, and no reuse may
            # exceed a boundary the store could still hold.
            engine.resetter(1)
            request(ids3k)
            previous = ids3k
            for step in range(4):
                target = previous + tail[:8] + expanded[4000 + step * 8:4000 + step * 8 + 8]
                warm, logits = request(target)
                record('budget-chain-' + str(step), logits == cold(target), warm['resume'] or 0, None,
                       extra={'checkpoint_mib': args.cache_mb})
                previous = target
    finally:
        engine.close()
    ok = not failures and all(r['finite_bitwise_equal'] and r['hit_ok'] and r.get('cancel_ok', True)
                              for r in rows)
    (out / 'summary.json').write_text(json.dumps(
        {'cases': len(rows), 'failures': failures,
         'all_finite_bitwise_equal': all(r['finite_bitwise_equal'] for r in rows),
         'all_hit_expectations_met': all(r['hit_ok'] for r in rows),
         'cancel_outcomes': sorted({r['finish'] for r in rows if 'finish' in r})}, indent=2) + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
