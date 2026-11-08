#!/usr/bin/env python3
"""GPU integration regression. Supply ROCm library/device environment externally.

Example: python tools/kolibri/test_unpinned.py --bin build-kolibri-hip/strata-kolibri --model /path/model.gguf
Logs and full distributions are kept in --out. No server is started.
"""
import argparse
import array
import pathlib
import subprocess

p = argparse.ArgumentParser(__doc__)
p.add_argument('--bin', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', default='/tmp/kolibri-unpinned-check')
p.add_argument('--phase', choices=['short', 'long', 'hot', 'all'], default='all')
p.add_argument('--hot', action='store_true', help='enable per-layer hot residency (required by demand profile)')
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
ids = pathlib.Path(__file__).with_name('oracle_ids512.txt')
base = [a.bin, '--model', a.model, '--expert-no-pin', '--ctx', '131072',
        '--vram-reserve-mib', '1536', '--tokens-file', str(ids), '--n', '4',
        '--force', '286,3515,310']
if a.hot:
    base += ['--expert-slots-hot']

def run(name, args, expected=0):
    with (out / (name + '.log')).open('w') as log:
        r = subprocess.run(base + args, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    assert r.returncode == expected, (name, r.returncode, out / (name + '.log'))
    return (out / (name + '.log')).read_text()

def logits(name):
    import math
    data = (out / (name + '.bin')).read_bytes()
    v = array.array('f')
    v.frombytes(data)
    assert len(v) == 4 * 128000, (name, len(v))
    assert all(math.isfinite(x) for x in v), name
    assert max(v) != min(v), name
    return data

# Invalid combinations must fail before model allocation, not dereference a null table.
if a.phase in ('short', 'all'):
    for flag in ['--grouped', '--cpu-threads']:
        run('reject-' + flag[2:], ['--expert-slots', '8', flag] + (['2'] if flag == '--cpu-threads' else []), 1)

short_ids = out / 'ids8.txt'
short_ids.write_text(','.join(ids.read_text().strip().split(',')[:8]))
cases = [('decode', 'auto', []), ('prefill', 'auto', ['--prefill']),
         ('prefill-repeat', 'auto', ['--prefill']), ('prefill-spill', '8', ['--prefill']),
         ('decode-short', 'auto', ['--tokens-file', str(short_ids)]),
         ('decode-short-spill', '8', ['--tokens-file', str(short_ids)])]
for name, slots, extra in (cases if a.phase in ('short', 'all') else []):
    text = run(name, ['--expert-slots', slots, '--dump-full-logits', str(out / (name + '.bin'))] + extra)
    logits(name)
    generated = next(line for line in text.splitlines() if line.startswith('gen:'))
    if name == 'decode':
        assert generated == 'gen: 286 3515 310 262', generated
    print(name, generated, flush=True)
if a.phase in ('short', 'all'):
    assert logits('prefill') == logits('prefill-repeat'), 'prefill not bit-identical to itself'
    assert logits('prefill') == logits('prefill-spill'), 'prefill changes with cache residency'
    assert logits('decode-short') == logits('decode-short-spill'), 'decode changes with cache residency'

# Cross the 513-row sliding window and prefill chunk boundary, on identical forced histories.
long_ids = out / 'ids1025.txt'
original = ids.read_text().strip().split(',')
long_ids.write_text(','.join((original * 3)[:1025]))
long_cases = [('long-prefill', 'auto'), ('long-prefill-spill', '8')]
for name, slots in (long_cases if a.phase in ('long', 'all') else []):
    text = run(name, ['--expert-slots', slots, '--prefill', '--tokens-file', str(long_ids),
                      '--dump-full-logits', str(out / (name + '.bin'))])
    logits(name)
    print(name, next(line for line in text.splitlines() if line.startswith('gen:')), flush=True)
if a.phase in ('long', 'all'):
    assert logits('long-prefill') == logits('long-prefill-spill'), 'multi-chunk prefill changes with cache residency'
if a.phase in ('hot', 'all'):
    profile = out / 'hot.profile'
    profile.unlink(missing_ok=True)
    for name in ['hot-cold', 'hot-warm']:
        text = run(name, ['--expert-slots', 'auto', '--expert-slots-hot', '--prefill',
                          '--expert-slots-profile', str(profile), '--tokens-file', str(short_ids),
                          '--dump-full-logits', str(out / (name + '.bin'))])
        logits(name)
        assert profile.exists() and len(profile.read_text().splitlines()) > 1, 'CLI did not save its hot profile'
        print(name, next(line for line in text.splitlines() if line.startswith('gen:')), flush=True)
    assert logits('hot-cold') == logits('hot-warm'), 'profile seeding changed full logits'
print('PASS:', a.phase, 'forced-history full-logit checks')
