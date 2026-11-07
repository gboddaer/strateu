#!/usr/bin/env python3
"""Reproducible offline Kolibri cache benchmark; never starts a server.

Uses fixed forced histories. Outputs raw logs, full logits and JSON. Supply HIP
library/device environment externally. Each run starts with an independent profile.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--prompt', type=int, default=512)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--slots', default='auto')
    ap.add_argument('--profile-input')
    ap.add_argument('--compare')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--env', action='append', default=[])
    a = ap.parse_args()
    if a.prompt < 1 or a.n < 1:
        ap.error('prompt and decode lengths must be positive')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = out / a.label
    ids = Path(__file__).with_name('oracle_ids512.txt').read_text().strip().split(',')
    prompt = (ids * ((a.prompt + len(ids) - 1) // len(ids)))[:a.prompt]
    prefix.with_suffix('.ids').write_text(','.join(prompt))
    profile = prefix.with_suffix('.profile')
    if a.profile_input and Path(a.profile_input).resolve() == profile.resolve():
        ap.error('--profile-input must differ from the output profile')
    profile.unlink(missing_ok=True)
    if a.profile_input:
        profile.write_bytes(Path(a.profile_input).read_bytes())
    forced = ([286,3515,310,262,14292,1343,1444,46,17156,32,51,48,58,4037,262,1890] * ((a.n+15)//16))[:a.n-1]
    cmd = [a.bin, '--model', a.model, '--expert-no-pin', '--expert-slots', a.slots,
           '--expert-slots-hot', '--expert-slots-profile', str(profile), '--prefill',
           '--ctx', '131072', '--vram-reserve-mib', '1536', '--tokens-file', str(prefix.with_suffix('.ids')),
           '--n', str(a.n), '--dump-full-logits', str(prefix.with_suffix('.bin')), '--counters']
    if forced:
        cmd += ['--force', ','.join(map(str, forced))]
    env = dict(os.environ, STRATA_KOLIBRI_BENCH='1', STRATA_KOLIBRI_NOPIN_STATS=str(int(a.stats)))
    for pair in a.env:
        k, v = pair.split('=', 1)
        env[k] = v
    start = time.monotonic()
    with prefix.with_suffix('.log').open('w') as log:
        r = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    wall = time.monotonic() - start
    assert r.returncode == 0, (r.returncode, str(prefix.with_suffix('.log')))
    text = prefix.with_suffix('.log').read_text()
    phases = {}
    for phase, tokens, ms in re.findall(r'NOPIN_PHASE (\w+) tokens=(\d+) ms=([0-9.]+)', text):
        phases[phase] = {'tokens': int(tokens), 'ms': float(ms), 'tok_s': int(tokens) * 1000 / max(float(ms), .001)}
    assert {'startup', 'prefill', 'decode'} <= phases.keys(), 'missing phase measurements'
    data = prefix.with_suffix('.bin').read_bytes()
    values = array.array('f')
    values.frombytes(data)
    assert len(values) == a.n * 128000 and all(map(math.isfinite, values)), 'invalid complete logits'
    if a.compare:
        assert data == Path(a.compare).read_bytes(), 'full logits differ on identical forced history'
    result = {'label': a.label, 'command': cmd, 'env_overrides': a.env, 'wall_s': wall,
              'phases': phases, 'logits_sha256': hashlib.sha256(data).hexdigest(),
              'binary_sha256': hashlib.sha256(Path(a.bin).read_bytes()).hexdigest(),
              'generated': next(x for x in text.splitlines() if x.startswith('gen:')),
              'diagnostics': [x for x in text.splitlines() if x.startswith(('NOPIN_STATS', 'NOPIN_IO')) or 'free (reserve' in x]}
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
