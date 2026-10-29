#!/usr/bin/env python3
"""Repeat an engine command and require finite, bit-identical logits and ids.

Example: HIP_VISIBLE_DEVICES=0 python tools/kolibri/check_repeat.py --runs 6 \
  --out /tmp/kolibri-repeat -- build-mmq/strata-kolibri --model MODEL \
  --tokens-file tools/kolibri/oracle_ids512.txt --n 16 --ctx 8192 \
  --cpu-threads 8 --prefill
"""
import argparse
import array
import math
import os
import re
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, default=6)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--require-mmq', action='store_true', help='fail if the engine falls back instead of activating MMQ')
    p.add_argument('--require-marker', default=None,
                   help='regex that must appear in every run\'s stdout+stderr, so an arm that is switched on but '
                        'inert cannot pass as six identical runs')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    cmd = a.command[1:] if a.command[:1] == ['--'] else a.command
    if a.runs < 2 or not cmd:
        p.error('need at least two runs and an engine command after --')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'command.txt').write_text(repr(cmd) + '\n' + '\n'.join(
        f'{k}={v}' for k, v in sorted(os.environ.items())
        if k.startswith(('STRATA_', 'HIP_VISIBLE_DEVICES'))))
    reference = None
    failed = False
    for i in range(a.runs):
        stem = a.out / str(i)
        logits = stem.with_suffix('.logits').resolve()
        with stem.with_suffix('.out').open('wb') as out, stem.with_suffix('.err').open('wb') as err:
            result = subprocess.run(cmd + ['--dump-full-logits', str(logits)], stdout=out, stderr=err, timeout=600)
        if result.returncode:
            print(f'FAIL run {i}: exit {result.returncode}', flush=True)
            return 1
        if a.require_mmq and 'MMQ active (direct shared-expert tuning verified)' not in stem.with_suffix('.err').read_text():
            print(f'FAIL run {i}: MMQ did not activate', flush=True)
            return 1
        if a.require_marker is not None:
            both = stem.with_suffix('.err').read_text() + stem.with_suffix('.out').read_text()
            if re.search(a.require_marker, both) is None:
                print(f'FAIL run {i}: the marker {a.require_marker!r} never appeared; the arm was not active', flush=True)
                return 1
        raw = logits.read_bytes()
        floats = array.array('f')
        floats.frombytes(raw)
        lines = stem.with_suffix('.out').read_text().splitlines()
        ids = [line for line in lines if line.startswith('gen:')]
        if not floats or not all(map(math.isfinite, floats)) or len(ids) != 1 or set(ids[0].split()[1:]) == {'0'}:
            print(f'FAIL run {i}: missing/nonfinite logits or invalid ids', flush=True)
            failed = True
        sample = (raw, ids)
        if reference is None:
            reference = sample
        elif sample != reference:
            print(f'FAIL run {i}: logits/ids differ from run 0', flush=True)
            failed = True
        else:
            print(f'run {i}: bit-identical to run 0', flush=True)
    print('REPEAT FAIL' if failed else 'REPEAT PASS', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
