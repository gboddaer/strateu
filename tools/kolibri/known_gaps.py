#!/usr/bin/env python3
"""RED tests for the known gaps in exact Kolibri prefix checkpoints.

Each case asserts the behaviour we WANT, so this runner exits 1 while a gap is
open. It is kept out of the passing unit/serve suites on purpose: a red test that
is wired into a green suite stops meaning anything. Nothing here changes engine
code; it only measures and reports.

  --store    compile and run tests/kolibri/prefix_cache_gaps_test.cpp (no GPU)
  --engine   run the four end-to-end gaps on a real engine (GPU, several minutes)

Exactness is not in question in any of these cases: every measured comparison so
far was bit-identical to a cold read. What fails is reuse - the work is correct and
paid for twice - or how promptly a cancelled prompt read stops.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path('/tmp/kolibri-known-gaps')

GAPS = {
    'truncated-target': 'a shorter exact prefix reuses nothing (older boundary evicted)',
    'middle-edit': 'an edit at 1500 of 3072 reuses nothing (older boundary evicted)',
    'token-abort-continue': 'a STOP during decoding discards the held history and all checkpoints',
    'prefill-stop-interrupts-read': 'a STOP during the prompt read lets the whole prompt finish',
}


def compile_store(binary: Path) -> None:
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = ['g++', '-std=c++17', '-O1', '-I', str(ROOT / 'src'),
               str(ROOT / 'tests/kolibri/prefix_cache_gaps_test.cpp'), '-o', str(binary)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit('compile failed:\n' + result.stdout + result.stderr)


def run_store() -> int:
    binary = OUT / 'prefix_cache_gaps_test'
    compile_store(binary)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    return result.returncode


def run_engine(binary: Path, model: str, output: Path, threads: int) -> int:
    from prefix_cache_stress_gate import ProbeEngine
    from degeneration_gate import read_records
    from reuse_gate import load

    output.mkdir(parents=True, exist_ok=False)
    dump = output / 'logits.bin'
    ids3k = load(HERE / 'reuse_ids3k.txt')
    tail = load(HERE / 'reuse_ids_tail64.txt')
    long_ids = (ids3k * 12)[:9216]
    for key in list(os.environ):
        if key.startswith('STRATA_KOLIBRI_'):
            del os.environ[key]
    engine = ProbeEngine(binary, model, 32768, threads,
                         {'STRATA_KOLIBRI_LOGITS_DUMP': str(dump),
                          'STRATA_KOLIBRI_PREFIX_CACHE_MB': '128',
                          'STRATA_KOLIBRI_EXPERT_STREAM': '1',
                          'STRATA_KOLIBRI_STREAM_DEV_ACT': '1',
                          'STRATA_KOLIBRI_EXPERT_F16': '0',
                          'STRATA_KOLIBRI_POOL_ALL': '1',
                          'STRATA_KOLIBRI_REUSE': '1',
                          'STRATA_KOLIBRI_REUSE_PREFIX': '1',
                          'STRATA_KOLIBRI_REUSE_ALIGN': '1'})
    rows, red = [], []

    def logits_of(ids, n=1, **kwargs):
        response = engine.run(ids, n, **kwargs)
        records = read_records(dump)
        if not records or len(records[-1][2]) != 128000 * 4:
            raise ValueError('missing complete Kolibri logits')
        return response, records[-1][2]

    def want(name, ok, detail):
        print(('GREEN  ' if ok else 'RED    ') + name + '  (' + detail + ')', flush=True)
        rows.append({'case': name, 'desired_behaviour_met': ok, 'detail': detail})
        if not ok:
            red.append(name)

    def cold(ids):
        engine.resetter(1)
        return logits_of(ids)[1]

    try:
        # (1) shorter exact prefix
        target = ids3k[:600]
        reference = cold(target)
        engine.resetter(1)
        logits_of(ids3k)
        warm, logits = logits_of(target)
        want('truncated-target', (warm['resume'] or 0) >= 512,
             'reused %s tokens, exact=%s' % (warm['resume'] or 0, reference == logits))

        # (2) edit in the middle
        target = ids3k[:1500] + [ids3k[1500] ^ 3] + ids3k[1501:] + tail[:8]
        reference = cold(target)
        engine.resetter(1)
        logits_of(ids3k)
        warm, logits = logits_of(target)
        want('middle-edit', (warm['resume'] or 0) >= 1024,
             'reused %s tokens, exact=%s' % (warm['resume'] or 0, reference == logits))

        # (3) STOP during decoding, then the next turn continues the real reply
        engine.resetter(1)
        logits_of(ids3k)
        stopped = engine.run(ids3k + tail[:8], 64, stop_after_tokens=3)
        follow = ids3k + stopped['gen'] + tail[:8]
        reference = cold(follow)
        engine.resetter(1)
        logits_of(ids3k)
        stopped = engine.run(ids3k + tail[:8], 64, stop_after_tokens=3)
        warm, logits = logits_of(ids3k + stopped['gen'] + tail[:8])
        want('token-abort-continue', (warm['resume'] or 0) >= 1024,
             'produced %s before cancel, next turn reused %s, exact=%s'
             % (stopped['gen'], warm['resume'] or 0, reference == logits))

        # (4) STOP during the prompt read
        engine.resetter(1)
        stopped = engine.run(long_ids, 8, stop_after_pp=1)
        chunks = (len(long_ids) - 1) // 512 + 1
        want('prefill-stop-interrupts-read', stopped['pp'] <= 2,
             'STOP after PP %s, %d of %d chunks still read, finish=%s'
             % (stopped['pp_at_stop'], stopped['pp'], chunks, stopped['done'][5]))
    finally:
        engine.close()
    (output / 'gaps.json').write_text(json.dumps(rows, indent=2) + '\n')
    return 0 if not red else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', action='store_true')
    parser.add_argument('--engine', action='store_true')
    parser.add_argument('--bin', type=Path, default=ROOT / 'build-cache-cuda/strata-kolibri')
    parser.add_argument('--model', default='/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf')
    parser.add_argument('--output', type=Path, default=OUT / 'engine')
    parser.add_argument('--cpu-threads', type=int, default=14)
    args = parser.parse_args()
    if not args.store and not args.engine:
        args.store = args.engine = True
    print('Known gaps (a RED line is a real, currently open gap):')
    status = 0
    if args.store:
        status = run_store()
    if args.engine:
        if args.output.exists():
            parser.error('%s exists; choose a new --output' % args.output)
        status = run_engine(args.bin, args.model, args.output, args.cpu_threads) or status
    print()
    for name, text in GAPS.items():
        print('  ' + name + ': ' + text)
    print('RED suite exit %d (0 would mean every gap was closed; update the docs then)' % status)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
