#!/usr/bin/env python3
"""Regression tests for the former gaps in exact Kolibri prefix checkpoints.

Each case retains its original desired behaviour and exits 1 on regression.
The two CPU retention cases also run in the normal pytest suite. The model gate
requires the same desired reuse AND finite byte-identical full logits.

  --store    compile and run tests/kolibri/prefix_cache_gaps_test.cpp (no GPU)
  --engine   run the four end-to-end gaps on a real engine (GPU, several minutes)

The original baseline failed reuse and cancellation assertions while keeping
exact logits. Fixes must satisfy both: a cache hit with different logits fails.
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
    'truncated-target': 'a shorter exact prefix retains a matching early boundary',
    'middle-edit': 'an edit at 1500 retains a canonical boundary below the edit',
    'token-abort-continue': 'decoding STOP preserves valid prefill checkpoints',
    'prefill-stop-interrupts-read': 'prompt STOP ends at a completed chunk boundary',
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
        before = len(read_records(dump)) if dump.exists() else 0
        response = engine.run(ids, n, **kwargs)
        records = read_records(dump)
        if len(records) != before + 1 or len(records[-1][2]) != 128000 * 4:
            raise ValueError('request did not produce its own complete Kolibri logits')
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
        want('truncated-target', (warm['resume'] or 0) >= 512 and reference == logits,
             'reused %s tokens, exact=%s' % (warm['resume'] or 0, reference == logits))

        # (2) edit in the middle
        target = ids3k[:1500] + [ids3k[1500] ^ 3] + ids3k[1501:] + tail[:8]
        reference = cold(target)
        engine.resetter(1)
        logits_of(ids3k)
        warm, logits = logits_of(target)
        want('middle-edit', (warm['resume'] or 0) >= 1024 and reference == logits,
             'reused %s tokens, exact=%s' % (warm['resume'] or 0, reference == logits))

        # (3) STOP during decoding, then the next turn continues the real reply
        engine.resetter(1)
        logits_of(ids3k)
        stopped = engine.run(ids3k + tail[:8], 64, stop_after_tokens=3)
        follow = ids3k + stopped['gen'] + tail[:8]
        warm, logits = logits_of(follow)
        reference = cold(follow)  # the actual cancelled IDs, never a second sampled history
        want('token-abort-continue',
             (warm['resume'] or 0) >= 1024 and reference == logits and stopped['done'][5] == 'cancel',
             'produced %s before cancel, next turn reused %s, exact=%s'
             % (stopped['gen'], warm['resume'] or 0, reference == logits))

        # (4) STOP during the prompt read
        engine.resetter(1)
        stopped = engine.run(long_ids, 8, stop_after_pp=1)
        chunks = (len(long_ids) - 1) // 512 + 1
        want('prefill-stop-interrupts-read',
             stopped['pp_at_stop'] == 1 and stopped['pp'] <= 2 and
             stopped['done'][5] == 'cancel' and not stopped['gen'] and
             0 < int(stopped['done'][14]) <= 1024,
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
    parser.add_argument('--bin', type=Path, help='required with --engine; select the build being qualified explicitly')
    parser.add_argument('--model', default='/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf')
    parser.add_argument('--output', type=Path, default=OUT / 'engine')
    parser.add_argument('--cpu-threads', type=int, default=14)
    args = parser.parse_args()
    if not args.store and not args.engine:
        args.store = args.engine = True
    if args.engine and args.bin is None:
        parser.error('--engine requires --bin; do not accidentally test a stale control build')
    print('Former gap regressions (GREEN requires the original desired behaviour):')
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
    print('Regression gate exit %d (0 = all requested properties satisfied)' % status)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
