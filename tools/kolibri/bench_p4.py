#!/usr/bin/env python3
"""Measure the same packaged binary with the gfx1201 MMQ default and its pool off switch.

Prefill is engine-reported PFchunk time. Generation is wall time from the unbuffered
CLI's `gen:` marker to its final newline, including token selection and output.
Model loading is excluded from both; process wall time is recorded separately.
"""
import argparse
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--tokens-file', required=True)
    p.add_argument('--ctx', type=int, default=8192)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--runs', type=int, default=2)
    p.add_argument('--arm', choices=('default', 'pool'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(HIP_VISIBLE_DEVICES='0', STRATA_KOLIBRI_EXPERT_F16='0',
               STRATA_KOLIBRI_ATTN_MB='512', STRATA_KOLIBRI_RES_TOP='56')
    for key in ('STRATA_KOLIBRI_PF_MMQ', 'STRATA_MMQ_KQ', 'STRATA_KOLIBRI_MMQ_ASYNC',
                'STRATA_HIPBLASLT_TUNING'):
        env.pop(key, None)
    if a.arm == 'pool':
        env['STRATA_KOLIBRI_PF_MMQ'] = '0'
    cmd = ['stdbuf', '-o0', str(Path(a.binary).resolve()), '--model', a.model,
           '--tokens-file', a.tokens_file, '--ctx', str(a.ctx), '--n', str(a.n),
           '--cpu-threads', '8', '--prefill']
    for run in range(a.runs):
        load_before = os.getloadavg()
        started = time.monotonic()
        gen_start = gen_end = None
        output = bytearray()
        with (a.out / f'{run}.err').open('wb') as err:
            child = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=err,
                                     start_new_session=True)
            try:
                with selectors.DefaultSelector() as sel:
                    sel.register(child.stdout, selectors.EVENT_READ)
                    while sel.get_map():
                        if time.monotonic() - started > 600:
                            raise TimeoutError('engine exceeded 600 seconds')
                        for key, _ in sel.select(5):
                            chunk = os.read(key.fd, 65536)
                            now = time.monotonic()
                            if not chunk:
                                sel.unregister(key.fileobj)
                                continue
                            output.extend(chunk)
                            at = output.find(b'gen:')
                            if at >= 0 and gen_start is None:
                                gen_start = now
                            if at >= 0 and gen_end is None and output.find(b'\n', at) >= 0:
                                gen_end = now
                rc = child.wait(timeout=10)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                child.stdout.close()
        wall = time.monotonic() - started
        (a.out / f'{run}.out').write_bytes(output)
        if rc or gen_start is None or gen_end is None:
            raise RuntimeError(f'run {run} failed: exit={rc}, missing generation timing?')
        stderr = (a.out / f'{run}.err').read_text()
        active = 'MMQ active (direct shared-expert tuning verified)' in stderr
        if active != (a.arm == 'default'):
            raise RuntimeError('wrong arm: MMQ activation did not match requested benchmark')
        chunks = re.findall(rb'PFchunk (\d+)/(\d+) (\d+)ms', output)
        ids = re.search(rb'^gen:([^\n]+)', output, re.M)
        if not chunks or not ids or len(ids[1].split()) != a.n or set(ids[1].split()) == {b'0'}:
            raise RuntimeError('missing prefill or invalid generation')
        tokens, _, pf_ms = map(int, chunks[-1])
        row = dict(arm=a.arm, run=run, ctx=a.ctx, chunk_size=int(chunks[0][0]),
                   expert_f16=0, pool_threads=8, gpu='card0/gfx1201',
                   load_before=load_before, load_after=os.getloadavg(),
                   prefill_tokens=tokens, prefill_engine_ms=pf_ms,
                   prefill_tok_s=1000 * tokens / pf_ms, generation_tokens=a.n,
                   generation_wall_ms=1000 * (gen_end - gen_start),
                   generation_tok_s=a.n / (gen_end - gen_start), process_wall_s=wall,
                   command=cmd, environment={k: v for k, v in env.items()
                                             if k.startswith(('STRATA_', 'HIP_VISIBLE_DEVICES'))})
        (a.out / f'{run}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
