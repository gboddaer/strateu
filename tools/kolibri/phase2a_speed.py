#!/usr/bin/env python3
"""Unprofiled paired Phase 2a timing sweep via short-lived stdio engines.
No HTTP service. These greedy speed histories are NOT qualification inputs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import phase2a_run as matrix
import state_check


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bin',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    calibration=json.loads((args.out/'manifest-control.json').read_text())
    binary=args.bin.resolve()
    if matrix.sha(binary)!=calibration['binary_sha256']:
        raise RuntimeError('speed binary differs from qualification binary')
    prefix=calibration['prefix']
    model=calibration['model']
    results={}
    for arm in ['f16','int8']:
        env=matrix.environment(arm)
        argv=prefix+[str(binary),'--serve','--model',model,'--prefill','--cpu-threads','14','--ctx','14336']
        stop=threading.Event(); samples=[]
        def monitor():
            while not stop.is_set():
                used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                                              '--format=csv,noheader,nounits'],text=True)
                samples.append((time.monotonic(),int(used.splitlines()[0])))
                stop.wait(.5)
        sampler=threading.Thread(target=monitor,daemon=True);sampler.start()
        t0=time.monotonic()
        eng=None
        record=dict(argv=argv,env=matrix.public_env(env),binary_sha256=calibration['binary_sha256'],
                    context_capacity=14336,note='32 greedy steps; not used for numerical fidelity; decode includes last prompt token')
        try:
            eng=state_check.Engine(argv,env,stderr_path=str(args.out/f'speed-{arm}.err'))
            record['load_to_ready_s']=time.monotonic()-t0
            print(f'READY {arm}: {record["load_to_ready_s"]:.1f}s',flush=True)
            record['runs']=[]
            for fam in ['512','8191','11000']:
                ids_path=Path(calibration['corpus'][fam]['ids_path'])
                ids=[int(x) for x in ids_path.read_text().strip().split(',') if x.strip()]
                for rep in [1,2]:
                    # Force a cold prompt independent of the preceding request's prefix.
                    eng.request(1,[1 if ids[0]!=1 else 2])
                    start=time.monotonic()
                    generated,done,_=eng.request(32,ids,timeout=1800)
                    fields=done.split()
                    n=int(fields[0]); prompt_ms=float(fields[2]); decode_ms=float(fields[3]); reused=int(fields[7])
                    if n!=32 or len(generated)!=32 or reused!=0:
                        raise RuntimeError(f'not a cold 32-step speed sample: {done}')
                    run=dict(family=fam,repeat=rep,prompt_tokens=len(ids),decode_steps=n,
                             prompt_ms=prompt_ms,decode_ms=decode_ms,
                             prefill_tok_s=len(ids)/(prompt_ms/1000),decode_tok_s=n/(decode_ms/1000),
                             ids=generated,done=done,wall_s=time.monotonic()-start,
                             sampled_vram_peak_mib=max(v for t,v in samples if t>=start))
                    record['runs'].append(run)
                    matrix.save(args.out/f'speed-{arm}.json',record)
                    print(f'SPEED {arm} {fam} r{rep}: prefill={run["prefill_tok_s"]:.2f} decode={run["decode_tok_s"]:.2f} tok/s',flush=True)
            record['sampled_process_vram_peak_mib']=max(v for _,v in samples)
            results[arm]=record
            matrix.save(args.out/f'speed-{arm}.json',record)
        finally:
            if eng is not None: eng.close()
            stop.set();sampler.join(timeout=5)
    matrix.save(args.out/'speed-summary.json',results)

if __name__=='__main__': main()
