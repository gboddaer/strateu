#!/usr/bin/env python3
"""Execute the approved Phase 2a matrix; stdio engines only, never an HTTP server.

Run in the performance worktree with its own build. All reference artifacts are
read-only inputs. Complete logs/manifests live outside the preserved FP16 tree.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time
import qualify
import state_check

CORPUS = [('16','ids16.txt',2048),('512','ids512.txt',2048),
          ('de512','ids512_de.txt',2048),('code512','ids512_code.txt',2048),
          ('8191','ids8191.txt',12288),('11000','ids11000.txt',14336)]
LIMITS = dict(kl_mean=1e-3, kl_p99=5e-3, dlogit_worst=.05,
              top1_agreement=1., margin_floor=.05)
FORCE = '286,3515,310,262,14292,1343,1444,46'


def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n')


def public_env(env):
    """Record relevant execution settings without storing inherited credentials."""
    def sensitive(key):
        key=key.upper()
        return (key.startswith('PI_') or bool(re.search(r'KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE',key))
                or ('TOKEN' in key and not key.startswith('STRATA_KOLIBRI_')))
    return {k:('<redacted>' if sensitive(k) else v) for k,v in env.items()}


def numeric_env(env):
    return {k:v for k,v in env.items() if k.startswith(('STRATA_','CUDA_','GGML_'))}


def environment(arm):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('STRATA_','KOLIBRI_'))}
    env.update(CUDA_VISIBLE_DEVICES='0',STRATA_GEMM_DETERMINISTIC='1',
               STRATA_KOLIBRI_EXPERT_F16='0',STRATA_KOLIBRI_PF_PARTIAL='0',
               STRATA_KOLIBRI_DEC_HANDOFF='0',STRATA_KOLIBRI_DEC_STREAM='0',
               STRATA_KOLIBRI_DEC_HANDOFF_GRAPH='0',STRATA_KOLIBRI_DEC_TOKEN_GRAPH='0',
               STRATA_KOLIBRI_KV_INT8=str(int(arm=='int8')),
               STRATA_KOLIBRI_ATTN_KT='0' if arm=='f16plain' else '1')
    return env


def comparison(a,b):
    m=qualify.metrics(qualify.load_exact(a,128000,9),qualify.load_exact(b,128000,9))
    checks=qualify.check_limits(m,LIMITS)
    return dict(metrics=m,checks=checks,pass_limits=all(x[3] for x in checks))


def prefix_check(argv, env, ids, stderr):
    eng=state_check.Engine(argv,env,stderr_path=stderr)
    try:
        own,_,_=eng.request(1,ids)
        extended=ids+own+ids[:64]
        warm,warm_done,_=eng.request(4,extended)
        # Reset by using a different first id, then compare the SAME extension cold.
        eng.request(1,[1 if ids[0]!=1 else 2])
        cold,cold_done,_=eng.request(4,extended)
        reused=int(warm_done.split()[7])
        return dict(warm=warm,cold=cold,warm_done=warm_done,cold_done=cold_done,
                    reused=reused,pass_check=reused>0 and warm==cold and len(warm)==4)
    finally:
        eng.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bin',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--model',default='/data4/models/Kolibri-1-Q4_K_M/Kolibri-1-Q4_K_M.gguf')
    ap.add_argument('--inputs',type=Path,default=Path('/tmp/f16-t0/inputs'))
    ap.add_argument('--refs',type=Path,default=Path('/data4/kolibri-f16-evidence/f16-t6/qual'))
    ap.add_argument('--stage',choices=['control','bridge','candidate','state'],required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    binary=args.bin.resolve()
    prefix=['/usr/bin/numactl','--cpunodebind=1','--membind=1']
    envs={arm:environment(arm) for arm in ['f16','f16plain','int8']}
    # Fail before spending GPU time if a preregistered input is absent or malformed.
    corpus={}
    for fam,ids,ctx in CORPUS:
        path=args.inputs/ids
        tokens=[int(x) for x in path.read_text().strip().split(',') if x.strip()]
        ref=args.refs/f'ref_{fam}.bin'
        qualify.load_exact(ref,128000,9)
        corpus[fam]=dict(ids_path=str(path),ids_sha256=sha(path),tokens=len(tokens),ctx=ctx,
                         reference_path=str(ref),reference_sha256=sha(ref),
                         reference_command=(args.refs/f'ref_{fam}.cmd').read_text())
    manifest=dict(binary=str(binary),binary_sha256=sha(binary),model=args.model,
                  model_size=Path(args.model).stat().st_size,limits=LIMITS,corpus=corpus,
                  env={arm:public_env(env) for arm,env in envs.items()},prefix=prefix,workers=14,stage=args.stage,
                  hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
                                                     '--format=csv,noheader'],text=True).strip())
    manifest_path=args.out/f'manifest-{args.stage}.json'
    if args.stage=='control' and manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if (old['binary_sha256']!=manifest['binary_sha256'] or old['corpus']!=manifest['corpus']
                or numeric_env(old['env']['f16'])!=numeric_env(envs['f16'])):
            raise RuntimeError('refuse overwriting a different control configuration; use --stage bridge')
    save(manifest_path,manifest)

    def run(arm,fam,ids,ctx,rep):
        stem=args.out/f'{arm}_{fam}_r{rep}'
        dump=Path(str(stem)+'.bin')
        record=Path(str(stem)+'.json')
        if record.exists():
            old=json.loads(record.read_text())
            if old.get('exit')==0 and old.get('binary_sha256')==manifest['binary_sha256']:
                qualify.load_exact(dump,128000,9)
                if sha(dump)==old['dump_sha256'] and numeric_env(old['env'])==numeric_env(envs[arm]): return dump
            raise RuntimeError(f'refuse overwriting incompatible/incomplete run: {stem}')
        argv=prefix+[str(binary),'--model',args.model,'--tokens-file',str(args.inputs/ids),
                     '--ctx',str(ctx),'--n','8','--force',FORCE,'--prefill','--cpu-threads','14',
                     '--counters','--dump-full-logits',str(dump)]
        data=dict(argv=argv,env=public_env(envs[arm]),binary_sha256=manifest['binary_sha256'],
                  started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        print(f'START {arm} {fam} repeat {rep}',flush=True)
        t=time.monotonic()
        with open(str(stem)+'.out','w') as out,open(str(stem)+'.err','w') as err:
            p=subprocess.Popen(argv,env=envs[arm],stdout=out,stderr=err)
            peak=0
            try:
                while p.poll() is None:
                    sample=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                                                    '--format=csv,noheader,nounits'],text=True).splitlines()[0]
                    peak=max(peak,int(sample))
                    if time.monotonic()-t>7200:
                        p.kill(); p.wait(); raise RuntimeError(f'timeout {stem}')
                    time.sleep(.5)
            finally:
                if p.poll() is None: p.kill(); p.wait()
        data.update(exit=p.returncode,elapsed_s=time.monotonic()-t,sampled_vram_peak_mib=peak)
        if p.returncode==0:
            qualify.load_exact(dump,128000,9)
            data['dump_sha256']=sha(dump)
        save(record,data)
        print(f'END {arm} {fam} r{rep} exit={p.returncode} elapsed={data["elapsed_s"]:.1f}s VRAM={peak}MiB',flush=True)
        if p.returncode: raise RuntimeError(f'engine failure: {stem}')
        return dump

    if args.stage=='control':
        calibration={}
        for fam,ids,ctx in CORPUS:
            a=run('f16',fam,ids,ctx,1)
            b=run('f16',fam,ids,ctx,2)
            if sha(a)!=sha(b): raise RuntimeError(f'nonrepeatable f16 {fam}')
            m=comparison(args.refs/f'ref_{fam}.bin',a)
            m['reference_role']='independent pinned llama.cpp f16; cached read-only reference'
            old=args.refs/f'ctl5b_{fam}.bin'
            m['preserved_control_bitwise']=sha(old)==sha(a)
            calibration[fam]=m
            save(args.out/'calibration.json',calibration)
            print(f'CALIBRATION {fam}: pass={m["pass_limits"]} old-control-identical={m["preserved_control_bitwise"]}',flush=True)
            if not m['preserved_control_bitwise']:
                raise RuntimeError(f'gate-off full logits changed against preserved control: {fam}')
        save(args.out/'calibration.json',calibration)
        print('CALIBRATION COMPLETE: '+('absolute usable' if all(x['pass_limits'] for x in calibration.values())
                                             else 'absolute verdict void; use same-engine comparisons'),flush=True)
    elif args.stage=='bridge':
        # Baseline mirrors were enabled by default. Prove their removal changes no logits
        # before using the original two-repeat control dumps for the mirror-off candidate.
        bridge={}
        for fam,ids,ctx in CORPUS:
            a=run('f16plain',fam,ids,ctx,1)
            control=args.out/f'f16_{fam}_r1.bin'
            exact=sha(a)==sha(control)
            bridge[fam]=dict(bitwise=exact,binary_sha256=manifest['binary_sha256'],
                             mirror_off_sha256=sha(a),control_sha256=sha(control))
            save(args.out/'bridge.json',bridge)
            print(f'BRIDGE {fam}: mirror-off == original f16 control: {exact}',flush=True)
            if not exact: raise RuntimeError(f'mirror removal changed full logits: {fam}')
    elif args.stage=='candidate':
        prior=json.loads((args.out/'manifest-control.json').read_text())
        if prior['binary_sha256']!=manifest['binary_sha256'] or prior['corpus']!=manifest['corpus']:
            raise RuntimeError('control/candidate identity mismatch')
        calibration=json.loads((args.out/'calibration.json').read_text())
        if set(calibration)!=set(corpus): raise RuntimeError('incomplete calibration')
        if numeric_env(prior['env']['f16'])!=numeric_env(manifest['env']['f16']):
            raise RuntimeError('control configuration changed; no mirror-off substitution is permitted')
        results={}
        for fam,ids,ctx in CORPUS:
            a=run('int8',fam,ids,ctx,1)
            b=run('int8',fam,ids,ctx,2)
            repeat=sha(a)==sha(b)
            m=comparison(args.out/f'f16_{fam}_r1.bin',a)
            m['repeat_bitwise']=repeat
            m['absolute_verdict']='void' if not calibration[fam]['pass_limits'] else comparison(args.refs/f'ref_{fam}.bin',a)
            results[fam]=m
            save(args.out/'candidate-metrics.json',results)
            print(f'QUAL {fam}: pass={m["pass_limits"]} repeat={repeat} meanKL={m["metrics"]["kl_mean"]:.7g} worst={m["metrics"]["dlogit_worst"]:.7g}',flush=True)
            if not repeat: raise RuntimeError(f'nonrepeatable int8 {fam}')
    else:
        ids=[int(x) for x in (args.inputs/'ids512.txt').read_text().strip().split(',') if x.strip()]
        for arm in ['f16','int8']:
            argv=prefix+[str(binary),'--serve','--model',args.model,'--prefill','--cpu-threads','14','--ctx','2048']
            log=dict(argv=argv,env=public_env(envs[arm]),note='stdio engine only; no network/HTTP server started')
            try:
                ok=state_check.run_checks(argv,envs[arm],ids,ids[:64],log,str(args.out/f'state-{arm}.err'))
                ok=ok and log['cancel']['done'].split()[4]=='cancel'
                log['prefix_cold_comparison']=prefix_check(argv,envs[arm],ids,str(args.out/f'prefix-{arm}.err'))
                ok=ok and log['prefix_cold_comparison']['pass_check']
                log['result']='PASS' if ok else 'FAIL'
            except state_check.Fail as ex:
                log.update(result='FAIL',error=str(ex))
            save(args.out/f'state-{arm}.json',log)
            print(f'STATE {arm}: {log["result"]}',flush=True)
            if log['result']!='PASS': raise RuntimeError(f'state failure {arm}')

if __name__=='__main__':
    main()
