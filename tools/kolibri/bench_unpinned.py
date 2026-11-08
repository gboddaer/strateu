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
import signal
import subprocess
import sys
import time

DEFAULT_CONTEXT = 131072
MAX_CONTEXT = 2**31 - 1
DEFAULT_PROMPT = 512
TOKEN_LIMIT = 128000
ORACLE_IDS = Path(__file__).with_name('oracle_ids512.txt')
# Only these settings are copied into evidence JSON; arbitrary prefixed variables may hold secrets.
EVIDENCE_ENV_KEYS = frozenset({
    'STRATA_KOLIBRI_BENCH', 'STRATA_KOLIBRI_NOPIN_STATS', 'STRATA_KOLIBRI_NOPIN_PROFILE',
    'STRATA_KOLIBRI_PF_CHUNK', 'STRATA_KOLIBRI_EXPERT_GPU', 'STRATA_KOLIBRI_EXPERT_STREAM',
    'STRATA_KOLIBRI_MMQ', 'STRATA_KOLIBRI_LAZY_CACHE', 'STRATA_KOLIBRI_KV_INT8',
    'STRATA_KOLIBRI_DEC_STREAM', 'STRATA_KOLIBRI_DEC_HANDOFF',
    'STRATA_KOLIBRI_MMVQ_SELECTED', 'STRATA_KOLIBRI_REUSE_Q8', 'STRATA_KOLIBRI_PREFILL_ADMIT',
    'STRATA_KOLIBRI_ORDERED_DECODE', 'STRATA_KOLIBRI_PROMOTE_BATCH', 'STRATA_KOLIBRI_PACK_THREADS',
    'STRATA_KOLIBRI_ZERO_BUFFER', 'STRATA_KOLIBRI_SKIP_ZERO_OVERFLOW', 'STRATA_KOLIBRI_POISON_ZERO_SLOT',
})


def evidence_env(environ):
    """Return only allowlisted benchmark settings from an environment mapping."""
    return {key: environ[key] for key in sorted(EVIDENCE_ENV_KEYS) if key in environ}


def recorded_env(env_values):
    """Keep values of known benchmark settings; redact values of everything else."""
    recorded = []
    for key, value in env_values:
        recorded.append(f'{key}={value}' if key in EVIDENCE_ENV_KEYS else f'{key}=<redacted>')
    return recorded


def parse_ids(text):
    """Return validated token IDs from comma/whitespace separated text."""
    tokens = [t for t in re.split(r'[,\s]+', text.strip()) if t]
    if not tokens:
        raise ValueError('token fixture is empty')
    ids = []
    for token in tokens:
        if not re.fullmatch(r'\d+', token):
            raise ValueError(f'invalid token id {token!r}: expected a non-negative integer')
        value = int(token)
        if value >= TOKEN_LIMIT:
            raise ValueError(f'invalid token id {value}: must be below {TOKEN_LIMIT}')
        ids.append(value)
    return ids


def generated_ids(length):
    ids = ORACLE_IDS.read_text().strip().split(',')
    if not ids:
        raise ValueError('oracle token fixture is empty')
    return (ids * ((length + len(ids) - 1) // len(ids)))[:length]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--bin', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--prompt', type=int, default=None)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--ctx', type=int, default=DEFAULT_CONTEXT)
    ap.add_argument('--timeout', type=float, default=600.0)
    ap.add_argument('--ids-file')
    ap.add_argument('--slots', default='auto')
    ap.add_argument('--profile-input')
    ap.add_argument('--compare')
    ap.add_argument('--stats', action='store_true')
    ap.add_argument('--env', action='append', default=[])
    a = ap.parse_args()
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        ap.error('--timeout must be finite and positive')
    if a.ctx <= 0:
        ap.error('--ctx must be positive')
    if a.ctx > MAX_CONTEXT:
        ap.error(f'--ctx must be at most {MAX_CONTEXT}')
    if a.n < 1:
        ap.error('decode length must be positive')
    if a.ids_file and a.prompt is not None:
        ap.error('--prompt cannot be combined with --ids-file; the fixture defines the input length')
    if a.ids_file:
        try:
            fixture_text = Path(a.ids_file).read_text()
        except OSError as exc:
            ap.error(f'--ids-file cannot be read: {exc}')
        try:
            prompt = parse_ids(fixture_text)
        except ValueError as exc:
            ap.error(f'--ids-file: {exc}')
    else:
        if a.prompt is None:
            a.prompt = DEFAULT_PROMPT
        if a.prompt < 1:
            ap.error('prompt length must be positive')
        prompt = generated_ids(a.prompt)
    if len(prompt) + a.n > a.ctx:
        ap.error(f'input length {len(prompt)} + decode {a.n} exceeds context {a.ctx}')
    out = Path(a.out)
    prefix = out / a.label
    ids_target = prefix.with_suffix('.ids')
    source_target = prefix.with_suffix('.ids-source')
    profile = prefix.with_suffix('.profile')
    log_target = prefix.with_suffix('.log')
    bin_target = prefix.with_suffix('.bin')
    json_target = prefix.with_suffix('.json')
    artifacts = (ids_target, source_target, profile, log_target, bin_target, json_target)

    def collides(source):
        source_path = Path(source)
        for artifact in artifacts:
            try:
                if source_path.exists() and artifact.exists() and os.path.samefile(source_path, artifact):
                    return True
            except OSError:
                pass
            if source_path.resolve() == artifact.resolve():
                return True
        return False

    sources = [('--ids-file', a.ids_file), ('--profile-input', a.profile_input),
               ('--compare', a.compare), ('--bin', a.bin), ('--model', a.model)]
    if not a.ids_file:
        sources.append(('token oracle', str(ORACLE_IDS)))
    for flag, source in sources:
        if source and collides(source):
            ap.error(f'{flag} must differ from every output artifact '
                     '(ids, ids-source, profile, log, bin, json)')
    if a.profile_input:
        try:
            profile_bytes = Path(a.profile_input).read_bytes()
        except OSError as exc:
            ap.error(f'--profile-input cannot be read: {exc}')
    else:
        profile_bytes = None
    env_values = []
    for pair in a.env:
        if '=' not in pair or not pair.split('=', 1)[0]:
            ap.error(f'--env must be NAME=VALUE, got {pair!r}')
        env_values.append(pair.split('=', 1))
    if not Path(a.bin).is_file():
        ap.error('--bin must be an existing file')
    out.mkdir(parents=True, exist_ok=True)
    ids_target.write_text(','.join(map(str, prompt)))
    if a.ids_file:
        source_target.write_text(fixture_text)
    profile.unlink(missing_ok=True)
    if profile_bytes is not None:
        profile.write_bytes(profile_bytes)
    forced = ([286,3515,310,262,14292,1343,1444,46,17156,32,51,48,58,4037,262,1890] * ((a.n+15)//16))[:a.n-1]
    cmd = [a.bin, '--model', a.model, '--expert-no-pin', '--expert-slots', a.slots,
           '--expert-slots-hot', '--expert-slots-profile', str(profile), '--prefill',
           '--ctx', str(a.ctx), '--vram-reserve-mib', '1536', '--tokens-file', str(prefix.with_suffix('.ids')),
           '--n', str(a.n), '--dump-full-logits', str(prefix.with_suffix('.bin')), '--counters']
    if forced:
        cmd += ['--force', ','.join(map(str, forced))]
    env = dict(os.environ, STRATA_KOLIBRI_BENCH='1', STRATA_KOLIBRI_NOPIN_STATS=str(int(a.stats)))
    for k, v in env_values:
        env[k] = v
    inherited_strata = evidence_env(os.environ)
    start = time.monotonic()
    with log_target.open('w') as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            returncode = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            print(f'engine timed out after {a.timeout:g}s; process group killed', file=sys.stderr)
            return 1
    wall = time.monotonic() - start
    assert returncode == 0, (returncode, str(log_target))
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
    result = {'label': a.label, 'command': cmd, 'env_overrides': recorded_env(env_values),
              'env_inherited_strata': inherited_strata, 'python': sys.version.split()[0],
              'wall_s': wall, 'input_tokens': len(prompt), 'context': a.ctx, 'timeout_s': a.timeout,
              'ids_source': a.ids_file, 'phases': phases,
              'logits_bytes': len(data), 'logits_floats': len(values),
              'logits_sha256': hashlib.sha256(data).hexdigest(),
              'binary_sha256': hashlib.sha256(Path(a.bin).read_bytes()).hexdigest(),
              'generated': next(x for x in text.splitlines() if x.startswith('gen:')),
              'diagnostics': [x for x in text.splitlines() if x.startswith(('NOPIN_STATS', 'NOPIN_IO')) or 'free (reserve' in x]}
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    sys.exit(main())
