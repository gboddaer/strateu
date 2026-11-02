import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync, rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';

function environment(t, values) {
  const home = mkdtempSync(join(tmpdir(), 'strata-demo-isolation-'));
  const entries = {...values, STRATA_HOME: home};
  const previous = Object.fromEntries(Object.keys(entries).map(k => [k, process.env[k]]));
  for (const [key, value] of Object.entries(entries)) {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key]; else process.env[key] = value;
    }
    rmSync(home, {recursive: true, force: true});
  });
}

test('explicit development demo never probes a real inference server', async t => {
  environment(t, {STRATA_FORCE_DEMO: '1', STRATA_PACKAGED: undefined, STRATA_REAL_ENGINE: '1'});
  const probe = t.mock.method(globalThis, 'fetch', async () => Response.json({service: 'strata', loaded: true, data: [{id: 'real-model'}]}));
  const {detectBackend} = await import('../server/inference.mjs');
  const backend = await detectBackend({force: true});
  assert.equal(backend.kind, 'demo');
  assert.equal(backend.simulated, true);
  assert.equal(probe.mock.callCount(), 0, 'demo must not contact the user\'s Ollama, Strata or other local servers');
});

test('packaged mode still forbids demo even when the development flag is set', async t => {
  environment(t, {STRATA_FORCE_DEMO: '1', STRATA_PACKAGED: '1', STRATA_REAL_ENGINE: '0'});
  t.mock.method(globalThis, 'fetch', async () => new Response('', {status: 404}));
  const {detectBackend, requiresRealEngine} = await import('../server/inference.mjs');
  assert.equal(requiresRealEngine(), true);
  assert.equal((await detectBackend({force: true})).kind, 'unavailable');
});
