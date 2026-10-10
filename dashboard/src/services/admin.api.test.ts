import { test } from 'node:test';
import assert from 'node:assert/strict';
import { keyProxyApi, translateApi } from './admin.api.ts';

async function capture(run: () => Promise<unknown>): Promise<{ url: string; init?: RequestInit }> {
  const realFetch = globalThis.fetch;
  let seen: { url: string; init?: RequestInit } = { url: '' };
  globalThis.fetch = async (url, init) => {
    seen = { url: String(url), init };
    return new Response('[]', { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  try {
    await run();
  } finally {
    globalThis.fetch = realFetch;
  }
  return seen;
}

test('updateConfig does not send the server-computed apiKeySet mask back', async () => {
  const { url, init } = await capture(() =>
    translateApi.updateConfig({ llmModel: 'm', apiKeySet: true }));
  assert.equal(url, '/api/translate/config');
  assert.equal(init?.method, 'PUT');
  assert.deepEqual(JSON.parse(String(init?.body)), { llmModel: 'm' });
});

test('keyProxyApi.remove encodes the provider path segment', async () => {
  const { url, init } = await capture(() => keyProxyApi.remove('a/b c', 2));
  assert.equal(url, '/api/keyproxy/keys/a%2Fb%20c/2');
  assert.equal(init?.method, 'DELETE');
});
