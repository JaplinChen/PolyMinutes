import { test } from 'node:test';
import assert from 'node:assert/strict';
import { API_BASE_URL, request } from './http.ts';
import { appApi } from './app.api.ts';

// Importing this module at all is half the point: `import.meta.env` is a Vite construct, and the
// bare `import.meta.env.VITE_API_URL` read ran at import time, so under `node --test` it threw
// before any assertion in any test that reached the service layer. These fail if that returns.

test('the service layer imports under plain node', () => {
  assert.equal(typeof request, 'function');
  assert.equal(typeof appApi.sessions, 'function');
});

test('with no VITE_API_URL set, the API is same-origin /api', () => {
  assert.equal(API_BASE_URL, '/api');
});

test('a failed request surfaces the FastAPI detail, not a generic HTTP status', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: 'lines must be between 1 and 20' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  try {
    await assert.rejects(request('/display'), (err: Error & { status?: number }) => {
      assert.equal(err.message, 'lines must be between 1 and 20');
      assert.equal(err.status, 400);
      return true;
    });
  } finally {
    globalThis.fetch = realFetch;
  }
});

async function withFetch<T>(stub: typeof fetch, run: () => Promise<T>): Promise<T> {
  const realFetch = globalThis.fetch;
  globalThis.fetch = stub;
  try {
    return await run();
  } finally {
    globalThis.fetch = realFetch;
  }
}

test('a 204 resolves to undefined without parsing a body', async () => {
  const result = await withFetch(async () => new Response(null, { status: 204 }), () => request('/x'));
  assert.equal(result, undefined);
});

test('a non-JSON error body falls back to HTTP <status>', async () => {
  await withFetch(
    async () => new Response('<html>bad gateway</html>', { status: 502 }),
    () => assert.rejects(request('/x'), (err: Error & { status?: number }) => {
      assert.equal(err.message, 'HTTP 502');
      assert.equal(err.status, 502);
      return true;
    }),
  );
});

test('a FormData body leaves Content-Type to fetch so the multipart boundary is set', async () => {
  let sent: HeadersInit | undefined;
  await withFetch(async (_url, init) => {
    sent = init?.headers;
    return new Response('{}', { status: 200 });
  }, () => request('/upload', { method: 'POST', body: new FormData() }));
  assert.equal(new Headers(sent).has('Content-Type'), false);
});

test('a 422 validation detail list becomes the joined messages', async () => {
  const detail = [
    { loc: ['body', 'name'], msg: 'Field required', type: 'missing' },
    { loc: ['query', 'session'], msg: 'Input should be a valid integer', type: 'int_parsing' },
  ];
  await withFetch(
    async () => new Response(JSON.stringify({ detail }), { status: 422 }),
    () => assert.rejects(request('/x'), (err: Error & { status?: number }) => {
      assert.equal(err.message, 'Field required; Input should be a valid integer');
      assert.equal(err.status, 422);
      return true;
    }),
  );
});
