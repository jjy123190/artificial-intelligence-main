import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import ts from "typescript";
const source = await readFile(
  new URL("../src/api.ts", import.meta.url),
  "utf8",
);
const code = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.ESNext,
    target: ts.ScriptTarget.ES2022,
  },
}).outputText;
const { request, ApiError } = await import(
  "data:text/javascript;base64," + Buffer.from(code).toString("base64")
);
const originalFetch = globalThis.fetch;
let checks = 0;
async function rejectsCall(fetcher, predicate, options = {}) {
  globalThis.fetch = fetcher;
  await assert.rejects(request("/test", options), predicate);
  checks++;
}
try {
  globalThis.fetch = async (_path, init) => {
    assert.equal(init.headers.get("Content-Type"), "application/json");
    assert.equal(init.headers.get("X-Test"), "kept");
    return new Response(JSON.stringify({ ok: true }), {
      headers: { "Content-Type": "application/json" },
    });
  };
  assert.deepEqual(
    await request("/test", {
      body: "{}",
      method: "POST",
      headers: new Headers({ "X-Test": "kept" }),
    }),
    { ok: true },
  );
  checks++;
  await rejectsCall(
    async () => {
      throw new TypeError("network down");
    },
    (e) => e instanceof ApiError && e.code === "network",
  );
  await rejectsCall(
    async () => new Response("<html>oops</html>"),
    (e) => e.code === "invalid_response",
  );
  await rejectsCall(
    async () => new Response("null", { status: 502 }),
    (e) => e instanceof ApiError && e.status === 502,
  );
  await rejectsCall(
    async () =>
      new Response(
        JSON.stringify({ error: { message: "已有任务", code: "busy" } }),
        { status: 429 },
      ),
    (e) => e.message === "已有任务" && e.code === "busy",
  );
  await rejectsCall(
    async () =>
      new Response(
        JSON.stringify({ error: { message: { untrusted: true } } }),
        { status: 500 },
      ),
    (e) =>
      typeof e.message === "string" && !e.message.includes("object Object"),
  );
  const hanging = (_url, { signal }) =>
    new Promise((_resolve, reject) => {
      if (signal.aborted) reject(signal.reason);
      else
        signal.addEventListener("abort", () => reject(signal.reason), {
          once: true,
        });
    });
  await rejectsCall(hanging, (e) => e.code === "timeout" && e.status === 408, {
    timeoutMs: 10,
  });
  const caller = new AbortController();
  setTimeout(() => caller.abort(), 5);
  await rejectsCall(hanging, (e) => e.name === "AbortError", {
    signal: caller.signal,
    timeoutMs: 100,
  });
  const early = new AbortController();
  early.abort();
  await rejectsCall(hanging, (e) => e.name === "AbortError", {
    signal: early.signal,
  });
  const bodyAbort = new AbortController();
  await rejectsCall(
    async () => ({
      ok: true,
      status: 200,
      json: async () => {
        bodyAbort.abort();
        throw bodyAbort.signal.reason;
      },
    }),
    (e) => e.name === "AbortError",
    { signal: bodyAbort.signal },
  );
  const afterParse = new AbortController();
  await rejectsCall(
    async () => ({
      ok: true,
      status: 200,
      json: async () => {
        afterParse.abort();
        return { late: true };
      },
    }),
    (e) => e.name === "AbortError",
    { signal: afterParse.signal },
  );
  await rejectsCall(
    async (_url, { signal }) => ({
      ok: true,
      status: 200,
      json: () =>
        new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(signal.reason), {
            once: true,
          });
        }),
    }),
    (e) => e.code === "timeout",
    { timeoutMs: 10 },
  );
  globalThis.fetch = async (_path, init) => {
    assert.equal(init.headers.get("Content-Type"), "text/plain");
    return new Response("{}");
  };
  await request("/test", {
    body: "plain",
    headers: { "Content-Type": "text/plain" },
  });
  checks++;
  console.log(
    checks +
      " API checks passed: headers, errors, network, bounded wait, cancellation during parsing and late responses.",
  );
} finally {
  globalThis.fetch = originalFetch;
}
