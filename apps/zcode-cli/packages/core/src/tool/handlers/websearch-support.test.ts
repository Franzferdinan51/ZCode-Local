/**
 * Local SearXNG search-fallback tests (see ./websearch-support.ts).
 *
 * The fallback keeps WebSearch working for models without native search
 * (local Ollama / LM Studio builds). Fail-open: a down/misbehaving
 * SearXNG returns undefined, never throws.
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";

import {
  fetchSearXngResult,
  resolveSearXngBaseUrl,
  SEARXNG_DEFAULT_BASE_URL,
} from "./websearch-support.ts";

test("resolveSearXngBaseUrl honors ZCODE_SEARXNG_URL and trims slashes", () => {
  assert.equal(resolveSearXngBaseUrl({}), SEARXNG_DEFAULT_BASE_URL);
  assert.equal(
    resolveSearXngBaseUrl({ ZCODE_SEARXNG_URL: "http://x:9999///" }),
    "http://x:9999",
  );
  assert.equal(
    resolveSearXngBaseUrl({ ZCODE_SEARXNG_URL: "  " }),
    SEARXNG_DEFAULT_BASE_URL,
  );
});

test("fetchSearXngResult shapes hits as sources plus a link summary", async () => {
  const payload = {
    results: [
      { url: "https://a.example/x", title: "X", content: "snippet x" },
      { url: "https://b.example/y", title: "", content: "snippet y" },
      { url: "", title: "skipped" },
      null,
    ],
  };
  let seenPath = "";
  const server = createServer((req, res) => {
    seenPath = req.url ?? "";
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const result = await fetchSearXngResult("some query", {
      baseUrl: `http://127.0.0.1:${address.port}`,
      timeoutMs: 2000,
    });
    assert.ok(result, "expected a result");
    assert.ok(seenPath.includes("format=json"), `path was ${seenPath}`);
    assert.ok(seenPath.includes("q=some%20query"), `path was ${seenPath}`);
    assert.equal(result.sources?.length, 2);
    assert.equal(result.sources?.[0]?.url, "https://a.example/x");
    assert.equal(result.sources?.[0]?.title, "X");
    assert.equal(result.sources?.[0]?.sourceType, "url");
    // Empty titles fall back to the URL.
    assert.equal(result.sources?.[1]?.title, "https://b.example/y");
    assert.ok(result.text.includes("[X](https://a.example/x)"));
    assert.ok(result.text.includes("snippet x"));
  } finally {
    server.close();
  }
});

test("fetchSearXngResult is fail-open: down server, bad status, bad body", async () => {
  // Nothing listens here: connection refused, fast.
  assert.equal(
    await fetchSearXngResult("x", {
      baseUrl: "http://127.0.0.1:1",
      timeoutMs: 1000,
    }),
    undefined,
  );
  assert.equal(await fetchSearXngResult("  "), undefined);

  const server = createServer((req, res) => {
    if (req.url?.includes("q=badstatus")) {
      res.writeHead(500);
      res.end("nope");
    } else if (req.url?.includes("q=badbody")) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ results: "nonsense" }));
    } else {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ results: [{ url: "" }] }));
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const baseUrl = `http://127.0.0.1:${address.port}`;
    assert.equal(
      await fetchSearXngResult("badstatus", { baseUrl, timeoutMs: 2000 }),
      undefined,
    );
    assert.equal(
      await fetchSearXngResult("badbody", { baseUrl, timeoutMs: 2000 }),
      undefined,
    );
    assert.equal(
      await fetchSearXngResult("nohits", { baseUrl, timeoutMs: 2000 }),
      undefined,
    );
  } finally {
    server.close();
  }
});
