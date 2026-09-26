// Tests for `zcode decide` — node:test, no external deps.
//
// Covers:
// - --help prints usage, exit 0
// - arg validation: missing --type/--state/--instructions, bad --criteria,
//   score labels must be 0..n-1, --gold must name an offered label
// - live-shaped decide flow against a stub shim (choice + --json)
// - shim down -> exit 1 with a clear stderr message (fail-open, no hang)
import assert from "node:assert/strict";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { test } from "node:test";
import type { RunContext } from "@zcode/shared-types";

import { runDecideCommand } from "./decide-command.js";

function makeCtx(argv: string[]): { ctx: RunContext; out: string[]; err: string[] } {
  const out: string[] = [];
  const err: string[] = [];
  const ctx = {
    argv,
    stdout: { write: (s: string) => { out.push(s); return true; } },
    stderr: { write: (s: string) => { err.push(s); return true; } },
    stdin: process.stdin,
  } as unknown as RunContext;
  return { ctx, out, err };
}

function stubShim(payload: unknown): Promise<{ url: string; close: () => void }> {
  return new Promise((resolve) => {
    const server = createServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(payload));
    });
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({ url: `http://127.0.0.1:${port}`, close: () => server.close() });
    });
  });
}

test("decide --help prints usage and exits 0", async () => {
  const { ctx, out } = makeCtx(["decide", "--help"]);
  const code = await runDecideCommand(ctx);
  assert.equal(code, 0);
  assert.match(out.join(""), /zcode decide --type choice/);
  assert.match(out.join(""), /--gold/);
});

test("decide requires --type/--state/--instructions", async () => {
  for (const argv of [
    ["decide"],
    ["decide", "--type", "choice"],
    ["decide", "--type", "choice", "--state", "s"],
    ["decide", "--type", "bogus", "--state", "s", "--instructions", "i"],
  ]) {
    const { ctx, err } = makeCtx(argv);
    const code = await runDecideCommand(ctx);
    assert.equal(code, 1, argv.join(" "));
    assert.ok(err.join("").length > 0, argv.join(" "));
  }
});

test("decide rejects malformed --criteria", async () => {
  const { ctx, err } = makeCtx([
    "decide", "--type", "choice", "--state", "s", "--instructions", "i",
    "--criteria", "no-equals-sign",
  ]);
  assert.equal(await runDecideCommand(ctx), 1);
  assert.match(err.join(""), /bad --criteria/);
});

test("decide rejects duplicate --criteria labels", async () => {
  const { ctx } = makeCtx([
    "decide", "--type", "choice", "--state", "s", "--instructions", "i",
    "--criteria", "a=first", "--criteria", "a=second",
  ]);
  assert.equal(await runDecideCommand(ctx), 1);
});

test("decide --type score requires 0..n-1 labels in order", async () => {
  const { ctx } = makeCtx([
    "decide", "--type", "score", "--state", "s", "--instructions", "i",
    "--criteria", "0=bad", "--criteria", "2=ok",
  ]);
  assert.equal(await runDecideCommand(ctx), 1);
});

test("decide rejects --criteria with --type noul", async () => {
  const { ctx } = makeCtx([
    "decide", "--type", "noul", "--state", "s", "--instructions", "i",
    "--criteria", "yes=sure",
  ]);
  assert.equal(await runDecideCommand(ctx), 1);
});

test("decide --gold must name an offered label", async () => {
  const { ctx } = makeCtx([
    "decide", "--type", "choice", "--state", "s", "--instructions", "i",
    "--criteria", "a=first", "--criteria", "b=second", "--gold", "zzz",
  ]);
  assert.equal(await runDecideCommand(ctx), 1);
});

test("decide answers a choice question against a stub shim", async () => {
  const shim = await stubShim({
    type: "choice",
    label: "b",
    probabilities: { a: 0.2, b: 0.8 },
    confidence: 0.8,
    latency_ms: 120,
    backend: "decider",
  });
  const saved = process.env.SYSTEMONE_SHIM_URL;
  const savedLog = process.env.ZCODE_SYSTEMONE_DECISION_LOG;
  process.env.SYSTEMONE_SHIM_URL = shim.url;
  process.env.ZCODE_SYSTEMONE_DECISION_LOG = "0";
  try {
    const { ctx, out } = makeCtx([
      "decide", "--type", "choice", "--state", "Pick one.", "--instructions", "Choose.",
      "--criteria", "a=first", "--criteria", "b=second",
    ]);
    const code = await runDecideCommand(ctx);
    assert.equal(code, 0);
    const text = out.join("");
    assert.match(text, /"b"/);
    assert.match(text, /confidence 0\.800/);
    assert.match(text, /decider/);
  } finally {
    if (saved === undefined) delete process.env.SYSTEMONE_SHIM_URL;
    else process.env.SYSTEMONE_SHIM_URL = saved;
    if (savedLog === undefined) delete process.env.ZCODE_SYSTEMONE_DECISION_LOG;
    else process.env.ZCODE_SYSTEMONE_DECISION_LOG = savedLog;
    shim.close();
  }
});

test("decide --json prints the raw answer", async () => {
  const shim = await stubShim({
    type: "noul",
    label: "yes",
    probabilities: { yes: 0.9, no: 0.1 },
    confidence: 0.9,
  });
  const saved = process.env.SYSTEMONE_SHIM_URL;
  const savedLog = process.env.ZCODE_SYSTEMONE_DECISION_LOG;
  process.env.SYSTEMONE_SHIM_URL = shim.url;
  process.env.ZCODE_SYSTEMONE_DECISION_LOG = "0";
  try {
    const { ctx, out } = makeCtx([
      "decide", "--type", "noul", "--state", "Is it up?", "--instructions", "Answer.",
      "--json",
    ]);
    const code = await runDecideCommand(ctx);
    assert.equal(code, 0);
    const parsed = JSON.parse(out.join(""));
    assert.equal(parsed.type, "noul");
    assert.equal(parsed.label, "yes");
    assert.equal(parsed.confidence, 0.9);
  } finally {
    if (saved === undefined) delete process.env.SYSTEMONE_SHIM_URL;
    else process.env.SYSTEMONE_SHIM_URL = saved;
    if (savedLog === undefined) delete process.env.ZCODE_SYSTEMONE_DECISION_LOG;
    else process.env.ZCODE_SYSTEMONE_DECISION_LOG = savedLog;
    shim.close();
  }
});

test("decide fails open with a clear message when the shim is down", async () => {
  const saved = process.env.SYSTEMONE_SHIM_URL;
  const savedLog = process.env.ZCODE_SYSTEMONE_DECISION_LOG;
  process.env.SYSTEMONE_SHIM_URL = "http://127.0.0.1:9";
  process.env.ZCODE_SYSTEMONE_DECISION_LOG = "0";
  try {
    const { ctx, err } = makeCtx([
      "decide", "--type", "noul", "--state", "s", "--instructions", "i",
      "--timeout-ms", "500",
    ]);
    const code = await runDecideCommand(ctx);
    assert.equal(code, 1);
    assert.match(err.join(""), /no answer from the SystemOne decide engine/);
  } finally {
    if (saved === undefined) delete process.env.SYSTEMONE_SHIM_URL;
    else process.env.SYSTEMONE_SHIM_URL = saved;
    if (savedLog === undefined) delete process.env.ZCODE_SYSTEMONE_DECISION_LOG;
    else process.env.ZCODE_SYSTEMONE_DECISION_LOG = savedLog;
  }
});
