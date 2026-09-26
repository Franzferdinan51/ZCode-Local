// Tests for speedstack/systemone-decide.ts — node:test, no external deps.
//
// Covers the decide client surface:
// - request building (choice criteria, state char budget)
// - response validation: choice/score/noul shapes, strict argmax +
//   probability-mass checks, fail-open on anything unexpected
// - fail-open transport: kill switch, non-200, fetch throw, empty state
// - observability: latency_ms/backend carried through
// - HTTP layer is injected (fetchImpl); no network ever.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  buildDecideChoiceRequest,
  decide,
  SYSTEMONE_DECIDE_STATE_CHARS,
  validateDecideAnswer,
  type DecideFetchImpl,
  type SystemOneDecideType,
} from "./systemone-decide.ts";

function okResponse(payload: unknown): Response {
  return {
    ok: true,
    json: async () => payload,
  } as Response;
}

function stubFetch(handler: () => Promise<Response> | Response): {
  impl: DecideFetchImpl;
  calls: Array<{ input: string; body: unknown }>;
} {
  const calls: Array<{ input: string; body: unknown }> = [];
  const impl: DecideFetchImpl = async (input, init) => {
    calls.push({
      input,
      body: init?.body ? JSON.parse(init.body as string) : undefined,
    });
    return handler();
  };
  return { impl, calls };
}

const ENV_ON: NodeJS.ProcessEnv = { ZCODE_SYSTEMONE: "1" };

test("buildDecideChoiceRequest slices state to the char budget", () => {
  const request = buildDecideChoiceRequest(
    "x".repeat(SYSTEMONE_DECIDE_STATE_CHARS + 100),
    "Pick one.",
    { a: "first", b: "second" },
  );
  assert.equal(request.state.length, SYSTEMONE_DECIDE_STATE_CHARS);
  assert.equal(request.type, "choice");
  assert.deepEqual(request.criteria, { a: "first", b: "second" });
});

test("decide parses a live-shaped choice response", async () => {
  const { impl, calls } = stubFetch(() =>
    okResponse({
      type: "choice",
      label: "a",
      probabilities: { a: 0.7111, b: 0.2889 },
      confidence: 0.4222,
      latency_ms: 200.6,
      backend: "decider",
    }),
  );
  const answer = await decide(
    {
      state: "which deployment?",
      instructions: "Pick the better deployment target.",
      criteria: { a: "canary", b: "full rollout" },
      type: "choice",
    },
    { env: ENV_ON, fetchImpl: impl },
  );
  assert.ok(answer);
  assert.equal(answer.type, "choice");
  assert.equal(answer.label, "a");
  assert.equal(answer.probabilities["a"], 0.7111);
  assert.equal(answer.confidence, 0.4222);
  assert.equal(answer.latencyMs, 200.6);
  assert.equal(answer.backend, "decider");
  assert.equal(calls.length, 1);
  assert.equal(calls[0]!.input, "http://127.0.0.1:8765/v1/systemone/decide");
  const body = calls[0]!.body as Record<string, unknown>;
  assert.equal(body["type"], "choice");
  assert.deepEqual(body["criteria"], { a: "canary", b: "full rollout" });
});

test("decide parses a noul response (no criteria sent)", async () => {
  const { impl, calls } = stubFetch(() =>
    okResponse({
      type: "noul",
      label: "yes",
      probabilities: { yes: 0.5892, no: 0.4108 },
      confidence: 0.5892,
      latency_ms: 77.4,
      backend: "decider",
    }),
  );
  const answer = await decide(
    {
      state: "run the full suite?",
      instructions: "Decide whether to run the full test suite.",
      type: "noul",
    },
    { env: ENV_ON, fetchImpl: impl },
  );
  assert.ok(answer);
  assert.equal(answer.type, "noul");
  assert.equal(answer.label, "yes");
  assert.equal(answer.confidence, 0.5892);
  const body = calls[0]!.body as Record<string, unknown>;
  assert.ok(!("criteria" in body));
});

test("decide parses a score response via the level/distribution shape", async () => {
  const { impl } = stubFetch(() =>
    okResponse({
      type: "score",
      level: "1",
      distribution: { "0": 0.4913, "1": 0.5087 },
      confidence: 0.5087,
      latency_ms: 84.9,
      backend: "decider",
    }),
  );
  const answer = await decide(
    {
      state: "rate this plan",
      instructions: "Score the plan.",
      criteria: { "0": "fast but risky", "1": "slow but safe" },
      type: "score",
    },
    { env: ENV_ON, fetchImpl: impl },
  );
  assert.ok(answer);
  assert.equal(answer.type, "score");
  assert.equal(answer.label, "1");
  assert.equal(answer.probabilities["0"], 0.4913);
});

test("validation rejects unoffered labels, bad mass, and argmax mismatch", () => {
  const type: SystemOneDecideType = "choice";
  const offered = ["a", "b"];
  const good = {
    type: "choice",
    label: "a",
    probabilities: { a: 0.7, b: 0.3 },
    confidence: 0.4,
  };
  assert.ok(validateDecideAnswer(good, type, offered));
  // unoffered label
  assert.equal(
    validateDecideAnswer({ ...good, label: "c" }, type, offered),
    null,
  );
  // mass doesn't sum to 1
  assert.equal(
    validateDecideAnswer(
      { ...good, probabilities: { a: 0.7, b: 0.2 } },
      type,
      offered,
    ),
    null,
  );
  // argmax mismatch
  assert.equal(
    validateDecideAnswer(
      { ...good, label: "b", probabilities: { a: 0.7, b: 0.3 } },
      type,
      offered,
    ),
    null,
  );
  // missing distribution key
  assert.equal(
    validateDecideAnswer(
      { ...good, probabilities: { a: 1 } },
      type,
      offered,
    ),
    null,
  );
  // score requires the level/distribution shape
  assert.equal(
    validateDecideAnswer(good, "score", offered),
    null,
  );
  assert.ok(
    validateDecideAnswer(
      {
        type: "score",
        level: "0",
        distribution: { "0": 0.6, "1": 0.4 },
        confidence: 0.6,
      },
      "score",
      ["0", "1"],
    ),
  );
  // garbage
  assert.equal(validateDecideAnswer(null, type, offered), null);
  assert.equal(validateDecideAnswer({}, type, offered), null);
});

test("decide fails open: kill switch, empty state, non-200, throw", async () => {
  const request = {
    state: "s",
    instructions: "i",
    criteria: { a: "A", b: "B" },
    type: "choice" as const,
  };
  let called = false;
  const counting = stubFetch(() => {
    called = true;
    return okResponse({});
  });
  // kill switch: never touches the network
  assert.equal(
    await decide(request, {
      env: { ZCODE_SYSTEMONE: "0" },
      fetchImpl: counting.impl,
    }),
    undefined,
  );
  assert.equal(called, false);
  // empty state
  assert.equal(
    await decide(
      { ...request, state: "  " },
      { env: ENV_ON, fetchImpl: counting.impl },
    ),
    undefined,
  );
  // non-200
  const nonOk = stubFetch(
    () => ({ ok: false, json: async () => ({}) }) as Response,
  );
  assert.equal(
    await decide(request, { env: ENV_ON, fetchImpl: nonOk.impl }),
    undefined,
  );
  // fetch throws: never propagates
  const throwing: DecideFetchImpl = async () => {
    throw new Error("boom");
  };
  assert.equal(
    await decide(request, { env: ENV_ON, fetchImpl: throwing }),
    undefined,
  );
  // invalid response body: fail-open
  const invalid = stubFetch(() => okResponse({ nonsense: true }));
  assert.equal(
    await decide(request, { env: ENV_ON, fetchImpl: invalid.impl }),
    undefined,
  );
});

test("decide accepts an endpoint override", async () => {
  const { impl, calls } = stubFetch(() =>
    okResponse({
      type: "choice",
      label: "b",
      probabilities: { a: 0.2, b: 0.8 },
      confidence: 0.6,
    }),
  );
  const answer = await decide(
    {
      state: "s",
      instructions: "i",
      criteria: { a: "A", b: "B" },
      type: "choice",
    },
    {
      env: ENV_ON,
      fetchImpl: impl,
      endpoint: "http://127.0.0.1:9999/v1/systemone/decide",
    },
  );
  assert.ok(answer);
  assert.equal(answer.label, "b");
  assert.equal(calls[0]!.input, "http://127.0.0.1:9999/v1/systemone/decide");
});
