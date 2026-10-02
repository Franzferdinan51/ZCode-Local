import assert from "node:assert/strict";
import test from "node:test";
import {
  heuristicAutoRouteScorer,
  suggestRoute,
  suggestRouteAsync,
  type AutoRouteCandidate,
} from "../src/auto-router.js";
import {
  buildRouteAgreement,
  buildRouteChoicePrompt,
  ML_ROUTE_DEFAULT_ENDPOINTS,
  routeAnswerToScored,
  SYSTEMONE_DECIDE_BACKENDS,
  SYSTEMONE_MAX_OPTIONS,
  SYSTEMONE_STATE_CHARS,
  validateRouteChoiceAnswer,
} from "../src/systemone-scorer.js";

function candidate(overrides: Partial<AutoRouteCandidate> = {}): AutoRouteCandidate {
  return {
    providerId: "p",
    modelId: "m",
    order: 0,
    enabled: true,
    accessType: "api-key",
    consentGranted: false,
    contextWindow: 200_000,
    supportsImage: false,
    supportsVideo: false,
    supportsAudio: false,
    supportsPdf: false,
    supportsToolCall: true,
    supportsJsonSchemaOutput: false,
    ...overrides,
  };
}

test("suggestRouteAsync matches suggestRoute ranking for the same scores", async () => {
  const candidates = [
    candidate({ providerId: "a", modelId: "chat", order: 0, supportsToolCall: false }),
    candidate({ providerId: "b", modelId: "code", order: 1, supportsToolCall: true }),
  ];
  const signals = { textSample: "Fix the crash:\n```\nTypeError: x\n```" };
  const sync = suggestRoute(candidates, signals);
  const asyncResult = await suggestRouteAsync(candidates, signals, async (c, s) =>
    heuristicAutoRouteScorer(c, s),
  );
  assert.deepEqual(asyncResult, sync);
});

test("buildRouteChoicePrompt maps survivors to single-letter aliases", () => {
  const survivors = [
    candidate({ providerId: "a", modelId: "chat", supportsToolCall: false }),
    candidate({ providerId: "b", modelId: "code", supportsToolCall: true }),
  ];
  const prompt = buildRouteChoicePrompt(survivors, {
    textSample: "x".repeat(SYSTEMONE_STATE_CHARS + 100),
    needsTools: true,
  });
  assert.ok(prompt);
  assert.equal(prompt.request.questions[0].id, "route");
  assert.deepEqual(
    prompt.request.questions[0].criteria.map((c) => c.id),
    ["A", "B"],
  );
  assert.match(prompt.request.questions[0].criteria[1].description, /b\/code/);
  assert.match(prompt.request.questions[0].criteria[1].description, /tools/);
  assert.equal(prompt.request.state.length, SYSTEMONE_STATE_CHARS);
  assert.match(prompt.request.questions[0].instructions, /tool calls/);
  assert.deepEqual(prompt.aliasToIndex, { A: 0, B: 1 });
});

test("buildRouteChoicePrompt refuses degenerate option counts", () => {
  assert.equal(buildRouteChoicePrompt([], {}), null);
  assert.equal(buildRouteChoicePrompt([candidate()], {}), null);
  const tooMany = Array.from({ length: SYSTEMONE_MAX_OPTIONS + 1 }, (_, order) =>
    candidate({ order, modelId: `m${order}` }),
  );
  assert.equal(buildRouteChoicePrompt(tooMany, {}), null);
});

test("validateRouteChoiceAnswer accepts a well-formed jev-style answer", () => {
  const survivors = [candidate({ modelId: "a" }), candidate({ modelId: "b" })];
  const prompt = buildRouteChoicePrompt(survivors, { textSample: "hi" });
  assert.ok(prompt);
  const answer = validateRouteChoiceAnswer(prompt, {
    answers: [{ choice: "B", probabilities: { A: 0.3, B: 0.7 }, confidence: 0.7 }],
  });
  assert.deepEqual(answer, {
    choice: "B",
    probabilities: { A: 0.3, B: 0.7 },
    confidence: 0.7,
  });
});

test("validateRouteChoiceAnswer rejects malformed answers", () => {
  const survivors = [candidate({ modelId: "a" }), candidate({ modelId: "b" })];
  const prompt = buildRouteChoicePrompt(survivors, { textSample: "hi" });
  assert.ok(prompt);
  const bad = [
    null,
    {},
    { answers: [] },
    // Unknown choice id.
    { answers: [{ choice: "Z", probabilities: { A: 0.5, B: 0.5 }, confidence: 0.5 }] },
    // Coverage mismatch.
    { answers: [{ choice: "A", probabilities: { A: 1 }, confidence: 1 }] },
    // Mass does not sum to 1.
    { answers: [{ choice: "A", probabilities: { A: 0.5, B: 0.1 }, confidence: 0.5 }] },
    // Choice is not the argmax.
    { answers: [{ choice: "A", probabilities: { A: 0.4, B: 0.6 }, confidence: 0.4 }] },
    // Non-numeric probability.
    { answers: [{ choice: "A", probabilities: { A: "high", B: 0.5 }, confidence: 0.5 }] },
  ];
  for (const raw of bad) {
    assert.equal(validateRouteChoiceAnswer(prompt, raw), null, JSON.stringify(raw));
  }
});

test("routeAnswerToScored gates on winner probability and scales scores", () => {
  const survivors = [candidate({ modelId: "a" }), candidate({ modelId: "b" })];
  const prompt = buildRouteChoicePrompt(survivors, { textSample: "hi" });
  assert.ok(prompt);
  const weak = validateRouteChoiceAnswer(prompt, {
    answers: [{ choice: "A", probabilities: { A: 0.34, B: 0.66 }, confidence: 0.34 }],
  });
  // Choice must be argmax, so craft a genuinely weak winner instead.
  assert.equal(weak, null);
  const uncertain = validateRouteChoiceAnswer(prompt, {
    answers: [{ choice: "B", probabilities: { A: 0.49, B: 0.51 }, confidence: 0.51 }],
  });
  assert.ok(uncertain);
  assert.equal(routeAnswerToScored(survivors, prompt, uncertain, 0.6), null);
  const confident = validateRouteChoiceAnswer(prompt, {
    answers: [{ choice: "B", probabilities: { A: 0.2, B: 0.8 }, confidence: 0.8 }],
  });
  assert.ok(confident);
  const scored = routeAnswerToScored(survivors, prompt, confident);
  assert.ok(scored);
  assert.equal(scored[1].score, 8);
  assert.match(scored[1].reasons[0], /80% confidence/);
  assert.deepEqual(scored[0].reasons, []);
});

test("buildRouteAgreement compares ML and heuristic picks", () => {
  const survivors = [
    candidate({ providerId: "a", modelId: "x" }),
    candidate({ providerId: "b", modelId: "y" }),
  ];
  const prompt = buildRouteChoicePrompt(survivors, { textSample: "hi" });
  assert.ok(prompt);
  const answer = validateRouteChoiceAnswer(prompt, {
    answers: [{ choice: "B", probabilities: { A: 0.1, B: 0.9 }, confidence: 0.9 }],
  });
  assert.ok(answer);
  assert.deepEqual(
    buildRouteAgreement({ providerId: "b", modelId: "y" }, answer, survivors, prompt),
    { agree: true, heuristicPick: "b/y", mlPick: "b/y", mlConfidence: 0.9 },
  );
  assert.deepEqual(
    buildRouteAgreement({ providerId: "a", modelId: "x" }, answer, survivors, prompt),
    { agree: false, heuristicPick: "a/x", mlPick: "b/y", mlConfidence: 0.9 },
  );
});

test("shim contract: :8765 primary, jeff-1 passthrough, decider|fallback backends", () => {
  assert.equal(ML_ROUTE_DEFAULT_ENDPOINTS["systemone"], "http://127.0.0.1:8765/v1/systemone");
  assert.equal(ML_ROUTE_DEFAULT_ENDPOINTS["jeff-1"], "http://127.0.0.1:8079/v1/systemone");
  assert.deepEqual([...SYSTEMONE_DECIDE_BACKENDS], ["decider", "fallback"]);
});
