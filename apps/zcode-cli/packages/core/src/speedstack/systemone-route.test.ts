/**
 * Speed Stack SystemOne routing wiring tests (Z1 effort + Z2 MCP attach policy).
 *
 * These tests cover the pure wiring helpers in speedstack/systemone-route.ts:
 * - route decision -> effort tier -> ModelOptions (reasoningLevel +
 *   maxOutputTokens)
 * - explicit session effort tier overrides the route decision
 * - malformed/unavailable route -> fail-open (no override)
 * - MCP attach policy: economy + confidence >= 0.8 prunes; lower
 *   confidence, non-economy, and missing decisions attach fully
 * - kill-switches: ZCODE_SPEEDSTACK_PRUNE=0 env and mcpPruning=false config
 *
 * The HTTP fetch path (fail-open on unreachable shim) is exercised by
 * `fetchSystemOneRouteDecision` against a guaranteed-dead port below.
 */
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Hermetic decision logging: the fetch path appends records; point it at
// /dev/null-equivalent so tests never touch the real log.
process.env.ZCODE_SYSTEMONE_DECISION_LOG = "0";

import {
  adjustEffortForUncertainty,
  applySystemOneEffortOverride,
  fetchSystemOneRouteDecision,
  isSystemOneDisabled,
  resolveEffortTierAndOptions,
  resolveMcpAttachPolicy,
  resolveSystemOneBehaviorPolicy,
  resolveSystemOneModelTarget,
  routeTierMargin,
  SYSTEMONE_PRUNE_CONFIDENCE_THRESHOLD,
  type EffortResolution,
  type SystemOneRouteDecision,
} from "./systemone-route.ts";

const TIER_MODEL = {
  optionSpecs: {
    reasoningLevel: { values: ["low", "medium", "high"] },
    maxOutputTokens: { max: 64000 },
  },
};

function route(overrides: Partial<SystemOneRouteDecision> = {}): SystemOneRouteDecision {
  return {
    tier: "economy",
    confidence: 0.8,
    effort: "low",
    task_labels: ["summarize"],
    rationale: "test",
    model_id: "test-model",
    ...overrides,
  };
}

function expectTier(resolved: EffortResolution | undefined): Extract<EffortResolution, { kind: "tier" }> {
  assert.ok(resolved, "expected a resolved effort");
  assert.equal(resolved.kind, "tier");
  return resolved;
}

test("route decision economy/low maps to low reasoning level and small output cap", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      route: route({ tier: "economy", effort: "low", confidence: 0.9 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "low");
  assert.equal(resolved.options.reasoningLevel, "low");
  // low -> LOW_TIER_MAX_OUTPUT_TOKENS (4000), capped by the model max
  assert.equal(resolved.options.maxOutputTokens, 4000);
});

test("route decision balanced/medium maps to medium reasoning level and medium budget", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      route: route({ tier: "balanced", effort: "medium", confidence: 0.9 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "medium");
  assert.equal(resolved.options.reasoningLevel, "medium");
  // medium -> MEDIUM_TIER_MAX_OUTPUT_TOKENS (12000)
  assert.equal(resolved.options.maxOutputTokens, 12000);
});

test("route decision heavy/high maps to the high budget (32000), capped by model max", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      route: route({ tier: "heavy", effort: "high", confidence: 0.9 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "high");
  assert.equal(resolved.options.reasoningLevel, "high");
  // high -> HIGH_TIER_MAX_OUTPUT_TOKENS (32000)
  assert.equal(resolved.options.maxOutputTokens, 32000);
});

test("route decision heavy/xhigh maps to the xhigh budget (64000), capped by model max", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      route: route({ tier: "heavy", effort: "xhigh", confidence: 0.95 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "xhigh");
  // xhigh -> XHIGH_TIER_MAX_OUTPUT_TOKENS (64000); model max is 64000
  assert.equal(resolved.options.maxOutputTokens, 64000);
});

test("route decision heavy/ultra maps to the model maximum (no ZCode-side cap)", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      route: route({ tier: "heavy", effort: "ultra", confidence: 0.99 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "ultra");
  assert.equal(resolved.options.maxOutputTokens, 64000);
});

test("explicit session effort tier overrides the route decision", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      config: { effortTier: "high" },
      route: route({ tier: "economy", effort: "low", confidence: 0.99 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "high");
});

test("thinkingMode pinned tier beats the route hint", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      config: { thinkingMode: "ultra" },
      route: route({ tier: "economy", effort: "low", confidence: 0.99 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "ultra");
});

test("thinkingMode off resolves to thinking-off (no tier)", () => {
  const resolved = resolveEffortTierAndOptions({
    config: { thinkingMode: "off" },
    route: route({ tier: "heavy", effort: "ultra", confidence: 0.99 }),
    model: TIER_MODEL,
  });
  assert.ok(resolved);
  assert.equal(resolved.kind, "off");
});

test("thinkingMode auto follows the route hint", () => {
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      config: { thinkingMode: "auto" },
      route: route({ tier: "balanced", effort: "medium", confidence: 0.9 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "medium");
});

test("modelRouting does not gate thinking resolution (independence)", () => {
  // Route moves the model AND thinking is pinned: both apply.
  const resolved = expectTier(
    resolveEffortTierAndOptions({
      config: { thinkingMode: "low", modelRouting: true },
      route: route({ tier: "heavy", effort: "ultra", confidence: 0.99 }),
      model: TIER_MODEL,
    }),
  );
  assert.equal(resolved.tier, "low");
  const target = resolveSystemOneModelTarget({
    route: route({ modelId: "other-model" }),
    modelRouting: true,
    currentModelId: "test-model",
  });
  assert.equal(target, "other-model");
});

test("missing route decision resolves nothing (fail-open)", () => {
  assert.equal(resolveEffortTierAndOptions({ route: undefined, model: TIER_MODEL }), undefined);
});

test("route without a usable effort resolves nothing (fail-open)", () => {
  const bad = route({ effort: "bogus" as unknown as "low" });
  assert.equal(resolveEffortTierAndOptions({ route: bad, model: TIER_MODEL }), undefined);
});

test("model target: routing off keeps the pinned model", () => {
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: "other-model" }),
      modelRouting: false,
      currentModelId: "test-model",
    }),
    undefined,
  );
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: "other-model" }),
      modelRouting: undefined,
      currentModelId: "test-model",
    }),
    undefined,
  );
});

test("model target: same model or missing model id resolves nothing", () => {
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: "test-model" }),
      modelRouting: true,
      currentModelId: "test-model",
    }),
    undefined,
  );
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: undefined }),
      modelRouting: true,
      currentModelId: "test-model",
    }),
    undefined,
  );
  assert.equal(
    resolveSystemOneModelTarget({
      route: undefined,
      modelRouting: true,
      currentModelId: "test-model",
    }),
    undefined,
  );
});

test("model target: thinkingMode never gates model routing", () => {
  // Thinking off + model routing on: the model still moves.
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: "other-model" }),
      modelRouting: true,
      currentModelId: "test-model",
    }),
    "other-model",
  );
});

test("applySystemOneEffortOverride binds the model with route options", () => {
  const bound: unknown[] = [];
  const fakeModel = {
    options: { reasoningLevel: "medium", maxOutputTokens: 64000 },
    optionSpecs: TIER_MODEL.optionSpecs,
    bind(options?: unknown) {
      bound.push(options);
      return this;
    },
  };
  const out = applySystemOneEffortOverride(
    {
      speedStackConfig: {},
      systemOneRouteValue: route({ tier: "economy", effort: "low" }),
    },
    fakeModel,
  );
  assert.equal(out, fakeModel);
  assert.equal(bound.length, 1);
  const options = bound[0] as { reasoningLevel: string; maxOutputTokens: number };
  assert.equal(options.reasoningLevel, "low");
  assert.equal(options.maxOutputTokens, 4000);
});

test("applySystemOneEffortOverride with thinkingMode off binds the off level", () => {
  const bound: unknown[] = [];
  const fakeModel = {
    options: { reasoningLevel: "high", maxOutputTokens: 64000 },
    optionSpecs: {
      reasoningLevel: { values: ["off", "low", "high"] },
      maxOutputTokens: { max: 64000 },
    },
    bind(options?: unknown) {
      bound.push(options);
      return this;
    },
  };
  const out = applySystemOneEffortOverride(
    {
      speedStackConfig: { thinkingMode: "off" },
      systemOneRouteValue: route({ tier: "heavy", effort: "ultra" }),
    },
    fakeModel,
  );
  assert.equal(out, fakeModel);
  assert.equal(bound.length, 1);
  const options = bound[0] as { reasoningLevel: string };
  assert.equal(options.reasoningLevel, "off");
});

test("applySystemOneEffortOverride with thinkingMode off and no off level leaves the model untouched", () => {
  let bindCalls = 0;
  const fakeModel = {
    options: {},
    optionSpecs: TIER_MODEL.optionSpecs,
    bind() {
      bindCalls += 1;
      return this;
    },
  };
  const out = applySystemOneEffortOverride(
    {
      speedStackConfig: { thinkingMode: "off" },
      systemOneRouteValue: route({ tier: "heavy", effort: "ultra" }),
    },
    fakeModel,
  );
  assert.equal(out, fakeModel);
  assert.equal(bindCalls, 0);
});

test("applySystemOneEffortOverride without a route returns the model untouched", () => {
  let bindCalls = 0;
  const fakeModel = {
    options: {},
    optionSpecs: TIER_MODEL.optionSpecs,
    bind() {
      bindCalls += 1;
      return this;
    },
  };
  const out = applySystemOneEffortOverride({ speedStackConfig: {} }, fakeModel);
  assert.equal(out, fakeModel);
  assert.equal(bindCalls, 0);
});

test("fetchSystemOneRouteDecision fails open on an unreachable shim", async () => {
  // Port 1 is guaranteed closed; the fetch must resolve undefined, not throw.
  const decision = await fetchSystemOneRouteDecision("hello world", {
    endpoint: "http://127.0.0.1:1/v1/systemone/route",
    timeoutMs: 250,
  });
  assert.equal(decision, undefined);
});

test("MCP attach policy: economy + confidence >= 0.8 skips MCP", () => {
  const policy = resolveMcpAttachPolicy(route({ tier: "economy", confidence: 0.8 }), {});
  assert.equal(policy.attachMcp, false);
  assert.match(policy.reason, /economy/);
});

test("MCP attach policy: economy below the confidence threshold attaches fully", () => {
  const policy = resolveMcpAttachPolicy(
    route({ tier: "economy", confidence: SYSTEMONE_PRUNE_CONFIDENCE_THRESHOLD - 0.01 }),
    {},
  );
  assert.equal(policy.attachMcp, true);
  assert.match(policy.reason, /confidence/);
});

test("MCP attach policy: non-economy tiers attach fully", () => {
  for (const tier of ["balanced", "heavy"] as const) {
    const policy = resolveMcpAttachPolicy(route({ tier, confidence: 0.99 }), {});
    assert.equal(policy.attachMcp, true, tier);
  }
});

test("MCP attach policy: missing route decision attaches fully (fail-open)", () => {
  assert.equal(resolveMcpAttachPolicy(undefined, {}).attachMcp, true);
});

test("MCP attach policy: env kill-switch disables pruning", () => {
  process.env.ZCODE_SPEEDSTACK_PRUNE = "0";
  try {
    const policy = resolveMcpAttachPolicy(route({ tier: "economy", confidence: 0.99 }), {});
    assert.equal(policy.attachMcp, true);
    assert.match(policy.reason, /kill-switch/);
  } finally {
    delete process.env.ZCODE_SPEEDSTACK_PRUNE;
  }
});

test("MCP attach policy: config kill-switch disables pruning", () => {
  const policy = resolveMcpAttachPolicy(route({ tier: "economy", confidence: 0.99 }), {
    mcpPruning: false,
  });
  assert.equal(policy.attachMcp, true);
  assert.match(policy.reason, /kill-switch/);
});

test("MCP attach policy: env var set to 1 keeps pruning live", () => {
  process.env.ZCODE_SPEEDSTACK_PRUNE = "1";
  try {
    const policy = resolveMcpAttachPolicy(route({ tier: "economy", confidence: 0.99 }), {});
    assert.equal(policy.attachMcp, false);
  } finally {
    delete process.env.ZCODE_SPEEDSTACK_PRUNE;
  }
});

test("ZCODE_SYSTEMONE=0 disables route lookups (master kill-switch)", async () => {
  process.env.ZCODE_SYSTEMONE = "0";
  try {
    assert.equal(isSystemOneDisabled(), true);
    // Even a live endpoint is never contacted when the kill-switch is set.
    const decision = await fetchSystemOneRouteDecision("summarize this", {
      endpoint: "http://127.0.0.1:8765/v1/systemone/route",
      timeoutMs: 100,
    });
    assert.equal(decision, undefined);
  } finally {
    delete process.env.ZCODE_SYSTEMONE;
  }
});

test("ZCODE_SYSTEMONE unset keeps route lookups live", async () => {
  delete process.env.ZCODE_SYSTEMONE;
  assert.equal(isSystemOneDisabled(), false);
  // Guaranteed-dead port: fail-open still yields undefined, but the lookup
  // was attempted (no kill-switch short-circuit).
  const decision = await fetchSystemOneRouteDecision("summarize this", {
    endpoint: "http://127.0.0.1:9/v1/systemone/route",
    timeoutMs: 500,
  });
  assert.equal(decision, undefined);
});

test("fetch preserves the shim's per-tier probabilities as tierScores", async () => {
  const payload = {
    route: {
      tier: "balanced",
      confidence: 0.68,
      effort: "medium",
      task_labels: ["files"],
      probabilities: { economy: 0.18, balanced: 0.54, heavy: 0.28 },
    },
  };
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const decision = await fetchSystemOneRouteDecision("rename a file", {
      endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
      timeoutMs: 2000,
    });
    assert.ok(decision, "expected a parsed decision");
    assert.deepEqual(decision.tierScores, { economy: 0.18, balanced: 0.54, heavy: 0.28 });
    assert.deepEqual(decision.taskLabels, ["files"]);
  } finally {
    server.close();
  }
});

test("fetch drops malformed probabilities but keeps the decision", async () => {
  const payload = { route: { tier: "economy", confidence: 0.9, probabilities: "nonsense" } };
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const decision = await fetchSystemOneRouteDecision("x", {
      endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
      timeoutMs: 2000,
    });
    assert.ok(decision);
    assert.equal(decision.tierScores, undefined);
  } finally {
    server.close();
  }
});

test("routeTierMargin returns the top-1 vs top-2 gap", () => {
  assert.equal(
    routeTierMargin({ tier: "balanced", confidence: 0.68, tierScores: { economy: 0.18, balanced: 0.54, heavy: 0.28 } }),
    0.54 - 0.28,
  );
  assert.equal(routeTierMargin({ tier: "balanced", confidence: 0.68 }), undefined);
  assert.equal(routeTierMargin(undefined), undefined);
  assert.equal(
    routeTierMargin({ tier: "balanced", confidence: 0.68, tierScores: { balanced: 1 } }),
    undefined,
  );
});

test("attach policy preserves tierScores/margin on the tool-level policy", () => {
  const policy = resolveMcpAttachPolicy(
    {
      tier: "balanced",
      confidence: 0.9,
      tierScores: { economy: 0.1, balanced: 0.7, heavy: 0.2 },
    },
    undefined,
    { serverNames: ["filesystem", "memory"] },
  );
  assert.deepEqual(policy.toolPolicy.tierScores, { economy: 0.1, balanced: 0.7, heavy: 0.2 });
  assert.equal(policy.toolPolicy.margin, 0.7 - 0.2);
});

// ---------------------------------------------------------------
// Phase 3: uncertain + ranked_tools decision surfaces
// ---------------------------------------------------------------

async function fetchFromPayload(payload: unknown) {
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    return await fetchSystemOneRouteDecision("do the thing", {
      endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
      timeoutMs: 2000,
    });
  } finally {
    server.close();
  }
}

test("fetch parses uncertain + ranked_tools from the shim", async () => {
  const decision = await fetchFromPayload({
    route: {
      tier: "balanced",
      confidence: 0.62,
      effort: "medium",
      uncertain: true,
      ranked_tools: [
        { id: "browserclaw", kind: "tool", relevance: 0.91 },
        { id: "weather", kind: "tool", relevance: 0.4 },
      ],
    },
  });
  assert.ok(decision);
  assert.equal(decision.uncertain, true);
  assert.deepEqual(decision.rankedTools, [
    { id: "browserclaw", kind: "tool", relevance: 0.91 },
    { id: "weather", kind: "tool", relevance: 0.4 },
  ]);
});

test("fetch leaves uncertain/rankedTools undefined on old-schema shims", async () => {
  const decision = await fetchFromPayload({
    route: { tier: "balanced", confidence: 0.7, effort: "medium" },
  });
  assert.ok(decision);
  assert.equal(decision.uncertain, undefined);
  assert.equal(decision.rankedTools, undefined);
});

test("fetch drops malformed ranked_tools entries but keeps good ones", async () => {
  const decision = await fetchFromPayload({
    route: {
      tier: "balanced",
      confidence: 0.7,
      ranked_tools: [
        { id: "", relevance: 0.9 },
        { id: "browserclaw", relevance: "high" },
        { id: "weather", relevance: 0.4 },
        "nonsense",
      ],
    },
  });
  assert.ok(decision);
  assert.deepEqual(decision.rankedTools, [{ id: "weather", relevance: 0.4 }]);
});

test("MCP attach policy: uncertain route never prunes, even at economy 0.95", () => {
  const policy = resolveMcpAttachPolicy(
    { tier: "economy", confidence: 0.95, uncertain: true },
    undefined,
    { serverNames: ["filesystem", "memory"] },
  );
  assert.equal(policy.attachMcp, true);
  assert.equal(policy.toolPolicy.mode, "all");
  assert.ok(policy.reason.includes("uncertain"));
});

test("MCP attach policy: certain economy 0.95 still prunes (unchanged)", () => {
  const policy = resolveMcpAttachPolicy(
    { tier: "economy", confidence: 0.95 },
    undefined,
    { serverNames: ["filesystem", "memory"] },
  );
  assert.equal(policy.attachMcp, false);
  assert.equal(policy.toolPolicy.mode, "none");
});

test("adjustEffortForUncertainty bumps route-driven effort one level", () => {
  const route: SystemOneRouteDecision = { tier: "balanced", confidence: 0.6, uncertain: true };
  const bumped = adjustEffortForUncertainty({ kind: "tier", tier: "low" }, route, { thinkingMode: "auto" });
  assert.equal(bumped.bumped, true);
  assert.equal(bumped.fromTier, "low");
  assert.equal(bumped.toTier, "medium");
  const bumpedHigh = adjustEffortForUncertainty({ kind: "tier", tier: "high" }, route, undefined);
  assert.equal(bumpedHigh.toTier, "xhigh");
});

test("adjustEffortForUncertainty never touches explicit pins", () => {
  const route: SystemOneRouteDecision = { tier: "balanced", confidence: 0.6, uncertain: true };
  assert.equal(
    adjustEffortForUncertainty({ kind: "tier", tier: "low" }, route, { thinkingMode: "high" }).bumped,
    false,
  );
  assert.equal(
    adjustEffortForUncertainty({ kind: "tier", tier: "low" }, route, { effortTier: "low" }).bumped,
    false,
  );
  assert.equal(
    adjustEffortForUncertainty({ kind: "off" }, route, undefined).bumped,
    false,
  );
  assert.equal(
    adjustEffortForUncertainty(
      { kind: "tier", tier: "low" },
      { tier: "balanced", confidence: 0.6 },
      undefined,
    ).bumped,
    false,
  );
  // ultra stays ultra: no phantom bump
  assert.equal(
    adjustEffortForUncertainty({ kind: "tier", tier: "ultra" }, route, undefined).bumped,
    false,
  );
});

test("resolveEffortTierAndOptions bumps the tier and marks the bump on uncertain routes", () => {
  const resolved = resolveEffortTierAndOptions({
    route: { tier: "balanced", confidence: 0.6, effort: "low", uncertain: true },
    config: { thinkingMode: "auto" },
    model: TIER_MODEL,
  });
  assert.ok(resolved && resolved.kind === "tier");
  assert.equal(resolved.tier, "medium");
  assert.deepEqual(resolved.uncertainBump, { from: "low", to: "medium" });
});

test("resolveEffortTierAndOptions leaves pinned thinking alone on uncertain routes", () => {
  const resolved = resolveEffortTierAndOptions({
    route: { tier: "balanced", confidence: 0.6, effort: "low", uncertain: true },
    config: { thinkingMode: "low" },
    model: TIER_MODEL,
  });
  assert.ok(resolved && resolved.kind === "tier");
  assert.equal(resolved.tier, "low");
  assert.equal(resolved.uncertainBump, undefined);
});

test("resolveSystemOneBehaviorPolicy applies the uncertain bump to tier, policy, and budgets", () => {
  const resolution = resolveSystemOneBehaviorPolicy(
    {
      speedStackConfig: { thinkingMode: "auto" },
      systemOneRouteValue: { tier: "balanced", confidence: 0.6, effort: "low", uncertain: true },
    },
    {},
  );
  assert.equal(resolution.tier, "medium");
  assert.equal(resolution.policy?.tier, "medium");
  assert.deepEqual(resolution.uncertainBump, { from: "low", to: "medium" });
});

test("resolveSystemOneBehaviorPolicy without uncertainty is unchanged", () => {
  const resolution = resolveSystemOneBehaviorPolicy(
    {
      speedStackConfig: { thinkingMode: "auto" },
      systemOneRouteValue: { tier: "balanced", confidence: 0.6, effort: "low" },
    },
    {},
  );
  assert.equal(resolution.tier, "low");
  assert.equal(resolution.uncertainBump, undefined);
});

test("fetch parses ranked_models in best-value order", async () => {
  const payload = {
    route: {
      tier: "balanced",
      confidence: 0.68,
      ranked_models: [
        { model_id: "model-a", tier: "balanced", utility: 0.91, quality: 0.8, cost: 0.2 },
        { model_id: "model-b", tier: "economy", utility: 0.42 },
        { not_a_model: true },
        { model_id: "  ", tier: "heavy", utility: 0.5 },
        { model_id: "model-c", tier: "heavy", utility: "high" },
      ],
    },
  };
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const decision = await fetchSystemOneRouteDecision("pick a model", {
      endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
      timeoutMs: 2000,
    });
    assert.ok(decision);
    assert.equal(decision.rankedModels?.length, 2);
    assert.equal(decision.rankedModels?.[0]?.modelId, "model-a");
    assert.equal(decision.rankedModels?.[0]?.utility, 0.91);
    assert.equal(decision.rankedModels?.[1]?.modelId, "model-b");
  } finally {
    server.close();
  }
});

test("fetch parses jeff1_second_opinion (decider-backed advisory)", async () => {
  const payload = {
    route: {
      tier: "balanced",
      confidence: 0.52,
      uncertain: true,
      jeff1_second_opinion: {
        tier: "heavy",
        confidence: 0.61,
        agree: false,
        rationale: "task mentions long-context synthesis",
      },
    },
  };
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const decision = await fetchSystemOneRouteDecision("tricky task", {
      endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
      timeoutMs: 2000,
    });
    assert.ok(decision);
    assert.deepEqual(decision.secondOpinion, {
      tier: "heavy",
      agree: false,
      confidence: 0.61,
      rationale: "task mentions long-context synthesis",
    });
  } finally {
    server.close();
  }
});

test("fetch ignores a malformed jeff1_second_opinion", async () => {
  const payload = { route: { tier: "economy", confidence: 0.9, jeff1_second_opinion: { agree: true } } };
  const server = createServer((_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify(payload));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    const decision = await fetchSystemOneRouteDecision("x", {
      endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
      timeoutMs: 2000,
    });
    assert.ok(decision);
    assert.equal(decision.secondOpinion, undefined);
  } finally {
    server.close();
  }
});

test("model target prefers the ranked best-value model over route.model_id", () => {
  const target = resolveSystemOneModelTarget({
    route: route({
      modelId: "routed-model",
      rankedModels: [
        { modelId: "best-value-model", tier: "balanced", utility: 0.95 },
        { modelId: "second", tier: "economy", utility: 0.5 },
      ],
    }),
    modelRouting: true,
    currentModelId: "test-model",
  });
  assert.equal(target, "best-value-model");
});

test("model target falls back to route.model_id when the ranking is absent", () => {
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: "routed-model", rankedModels: undefined }),
      modelRouting: true,
      currentModelId: "test-model",
    }),
    "routed-model",
  );
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({ modelId: "routed-model", rankedModels: [] }),
      modelRouting: true,
      currentModelId: "test-model",
    }),
    "routed-model",
  );
});

test("model target skips the retarget when the ranked pick is already loaded", () => {
  assert.equal(
    resolveSystemOneModelTarget({
      route: route({
        modelId: "routed-model",
        rankedModels: [{ modelId: "test-model", tier: "economy", utility: 0.9 }],
      }),
      modelRouting: true,
      currentModelId: "test-model",
    }),
    undefined,
  );
});

test("fetch appends a route decision record to the log path", async () => {
  const dir = mkdtempSync(join(tmpdir(), "zcode-decision-log-"));
  const logPath = join(dir, "records.jsonl");
  process.env.ZCODE_SYSTEMONE_DECISION_LOG = logPath;
  try {
    const payload = {
      route: {
        tier: "balanced",
        confidence: 0.68,
        probabilities: { economy: 0.18, balanced: 0.54, heavy: 0.28 },
        ranked_models: [{ model_id: "model-a", tier: "balanced", utility: 0.91 }],
      },
    };
    const server = createServer((_req, res) => {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(payload));
    });
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    try {
      const address = server.address();
      assert.ok(address && typeof address === "object");
      const decision = await fetchSystemOneRouteDecision("summarize this doc", {
        endpoint: `http://127.0.0.1:${address.port}/v1/systemone/route`,
        timeoutMs: 2000,
      });
      assert.ok(decision);
    } finally {
      server.close();
    }
    const lines = readFileSync(logPath, "utf8").trim().split("\n");
    assert.equal(lines.length, 1);
    const record = JSON.parse(lines[0]);
    assert.equal(record.kind, "route");
    assert.equal(record.tier, "balanced");
    assert.equal(record.confidence, 0.68);
    assert.equal(record.taskChars, "summarize this doc".length);
    assert.ok(Math.abs(record.margin - (0.54 - 0.28)) < 1e-9);
    assert.deepEqual(record.rankedModels, [
      { modelId: "model-a", tier: "balanced", utility: 0.91 },
    ]);
    assert.ok(typeof record.ts === "string");
  } finally {
    process.env.ZCODE_SYSTEMONE_DECISION_LOG = "0";
  }
});
