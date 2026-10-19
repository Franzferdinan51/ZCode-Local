// Tests for speedstack/effort-tiers.ts — node:test, no external deps.
import test from "node:test";
import assert from "node:assert/strict";
import {
  bumpEffortTier,
  effortTierToModelOptions,
  findThinkingOffLevel,
  normalizeSpeedStackSessionConfig,
  parseEffortTier,
  parseThinkingMode,
  resolveEffectiveEffortTier,
  resolveThinkingTier,
} from "./effort-tiers.ts";

const MODEL = {
  optionSpecs: {
    reasoningLevel: { values: ["low", "medium", "high"] },
    maxOutputTokens: { max: 32_000 },
  },
};

test("parseEffortTier accepts all five tiers case-insensitively", () => {
  assert.equal(parseEffortTier("low"), "low");
  assert.equal(parseEffortTier("MEDIUM"), "medium");
  assert.equal(parseEffortTier(" High "), "high");
  assert.equal(parseEffortTier("XHigh"), "xhigh");
  assert.equal(parseEffortTier("ULTRA"), "ultra");
});

test("parseEffortTier fails open to undefined", () => {
  assert.equal(parseEffortTier("bogus"), undefined);
  assert.equal(parseEffortTier(""), undefined);
  assert.equal(parseEffortTier(undefined), undefined);
  assert.equal(parseEffortTier(42), undefined);
});

test("parseThinkingMode accepts auto/off/tiers case-insensitively", () => {
  assert.equal(parseThinkingMode("auto"), "auto");
  assert.equal(parseThinkingMode("OFF"), "off");
  assert.equal(parseThinkingMode("Xhigh"), "xhigh");
  assert.equal(parseThinkingMode("bogus"), undefined);
  assert.equal(parseThinkingMode(undefined), undefined);
});

test("effortTierToModelOptions maps tiers onto the model's own specs", () => {
  assert.deepEqual(effortTierToModelOptions(MODEL, "low"), {
    reasoningLevel: "low",
    maxOutputTokens: 4_000,
  });
  assert.deepEqual(effortTierToModelOptions(MODEL, "medium"), {
    reasoningLevel: "medium",
    maxOutputTokens: 12_000,
  });
  assert.deepEqual(effortTierToModelOptions(MODEL, "high"), {
    reasoningLevel: "high",
    maxOutputTokens: 32_000,
  });
});

test("effortTierToModelOptions budgets: high 32k, xhigh 64k, ultra model max", () => {
  const big = {
    optionSpecs: {
      reasoningLevel: { values: ["a", "b", "c", "d", "e"] },
      maxOutputTokens: { max: 128_000 },
    },
  };
  assert.equal(effortTierToModelOptions(big, "high").maxOutputTokens, 32_000);
  assert.equal(effortTierToModelOptions(big, "xhigh").maxOutputTokens, 64_000);
  assert.equal(effortTierToModelOptions(big, "ultra").maxOutputTokens, 128_000);
  // xhigh spreads to a distinct upper-half level on 5-level models.
  assert.equal(effortTierToModelOptions(big, "low").reasoningLevel, "a");
  assert.equal(effortTierToModelOptions(big, "medium").reasoningLevel, "c");
  assert.equal(effortTierToModelOptions(big, "high").reasoningLevel, "e");
  assert.equal(effortTierToModelOptions(big, "xhigh").reasoningLevel, "d");
  assert.equal(effortTierToModelOptions(big, "ultra").reasoningLevel, "e");
});

test("effortTierToModelOptions respects a smaller model max", () => {
  const small = {
    optionSpecs: {
      reasoningLevel: { values: ["off", "on"] },
      maxOutputTokens: { max: 2_000 },
    },
  };
  const options = effortTierToModelOptions(small, "high");
  assert.equal(options.maxOutputTokens, 2_000);
  assert.equal(options.reasoningLevel, "on");
  const medium = effortTierToModelOptions(small, "medium");
  assert.equal(medium.reasoningLevel, "on"); // middle of two values
});

test("resolveEffectiveEffortTier prefers explicit config over route hint", () => {
  assert.equal(resolveEffectiveEffortTier({ effortTier: "low" }, "high"), "low");
  assert.equal(resolveEffectiveEffortTier({}, "high"), "high");
  assert.equal(resolveEffectiveEffortTier(undefined, undefined), undefined);
});

test("normalizeSpeedStackSessionConfig keeps only valid fields", () => {
  const config = normalizeSpeedStackSessionConfig({
    effortTier: "HIGH",
    modelRouting: true,
    thinkingMode: "Auto",
    planThenExecute: true,
    mcpServerAllowlist: ["fs", "", 42],
    unknown: "dropped",
  });
  // modelRouting is ignored entirely: ZCode-Local never routes models.
  assert.deepEqual(config, {
    effortTier: "high",
    thinkingMode: "auto",
    planThenExecute: true,
    mcpServerAllowlist: ["fs"],
  });
  assert.deepEqual(normalizeSpeedStackSessionConfig(null), {});
  // Invalid thinking mode is dropped (fail-open).
  assert.deepEqual(
    normalizeSpeedStackSessionConfig({ thinkingMode: "bogus", modelRouting: "yes" }),
    {},
  );
});

test("resolveThinkingTier precedence: off > pinned tier > explicit > route hint", () => {
  assert.deepEqual(
    resolveThinkingTier({ thinkingMode: "off", explicitTier: "ultra", routeHint: "low" }),
    { kind: "off" },
  );
  assert.deepEqual(
    resolveThinkingTier({ thinkingMode: "high", explicitTier: "ultra", routeHint: "low" }),
    { kind: "tier", tier: "high" },
  );
  assert.deepEqual(
    resolveThinkingTier({ thinkingMode: "auto", explicitTier: "ultra", routeHint: "low" }),
    { kind: "tier", tier: "ultra" },
  );
  assert.deepEqual(
    resolveThinkingTier({ thinkingMode: "auto", routeHint: "xhigh" }),
    { kind: "tier", tier: "xhigh" },
  );
  assert.equal(
    resolveThinkingTier({ thinkingMode: "auto", routeHint: undefined }),
    undefined,
  );
  assert.equal(resolveThinkingTier({}), undefined);
});

test("findThinkingOffLevel detects off-like values case-insensitively", () => {
  assert.equal(findThinkingOffLevel(["off", "low", "high"]), "off");
  assert.equal(findThinkingOffLevel(["Disabled", "low"]), "Disabled");
  assert.equal(findThinkingOffLevel(["no_think", "low"]), "no_think");
  assert.equal(findThinkingOffLevel(["low", "high"]), undefined);
});

// ---------------------------------------------------------------
// Phase 3: uncertain effort bump + plan config surface
// ---------------------------------------------------------------

test("bumpEffortTier raises one level, ultra stays ultra", () => {
  assert.equal(bumpEffortTier("low"), "medium");
  assert.equal(bumpEffortTier("medium"), "high");
  assert.equal(bumpEffortTier("high"), "xhigh");
  assert.equal(bumpEffortTier("xhigh"), "ultra");
  assert.equal(bumpEffortTier("ultra"), "ultra");
});

test("normalizeSpeedStackSessionConfig keeps planPin/planCandidates", () => {
  const config = normalizeSpeedStackSessionConfig({
    planPin: true,
    planCandidates: 3,
    thinkingMode: "auto",
  });
  assert.equal(config.planPin, true);
  assert.equal(config.planCandidates, 3);
  assert.equal(config.thinkingMode, "auto");
  const invalid = normalizeSpeedStackSessionConfig({
    planPin: "yes",
    planCandidates: 0,
  });
  assert.equal(invalid.planPin, undefined);
  assert.equal(invalid.planCandidates, undefined);
});
