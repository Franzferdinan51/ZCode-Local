// Tests for the effort behavior policy table (speedstack/effort-tiers.ts,
// Z2 "effort means behavior").
// Run: tsx --test src/speedstack/effort-policy.test.ts
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_EFFORT_TIER,
  getEffortBehaviorPolicy,
  isPlanThenExecuteEligible,
  isSubagentSpawningAllowed,
  resolveEffectiveTurnBudgets,
  resolveSystemOneBehaviorPolicy,
  type EffortBehaviorPolicy,
  type EffortTier,
} from "./index.js";

function envOf(values: Record<string, string>): NodeJS.ProcessEnv {
  return { ...values };
}

const TIERS: EffortTier[] = ["low", "medium", "high", "xhigh", "ultra"];

describe("getEffortBehaviorPolicy defaults", () => {
  it("returns the raised step/call defaults per tier", () => {
    assert.deepEqual(
      TIERS.map((tier) => {
        const p = getEffortBehaviorPolicy(tier, envOf({}));
        return [p.maxSteps, p.maxToolCalls];
      }),
      [
        [60, 150],
        [120, 300],
        [250, 600],
        [500, 1200],
        [1000, 2500],
      ],
    );
  });

  it("gates subagents by tier: never for low, parallel for xhigh/ultra", () => {
    assert.equal(getEffortBehaviorPolicy("low", envOf({})).subagents, "never");
    assert.equal(
      getEffortBehaviorPolicy("medium", envOf({})).subagents,
      "conservative",
    );
    assert.equal(
      getEffortBehaviorPolicy("high", envOf({})).subagents,
      "conservative",
    );
    assert.equal(getEffortBehaviorPolicy("xhigh", envOf({})).subagents, "parallel");
    assert.equal(getEffortBehaviorPolicy("ultra", envOf({})).subagents, "parallel");
  });

  it("scales subagent maxTurns with the tier", () => {
    const turns = TIERS.map(
      (tier) => getEffortBehaviorPolicy(tier, envOf({})).subagentMaxTurns,
    );
    assert.deepEqual(turns, [8, 16, 32, 48, 64]);
    for (let i = 1; i < turns.length; i += 1) {
      assert.ok(turns[i] > turns[i - 1], "monotonic increase");
    }
  });

  it("grants verification passes only to xhigh/ultra", () => {
    assert.deepEqual(
      TIERS.map((tier) => getEffortBehaviorPolicy(tier, envOf({})).verificationPasses),
      [0, 0, 0, 1, 1],
    );
  });

  it("compacts more aggressively at higher tiers", () => {
    assert.deepEqual(
      TIERS.map(
        (tier) => getEffortBehaviorPolicy(tier, envOf({})).compactionAggressiveness,
      ),
      [1.0, 1.0, 1.0, 0.9, 0.85],
    );
  });

  it("marks plan-execute eligible only for xhigh/ultra", () => {
    assert.deepEqual(
      TIERS.map(
        (tier) => getEffortBehaviorPolicy(tier, envOf({})).planThenExecuteEligible,
      ),
      [false, false, false, true, true],
    );
  });
});

describe("getEffortBehaviorPolicy env overrides", () => {
  it("overrides every dimension independently", () => {
    const policy = getEffortBehaviorPolicy(
      "medium",
      envOf({
        ZCODE_EFFORT_MEDIUM_SUBAGENTS: "never",
        ZCODE_EFFORT_MEDIUM_SUBAGENT_MAX_TURNS: "9",
        ZCODE_EFFORT_MEDIUM_READ_BREADTH: "42",
        ZCODE_EFFORT_MEDIUM_VERIFICATION_PASSES: "2",
        ZCODE_EFFORT_MEDIUM_COMPACTION: "0.7",
        ZCODE_EFFORT_MEDIUM_PLAN_EXECUTE: "1",
      }),
    );
    assert.equal(policy.subagents, "never");
    assert.equal(policy.subagentMaxTurns, 9);
    assert.equal(policy.readBreadth, 42);
    assert.equal(policy.verificationPasses, 2);
    assert.equal(policy.compactionAggressiveness, 0.7);
    assert.equal(policy.planThenExecuteEligible, true);
  });

  it("scopes overrides to the named tier only", () => {
    const env = envOf({ ZCODE_EFFORT_LOW_SUBAGENTS: "parallel" });
    assert.equal(getEffortBehaviorPolicy("low", env).subagents, "parallel");
    assert.equal(
      getEffortBehaviorPolicy("medium", env).subagents,
      "conservative",
    );
  });

  it("fails open on garbage: keeps table defaults", () => {
    const policy = getEffortBehaviorPolicy(
      "high",
      envOf({
        ZCODE_EFFORT_HIGH_SUBAGENTS: "sometimes",
        ZCODE_EFFORT_HIGH_SUBAGENT_MAX_TURNS: "many",
        ZCODE_EFFORT_HIGH_READ_BREADTH: "0",
        ZCODE_EFFORT_HIGH_VERIFICATION_PASSES: "-1",
        ZCODE_EFFORT_HIGH_COMPACTION: "huge",
        ZCODE_EFFORT_HIGH_PLAN_EXECUTE: "maybe",
      }),
    );
    assert.equal(policy.subagents, "conservative");
    assert.equal(policy.subagentMaxTurns, 32);
    assert.equal(policy.readBreadth, 15);
    assert.equal(policy.verificationPasses, 0);
    assert.equal(policy.compactionAggressiveness, 1.0);
    assert.equal(policy.planThenExecuteEligible, false);
  });
});

describe("resolveEffectiveTurnBudgets", () => {
  it("uses the policy raised defaults when no budget env is set", () => {
    assert.deepEqual(
      resolveEffectiveTurnBudgets({ routeTier: "balanced", effortTier: "medium", env: envOf({}) }),
      { maxSteps: 120, maxToolCalls: 300 },
    );
  });

  it("lets the package-1 env surface win over policy defaults, per dimension", () => {
    assert.deepEqual(
      resolveEffectiveTurnBudgets({
        routeTier: "balanced",
        effortTier: "medium",
        env: envOf({ ZCODE_BUDGET_BALANCED_MAX_STEPS: "20" }),
      }),
      { maxSteps: 20, maxToolCalls: 300 },
    );
  });

  it("degrades to the package-1 env-only resolution when no effort tier resolved", () => {
    assert.deepEqual(
      resolveEffectiveTurnBudgets({ routeTier: "balanced", env: envOf({}) }),
      {},
    );
    assert.deepEqual(
      resolveEffectiveTurnBudgets({
        routeTier: "balanced",
        env: envOf({ ZCODE_BUDGET_BALANCED_MAX_TOOL_CALLS: "40" }),
      }),
      { maxToolCalls: 40 },
    );
  });
});

describe("resolveSystemOneBehaviorPolicy", () => {
  it("resolves the tier, policy, and budgets from a route hint", () => {
    const res = resolveSystemOneBehaviorPolicy(
      { systemOneRouteValue: { tier: "balanced", effort: "high" } },
      envOf({}),
    );
    assert.equal(res.tier, "high");
    assert.equal(res.policy?.tier, "high");
    assert.equal(res.policy?.maxSteps, 250);
    assert.deepEqual(res.budgets, { maxSteps: 250, maxToolCalls: 600 });
  });

  it("maps thinking off to the low policy row", () => {
    const res = resolveSystemOneBehaviorPolicy(
      {
        speedStackConfig: { thinkingMode: "off" },
        systemOneRouteValue: { tier: "heavy", effort: "ultra" },
      },
      envOf({}),
    );
    assert.equal(res.tier, "low");
    assert.equal(res.policy?.subagents, "never");
  });

  it("honors an explicitly pinned effort tier over the route hint", () => {
    const res = resolveSystemOneBehaviorPolicy(
      {
        speedStackConfig: { effortTier: "ultra" },
        systemOneRouteValue: { tier: "economy", effort: "low" },
      },
      envOf({}),
    );
    assert.equal(res.tier, "ultra");
    assert.equal(res.policy?.verificationPasses, 1);
  });

  it("fails open to empty budgets when nothing resolves", () => {
    const res = resolveSystemOneBehaviorPolicy({}, envOf({}));
    assert.equal(res.tier, undefined);
    assert.equal(res.policy, undefined);
    assert.deepEqual(res.budgets, {});
  });
});

describe("isSubagentSpawningAllowed / isPlanThenExecuteEligible", () => {
  const policy = (overrides: Partial<EffortBehaviorPolicy>): EffortBehaviorPolicy => ({
    ...getEffortBehaviorPolicy(DEFAULT_EFFORT_TIER, envOf({})),
    ...overrides,
  });

  it("denies spawning for never, allows otherwise", () => {
    assert.equal(isSubagentSpawningAllowed(policy({ subagents: "never" })), false);
    assert.equal(
      isSubagentSpawningAllowed(policy({ subagents: "conservative" })),
      true,
    );
    assert.equal(isSubagentSpawningAllowed(policy({ subagents: "parallel" })), true);
    assert.equal(isSubagentSpawningAllowed(undefined), true);
  });

  it("gates plan-execute: policy flag or heavy route tier", () => {
    assert.equal(
      isPlanThenExecuteEligible({ policy: policy({ planThenExecuteEligible: true }) }),
      true,
    );
    assert.equal(
      isPlanThenExecuteEligible({
        policy: policy({ planThenExecuteEligible: false }),
        routeTier: "heavy",
      }),
      true,
    );
    assert.equal(
      isPlanThenExecuteEligible({
        policy: policy({ planThenExecuteEligible: false }),
        routeTier: "balanced",
      }),
      false,
    );
    assert.equal(isPlanThenExecuteEligible({}), false);
  });
});
