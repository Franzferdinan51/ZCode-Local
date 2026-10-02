// ============================================================
// Speed Stack: effort tiers per task
// ============================================================
//
// A coarse low/medium/high/xhigh/ultra effort setting per task, mapped onto
// the model's OWN optionSpecs (never provider-specific level names, never
// cloud model names). Surfacing: SpeedStackSessionConfig, carried by the
// task/session.
//
// SystemOne extension: extractRouteEffortHint reads an optional `effort`
// field from a route response object, fail-open (undefined = current
// behavior unchanged).
//
// Thinking modes (3.24.0): the desktop thinking selector offers
// Off / Low / Medium / High / XHigh / Ultra / Auto. `ThinkingMode` captures
// that choice: "off" disables thinking, a tier pins the effort, "auto" lets
// the SystemOne route decide per task. Model routing (`modelRouting`) is a
// fully independent switch: the route may retarget the session model only
// when it is on, and the thinking mode never gates model routing (nor the
// reverse).

import type { ModelOptions } from "@zcode/contracts";
import { resolveTurnBudgets, type TurnBudgets } from "./turn-budgets.js";

export const EFFORT_TIERS = ["low", "medium", "high", "xhigh", "ultra"] as const;

/** Per-task reasoning effort tier. */
export type EffortTier = (typeof EFFORT_TIERS)[number];

export const DEFAULT_EFFORT_TIER: EffortTier = "medium";

/**
 * Thinking selector mode. "auto" = the SystemOne route picks the effort per
 * task; "off" = thinking disabled; a tier pins that effort for every task.
 */
export const THINKING_MODES = ["auto", "off", ...EFFORT_TIERS] as const;
export type ThinkingMode = (typeof THINKING_MODES)[number];

/** Output-token budgets per tier; the model's own max always wins. */
const LOW_TIER_MAX_OUTPUT_TOKENS = 4_000;
const MEDIUM_TIER_MAX_OUTPUT_TOKENS = 12_000;
const HIGH_TIER_MAX_OUTPUT_TOKENS = 32_000;
const XHIGH_TIER_MAX_OUTPUT_TOKENS = 64_000;
/** ultra -> the model's own maximum output tokens (no ZCode-side cap). */

/** Reasoning-level values that mean "thinking off" (case-insensitive). */
const OFF_LEVEL_PATTERN = /^(off|disabled|false|no|none|nothink|no[-_]?think)$/i;

/** Minimal structural view of a model needed for tier mapping. */
export interface EffortTierModel {
  readonly optionSpecs: {
    readonly reasoningLevel: { readonly values: readonly string[] };
    readonly maxOutputTokens: { readonly max: number };
  };
}

/**
 * Parse user input into an EffortTier. Case-insensitive, fail-open to
 * undefined (caller keeps current behavior).
 */
export function parseEffortTier(input: unknown): EffortTier | undefined {
  if (typeof input !== "string") return undefined;
  const normalized = input.trim().toLowerCase();
  return (EFFORT_TIERS as readonly string[]).includes(normalized)
    ? (normalized as EffortTier)
    : undefined;
}

/**
 * Parse user input into a ThinkingMode. Case-insensitive, fail-open to
 * undefined (caller keeps current behavior).
 */
export function parseThinkingMode(input: unknown): ThinkingMode | undefined {
  if (typeof input !== "string") return undefined;
  const normalized = input.trim().toLowerCase();
  if (normalized === "auto" || normalized === "off") return normalized;
  return parseEffortTier(normalized);
}

/**
 * Find the model's own "thinking off" reasoning level, if it declares one.
 * Mirrors the UI's off-entry detection so runtime "Off" binds the same value
 * the user would pick manually. Returns undefined when the model has no
 * off-like level (caller then leaves the model untouched: fail-open).
 */
export function findThinkingOffLevel(
  values: readonly string[],
): string | undefined {
  return values.find((value) => OFF_LEVEL_PATTERN.test(value.trim()));
}

/**
 * Map an effort tier onto the index of the model's own ordered reasoning
 * levels (spec order is low..high per Model Config). Legacy low/medium/high
 * keep their exact historical indices; xhigh/ultra spread across the upper
 * half so models with 4+ levels get distinct depths. Models with fewer
 * levels collapse adjacent tiers onto the same value.
 */
function tierLevelIndex(tier: EffortTier, levelCount: number): number {
  if (levelCount <= 1) return 0;
  if (tier === "low") return 0;
  if (tier === "medium") return Math.floor(levelCount / 2);
  if (tier === "high" || tier === "ultra") return levelCount - 1;
  // xhigh: halfway between the medium index and the max.
  const mediumIndex = Math.floor(levelCount / 2);
  return Math.min(levelCount - 1, Math.round((mediumIndex + (levelCount - 1)) / 2));
}

/**
 * Map an effort tier onto concrete ModelOptions using the model's own
 * optionSpecs:
 * - reasoningLevel: tierLevelIndex(tier) into the model's ordered values.
 * - maxOutputTokens: tier budget capped by the model's max
 *   (low 4k / medium 12k / high 32k / xhigh 64k / ultra model max).
 */
export function effortTierToModelOptions(
  model: EffortTierModel,
  tier: EffortTier = DEFAULT_EFFORT_TIER,
): Required<ModelOptions> {
  const levels = model.optionSpecs.reasoningLevel.values;
  const reasoningLevel = levels[tierLevelIndex(tier, levels.length)]!;
  const modelMax = model.optionSpecs.maxOutputTokens.max;
  const tierBudget =
    tier === "low"
      ? LOW_TIER_MAX_OUTPUT_TOKENS
      : tier === "medium"
        ? MEDIUM_TIER_MAX_OUTPUT_TOKENS
        : tier === "high"
          ? HIGH_TIER_MAX_OUTPUT_TOKENS
          : tier === "xhigh"
            ? XHIGH_TIER_MAX_OUTPUT_TOKENS
            : modelMax;
  return {
    reasoningLevel,
    maxOutputTokens: Math.min(tierBudget, modelMax),
  };
}

/**
 * Per-task/per-session speed-stack config surface. All fields optional;
 * absent fields mean "current behavior unchanged".
 */
export interface SpeedStackSessionConfig {
  /** Reasoning effort tier for this task (legacy explicit pin). */
  effortTier?: EffortTier;
  /**
   * Model routing switch (3.24.0). When true, a SystemOne route decision may
   * retarget the session model per task ("Auto (SystemOne)" in the desktop
   * model picker). False/undefined = the pinned model never moves.
   * Independent of thinkingMode: neither gates the other.
   */
  modelRouting?: boolean;
  /**
   * Thinking selector mode (3.24.0): "off" disables thinking, a tier pins
   * the effort, "auto"/undefined lets the route decide per task.
   * Independent of modelRouting: neither gates the other.
   */
  thinkingMode?: ThinkingMode;
  /** When true, run plan-then-execute instead of a single reactive loop. */
  planThenExecute?: boolean;
  /**
   * Plan pin (Phase 3): when true, plan-then-execute produces exactly one
   * candidate plan and executes it as-is — the SystemOne rank-plans call
   * is skipped. Explicit deterministic config; wins over everything.
   * Env equivalent: ZCODE_PLAN_PIN=1.
   */
  planPin?: boolean;
  /**
   * Candidate plans (Phase 3): how many candidate plans the planner
   * produces for rank-plans scoring. Undefined = the default (2) when
   * unpinned. Env ZCODE_PLAN_CANDIDATES wins over this.
   */
  planCandidates?: number;
  /** MCP server allowlist; undefined/empty = attach everything (default). */
  mcpServerAllowlist?: readonly string[];
  /** MCP tool allowlist ("server.tool" or "tool"); undefined/empty = all. */
  mcpToolAllowlist?: readonly string[];
  /**
   * MCP pruning kill-switch (config level). `false` disables route-driven
   * MCP pruning for the session; undefined/true leaves the default policy
   * live. The env-level kill-switch is ZCODE_SPEEDSTACK_PRUNE=0 — see
   * resolveMcpAttachPolicy in speedstack/systemone-route.ts, which documents
   * both.
   */
  mcpPruning?: boolean;
}

export function normalizeSpeedStackSessionConfig(
  input: unknown,
): SpeedStackSessionConfig {
  if (!input || typeof input !== "object") return {};
  const raw = input as Record<string, unknown>;
  const config: SpeedStackSessionConfig = {};
  const tier = parseEffortTier(raw["effortTier"]);
  if (tier) config.effortTier = tier;
  if (raw["modelRouting"] === true) config.modelRouting = true;
  const thinkingMode = parseThinkingMode(raw["thinkingMode"]);
  if (thinkingMode) config.thinkingMode = thinkingMode;
  // Z3: preserve an explicit boolean pin — false forces direct execution
  // in the Rank-3 gate (previously only `true` survived normalization).
  if (typeof raw["planThenExecute"] === "boolean") {
    config.planThenExecute = raw["planThenExecute"];
  }
  if (typeof raw["planPin"] === "boolean") {
    config.planPin = raw["planPin"];
  }
  const planCandidates = raw["planCandidates"];
  if (
    typeof planCandidates === "number" &&
    Number.isInteger(planCandidates) &&
    planCandidates > 0
  ) {
    config.planCandidates = planCandidates;
  }
  if (Array.isArray(raw["mcpServerAllowlist"])) {
    config.mcpServerAllowlist = raw["mcpServerAllowlist"].filter(
      (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
    );
  }
  if (Array.isArray(raw["mcpToolAllowlist"])) {
    config.mcpToolAllowlist = raw["mcpToolAllowlist"].filter(
      (entry): entry is string => typeof entry === "string" && entry.trim().length > 0,
    );
  }
  if (typeof raw["mcpPruning"] === "boolean") {
    config.mcpPruning = raw["mcpPruning"];
  }
  return config;
}

/**
 * Resolve the effective effort tier: explicit session config wins, then the
 * SystemOne route hint (see extractRouteEffortHint in
 * packages/shared/src/systemone-scorer.ts), then undefined (caller keeps
 * current behavior). Legacy helper kept for the explicit-pin path; new code
 * should prefer resolveThinkingTier, which also honors thinkingMode/off.
 */
export function resolveEffectiveEffortTier(
  config: SpeedStackSessionConfig | undefined,
  routeHint: EffortTier | undefined,
): EffortTier | undefined {
  return config?.effortTier ?? routeHint ?? undefined;
}

/**
 * Bump an effort tier up one level (low→medium→high→xhigh→ultra). The top
 * tier stays put (fail-open: never invents a tier above ultra). Used by the
 * SystemOne "uncertain" rule: an uncertain route raises effort one level
 * instead of pruning tools.
 */
export function bumpEffortTier(tier: EffortTier): EffortTier {
  const index = EFFORT_TIERS.indexOf(tier);
  if (index < 0) return DEFAULT_EFFORT_TIER;
  return EFFORT_TIERS[Math.min(EFFORT_TIERS.length - 1, index + 1)] as EffortTier;
}

/** Outcome of thinking-mode resolution for one task. */
export type ResolvedThinking =
  | { readonly kind: "off" }
  | { readonly kind: "tier"; readonly tier: EffortTier };

/**
 * Resolve what thinking to apply for one task.
 *
 * Precedence (model/thinking independence: this never looks at modelRouting):
 *   1. thinkingMode === "off"            -> thinking off
 *   2. thinkingMode is a tier             -> pinned tier (user's explicit choice)
 *   3. config.effortTier (legacy pin)     -> that tier
 *   4. thinkingMode "auto"/unset + route  -> the route's effort hint
 *   5. otherwise                          -> undefined (caller keeps behavior)
 */
export function resolveThinkingTier(input: {
  thinkingMode?: ThinkingMode;
  explicitTier?: EffortTier;
  routeHint?: EffortTier;
}): ResolvedThinking | undefined {
  const mode = input.thinkingMode;
  if (mode === "off") return { kind: "off" };
  if (mode !== undefined && mode !== "auto") return { kind: "tier", tier: mode };
  if (input.explicitTier) return { kind: "tier", tier: input.explicitTier };
  if (input.routeHint) return { kind: "tier", tier: input.routeHint };
  return undefined;
}

// -- Z2: effort behavior policy ("effort means behavior") -----------------
//
// Ryan's explicit ask (3.25.0 Rank 2): effort must change agent BEHAVIOR,
// not just reasoningLevel/maxOutputTokens. This table is the single policy
// surface the runtime consults per turn, resolved alongside
// applySystemOneEffortOverride in speedstack/systemone-route.ts
// (resolveSystemOneBehaviorPolicy).
//
// Dimensions per tier:
//   maxSteps / maxToolCalls  - per-turn budgets. DEFAULTS ONLY: the
//     package-1 config surface (ZCODE_BUDGET_<ROUTE_TIER>_MAX_STEPS /
//     _MAX_TOOL_CALLS, ZCODE_BUDGET_DEFAULT_*) always wins when set.
//     There is deliberately no second config for caps.
//   subagents                - "never" (low/off: the dispatch tools are
//     hidden from the model), "conservative" (allowed, modest maxTurns),
//     "parallel" (ultra: allowed for parallel exploration).
//   subagentMaxTurns         - tier-scaled maxTurns for spawned subagents
//     (runtime/methods/subagent.ts).
//   readBreadth              - advisory cap on parallel reads/searches per
//     exploration step. Resolved, logged, and harness-reported; hard
//     enforcement lands with the Rank 7/8 workflow/doom-loop work.
//   verificationPasses       - review passes after edits (xhigh/ultra: one
//     review reminder once per turn when Edit/Write ran).
//   compactionAggressiveness  - multiplier on the auto-compact threshold
//     (1.0 = default; <1 compacts earlier). Wired via
//     thresholdPercentOverride in compact/policy.ts.
//   planThenExecuteEligible  - gate for the Rank-3 plan-execute wiring;
//     see isPlanThenExecuteEligible (route-tier heavy also qualifies).
//
// Every dimension is separately overridable via env (documented below);
// garbage fails open to the table default. Thinking "off" resolves to the
// low row (minimal agent behavior).
//
// Env overrides (all optional):
//   ZCODE_EFFORT_<TIER>_SUBAGENTS          never|conservative|parallel
//   ZCODE_EFFORT_<TIER>_SUBAGENT_MAX_TURNS positive int
//   ZCODE_EFFORT_<TIER>_READ_BREADTH       positive int
//   ZCODE_EFFORT_<TIER>_VERIFICATION_PASSES non-negative int
//   ZCODE_EFFORT_<TIER>_COMPACTION         positive float (0.85 = compact at 85% of normal threshold)
//   ZCODE_EFFORT_<TIER>_PLAN_EXECUTE       1|0
// <TIER> is the uppercase effort tier: LOW, MEDIUM, HIGH, XHIGH, ULTRA.

/** Subagent allowance for an effort tier. */
export const SUBAGENT_ALLOWANCES = ["never", "conservative", "parallel"] as const;
/** Subagent allowance for an effort tier. */
export type SubagentAllowance = (typeof SUBAGENT_ALLOWANCES)[number];

/** Per-tier agent behavior policy. All fields have table defaults; see env overrides above. */
export interface EffortBehaviorPolicy {
  /** The effort tier this row describes. */
  readonly tier: EffortTier;
  /** Default max model steps per turn (package-1 ZCODE_BUDGET_* env wins). */
  readonly maxSteps: number;
  /** Default max tool calls per turn (package-1 ZCODE_BUDGET_* env wins). */
  readonly maxToolCalls: number;
  /** Subagent allowance: never (low/off), conservative, parallel (ultra). */
  readonly subagents: SubagentAllowance;
  /** Tier-scaled maxTurns for spawned subagents. */
  readonly subagentMaxTurns: number;
  /** Advisory cap on parallel reads/searches per exploration step. */
  readonly readBreadth: number;
  /** Review passes after edits (xhigh/ultra: 1). */
  readonly verificationPasses: number;
  /** Multiplier on the auto-compact threshold (1.0 = default). */
  readonly compactionAggressiveness: number;
  /** Eligible for plan-then-execute gating (Rank 3 consumes this). */
  readonly planThenExecuteEligible: boolean;
}

/**
 * The behavior table. Step/call defaults are roomy on purpose: agentic
 * coding loops burn a step per tool round-trip, so tight budgets abort
 * long tasks before they converge. Tunable without a release via the env
 * overrides above (steps/calls) and the package-1 ZCODE_BUDGET_* surface.
 */
const EFFORT_BEHAVIOR_POLICY_TABLE: Record<EffortTier, EffortBehaviorPolicy> = {
  low: {
    tier: "low",
    maxSteps: 60,
    maxToolCalls: 150,
    subagents: "never",
    subagentMaxTurns: 8,
    readBreadth: 5,
    verificationPasses: 0,
    compactionAggressiveness: 1.0,
    planThenExecuteEligible: false,
  },
  medium: {
    tier: "medium",
    maxSteps: 120,
    maxToolCalls: 300,
    subagents: "conservative",
    subagentMaxTurns: 16,
    readBreadth: 10,
    verificationPasses: 0,
    compactionAggressiveness: 1.0,
    planThenExecuteEligible: false,
  },
  high: {
    tier: "high",
    maxSteps: 250,
    maxToolCalls: 600,
    subagents: "conservative",
    subagentMaxTurns: 32,
    readBreadth: 15,
    verificationPasses: 0,
    compactionAggressiveness: 1.0,
    planThenExecuteEligible: false,
  },
  xhigh: {
    tier: "xhigh",
    maxSteps: 500,
    maxToolCalls: 1200,
    subagents: "parallel",
    subagentMaxTurns: 48,
    readBreadth: 25,
    verificationPasses: 1,
    compactionAggressiveness: 0.9,
    planThenExecuteEligible: true,
  },
  ultra: {
    tier: "ultra",
    maxSteps: 1000,
    maxToolCalls: 2500,
    subagents: "parallel",
    subagentMaxTurns: 64,
    readBreadth: 40,
    verificationPasses: 1,
    compactionAggressiveness: 0.85,
    planThenExecuteEligible: true,
  },
};

function parsePositiveIntValue(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number.parseInt(trimmed, 10);
  return value > 0 ? value : undefined;
}

function parseNonNegativeIntValue(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) return undefined;
  return Number.parseInt(trimmed, 10);
}

function parsePositiveFloatValue(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (!/^\d+(\.\d+)?$/.test(trimmed)) return undefined;
  const value = Number.parseFloat(trimmed);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function parseSubagentAllowance(raw: string | undefined): SubagentAllowance | undefined {
  if (raw === undefined) return undefined;
  const normalized = raw.trim().toLowerCase();
  return (SUBAGENT_ALLOWANCES as readonly string[]).includes(normalized)
    ? (normalized as SubagentAllowance)
    : undefined;
}

function parseBoolean01(raw: string | undefined): boolean | undefined {
  if (raw === undefined) return undefined;
  const normalized = raw.trim().toLowerCase();
  if (normalized === "1" || normalized === "true") return true;
  if (normalized === "0" || normalized === "false") return false;
  return undefined;
}

/**
 * Resolve the behavior policy for an effort tier: table defaults with
 * per-dimension env overrides applied. Never throws; garbage fails open
 * to the table row.
 */
export function getEffortBehaviorPolicy(
  tier: EffortTier,
  env: NodeJS.ProcessEnv = process.env,
): EffortBehaviorPolicy {
  try {
    const base =
      EFFORT_BEHAVIOR_POLICY_TABLE[tier] ??
      EFFORT_BEHAVIOR_POLICY_TABLE[DEFAULT_EFFORT_TIER];
    const prefix = `ZCODE_EFFORT_${tier.toUpperCase()}`;
    return {
      tier: base.tier,
      maxSteps: base.maxSteps,
      maxToolCalls: base.maxToolCalls,
      subagents: parseSubagentAllowance(env[`${prefix}_SUBAGENTS`]) ?? base.subagents,
      subagentMaxTurns:
        parsePositiveIntValue(env[`${prefix}_SUBAGENT_MAX_TURNS`]) ?? base.subagentMaxTurns,
      readBreadth:
        parsePositiveIntValue(env[`${prefix}_READ_BREADTH`]) ?? base.readBreadth,
      verificationPasses:
        parseNonNegativeIntValue(env[`${prefix}_VERIFICATION_PASSES`]) ??
        base.verificationPasses,
      compactionAggressiveness:
        parsePositiveFloatValue(env[`${prefix}_COMPACTION`]) ??
        base.compactionAggressiveness,
      planThenExecuteEligible:
        parseBoolean01(env[`${prefix}_PLAN_EXECUTE`]) ?? base.planThenExecuteEligible,
    };
  } catch {
    return (
      EFFORT_BEHAVIOR_POLICY_TABLE[tier] ??
      EFFORT_BEHAVIOR_POLICY_TABLE[DEFAULT_EFFORT_TIER]
    );
  }
}

/**
 * Resolve the EFFECTIVE per-turn budgets: the policy table's raised
 * defaults for the effort tier, with the package-1 env surface
 * (ZCODE_BUDGET_<ROUTE_TIER>_MAX_STEPS/_MAX_TOOL_CALLS, ZCODE_BUDGET_DEFAULT_*)
 * winning wherever it is set. When no effort tier resolved, this is exactly
 * the package-1 env-only resolution ({} when unconfigured = unbounded =
 * today's behavior). Never throws.
 */
export function resolveEffectiveTurnBudgets(input: {
  routeTier?: string | undefined;
  effortTier?: EffortTier | undefined;
  env?: NodeJS.ProcessEnv;
}): TurnBudgets {
  try {
    const env = input.env ?? process.env;
    const envBudgets = resolveTurnBudgets(input.routeTier, env);
    if (!input.effortTier) return envBudgets;
    const policy = getEffortBehaviorPolicy(input.effortTier, env);
    return {
      maxSteps: envBudgets.maxSteps ?? policy.maxSteps,
      maxToolCalls: envBudgets.maxToolCalls ?? policy.maxToolCalls,
    };
  } catch {
    return {};
  }
}

/**
 * Whether the model may spawn subagents under this policy. "never" (low/off)
 * hides the dispatch tools from the model; anything else allows spawning
 * (modest vs generous maxTurns carries the conservative/parallel split).
 */
export function isSubagentSpawningAllowed(
  policy: EffortBehaviorPolicy | undefined,
): boolean {
  return policy?.subagents !== "never";
}

/**
 * Plan-then-execute eligibility gate (Rank 3 consumes this; Rank 2 only
 * defines it). Eligible when the effort policy says so (xhigh/ultra) OR the
 * route tier is heavy; below that, direct execution. Fail-open to false on
 * garbage input.
 */
export function isPlanThenExecuteEligible(input: {
  policy?: EffortBehaviorPolicy | undefined;
  routeTier?: string | undefined;
}): boolean {
  try {
    if (input.policy?.planThenExecuteEligible === true) return true;
    return (input.routeTier ?? "").trim().toLowerCase() === "heavy";
  } catch {
    return false;
  }
}
