// ============================================================
// Speed Stack: SystemOne route client + per-task policies
// ============================================================
//
// One lightweight client for the local SystemOne shim
// (POST http://127.0.0.1:8765/v1/systemone/route), plus the two per-task
// policies driven by its route decision:
//
//   Z0 model routing  - route.model_id -> retarget the session model for
//                     this task, ONLY when SpeedStackSessionConfig.modelRouting
//                     is on ("Auto (SystemOne)"). Fully independent of Z1.
//   Z1 effort wiring  - route.effort -> EffortTier -> ModelOptions
//                     (reasoningLevel + maxOutputTokens) via
//                     resolveEffortTierAndOptions / applySystemOneEffortOverride.
//                     Honors thinkingMode (off/pinned/auto). Fully independent
//                     of Z0.
//   Z2 MCP pruning    - resolveMcpAttachPolicy decides whether MCP servers
//                     attach at all; pruneMcpServerMap / pruneMcpTools apply
//                     explicit allowlists from SpeedStackSessionConfig.
//                     Independent of Z0/Z1; kill-switches documented below.
//
// Everything here is fail-open: the shim being down, slow, or returning
// garbage yields `undefined` / "attach everything", i.e. today's behavior.

import type { Logger, ModelOptions } from "@zcode/contracts";
import { extractRouteEffortHint } from "@zcode/shared/systemone-scorer";
import {
  bumpEffortTier,
  effortTierToModelOptions,
  findThinkingOffLevel,
  getEffortBehaviorPolicy,
  resolveEffectiveEffortTier,
  resolveEffectiveTurnBudgets,
  resolveThinkingTier,
  type EffortBehaviorPolicy,
  type EffortTier,
  type EffortTierModel,
  type ResolvedThinking,
  type SpeedStackSessionConfig,
  type ThinkingMode,
} from "./effort-tiers.js";
import type { TurnBudgets } from "./turn-budgets.js";
import { computeServerShortlist } from "./tool-packs.js";
import {
  DEFAULT_SYSTEMONE_SHIM_URL,
  systemOneShimEndpoint,
} from "./systemone-shim-url.js";
import { appendDecisionRecord } from "./systemone-decision-log.js";

/** Local SystemOne shim route endpoint (see systemone/shim.py).
 *
 * Default when $SYSTEMONE_SHIM_URL is unset — prefer
 * resolveSystemOneRouteEndpoint() for the live value, which honors the
 * env override. Kept as a const for callers that need a stable default
 * (eval harness fixtures, tests).
 */
export const SYSTEMONE_ROUTE_ENDPOINT = `${DEFAULT_SYSTEMONE_SHIM_URL}/v1/systemone/route`;

/**
 * Resolve the route endpoint from $SYSTEMONE_SHIM_URL
 * (see ./systemone-shim-url.js), defaulting to the localhost shim.
 */
export function resolveSystemOneRouteEndpoint(
  env: NodeJS.ProcessEnv = process.env,
): string {
  return systemOneShimEndpoint("/v1/systemone/route", env);
}

/** Hard bound on the route lookup; the shim answers in ~100ms when healthy. */
export const SYSTEMONE_ROUTE_TIMEOUT_MS = 3_000;

/**
 * Kill-switch (env): set `ZCODE_SPEEDSTACK_PRUNE=0` to force full MCP attach,
 * disabling route-driven MCP pruning for the process. Documented alongside
 * the config-level kill-switch `SpeedStackSessionConfig.mcpPruning`.
 */
export const SYSTEMONE_PRUNE_KILL_SWITCH_ENV = "ZCODE_SPEEDSTACK_PRUNE";

/**
 * Master kill-switch (env): set `ZCODE_SYSTEMONE=0` to disable both the
 * bundled-shim auto-start (see the CLI's systemone-shim-bootstrap) and all
 * route lookups. The session then behaves exactly as if the shim never
 * existed. Fail-open by construction.
 */
export const SYSTEMONE_KILL_SWITCH_ENV = "ZCODE_SYSTEMONE";

/** True when the master SystemOne kill-switch is engaged. */
export function isSystemOneDisabled(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  return env[SYSTEMONE_KILL_SWITCH_ENV] === "0";
}

/**
 * Default pruning policy: only skip MCP servers when the router confidently
 * calls the task trivial. Conservative on purpose — anything ambiguous keeps
 * the full tool surface.
 */
export const SYSTEMONE_PRUNE_CONFIDENCE_THRESHOLD = 0.8;

/** Minimal route decision consumed by the Z1/Z2 policies. */
export interface SystemOneRouteDecision {
  readonly tier: string;
  readonly confidence: number;
  readonly effort?: string;
  readonly taskLabels?: readonly string[];
  readonly modelId?: string;
  /**
   * True when the shim's calibration flagged the route as uncertain
   * (narrow top-1/top-2 margin). Phase 3 rule: an uncertain route NEVER
   * prunes tools/MCP servers and bumps effort one level. Absent on older
   * shims — "not present", never an error.
   */
  readonly uncertain?: boolean | undefined;
  /**
   * Shim-scored tool/MCP relevance ranking (advisory): entries are
   * keep-signals only, never prune-signals. Absent/empty on older shims.
   */
  readonly rankedTools?: readonly RankedTool[] | undefined;
  /**
   * Shim's expected-utility model ranking (advisory, best-value first).
   * Z0 model routing consumes the top entry; absent/empty on older shims
   * or when the registry can't rank — never an error.
   */
  readonly rankedModels?: readonly RankedModel[] | undefined;
  /**
   * Advisory tier second opinion the shim computed for uncertain routes
   * (wire name `jeff1_second_opinion`; decider-backed now, the name is
   * historical). Never changes the routed tier. Absent unless the route
   * was uncertain.
   */
  readonly secondOpinion?: SecondOpinion | undefined;
  /**
   * Per-tier classification scores, preserved verbatim from the shim's
   * `probabilities` payload when present. Never consumed by current
   * policy thresholds (those stay confidence-gated); carried end-to-end
   * so the future scored-classification/ranking work can condition on
   * them without re-fetching the route.
   */
  readonly tierScores?: Readonly<Record<string, number>>;
}

/** One entry of the shim's ranked_tools surface (advisory keep-signal). */
export interface RankedTool {
  readonly id: string;
  readonly relevance: number;
  readonly kind?: string | undefined;
}

/**
 * One entry of the shim's ranked_models surface (best-value first).
 * Model ids come from the shim's registry — never hard-coded here.
 */
export interface RankedModel {
  readonly modelId: string;
  readonly tier: string;
  readonly utility: number;
  readonly quality?: number | undefined;
  readonly cost?: number | undefined;
}

/**
 * The shim's advisory tier second opinion on an uncertain route
 * (wire name `jeff1_second_opinion`; decider-backed since the
 * decider-only sidecar — the name is historical).
 */
export interface SecondOpinion {
  readonly tier: string;
  readonly agree: boolean;
  readonly confidence?: number | undefined;
  readonly rationale?: string | undefined;
}

/**
 * POST {task} to the SystemOne route endpoint and return the parsed route
 * decision. Fail-open: any failure (shim down, timeout, non-200, malformed
 * body) returns undefined and never throws.
 */
export async function fetchSystemOneRouteDecision(
  task: string,
  options?: { endpoint?: string; timeoutMs?: number },
): Promise<SystemOneRouteDecision | undefined> {
  if (isSystemOneDisabled()) return undefined;
  const endpoint = options?.endpoint ?? resolveSystemOneRouteEndpoint();
  const timeoutMs = options?.timeoutMs ?? SYSTEMONE_ROUTE_TIMEOUT_MS;
  if (!task || !task.trim()) return undefined;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ task }),
      signal: controller.signal,
    });
    if (!response.ok) return undefined;
    const decision = parseRouteDecision(await response.json());
    logRouteDecision(task, decision);
    return decision;
  } catch {
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Best-effort decision-record logging for the calibration battery
 * (see ./systemone-decision-log.ts). Never throws; a logging failure
 * must not affect routing.
 */
function logRouteDecision(
  task: string,
  decision: SystemOneRouteDecision | undefined,
): void {
  try {
    if (!decision) return;
    const margin = routeTierMargin(decision);
    appendDecisionRecord({
      kind: "route",
      ts: new Date().toISOString(),
      tier: decision.tier,
      confidence: decision.confidence,
      ...(decision.uncertain !== undefined
        ? { uncertain: decision.uncertain }
        : {}),
      ...(margin !== undefined ? { margin } : {}),
      ...(decision.modelId ? { modelId: decision.modelId } : {}),
      ...(decision.rankedModels
        ? {
            rankedModels: decision.rankedModels.map((m) => ({
              modelId: m.modelId,
              tier: m.tier,
              utility: m.utility,
            })),
          }
        : {}),
      taskChars: task.length,
    });
  } catch {
    // logging never breaks routing
  }
}

function parseRouteDecision(payload: unknown): SystemOneRouteDecision | undefined {
  if (!payload || typeof payload !== "object") return undefined;
  const route = (payload as Record<string, unknown>)["route"];
  if (!route || typeof route !== "object") return undefined;
  const record = route as Record<string, unknown>;
  const tier = record["tier"];
  const confidence = record["confidence"];
  if (typeof tier !== "string" || typeof confidence !== "number") return undefined;
  const decision: {
    tier: string;
    confidence: number;
    effort?: string;
    taskLabels?: readonly string[];
    modelId?: string;
    uncertain?: boolean;
    rankedTools?: RankedTool[];
    rankedModels?: RankedModel[];
    secondOpinion?: SecondOpinion;
    tierScores?: Readonly<Record<string, number>>;
  } = { tier, confidence };
  if (typeof record["effort"] === "string") decision.effort = record["effort"];
  if (record["uncertain"] === true) decision.uncertain = true;
  const rankedTools = record["ranked_tools"];
  if (Array.isArray(rankedTools)) {
    const parsed: RankedTool[] = [];
    for (const item of rankedTools) {
      if (!item || typeof item !== "object") continue;
      const entry = item as Record<string, unknown>;
      const id = entry["id"];
      const relevance = entry["relevance"];
      if (typeof id !== "string" || id.trim().length === 0) continue;
      if (typeof relevance !== "number" || !Number.isFinite(relevance)) {
        continue;
      }
      const kind = entry["kind"];
      parsed.push({
        id: id.trim(),
        relevance,
        ...(typeof kind === "string" ? { kind } : {}),
      });
    }
    if (parsed.length > 0) decision.rankedTools = parsed;
  }
  const rankedModels = record["ranked_models"];
  if (Array.isArray(rankedModels)) {
    const parsed: RankedModel[] = [];
    for (const item of rankedModels) {
      if (!item || typeof item !== "object") continue;
      const entry = item as Record<string, unknown>;
      const modelId = entry["model_id"];
      const tier = entry["tier"];
      const utility = entry["utility"];
      if (typeof modelId !== "string" || modelId.trim().length === 0) {
        continue;
      }
      if (typeof tier !== "string") continue;
      if (typeof utility !== "number" || !Number.isFinite(utility)) continue;
      const quality = entry["quality"];
      const cost = entry["cost"];
      parsed.push({
        modelId: modelId.trim(),
        tier,
        utility,
        ...(typeof quality === "number" && Number.isFinite(quality)
          ? { quality }
          : {}),
        ...(typeof cost === "number" && Number.isFinite(cost) ? { cost } : {}),
      });
    }
    if (parsed.length > 0) decision.rankedModels = parsed;
  }
  const rawOpinion = record["jeff1_second_opinion"];
  if (rawOpinion && typeof rawOpinion === "object") {
    const entry = rawOpinion as Record<string, unknown>;
    const tier = entry["tier"];
    if (typeof tier === "string" && tier.length > 0) {
      const confidence = entry["confidence"];
      const rationale = entry["rationale"];
      decision.secondOpinion = {
        tier,
        agree: entry["agree"] !== false,
        ...(typeof confidence === "number" && Number.isFinite(confidence)
          ? { confidence }
          : {}),
        ...(typeof rationale === "string" ? { rationale } : {}),
      };
    }
  }
  if (
    typeof record["model_id"] === "string" &&
    record["model_id"].trim().length > 0
  ) {
    decision.modelId = record["model_id"].trim();
  }
  if (Array.isArray(record["task_labels"])) {
    decision.taskLabels = record["task_labels"].filter(
      (label): label is string => typeof label === "string",
    );
  }
  const probabilities = record["probabilities"];
  if (probabilities && typeof probabilities === "object") {
    const tierScores: Record<string, number> = {};
    for (const [key, value] of Object.entries(
      probabilities as Record<string, unknown>,
    )) {
      if (typeof value === "number" && Number.isFinite(value)) {
        tierScores[key] = value;
      }
    }
    if (Object.keys(tierScores).length > 0) decision.tierScores = tierScores;
  }
  return decision;
}

/**
 * Margin signal for a route decision: the top-1 vs top-2 gap of the
 * preserved per-tier scores. Wide margin -> the classifier committed;
 * narrow margin -> hedge (wider tool pack, deeper verification). Pure;
 * undefined when the shim did not provide scores. This only surfaces the
 * signal -- no policy conditions on it yet.
 */
export function routeTierMargin(
  decision: SystemOneRouteDecision | undefined,
): number | undefined {
  try {
    const scores = decision?.tierScores;
    if (!scores) return undefined;
    const ranked = Object.values(scores)
      .filter((value) => Number.isFinite(value))
      .sort((a, b) => b - a);
    if (ranked.length < 2) return undefined;
    return ranked[0] - ranked[1];
  } catch {
    return undefined;
  }
}

// -- Z0: model routing -----------------------------------------------

/**
 * Resolve the model the route wants for this task. Returns a model id only
 * when model routing is on (SpeedStackSessionConfig.modelRouting) AND the
 * route named a model AND it differs from the current one. Otherwise
 * undefined: the pinned model stays. Never throws (fail-open).
 *
 * Independence: thinkingMode never gates model routing; modelRouting never
 * gates the Z1 effort wiring.
 */
export function resolveSystemOneModelTarget(input: {
  route: SystemOneRouteDecision | undefined;
  modelRouting?: boolean;
  currentModelId?: string;
}): string | undefined {
  try {
    if (!input.modelRouting) return undefined;
    // Best-value pick first: the shim's expected-utility ranking names the
    // top model. Falls back to the route's own model_id when the shim
    // didn't rank (older shims). The id always comes from the shim — never
    // hard-coded here.
    const ranked = input.route?.rankedModels?.[0]?.modelId;
    const routed = input.route?.modelId;
    const modelId =
      ranked && ranked.length > 0 ? ranked : routed;
    if (!modelId || modelId.length === 0) return undefined;
    if (input.currentModelId && modelId === input.currentModelId) return undefined;
    return modelId;
  } catch {
    return undefined;
  }
}

// -- Z1: effort wiring ------------------------------------------------

/**
 * Outcome of resolving what thinking to apply for one task:
 * - { kind: "tier", ... } -> bind the tier's reasoningLevel + token budget.
 * - { kind: "off" }        -> bind the model's own "thinking off" level.
 * - undefined              -> nothing to apply; keep the model untouched.
 */
export type EffortResolution =
  | {
      readonly kind: "tier";
      readonly tier: EffortTier;
      readonly options: Required<ModelOptions>;
      /**
       * Set when the SystemOne uncertain rule raised the tier one level
       * (Phase 3). Present for diagnostics only; the bound options already
       * reflect `tier`.
       */
      readonly uncertainBump?: { readonly from: EffortTier; readonly to: EffortTier };
    }
  | { readonly kind: "off" };

/** Outcome of the Phase-3 uncertain adjustment (pure). */
export interface UncertainEffortAdjustment {
  readonly thinking: ResolvedThinking | undefined;
  /** True when the uncertain rule raised the tier one level. */
  readonly bumped: boolean;
  readonly fromTier?: EffortTier;
  readonly toTier?: EffortTier;
}

/**
 * Phase-3 uncertain rule (effort half): when the route says
 * `uncertain === true`, route-driven effort moves up one tier
 * (low→medium→high→xhigh→ultra; ultra stays).
 *
 * Explicit user choices are sacred and never bumped: thinkingMode
 * "off", a pinned thinking tier, or a legacy config.effortTier pin all
 * pass through untouched. Thinking "off" resolves to the low policy row
 * downstream, unchanged. Pure; never throws.
 */
export function adjustEffortForUncertainty(
  thinking: ResolvedThinking | undefined,
  route: SystemOneRouteDecision | undefined,
  config: SpeedStackSessionConfig | undefined,
): UncertainEffortAdjustment {
  const passthrough: UncertainEffortAdjustment = { thinking, bumped: false };
  try {
    if (route?.uncertain !== true) return passthrough;
    if (!thinking || thinking.kind !== "tier") return passthrough;
    const mode: ThinkingMode | undefined = config?.thinkingMode;
    if (mode !== undefined && mode !== "auto") return passthrough;
    if (config?.effortTier) return passthrough;
    const toTier = bumpEffortTier(thinking.tier);
    if (toTier === thinking.tier) return passthrough;
    return {
      thinking: { kind: "tier", tier: toTier },
      bumped: true,
      fromTier: thinking.tier,
      toTier,
    };
  } catch {
    return passthrough;
  }
}

/**
 * Resolve the effective thinking for one task and its concrete ModelOptions
 * from a route decision + session config. Returns undefined when there is
 * nothing to apply (no route, no usable hint, no explicit tier/mode) — the
 * caller then keeps today's model options untouched.
 *
 * Thinking-mode precedence (see resolveThinkingTier; independent of Z0
 * model routing): thinkingMode "off" -> off; a pinned tier -> that tier;
 * legacy config.effortTier -> that tier; thinkingMode "auto"/unset -> the
 * route's effort hint.
 */
export function resolveEffortTierAndOptions(input: {
  route: SystemOneRouteDecision | undefined;
  config: SpeedStackSessionConfig | undefined;
  model: EffortTierModel;
}): EffortResolution | undefined {
  const hint = extractRouteEffortHint(input.route);
  const thinking = resolveThinkingTier({
    thinkingMode: input.config?.thinkingMode,
    explicitTier: input.config?.effortTier,
    routeHint: hint,
  });
  if (!thinking) return undefined;
  if (thinking.kind === "off") return { kind: "off" };
  const adjusted = adjustEffortForUncertainty(thinking, input.route, input.config);
  const tier = adjusted.thinking?.kind === "tier" ? adjusted.thinking.tier : thinking.tier;
  return {
    kind: "tier",
    tier,
    options: effortTierToModelOptions(input.model, tier),
    ...(adjusted.bumped && adjusted.fromTier && adjusted.toTier
      ? { uncertainBump: { from: adjusted.fromTier, to: adjusted.toTier } }
      : {}),
  };
}

/** Structural session state needed by the effort override. */
export interface SystemOneRouteHolder {
  readonly speedStackConfig?: SpeedStackSessionConfig;
  systemOneRouteValue?: SystemOneRouteDecision | undefined;
}

/**
 * Apply the route-driven effort override to a turn model. Returns the model
 * with reasoningLevel + maxOutputTokens bound when a tier resolves,
 * otherwise the model unchanged. Never throws (fail-open).
 */
export function applySystemOneEffortOverride<
  ModelT extends EffortTierModel & {
    bind(options?: ModelOptions): ModelT;
    readonly options: ModelOptions;
  },
>(
  holder: SystemOneRouteHolder,
  model: ModelT,
  logger?: Logger,
): ModelT {
  try {
    const resolved = resolveEffortTierAndOptions({
      route: holder.systemOneRouteValue,
      config: holder.speedStackConfig,
      model,
    });
    if (!resolved) return model;
    if (resolved.kind === "off") {
      // Thinking off: bind the model's own off level. Fail-open when the
      // model declares no off-like level (leave today's options untouched).
      const offLevel = findThinkingOffLevel(
        model.optionSpecs.reasoningLevel.values,
      );
      if (!offLevel) return model;
      logger?.debug("SystemOne thinking off applied", {
        event: "systemone.thinking.off",
        module: "core.speedstack",
        reasoningLevel: offLevel,
      });
      return model.bind({ ...model.options, reasoningLevel: offLevel });
    }
    if (resolved.uncertainBump) {
      logger?.debug("SystemOne uncertain effort bump applied", {
        event: "systemone.effort.uncertain_bump",
        module: "core.speedstack",
        fromTier: resolved.uncertainBump.from,
        toTier: resolved.uncertainBump.to,
        reason: "route uncertain=true: no tool pruning, effort +1 tier",
      });
    }
    logger?.debug("SystemOne effort override applied", {
      effortTier: resolved.tier,
      event: "systemone.effort.applied",
      maxOutputTokens: resolved.options.maxOutputTokens,
      module: "core.speedstack",
      reasoningLevel: resolved.options.reasoningLevel,
    });
    return model.bind({
      ...model.options,
      reasoningLevel: resolved.options.reasoningLevel,
      maxOutputTokens: resolved.options.maxOutputTokens,
    });
  } catch {
    return model;
  }
}

// -- Z2: MCP attach policy --------------------------------------------

/**
 * Tool-level MCP policy for one task: which configured MCP servers are
 * worth attaching. Consumed at MCP startup (runtime/methods/mcp.ts) to skip
 * starting clearly-irrelevant servers, and at request construction
 * (runtime/methods/turn-loop.ts) where the full per-tool shortlist is
 * applied via speedstack/tool-packs.ts.
 */
export interface McpToolLevelPolicy {
  /**
   * - "all": keep every server (fail-open default).
   * - "none": skip all MCP servers (confident trivial task).
   * - "subset": start only `keepServers`.
   */
  readonly mode: "all" | "none" | "subset";
  /** Human-readable reason; always logged with the decision. */
  readonly reason: string;
  /** Servers worth starting when mode === "subset". */
  readonly keepServers?: readonly string[];
  /** Servers pruned when mode === "subset". */
  readonly prunedServers?: readonly string[];
  /**
   * Preserved route signals (see SystemOneRouteDecision.tierScores):
   * per-tier scores + the top-1/top-2 margin. Informational for now --
   * the future ranking work consumes these; current thresholds ignore them.
   */
  readonly tierScores?: Readonly<Record<string, number>>;
  readonly margin?: number;
}

/** Outcome of the per-task MCP attach policy. */
export interface McpAttachPolicy {
  /** False -> skip MCP servers entirely (built-in tools only). */
  readonly attachMcp: boolean;
  /** Human-readable reason; always logged with the decision. */
  readonly reason: string;
  /** Tool-level policy: which servers are relevant for this task. */
  readonly toolPolicy: McpToolLevelPolicy;
}

export interface ResolveMcpAttachPolicyOptions {
  /** Configured MCP server names; enables the per-server subset policy. */
  readonly serverNames?: readonly string[];
  /** Task text; feeds label inference alongside route.taskLabels. */
  readonly taskText?: string;
}

/**
 * Decide whether MCP servers attach for this task, and -- when the caller
 * passes `serverNames` -- which of them are worth starting.
 *
 * Kill-switches (either forces full attach, i.e. today's behavior):
 *   1. Env: ZCODE_SPEEDSTACK_PRUNE=0
 *   2. Session config: SpeedStackSessionConfig.mcpPruning === false
 *
 * Default policy (conservative, live by default): attach built-in tools
 * only when the route says tier == "economy" with confidence >= 0.8.
 * Otherwise every server attaches, but with a confident route the policy
 * additionally names the per-task server subset so startup can skip
 * clearly-irrelevant servers (fail-open: anything ambiguous keeps all).
 * Everything else -- including a missing route decision (shim down:
 * fail-open) -- attaches the full MCP surface exactly like today.
 *
 * Never throws (fail-open).
 */
/** Preserved score signals for a tool-level policy, from the route. */
function routeScoreSignals(route: SystemOneRouteDecision | undefined): {
  tierScores?: Readonly<Record<string, number>>;
  margin?: number;
} {
  const tierScores = route?.tierScores;
  const margin = routeTierMargin(route);
  return {
    ...(tierScores ? { tierScores } : {}),
    ...(margin !== undefined ? { margin } : {}),
  };
}

export function resolveMcpAttachPolicy(
  route: SystemOneRouteDecision | undefined,
  config: SpeedStackSessionConfig | undefined,
  options?: ResolveMcpAttachPolicyOptions,
): McpAttachPolicy {
  try {
    const scoreSignals = routeScoreSignals(route);
    if (process.env[SYSTEMONE_PRUNE_KILL_SWITCH_ENV] === "0") {
      const reason = `kill-switch: ${SYSTEMONE_PRUNE_KILL_SWITCH_ENV}=0`;
      return { attachMcp: true, reason, toolPolicy: { mode: "all", reason } };
    }
    if (config?.mcpPruning === false) {
      const reason = "kill-switch: SpeedStackSessionConfig.mcpPruning=false";
      return { attachMcp: true, reason, toolPolicy: { mode: "all", reason } };
    }
    if (route?.uncertain === true) {
      // Phase-3 uncertain rule: an uncertain route NEVER prunes. The
      // classifier didn't commit, so the full tool surface stays available
      // and effort is raised elsewhere (adjustEffortForUncertainty).
      const reason =
        "route uncertain=true: no tool/MCP pruning (fail-open), effort bumped separately";
      return { attachMcp: true, reason, toolPolicy: { mode: "all", reason } };
    }
    if (
      route &&
      route.tier === "economy" &&
      route.confidence >= SYSTEMONE_PRUNE_CONFIDENCE_THRESHOLD
    ) {
      const reason =
        `route tier=economy confidence=${route.confidence} ` +
        `>= ${SYSTEMONE_PRUNE_CONFIDENCE_THRESHOLD}`;
      return {
        attachMcp: false,
        reason,
        toolPolicy: { mode: "none", reason, ...scoreSignals },
      };
    }
    const serverNames = options?.serverNames ?? [];
    if (serverNames.length > 0) {
      const shortlist = computeServerShortlist({
        route,
        config,
        taskText: options?.taskText,
        serverNames,
      });
      if (shortlist.pruned) {
        return {
          attachMcp: true,
          reason:
            `route tier=${route?.tier} confidence=${route?.confidence}: ` +
            `attaching all MCP servers, starting subset per task labels`,
          toolPolicy: {
            mode: "subset",
            reason: shortlist.reason,
            keepServers: shortlist.keepServers,
            prunedServers: shortlist.prunedServers,
            ...scoreSignals,
          },
        };
      }
      return {
        attachMcp: true,
        reason: shortlist.reason,
        toolPolicy: { mode: "all", reason: shortlist.reason, ...scoreSignals },
      };
    }
    const reason = route
      ? route.tier === "economy"
        ? `route tier=economy confidence=${route.confidence} below prune threshold ${SYSTEMONE_PRUNE_CONFIDENCE_THRESHOLD}`
        : `route tier=${route.tier} is not economy (pruning is economy-only)`
      : "no route decision (fail-open)";
    return { attachMcp: true, reason, toolPolicy: { mode: "all", reason } };
  } catch {
    const reason = "attach-policy computation threw (fail-open)";
    return { attachMcp: true, reason, toolPolicy: { mode: "all", reason } };
  }
}

// -- Z2: effort behavior policy ("effort means behavior") -----------------
//
// Companion to applySystemOneEffortOverride: where that function binds
// reasoningLevel/maxOutputTokens, this one resolves the full behavior
// policy (budgets, subagent allowance, verification, compaction, ...) for
// the turn. Same thinking precedence (off/pinned/auto -> route hint);
// thinking "off" resolves to the low policy row. Fail-open: no resolved
// effort and no configured budgets -> empty budgets (unbounded, today's
// behavior).

/** Resolved behavior policy + effective budgets for one task. */
export interface SystemOneBehaviorPolicyResolution {
  /**
   * The resolved effort tier, or undefined when nothing resolved (caller
   * keeps today's behavior). Thinking "off" maps to "low".
   */
  readonly tier: EffortTier | undefined;
  /** The behavior policy row, or undefined when no tier resolved. */
  readonly policy: EffortBehaviorPolicy | undefined;
  /**
   * Set when the SystemOne uncertain rule raised the tier one level
   * (Phase 3) — diagnostics for the planner rationale. The tier/policy/
   * budgets above already reflect the bumped tier.
   */
  readonly uncertainBump?: { readonly from: EffortTier; readonly to: EffortTier };
  /**
   * Effective per-turn budgets: policy-table raised defaults merged with the
   * package-1 ZCODE_BUDGET_* env surface (env wins). {} = unbounded.
   */
  readonly budgets: TurnBudgets;
}

/**
 * Resolve the effort behavior policy + effective turn budgets for the
 * current task, alongside applySystemOneEffortOverride. Applies the
 * Phase-3 uncertain rule (uncertain route -> effort +1 tier, logged for
 * the planner rationale). Never throws.
 */
export function resolveSystemOneBehaviorPolicy(
  holder: SystemOneRouteHolder,
  env: NodeJS.ProcessEnv = process.env,
  logger?: Logger,
): SystemOneBehaviorPolicyResolution {
  try {
    const thinking = resolveThinkingTier({
      thinkingMode: holder.speedStackConfig?.thinkingMode,
      explicitTier: holder.speedStackConfig?.effortTier,
      routeHint: extractRouteEffortHint(holder.systemOneRouteValue),
    });
    const adjusted = adjustEffortForUncertainty(
      thinking,
      holder.systemOneRouteValue,
      holder.speedStackConfig,
    );
    const adjustedThinking = adjusted.thinking;
    const tier =
      adjustedThinking?.kind === "tier"
        ? adjustedThinking.tier
        : adjustedThinking?.kind === "off"
          ? ("low" as EffortTier)
          : undefined;
    const policy = tier ? getEffortBehaviorPolicy(tier, env) : undefined;
    const budgets = resolveEffectiveTurnBudgets({
      routeTier: holder.systemOneRouteValue?.tier,
      effortTier: tier,
      env,
    });
    const uncertainBump =
      adjusted.bumped && adjusted.fromTier && adjusted.toTier
        ? { from: adjusted.fromTier, to: adjusted.toTier }
        : undefined;
    if (uncertainBump) {
      logger?.debug("SystemOne uncertain effort bump applied", {
        event: "systemone.behavior.uncertain_bump",
        module: "core.speedstack",
        fromTier: uncertainBump.from,
        toTier: uncertainBump.to,
        reason: "route uncertain=true: no tool pruning, effort +1 tier",
      });
    }
    return { tier, policy, budgets, ...(uncertainBump ? { uncertainBump } : {}) };
  } catch {
    return { tier: undefined, policy: undefined, budgets: {} };
  }
}
