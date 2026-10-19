// ============================================================
// Speed Stack eval harness (Packages 1+2)
// ============================================================
//
// Deterministic policy/schema measurement + live-stack smoke + budget
// replay. See eval/README.md for the tune loop.
//
// Modes:
//   policy  (default) For each battery task: route (live shim or
//           --offline fixture) -> computeToolShortlist over the REAL
//           built-in tool schemas -> resolveTurnBudgets -> record
//           schema-token deltas, labels/scores/margin, budgets, and
//           optional simulated usage -> budget-hit verdict.
//   smoke   Verify the local stack answers without running inference:
//           POST the SystemOne shim route endpoint, GET LM Studio
//           /v1/models. Never switches or unloads models.
//   replay  Re-evaluate a previous report's recorded usage against a NEW
//           budget config (--budget flags): the tune loop. No shim needed.
//
// Budgets are config-driven (see ../turn-budgets.ts). --budget flags set
// the ZCODE_BUDGET_* env vars in-process for the run. Per-task EFFECTIVE
// budgets (Package 2) merge the effort behavior policy's raised defaults
// (see ../effort-tiers.ts) with the ZCODE_BUDGET_* env surface, which wins
// wherever set; the effective config is always printed in the report so a
// tuning run is reproducible.

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  fetchSystemOneRouteDecision,
  routeTierMargin,
  SYSTEMONE_ROUTE_ENDPOINT,
  type SystemOneRouteDecision,
} from "../systemone-route.js";
import {
  computeToolShortlist,
  type ToolSchemaDescriptor,
  type ToolShortlist,
} from "../tool-packs.js";
import {
  checkBudgetHit,
  describeTurnBudgets,
  resolveTurnBudgets,
  type BudgetHit,
  type BudgetUsage,
  type TurnBudgets,
} from "../turn-budgets.js";
import {
  getEffortBehaviorPolicy,
  isPlanThenExecuteEligible,
  parseEffortTier,
  resolveEffectiveTurnBudgets,
} from "../effort-tiers.js";
import { decidePlanThenExecute } from "../plan-execute-gate.js";
import { EVAL_TASKS, type EvalTask } from "./tasks.js";

const here = dirname(fileURLToPath(import.meta.url));
const LM_STUDIO_MODELS_ENDPOINT = "http://127.0.0.1:1234/v1/models";

export type HarnessMode = "policy" | "smoke" | "replay";

export interface HarnessOptions {
  mode: HarnessMode;
  offline: boolean;
  routeFixturePath?: string;
  schemaSizesPath?: string;
  simulatePath?: string;
  runPath?: string;
  outPath?: string;
  shimEndpoint: string;
  budgetFlags: string[];
  budgetDefaultFlag?: string;
}

export interface RouteFixtureFile {
  recordedAt?: string;
  tasks: Record<string, SystemOneRouteDecision>;
}

export interface SchemaSizesFile {
  toolCount: number;
  totalChars: number;
  tools: { name: string; chars: number; descriptionChars: number; schemaChars: number }[];
}

export interface HarnessRouteRecord {
  tier: string;
  confidence: number;
  labels: string[];
  tierScores?: Record<string, number>;
  margin?: number;
  effort?: string;
  modelId?: string;
  source: "live-shim" | "fixture";
  latencyMs?: number;
}

export interface HarnessTaskResult {
  id: string;
  task: string;
  route: HarnessRouteRecord | null;
  toolPack: {
    pruned: boolean;
    keepCount: number;
    totalCount: number;
    schemaTokensBefore: number;
    schemaTokensAfter: number;
    tokensSaved: number;
    savedPct: number;
    reason: string;
  } | null;
  budgets: TurnBudgets;
  budgetDescription: string;
  /** Package 2: the resolved effort behavior policy row (null when no effort resolved). */
  effortPolicy: {
    tier: string;
    subagents: string;
    subagentMaxTurns: number;
    readBreadth: number;
    verificationPasses: number;
    compactionAggressiveness: number;
  } | null;
  /** Package 2: plan-then-execute eligibility gate for this task. */
  planThenExecuteEligible: boolean;
  /** Package 3: route-driven plan-vs-direct decision (pure gate, no inference). */
  planExecute: { plan: boolean; reason: string } | null;
  usage: (BudgetUsage & { simulated: boolean }) | null;
  budgetHit: BudgetHit | null;
  error?: string;
}

export interface HarnessReport {
  harness: "zcode-speedstack-eval/1";
  mode: HarnessMode;
  timestamp: string;
  replayOf?: string;
  shim: { endpoint: string; reachable: boolean; latencyMs?: number };
  lmstudio: { endpoint: string; reachable: boolean; models?: string[]; skipped?: string };
  budgets: Record<string, TurnBudgets>;
  tasks: HarnessTaskResult[];
  summary: {
    taskCount: number;
    routedCount: number;
    prunedCount: number;
    totalTokensBefore: number;
    totalTokensAfter: number;
    totalTokensSaved: number;
    budgetHitCount: number;
    planExecuteCount: number;
    errors: string[];
  };
}

/** Apply --budget flags by setting ZCODE_BUDGET_* in-process for this run. */
export function applyBudgetFlags(flags: string[], budgetDefaultFlag?: string): void {
  for (const flag of flags) {
    const parts = flag.split(":");
    if (parts.length !== 3) {
      throw new Error(
        `--budget expects <tier>:<maxSteps>:<maxToolCalls>, got ${JSON.stringify(flag)}`,
      );
    }
    const [tierRaw, stepsRaw, callsRaw] = parts as [string, string, string];
    const tier = tierRaw.trim().toUpperCase();
    if (!/^[A-Z]+$/.test(tier)) {
      throw new Error(`--budget tier must be a-z, got ${JSON.stringify(tierRaw)}`);
    }
    setBudgetVar(`ZCODE_BUDGET_${tier}_MAX_STEPS`, stepsRaw);
    setBudgetVar(`ZCODE_BUDGET_${tier}_MAX_TOOL_CALLS`, callsRaw);
  }
  if (budgetDefaultFlag) {
    const parts = budgetDefaultFlag.split(":");
    if (parts.length !== 2) {
      throw new Error(
        `--budget-default expects <maxSteps>:<maxToolCalls>, got ${JSON.stringify(budgetDefaultFlag)}`,
      );
    }
    setBudgetVar("ZCODE_BUDGET_DEFAULT_MAX_STEPS", parts[0] as string);
    setBudgetVar("ZCODE_BUDGET_DEFAULT_MAX_TOOL_CALLS", parts[1] as string);
  }
}

function setBudgetVar(name: string, raw: string): void {
  const value = raw.trim();
  if (value === "-" || value === "") {
    delete process.env[name];
    return;
  }
  process.env[name] = value;
}

/**
 * Build tool descriptors whose char counts exactly match the measured
 * sizes, so estimateToolSchemaTokens() reports real numbers. The filler
 * content is meaningless; only the lengths matter (documented).
 */
export function buildDescriptorsFromSizes(sizes: SchemaSizesFile): ToolSchemaDescriptor[] {
  return sizes.tools.map((tool) => ({
    name: tool.name,
    description: "x".repeat(Math.max(0, tool.descriptionChars)),
    inputSchema:
      tool.schemaChars > 2 ? "y".repeat(tool.schemaChars - 2) : undefined,
  }));
}

function toRouteRecord(
  decision: SystemOneRouteDecision,
  source: "live-shim" | "fixture",
  latencyMs?: number,
): HarnessRouteRecord {
  const record: HarnessRouteRecord = {
    tier: decision.tier,
    confidence: decision.confidence,
    labels: [...(decision.taskLabels ?? [])],
    source,
  };
  if (decision.tierScores) record.tierScores = { ...decision.tierScores };
  const margin = routeTierMargin(decision);
  if (margin !== undefined) record.margin = margin;
  if (decision.effort !== undefined) record.effort = decision.effort;
  if (latencyMs !== undefined) record.latencyMs = latencyMs;
  return record;
}

async function checkHttp(
  url: string,
  init?: RequestInit,
  timeoutMs = 5_000,
): Promise<{ reachable: boolean; latencyMs?: number; json?: unknown }> {
  const start = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const latencyMs = Date.now() - start;
    if (!response.ok) return { reachable: false, latencyMs };
    let json: unknown;
    try {
      json = await response.json();
    } catch {
      json = undefined;
    }
    return { reachable: true, latencyMs, json };
  } catch {
    return { reachable: false, latencyMs: Date.now() - start };
  } finally {
    clearTimeout(timer);
  }
}

function loadJson<T>(path: string): T {
  return JSON.parse(readFileSync(path, "utf8")) as T;
}

function shortlistToRecord(shortlist: ToolShortlist, totalCount: number): NonNullable<HarnessTaskResult["toolPack"]> {
  const saved = shortlist.schemaTokensBefore - shortlist.schemaTokensAfter;
  return {
    pruned: shortlist.pruned,
    keepCount: shortlist.keepNames.size,
    totalCount,
    schemaTokensBefore: shortlist.schemaTokensBefore,
    schemaTokensAfter: shortlist.schemaTokensAfter,
    tokensSaved: saved,
    savedPct:
      shortlist.schemaTokensBefore > 0
        ? Math.round((saved / shortlist.schemaTokensBefore) * 1000) / 10
        : 0,
    reason: shortlist.reason,
  };
}

async function runPolicyTask(
  task: EvalTask,
  ctx: {
    descriptors: ToolSchemaDescriptor[];
    fixtures: RouteFixtureFile | null;
    shimEndpoint: string;
    shimReachable: { value: boolean; latencyMs?: number };
    simulate: Record<string, BudgetUsage>;
  },
): Promise<HarnessTaskResult> {
  const base = {
    id: task.id,
    task: task.task,
    budgets: {} as TurnBudgets,
    budgetDescription: "",
    effortPolicy: null as HarnessTaskResult["effortPolicy"],
    planThenExecuteEligible: false,
    planExecute: null,
    usage: null as HarnessTaskResult["usage"],
    budgetHit: null as BudgetHit | null,
  };
  try {
    let decision: SystemOneRouteDecision | undefined;
    let source: "live-shim" | "fixture" = "live-shim";
    let latencyMs: number | undefined;
    if (ctx.fixtures?.tasks[task.id]) {
      decision = ctx.fixtures.tasks[task.id];
      source = "fixture";
    } else {
      const start = Date.now();
      decision = await fetchSystemOneRouteDecision(task.task, {
        endpoint: ctx.shimEndpoint,
      });
      latencyMs = Date.now() - start;
      ctx.shimReachable.value = decision !== undefined;
      if (latencyMs !== undefined && decision !== undefined) {
        ctx.shimReachable.latencyMs = latencyMs;
      }
    }
    if (!decision) {
      return {
        ...base,
        route: null,
        toolPack: null,
        error: `no route decision (${source === "fixture" ? "missing fixture" : "shim unreachable"})`,
      };
    }
    const route = toRouteRecord(decision, source, latencyMs);
    const shortlist = computeToolShortlist({
      route: decision,
      taskText: task.task,
      tools: ctx.descriptors,
    });
    // Package 2: effective per-turn budgets = policy raised defaults merged
    // with the ZCODE_BUDGET_* env surface (env wins wherever set).
    const effortTier = parseEffortTier(decision.effort);
    const behaviorPolicy = effortTier
      ? getEffortBehaviorPolicy(effortTier, process.env)
      : undefined;
    const budgets = resolveEffectiveTurnBudgets({
      routeTier: decision.tier,
      effortTier,
      env: process.env,
    });
    const simulated = ctx.simulate[task.id];
    const usage = simulated
      ? { steps: simulated.steps, toolCalls: simulated.toolCalls, simulated: true }
      : null;
    return {
      ...base,
      route,
      toolPack: shortlistToRecord(shortlist, ctx.descriptors.length),
      budgets,
      budgetDescription: describeTurnBudgets(decision.tier, budgets),
      effortPolicy: behaviorPolicy
        ? {
            tier: behaviorPolicy.tier,
            subagents: behaviorPolicy.subagents,
            subagentMaxTurns: behaviorPolicy.subagentMaxTurns,
            readBreadth: behaviorPolicy.readBreadth,
            verificationPasses: behaviorPolicy.verificationPasses,
            compactionAggressiveness: behaviorPolicy.compactionAggressiveness,
          }
        : null,
      planThenExecuteEligible: isPlanThenExecuteEligible({
        policy: behaviorPolicy,
        routeTier: decision.tier,
      }),
      // Package 3: route-driven plan-then-execute gate (pure, deterministic).
      planExecute: (() => {
        const gate = decidePlanThenExecute({
          policy: behaviorPolicy,
          routeTier: decision.tier,
          routeConfidence: decision.confidence,
          taskLabels: decision.taskLabels,
          taskText: task.task,
        });
        return { plan: gate.plan, reason: gate.reason };
      })(),
      usage,
      budgetHit: usage ? checkBudgetHit(usage, budgets) : null,
    };
  } catch (error) {
    return {
      ...base,
      route: null,
      toolPack: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export async function runPolicyMode(options: HarnessOptions): Promise<HarnessReport> {
  applyBudgetFlags(options.budgetFlags, options.budgetDefaultFlag);
  const sizes = loadJson<SchemaSizesFile>(
    options.schemaSizesPath ?? join(here, "tool-schema-sizes.json"),
  );
  const descriptors = buildDescriptorsFromSizes(sizes);
  const fixtures = options.offline
    ? loadJson<RouteFixtureFile>(
        options.routeFixturePath ?? join(here, "route-fixtures.json"),
      )
    : null;
  const simulate: Record<string, BudgetUsage> = options.simulatePath
    ? loadJson<Record<string, BudgetUsage>>(resolve(options.simulatePath))
    : {};
  const shimReachable: { value: boolean; latencyMs?: number } = { value: false };

  const tasks: HarnessTaskResult[] = [];
  for (const task of EVAL_TASKS) {
    tasks.push(await runPolicyTask(task, { descriptors, fixtures, shimEndpoint: options.shimEndpoint, shimReachable, simulate }));
  }
  return finalizeReport("policy", options, tasks, shimReachable, undefined);
}

export async function runSmokeMode(options: HarnessOptions): Promise<HarnessReport> {
  applyBudgetFlags(options.budgetFlags, options.budgetDefaultFlag);
  const shim = await checkHttp(options.shimEndpoint, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ task: "smoke test" }),
  });
  // Live LM Studio probe is opt-in. Default: eval tooling never touches the
  // model server — Ryan's rule is that nothing invokes the models unless the
  // app is being used. Set ZCODE_EVAL_LIVE=1 to probe the live server.
  const evalLive = process.env.ZCODE_EVAL_LIVE === "1";
  const lmstudio = evalLive
    ? await checkHttp(LM_STUDIO_MODELS_ENDPOINT)
    : { reachable: false, latencyMs: undefined as number | undefined, json: undefined as unknown };
  const models =
    lmstudio.json && typeof lmstudio.json === "object" && Array.isArray((lmstudio.json as { data?: unknown }).data)
      ? ((lmstudio.json as { data: { id?: string }[] }).data.map((m) => m.id ?? "?"))
      : undefined;
  const report = finalizeReport(
    "smoke",
    options,
    [],
    { value: shim.reachable, latencyMs: shim.latencyMs },
    { value: lmstudio.reachable, models },
  );
  report.lmstudio = {
    endpoint: LM_STUDIO_MODELS_ENDPOINT,
    reachable: lmstudio.reachable,
    ...(models ? { models } : {}),
    ...(!evalLive ? { skipped: "set ZCODE_EVAL_LIVE=1 to probe the live model server" } : {}),
  };
  return report;
}

export async function runReplayMode(options: HarnessOptions): Promise<HarnessReport> {
  if (!options.runPath) throw new Error("replay mode requires --run <report.json>");
  applyBudgetFlags(options.budgetFlags, options.budgetDefaultFlag);
  const prior = loadJson<HarnessReport>(resolve(options.runPath));
  const tasks: HarnessTaskResult[] = prior.tasks.map((task) => {
    if (!task.usage) {
      return { ...task, error: "replay needs recorded usage; task has none" };
    }
    const replayEffortTier = parseEffortTier(task.route?.effort);
    const budgets = resolveEffectiveTurnBudgets({
      routeTier: task.route?.tier,
      effortTier: replayEffortTier,
      env: process.env,
    });
    return {
      ...task,
      budgets,
      budgetDescription: describeTurnBudgets(task.route?.tier, budgets),
      budgetHit: checkBudgetHit(task.usage, budgets),
    };
  });
  const report = finalizeReport("replay", options, tasks, {
    value: prior.shim.reachable,
    latencyMs: prior.shim.latencyMs,
  }, undefined);
  report.replayOf = options.runPath;
  report.shim = prior.shim;
  report.lmstudio = prior.lmstudio;
  return report;
}

function effectiveBudgets(): Record<string, TurnBudgets> {
  const tiers = ["economy", "balanced", "heavy"];
  const out: Record<string, TurnBudgets> = {};
  for (const tier of tiers) out[tier] = resolveTurnBudgets(tier);
  out["default"] = resolveTurnBudgets(undefined);
  return out;
}

function finalizeReport(
  mode: HarnessMode,
  options: HarnessOptions,
  tasks: HarnessTaskResult[],
  shimReachable: { value: boolean; latencyMs?: number },
  _lmstudio: { value: boolean; models?: string[] } | undefined,
): HarnessReport {
  const summary = {
    taskCount: tasks.length,
    routedCount: tasks.filter((t) => t.route !== null).length,
    prunedCount: tasks.filter((t) => t.toolPack?.pruned).length,
    totalTokensBefore: tasks.reduce((s, t) => s + (t.toolPack?.schemaTokensBefore ?? 0), 0),
    totalTokensAfter: tasks.reduce((s, t) => s + (t.toolPack?.schemaTokensAfter ?? 0), 0),
    totalTokensSaved: 0,
    budgetHitCount: tasks.filter((t) => t.budgetHit?.anyHit).length,
    planExecuteCount: tasks.filter((t) => t.planExecute?.plan === true).length,
    errors: tasks.flatMap((t) => (t.error ? [`${t.id}: ${t.error}`] : [])),
  };
  summary.totalTokensSaved = summary.totalTokensBefore - summary.totalTokensAfter;
  return {
    harness: "zcode-speedstack-eval/1",
    mode,
    timestamp: new Date().toISOString(),
    shim: {
      endpoint: options.shimEndpoint,
      reachable: shimReachable.value,
      ...(shimReachable.latencyMs !== undefined ? { latencyMs: shimReachable.latencyMs } : {}),
    },
    lmstudio: { endpoint: LM_STUDIO_MODELS_ENDPOINT, reachable: false },
    budgets: effectiveBudgets(),
    tasks,
    summary,
  };
}

export function writeReport(report: HarnessReport, outPath?: string): void {
  const json = JSON.stringify(report, null, 2) + "\n";
  if (outPath) {
    writeFileSync(resolve(outPath), json);
    console.error(`report written to ${resolve(outPath)}`);
  } else {
    process.stdout.write(json);
  }
}

export function printSummary(report: HarnessReport): void {
  const lines = [
    `mode=${report.mode} tasks=${report.summary.taskCount} routed=${report.summary.routedCount} pruned=${report.summary.prunedCount}`,
    `schema tokens: before=${report.summary.totalTokensBefore} after=${report.summary.totalTokensAfter} saved=${report.summary.totalTokensSaved}`,
    `budgets: ${Object.entries(report.budgets).map(([t, b]) => `${t}=[steps:${b.maxSteps ?? "-"},calls:${b.maxToolCalls ?? "-"}]`).join(" ")}`,
    `budget hits: ${report.summary.budgetHitCount}/${report.summary.taskCount}`,
    `plan-then-execute: ${report.summary.planExecuteCount}/${report.summary.taskCount} tasks`,
  ];
  for (const task of report.tasks) {
    const tp = task.toolPack;
    const hit = task.budgetHit;
    lines.push(
      `  ${task.id}: tier=${task.route?.tier ?? "?"} conf=${task.route?.confidence ?? "?"}` +
        ` margin=${task.route?.margin?.toFixed(3) ?? "-"}` +
        (tp ? ` pack=${tp.keepCount}/${tp.totalCount} saved=${tp.tokensSaved}` : " pack=n/a") +
        ` budget=[${task.budgets.maxSteps ?? "-"},${task.budgets.maxToolCalls ?? "-"}]` +
        (task.usage ? ` usage=[${task.usage.steps},${task.usage.toolCalls}]${task.usage.simulated ? "(sim)" : ""} hit=${hit?.anyHit}` : " usage=n/a") +
        (task.planExecute ? ` plan=${task.planExecute.plan ? "yes" : "no"}(${task.planExecute.reason})` : "") +
        (task.error ? ` ERROR: ${task.error}` : ""),
    );
  }
  for (const error of report.summary.errors) lines.push(`  ERROR ${error}`);
  console.error(lines.join("\n"));
}

// Re-export for the CLI entry.
export { SYSTEMONE_ROUTE_ENDPOINT };
