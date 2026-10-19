// ============================================================
// Speed Stack: plan-then-execute mode
// ============================================================
//
// A strong planner model produces a plan artifact (PLAN.md); a fast executor
// model implements the steps. Per-task toggle via
// SpeedStackSessionConfig.planThenExecute, or the route-driven gate in
// plan-execute-gate.ts (heavy routes / risky labels / multi-file signals).
//
// LOCAL-ONLY model selection (Ryan's standing rule):
// - Role models are chosen ONLY from the locally available model list.
// - This module NEVER loads or unloads a model (pure selection).
// - NO hard-coded model ids anywhere. Resolution order:
//     1. router advisory (SystemOne route decision modelId / model mapping)
//     2. registry/config-driven preferences (explicit planner/executor picks,
//        config executor preference list)
//     3. explicit user configuration (ZCODE_PLAN_EXECUTE_* env)
//     4. first usable locally available fallback (planner) / smallest
//        usable locally available fallback (executor)
// - With one pinned model, planner and executor both use it. The runtime
//   never unloads/switches the loaded model for either pass.

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export const PLAN_ARTIFACT_FILENAME = "PLAN.md";

/** Kill switch: ZCODE_PLAN_EXECUTE=0 disables route-driven plan-then-execute. */
export const PLAN_EXECUTE_KILL_SWITCH_ENV = "ZCODE_PLAN_EXECUTE";

/** Explicit user config: planner model override (must be locally available). */
export const PLAN_EXECUTE_PLANNER_MODEL_ENV = "ZCODE_PLAN_EXECUTE_PLANNER_MODEL";

/** Explicit user config: executor model override (must be locally available). */
export const PLAN_EXECUTE_EXECUTOR_MODEL_ENV = "ZCODE_PLAN_EXECUTE_EXECUTOR_MODEL";

/**
 * Explicit user config: comma-separated, small-first executor preference
 * list. Must come from config/registry, not code — this is only the
 * env-var *name*, never a model id.
 */
export const PLAN_EXECUTE_EXECUTOR_PREFS_ENV = "ZCODE_PLAN_EXECUTE_EXECUTOR_PREFS";

/** Explicit user config: planner pass step cap (default below). */
export const PLAN_EXECUTE_PLANNER_MAX_STEPS_ENV = "ZCODE_PLAN_EXECUTE_PLANNER_MAX_STEPS";

/** Explicit user config: planner pass tool-call cap (default below). */
export const PLAN_EXECUTE_PLANNER_MAX_TOOL_CALLS_ENV =
  "ZCODE_PLAN_EXECUTE_PLANNER_MAX_TOOL_CALLS";

/** Planner budget: small by design, so a runaway plan can't eat the turn. */
export const DEFAULT_PLANNER_MAX_STEPS = 12;
export const DEFAULT_PLANNER_MAX_TOOL_CALLS = 30;

export interface PlanExecuteRouteInput {
  /** Route tier (advisory context only; never selects a model). */
  readonly routeTier?: string | undefined;
}

export interface PlanExecuteModelPreferences {
  /** Registry/config planner pick; wins when locally available. */
  readonly plannerModelId?: string | undefined;
  /** Registry/config executor pick; wins when locally available. */
  readonly executorModelId?: string | undefined;
  /**
   * Registry/config executor preference list (small-first, config-driven).
   * The executor takes the first entry that is locally available.
   */
  readonly executorModelPreferences?: readonly string[] | undefined;
}

export interface PlanExecuteRoleModels {
  readonly plannerModelId: string;
  readonly executorModelId: string;
}

export interface PlanExecutePlannerBudgets {
  readonly maxSteps: number;
  readonly maxToolCalls: number;
}

const EMBEDDING_MODEL_ID_PATTERN = /embed/i;

function isUsableModelId(modelId: string): boolean {
  return modelId.trim().length > 0 && !EMBEDDING_MODEL_ID_PATTERN.test(modelId);
}

function isAvailableModelId(
  available: readonly string[],
  modelId: string | undefined,
): modelId is string {
  return (
    typeof modelId === "string" &&
    isUsableModelId(modelId) &&
    available.includes(modelId)
  );
}

/** Parse a parameter count like "35b"/"9b"/"12b"/"1.5b" out of a model id. */
export function parseModelSizeBillions(modelId: string): number | undefined {
  const match = /(\d+(?:\.\d+)?)\s*b\b/i.exec(modelId);
  if (!match) return undefined;
  const value = Number.parseFloat(match[1] as string);
  return Number.isFinite(value) && value > 0 ? value : undefined;
}

function firstUsableModel(available: readonly string[]): string | undefined {
  return available.find(isUsableModelId);
}

/**
 * Smallest-first pick from the locally available list (no hard-coded ids).
 * Models whose size can't be parsed sort last (after every sized model).
 */
function smallestAvailableModel(
  available: readonly string[],
): string | undefined {
  const usable = available.filter(isUsableModelId);
  if (usable.length === 0) return undefined;
  return [...usable].sort((a, b) => {
    const sizeA = parseModelSizeBillions(a) ?? Number.POSITIVE_INFINITY;
    const sizeB = parseModelSizeBillions(b) ?? Number.POSITIVE_INFINITY;
    return sizeA - sizeB;
  })[0];
}

/**
 * Resolve planner/executor model ids from the LOCALLY AVAILABLE list.
 * Pure selection — never loads, unloads, or otherwise touches model state.
 * Order: explicit preferences (registry/config/env) > router advisory >
 * first/smallest usable locally available fallback.
 */
export function resolvePlanExecuteModels(
  availableModelIds: readonly string[],
  route: PlanExecuteRouteInput = {},
  preferred: PlanExecuteModelPreferences = {},
): PlanExecuteRoleModels {
  const available = availableModelIds.filter((id) => typeof id === "string");
  const planner =
    (isAvailableModelId(available, preferred.plannerModelId)
      ? preferred.plannerModelId
      : undefined) ??
    firstUsableModel(available);
  if (!planner) {
    throw new Error(
      "plan-then-execute needs at least one locally available model",
    );
  }
  const executor =
    (isAvailableModelId(available, preferred.executorModelId)
      ? preferred.executorModelId
      : undefined) ??
    preferred.executorModelPreferences?.find((id) =>
      isAvailableModelId(available, id),
    ) ??
    smallestAvailableModel(available) ??
    planner;
  return { plannerModelId: planner, executorModelId: executor };
}

function isFalsyEnvValue(value: string | undefined): boolean {
  if (value === undefined) return false;
  const normalized = value.trim().toLowerCase();
  return (
    normalized === "0" ||
    normalized === "false" ||
    normalized === "no" ||
    normalized === "off"
  );
}

/**
 * Global kill switch for route-driven plan-then-execute. Default: enabled.
 * Fail-open: unreadable env never disables.
 */
export function isPlanThenExecuteDisabled(
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  try {
    return isFalsyEnvValue(env[PLAN_EXECUTE_KILL_SWITCH_ENV]);
  } catch {
    return false;
  }
}

/**
 * Explicit user configuration for role models (ZCODE_PLAN_EXECUTE_* env).
 * Overrides the router advisory; still must be locally available.
 */
export function readPlanExecuteModelPreferences(
  env: NodeJS.ProcessEnv = process.env,
): PlanExecuteModelPreferences {
  try {
    const prefs: {
      plannerModelId?: string;
      executorModelId?: string;
      executorModelPreferences?: string[];
    } = {};
    const planner = env[PLAN_EXECUTE_PLANNER_MODEL_ENV]?.trim();
    if (planner) prefs.plannerModelId = planner;
    const executor = env[PLAN_EXECUTE_EXECUTOR_MODEL_ENV]?.trim();
    if (executor) prefs.executorModelId = executor;
    const executorPrefs = env[PLAN_EXECUTE_EXECUTOR_PREFS_ENV]
      ?.split(",")
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0);
    if (executorPrefs && executorPrefs.length > 0) {
      prefs.executorModelPreferences = executorPrefs;
    }
    return prefs;
  } catch {
    return {};
  }
}

function parsePositiveInt(
  raw: string | undefined,
  fallback: number,
): number {
  if (raw === undefined) return fallback;
  const parsed = Number.parseInt(raw.trim(), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/**
 * Planner-pass budgets (small by design). Env overrides; invalid values
 * fall back to the defaults. Fail-open.
 */
export function resolvePlannerBudgets(
  env: NodeJS.ProcessEnv = process.env,
): PlanExecutePlannerBudgets {
  try {
    return {
      maxSteps: parsePositiveInt(
        env[PLAN_EXECUTE_PLANNER_MAX_STEPS_ENV],
        DEFAULT_PLANNER_MAX_STEPS,
      ),
      maxToolCalls: parsePositiveInt(
        env[PLAN_EXECUTE_PLANNER_MAX_TOOL_CALLS_ENV],
        DEFAULT_PLANNER_MAX_TOOL_CALLS,
      ),
    };
  } catch {
    return {
      maxSteps: DEFAULT_PLANNER_MAX_STEPS,
      maxToolCalls: DEFAULT_PLANNER_MAX_TOOL_CALLS,
    };
  }
}

/** Session-scoped plan directory (shared ground truth on disk). */
export function resolvePlanExecuteSessionDir(
  homeDir: string = homedir(),
  sessionId: string,
): string {
  const safeSessionId =
    sessionId.replace(/[^a-zA-Z0-9_-]/g, "_") || "session";
  return join(homeDir, ".zcode-local", "speedstack", "plans", safeSessionId);
}

/** Prompt for the planner pass: produce an explicit step-by-step plan. */
export function buildPlannerPrompt(task: string): string {
  return [
    "You are the PLANNER in a plan-then-execute workflow.",
    "Produce a concrete, step-by-step implementation plan and nothing else.",
    "",
    "Rules:",
    "- One numbered step per action; each step names the files it touches.",
    "- Mark steps that can run in parallel with [parallel].",
    "- End the plan with a verification step (build + tests to run).",
    "- Do NOT write code in the plan; the executor implements it.",
    "",
    `Task: ${task}`,
  ].join("\n");
}

/** Prompt for the executor pass: implement the given plan step by step. */
export function buildExecutorPrompt(planMarkdown: string): string {
  return [
    "You are the EXECUTOR in a plan-then-execute workflow.",
    "Implement the plan below exactly, step by step.",
    "If a step is impossible as written, stop and explain why instead of improvising.",
    "",
    "Plan:",
    "```markdown",
    planMarkdown,
    "```",
  ].join("\n");
}

/** Persist the plan artifact (file-based state, Ralph-compatible). */
export async function writePlanArtifact(
  sessionDir: string,
  planMarkdown: string,
): Promise<string> {
  await mkdir(sessionDir, { recursive: true });
  const artifactPath = join(sessionDir, PLAN_ARTIFACT_FILENAME);
  await writeFile(artifactPath, planMarkdown, "utf8");
  return artifactPath;
}

/** Read the plan artifact back; undefined when absent. */
export async function readPlanArtifact(
  sessionDir: string,
): Promise<string | undefined> {
  try {
    return await readFile(join(sessionDir, PLAN_ARTIFACT_FILENAME), "utf8");
  } catch {
    return undefined;
  }
}
