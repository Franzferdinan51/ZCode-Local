// ============================================================
// Speed Stack: SystemOne plan ranking (Phase 3)
// ============================================================
//
// When plan-then-execute runs, the planner can produce MULTIPLE candidate
// plans instead of one; the SystemOne shim scores them
// (POST /v1/systemone/rank-plans -> P(plan succeeds | task) minus a cost
// penalty) and the executor runs the highest-ranked candidate. Advisory
// only, fail-open everywhere:
//
//   - The planner prompt asks for N candidates separated by
//     `=== CANDIDATE k ===` markers; if the response can't be parsed into
//     >= 2 candidates, the turn runs today's single-plan path unchanged.
//   - The rank-plans call never throws: shim down / timeout / non-200 /
//     old schema (no `ranking` key) -> fall back to the decide engine
//     (POST /v1/systemone/decide, Mapika/decider-4b via ./systemone-decide.js)
//     which picks the best candidate; only when that also fails open does
//     the first candidate run, exactly like today's single plan.
//   - Pinned/deterministic config (ZCODE_PLAN_PIN=1 or
//     SpeedStackSessionConfig.planPin) forces one candidate and skips the
//     ranking call entirely.
//   - The full ranking is logged in the run diagnostics
//     (event "plan_execute.ranked_plans") before execution.
//
// This module is intentionally pure (no runtime imports) so it stays
// runnable under plain `node --test` type-stripping like its speedstack
// siblings. No model IDs appear anywhere here: selection is by rank only.

import {
  decide,
  SYSTEMONE_DECIDE_MIN_WINNER_PROBABILITY,
  SYSTEMONE_DECIDE_STATE_CHARS,
  type SystemOneDecideAnswer,
} from "./systemone-decide.js";

/**
 * Explicit user pin: deterministic single-plan behavior. Any of
 * 1/true/yes/on forces one candidate and skips ranking entirely.
 */
export const PLAN_PIN_ENV = "ZCODE_PLAN_PIN";

/** Explicit user config: how many candidate plans the planner produces. */
export const PLAN_CANDIDATES_ENV = "ZCODE_PLAN_CANDIDATES";

/** Default candidate count when plan-then-execute runs unpinned. */
export const DEFAULT_PLAN_CANDIDATE_COUNT = 2;

/** Sanity cap: never ask the planner for more candidates than this. */
export const MAX_PLAN_CANDIDATE_COUNT = 8;

/** Local SystemOne shim plan-ranking endpoint. */
export const SYSTEMONE_RANK_PLANS_ENDPOINT =
  "http://127.0.0.1:8765/v1/systemone/rank-plans";

/** Hard bound on the rank-plans lookup; batched, one engine call. */
export const SYSTEMONE_RANK_PLANS_TIMEOUT_MS = 8_000;

/** Char budget per plan when a candidate plan becomes decider criteria. */
export const PLAN_DECIDE_CRITERION_CHARS = 2_000;

/** Instructions sent with the decider plan-choice question. */
export const PLAN_DECIDE_INSTRUCTIONS =
  "Pick the single best implementation plan for this task. " +
  "Reply with exactly one option id.";

// NOTE: the canonical master kill-switch export lives in
// ./systemone-route.ts as SYSTEMONE_KILL_SWITCH_ENV; the literal is
// duplicated here so this module keeps zero runtime imports
// (node --test type-stripping).
const SYSTEMONE_MASTER_KILL_SWITCH_ENV = "ZCODE_SYSTEMONE";

/** One candidate plan produced by the planner pass. */
export interface PlanCandidate {
  readonly id: string;
  readonly text: string;
}

/** One scored entry from the shim's rank-plans response. */
export interface PlanRankEntry {
  readonly id: string;
  /** Combined score (higher = better); null when the shim couldn't score. */
  readonly score: number | null;
  readonly pSuccess?: number | undefined;
  readonly costPenalty?: number | undefined;
  readonly estSteps?: number | undefined;
}

/** Parsed rank-plans response. */
export interface PlanRanking {
  readonly task: string;
  readonly tier?: string | undefined;
  /** Where the ranking came from: rank-plans, or the decider fallback. */
  readonly source?: "rank-plans" | "decide" | undefined;
  readonly ranking: readonly PlanRankEntry[];
}

/** Outcome of picking the plan to execute. */
export interface PlanSelection {
  /** Index into the candidates array. */
  readonly index: number;
  readonly plan: PlanCandidate;
  readonly reason:
    | "ranked-winner"
    | "decided-winner"
    | "no-usable-scores"
    | "no-ranking"
    | "single-plan";
  /** Shim ranking order (plan ids), for diagnostics. */
  readonly rankedIds: readonly string[];
}

/** Minimal structural view of the pin/candidate config surface. */
export interface PlanRankingConfigView {
  readonly planPin?: boolean | undefined;
  readonly planCandidates?: number | undefined;
}

/**
 * True when plan ranking is pinned off: explicit deterministic config wins
 * over the route/shim every time. Fail-open: unreadable env/config never
 * pins (ranking stays live).
 */
export function isPlanPinned(
  env: NodeJS.ProcessEnv = process.env,
  config?: PlanRankingConfigView | undefined,
): boolean {
  try {
    if (config?.planPin === true) return true;
    const raw = env[PLAN_PIN_ENV]?.trim().toLowerCase();
    return (
      raw === "1" || raw === "true" || raw === "yes" || raw === "on"
    );
  } catch {
    return false;
  }
}

function parsePositiveInt(raw: string | undefined): number | undefined {
  if (raw === undefined) return undefined;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) return undefined;
  const value = Number.parseInt(trimmed, 10);
  return value > 0 ? value : undefined;
}

/**
 * How many candidate plans the planner should produce. Pinned -> 1.
 * Env ZCODE_PLAN_CANDIDATES wins, then config.planCandidates, then the
 * default (2). Garbage fails open to the default; internal errors to 1
 * (today's single-plan behavior).
 */
export function resolvePlanCandidateCount(
  config?: PlanRankingConfigView | undefined,
  env: NodeJS.ProcessEnv = process.env,
): number {
  try {
    if (isPlanPinned(env, config)) return 1;
    const fromEnv = parsePositiveInt(env[PLAN_CANDIDATES_ENV]);
    if (fromEnv !== undefined) {
      return Math.min(fromEnv, MAX_PLAN_CANDIDATE_COUNT);
    }
    const fromConfig = config?.planCandidates;
    if (
      typeof fromConfig === "number" &&
      Number.isInteger(fromConfig) &&
      fromConfig > 0
    ) {
      return Math.min(fromConfig, MAX_PLAN_CANDIDATE_COUNT);
    }
    return DEFAULT_PLAN_CANDIDATE_COUNT;
  } catch {
    return 1;
  }
}

/** Marker separating candidate plans in the planner's response. */
export const PLAN_CANDIDATE_MARKER = "=== CANDIDATE";

const CANDIDATE_MARKER_PATTERN = /^===\s*CANDIDATE\s+(\d+)\s*===\s*$/gim;

/**
 * Prompt for the planner pass: produce N genuinely distinct candidate
 * plans, separated by `=== CANDIDATE k ===` marker lines. Same plan rules
 * as the single-plan prompt (numbered steps, files, [parallel],
 * verification step, no code).
 */
export function buildPlannerCandidatesPrompt(
  task: string,
  count: number,
): string {
  const safeCount = Math.max(2, Math.min(count, MAX_PLAN_CANDIDATE_COUNT));
  return [
    "You are the PLANNER in a plan-then-execute workflow.",
    `Produce ${safeCount} DISTINCT candidate implementation plans for the task below, then stop.`,
    "Format rules:",
    "- Start each candidate with a line exactly `=== CANDIDATE k ===` (k = 1..N), then the plan.",
    "- Make the candidates genuinely different approaches, not rewordings of one plan.",
    "- One numbered step per action; each step names the files it touches.",
    "- Mark steps that can run in parallel with [parallel].",
    "- End each plan with a verification step (build + tests to run).",
    "- Do NOT write code in the plans; the executor implements the winner.",
    "",
    `Task: ${task}`,
  ].join("\n");
}

/**
 * Split a planner response into candidates on the `=== CANDIDATE k ===`
 * markers. Fail-open: fewer than 2 markers (or unparseable input) returns
 * the whole response as a single candidate, i.e. today's single-plan path.
 */
export function parseCandidatePlans(text: string): PlanCandidate[] {
  try {
    const source = typeof text === "string" ? text : "";
    if (source.trim().length === 0) return [];
    const matches = [...source.matchAll(CANDIDATE_MARKER_PATTERN)];
    if (matches.length >= 2) {
      const candidates: PlanCandidate[] = [];
      for (let i = 0; i < matches.length; i++) {
        const match = matches[i] as RegExpMatchArray;
        const start = (match.index ?? 0) + match[0].length;
        const next = matches[i + 1] as RegExpMatchArray | undefined;
        const end = next?.index ?? source.length;
        const body = source.slice(start, end).trim();
        if (body.length === 0) continue;
        candidates.push({ id: `plan_${match[1]}`, text: body });
      }
      if (candidates.length >= 2) return candidates;
    }
    return [{ id: "plan_1", text: source.trim() }];
  } catch {
    const trimmed = typeof text === "string" ? text.trim() : "";
    return trimmed.length > 0 ? [{ id: "plan_1", text: trimmed }] : [];
  }
}

function parseRankEntry(item: unknown): PlanRankEntry | undefined {
  if (!item || typeof item !== "object") return undefined;
  const record = item as Record<string, unknown>;
  const id = record["id"];
  if (typeof id !== "string" || id.length === 0) return undefined;
  const score = record["score"];
  const entry: {
    id: string;
    score: number | null;
    pSuccess?: number;
    costPenalty?: number;
    estSteps?: number;
  } = {
    id,
    score:
      typeof score === "number" && Number.isFinite(score) ? score : null,
  };
  const pSuccess = record["p_success"];
  if (typeof pSuccess === "number" && Number.isFinite(pSuccess)) {
    entry.pSuccess = pSuccess;
  }
  const costPenalty = record["cost_penalty"];
  if (typeof costPenalty === "number" && Number.isFinite(costPenalty)) {
    entry.costPenalty = costPenalty;
  }
  const estSteps = record["est_steps"];
  if (typeof estSteps === "number" && Number.isFinite(estSteps)) {
    entry.estSteps = estSteps;
  }
  return entry;
}

/**
 * Parse a rank-plans response body. Old-schema shims (no `ranking` array)
 * yield undefined: "not present", never an error.
 */
export function parsePlanRanking(payload: unknown): PlanRanking | undefined {
  try {
    if (!payload || typeof payload !== "object") return undefined;
    const record = payload as Record<string, unknown>;
    const ranking = record["ranking"];
    if (!Array.isArray(ranking) || ranking.length === 0) return undefined;
    const entries: PlanRankEntry[] = [];
    for (const item of ranking) {
      const entry = parseRankEntry(item);
      if (entry) entries.push(entry);
    }
    if (entries.length === 0) return undefined;
    const task = record["task"];
    const tier = record["tier"];
    return {
      task: typeof task === "string" ? task : "",
      ...(typeof tier === "string" ? { tier } : {}),
      ranking: entries,
    };
  } catch {
    return undefined;
  }
}

/**
 * POST candidate plans to the SystemOne rank-plans endpoint. Fail-open:
 * master kill-switch, empty task, < 2 plans, shim down, timeout, non-200,
 * or an unparseable body all return undefined (caller runs the first
 * candidate). Never throws.
 */
export async function fetchSystemOnePlanRanking(
  task: string,
  plans: readonly PlanCandidate[],
  options?: {
    endpoint?: string;
    timeoutMs?: number;
    env?: NodeJS.ProcessEnv;
    /** Override for the decider-fallback endpoint (same shim base URL). */
    decideEndpoint?: string;
  },
): Promise<PlanRanking | undefined> {
  try {
    const env = options?.env ?? process.env;
    if (env[SYSTEMONE_MASTER_KILL_SWITCH_ENV] === "0") return undefined;
    if (!task || !task.trim()) return undefined;
    if (!plans || plans.length < 2) return undefined;
    const endpoint = options?.endpoint ?? SYSTEMONE_RANK_PLANS_ENDPOINT;
    const timeoutMs = options?.timeoutMs ?? SYSTEMONE_RANK_PLANS_TIMEOUT_MS;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          task,
          plans: plans.map((plan) => ({ id: plan.id, text: plan.text })),
        }),
        signal: controller.signal,
      });
      if (response.ok) {
        const ranking = parsePlanRanking(await response.json());
        if (ranking) return { ...ranking, source: "rank-plans" as const };
      }
      // Non-200 or old schema: fall through to the decider fallback below.
    } catch {
      // Shim down / timeout: fall through to the decider fallback below.
    } finally {
      clearTimeout(timer);
    }
    return await fetchDeciderPlanRanking(task, plans, {
      endpoint: options?.decideEndpoint,
      timeoutMs,
      env,
    });
  } catch {
    return undefined;
  }
}

/**
 * Candidate plans as decider choice criteria: plan id -> plan text
 * (truncated to the criteria char budget). Plan ids travel as the
 * decider's labels and map back 1:1 to candidates.
 */
function buildPlanDecideCriteria(
  plans: readonly PlanCandidate[],
): Record<string, string> {
  const criteria: Record<string, string> = {};
  for (const plan of plans) {
    criteria[plan.id] = plan.text.slice(0, PLAN_DECIDE_CRITERION_CHARS);
  }
  return criteria;
}

/**
 * Convert a validated decider choice answer into a PlanRanking so the
 * winner selection below works unchanged. Score = winning probability
 * (highest wins), carried as pSuccess for the run diagnostics. Returns
 * undefined when the winner is too uncertain — the caller keeps the
 * first candidate.
 */
export function decideAnswerToPlanRanking(
  task: string,
  plans: readonly PlanCandidate[],
  answer: SystemOneDecideAnswer,
  minWinnerProbability: number = SYSTEMONE_DECIDE_MIN_WINNER_PROBABILITY,
): PlanRanking | undefined {
  try {
    const winnerProb = answer.probabilities[answer.label] ?? 0;
    if (winnerProb < minWinnerProbability) return undefined;
    const ranking: PlanRankEntry[] = [];
    for (const plan of plans) {
      const prob = answer.probabilities[plan.id];
      if (typeof prob !== "number" || !Number.isFinite(prob)) {
        return undefined;
      }
      ranking.push({ id: plan.id, score: prob, pSuccess: prob });
    }
    return { task, tier: "systemone-decide", source: "decide", ranking };
  } catch {
    return undefined;
  }
}

/**
 * Decider-backed plan ranking (Mapika/decider-4b via the shim's
 * /v1/systemone/decide endpoint): asks the decision engine to pick the
 * best candidate plan. Advisory only, fail-open: any error, timeout, or
 * validation failure returns undefined and the caller runs the first
 * candidate. Never throws.
 */
export async function fetchDeciderPlanRanking(
  task: string,
  plans: readonly PlanCandidate[],
  options?: {
    endpoint?: string;
    timeoutMs?: number;
    env?: NodeJS.ProcessEnv;
  },
): Promise<PlanRanking | undefined> {
  try {
    const env = options?.env ?? process.env;
    if (env[SYSTEMONE_MASTER_KILL_SWITCH_ENV] === "0") return undefined;
    if (!task || !task.trim()) return undefined;
    if (!plans || plans.length < 2) return undefined;
    const answer = await decide(
      {
        state: task.slice(0, SYSTEMONE_DECIDE_STATE_CHARS),
        instructions: PLAN_DECIDE_INSTRUCTIONS,
        criteria: buildPlanDecideCriteria(plans),
        type: "choice",
      },
      {
        endpoint: options?.endpoint,
        timeoutMs: options?.timeoutMs,
        env,
      },
    );
    if (!answer) return undefined;
    return decideAnswerToPlanRanking(task, plans, answer);
  } catch {
    return undefined;
  }
}

/**
 * Pick the plan to execute: highest finite score wins; ties and missing
 * scores fall back to the earliest candidate (deterministic). No ranking or
 * no usable scores -> the first candidate, i.e. today's behavior.
 * Never throws; undefined only when there are no candidates at all.
 */
export function selectRankedPlan(
  plans: readonly PlanCandidate[],
  ranking?: PlanRanking | undefined,
): PlanSelection | undefined {
  try {
    if (!plans || plans.length === 0) return undefined;
    if (plans.length === 1) {
      return {
        index: 0,
        plan: plans[0] as PlanCandidate,
        reason: "single-plan",
        rankedIds: [],
      };
    }
    if (!ranking) {
      return {
        index: 0,
        plan: plans[0] as PlanCandidate,
        reason: "no-ranking",
        rankedIds: [],
      };
    }
    const rankedIds = ranking.ranking.map((entry) => entry.id);
    const byId = new Map<string, PlanRankEntry>();
    for (const entry of ranking.ranking) {
      if (!byId.has(entry.id)) byId.set(entry.id, entry);
    }
    let bestIndex = 0;
    let bestScore = Number.NEGATIVE_INFINITY;
    let sawFiniteScore = false;
    for (let i = 0; i < plans.length; i++) {
      const score = byId.get((plans[i] as PlanCandidate).id)?.score;
      if (typeof score === "number" && Number.isFinite(score)) {
        sawFiniteScore = true;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = i;
        }
      }
    }
    return {
      index: bestIndex,
      plan: plans[bestIndex] as PlanCandidate,
      reason: sawFiniteScore
        ? ranking.source === "decide"
          ? "decided-winner"
          : "ranked-winner"
        : "no-usable-scores",
      rankedIds,
    };
  } catch {
    if (!plans || plans.length === 0) return undefined;
    return {
      index: 0,
      plan: plans[0] as PlanCandidate,
      reason: "no-ranking",
      rankedIds: [],
    };
  }
}
