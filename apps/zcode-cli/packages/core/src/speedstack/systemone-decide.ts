// ============================================================
// Speed Stack: SystemOne decision-engine client (decide endpoint)
// ============================================================
//
// Lightweight, fail-open client for the SystemOne shim's decision
// engine endpoint (POST /v1/systemone/decide). Where the older
// /v1/systemone route/rank-plans endpoints re-rank heuristic
// survivors, the decide endpoint answers typed decision questions
// directly — choice, score, or noul (yes/no) — backed by
// Mapika/decider-4b (https://huggingface.co/Mapika/decider-4b,
// Apache 2.0, decider-4b v2.1 beat Jeff-1 0.640 vs 0.405 on the
// 111-item JevBench hard set). Anything the shim can't answer falls
// back to its local fallback, and anything this client can't parse
// falls open to the caller (undefined), never throws.
//
// Contract (verified against the live shim):
//   request:  { state, instructions, criteria?, type }
//     - choice: criteria = { label: description }
//     - score:  criteria keyed "0".."n-1"
//     - noul:   no criteria
//   response: { type, label | level, probabilities | distribution,
//               confidence, latency_ms?, backend? }
//
// This module is intentionally pure (no runtime imports) so it stays
// runnable under plain `node --test` type-stripping like its speedstack
// siblings. No model IDs appear anywhere here: selection is by the
// decider's own labels only.

/** Decision question types the shim's decide endpoint answers. */
export type SystemOneDecideType = "choice" | "score" | "noul";

// The shim-url module is itself import-free, so importing it keeps this
// module runnable under plain `node --test` type-stripping.
import {
  DEFAULT_SYSTEMONE_SHIM_URL,
  systemOneShimEndpoint,
} from "./systemone-shim-url.js";
import { appendDecisionRecord } from "./systemone-decision-log.js";

/** Local SystemOne shim decide endpoint. The shim base URL is resolved
 * from $SYSTEMONE_SHIM_URL (see ./systemone-shim-url.js) — this constant
 * is only the default; prefer resolveSystemOneDecideEndpoint() for the
 * live value. */
export const SYSTEMONE_DECIDE_ENDPOINT =
  `${DEFAULT_SYSTEMONE_SHIM_URL}/v1/systemone/decide`;

/**
 * Resolve the decide endpoint from $SYSTEMONE_SHIM_URL
 * (see ./systemone-shim-url.js), defaulting to the localhost shim.
 */
export function resolveSystemOneDecideEndpoint(
  env: NodeJS.ProcessEnv = process.env,
): string {
  return systemOneShimEndpoint("/v1/systemone/decide", env);
}

/** Hard bound on one decide lookup; the shim answers in ~100ms healthy. */
export const SYSTEMONE_DECIDE_TIMEOUT_MS = 8_000;

/** State char budget (SystemOne-class servers cap state near 6000 chars). */
export const SYSTEMONE_DECIDE_STATE_CHARS = 6_000;

/**
 * Minimum winner probability before a decider choice beats the
 * fail-open fallback. Mirrors SYSTEMONE_MIN_WINNER_PROBABILITY in
 * @zcode/shared/systemone-scorer (literal duplicated so this module
 * keeps zero runtime imports).
 */
export const SYSTEMONE_DECIDE_MIN_WINNER_PROBABILITY = 0.35;

/** Probability mass tolerance for decide response validation. */
const DECIDE_PROBABILITY_SUM_TOLERANCE = 0.02;

// NOTE: the canonical master kill-switch export lives in
// ./systemone-route.ts as SYSTEMONE_KILL_SWITCH_ENV; the literal is
// duplicated here so this module keeps zero runtime imports
// (node --test type-stripping).
const SYSTEMONE_MASTER_KILL_SWITCH_ENV = "ZCODE_SYSTEMONE";

/**
 * Kill-switch (env): set `ZCODE_SYSTEMONE_DECIDE=0` to disable the
 * decide engine — the plan-ranking decider fallback and direct decide()
 * calls both return undefined (fail-open). Route lookups are unaffected.
 */
export const SYSTEMONE_DECIDE_KILL_SWITCH_ENV = "ZCODE_SYSTEMONE_DECIDE";

/**
 * Env var disabling the decision sidecar's second-opinion (`=0`). Read by
 * the SystemOne shim (uncertain routes + rank-plans blending); honored by
 * shims ZCode spawns since the child inherits process.env.
 */
export const SYSTEMONE_JEFF1_ENV = "SYSTEMONE_JEFF1";

/** One decide request. */
export interface SystemOneDecideRequest {
  readonly state: string;
  readonly instructions: string;
  /** choice: { label: description }; score: keyed "0".."n-1"; noul: omit. */
  readonly criteria?: Readonly<Record<string, string>> | undefined;
  readonly type: SystemOneDecideType;
}

/**
 * Validated decide answer. `label` is the decider's winning label for
 * choice and noul ("yes"/"no"), and the winning score `level` key for
 * score — normalized to one shape so callers branch on one field.
 */
export interface SystemOneDecideAnswer {
  readonly type: SystemOneDecideType;
  readonly label: string;
  /** Winning distribution (choice/noul: probabilities, score: levels). */
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
  /** Observability: server-reported latency and decision backend. */
  readonly latencyMs?: number | undefined;
  readonly backend?: string | undefined;
}

/** Build a choice decide request over labeled options. */
export function buildDecideChoiceRequest(
  state: string,
  instructions: string,
  criteria: Readonly<Record<string, string>>,
): SystemOneDecideRequest {
  return {
    state: (typeof state === "string" ? state : "").slice(
      0,
      SYSTEMONE_DECIDE_STATE_CHARS,
    ),
    instructions,
    criteria,
    type: "choice",
  };
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Parse + validate an untrusted decide response against the offered
 * labels: the label must be offered, the distribution must cover exactly
 * the offered ids, sum to 1 within tolerance, and the label must be the
 * argmax. Returns the validated answer or null (caller fails open).
 */
export function validateDecideAnswer(
  raw: unknown,
  type: SystemOneDecideType,
  offeredIds: readonly string[],
): SystemOneDecideAnswer | null {
  try {
    if (!isRecord(raw)) return null;
    if (offeredIds.length === 0) return null;
    const offered = new Set(offeredIds);
    const responseType = raw["type"];
    if (responseType !== type) return null;
    // choice: `label`; noul: `label` ("yes"/"no"); score: `level`.
    const labelField = type === "score" ? "level" : "label";
    const label = raw[labelField];
    if (typeof label !== "string" || !offered.has(label)) return null;
    // choice/noul: `probabilities`; score: `distribution`.
    const distribution = raw[type === "score" ? "distribution" : "probabilities"];
    if (!isRecord(distribution)) return null;
    const covered = Object.keys(distribution);
    if (covered.length !== offered.size) return null;
    let sum = 0;
    let argmax = "";
    let argmaxProb = -Infinity;
    const clean: Record<string, number> = {};
    for (const id of offeredIds) {
      const prob = distribution[id];
      if (typeof prob !== "number" || !Number.isFinite(prob) || prob < 0) {
        return null;
      }
      clean[id] = prob;
      sum += prob;
      if (prob > argmaxProb) {
        argmaxProb = prob;
        argmax = id;
      }
    }
    if (Math.abs(sum - 1) > DECIDE_PROBABILITY_SUM_TOLERANCE) return null;
    if (argmax !== label) return null;
    const confidence = raw["confidence"];
    if (typeof confidence !== "number" || !Number.isFinite(confidence)) {
      return null;
    }
    const answer: {
      type: SystemOneDecideType;
      label: string;
      probabilities: Readonly<Record<string, number>>;
      confidence: number;
      latencyMs?: number;
      backend?: string;
    } = { type, label, probabilities: clean, confidence };
    const latencyMs = raw["latency_ms"];
    if (typeof latencyMs === "number" && Number.isFinite(latencyMs)) {
      answer.latencyMs = latencyMs;
    }
    const backend = raw["backend"];
    if (typeof backend === "string" && backend.length > 0) {
      answer.backend = backend;
    }
    return answer;
  } catch {
    return null;
  }
}

/** Injectable fetch so unit tests never touch the network. */
export type DecideFetchImpl = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

/**
 * POST a decide question to the SystemOne shim. Fail-open: master
 * kill-switch, decide kill-switch (ZCODE_SYSTEMONE_DECIDE=0), empty
 * state, shim down, timeout, non-200, or an unparseable answer all
 * return undefined. Never throws.
 */
export async function decide(
  request: SystemOneDecideRequest,
  options?: {
    endpoint?: string;
    timeoutMs?: number;
    env?: NodeJS.ProcessEnv;
    fetchImpl?: DecideFetchImpl;
    /**
     * Label id of the known-correct answer. Recorded as `gold` in the
     * decision log so SystemOne's calibration battery (fit_types.py) can
     * fit per-type temperatures from real outcomes. Omit when the outcome
     * is not knowable.
     */
    goldLabel?: string;
  },
): Promise<SystemOneDecideAnswer | undefined> {
  try {
    const env = options?.env ?? process.env;
    if (env[SYSTEMONE_MASTER_KILL_SWITCH_ENV] === "0") return undefined;
    if (env[SYSTEMONE_DECIDE_KILL_SWITCH_ENV] === "0") return undefined;
    const state = typeof request?.state === "string" ? request.state : "";
    const instructions =
      typeof request?.instructions === "string" ? request.instructions : "";
    const type = request?.type;
    if (!state.trim() || !instructions.trim()) return undefined;
    if (type !== "choice" && type !== "score" && type !== "noul") {
      return undefined;
    }
    const offeredIds =
      type === "noul"
        ? ["yes", "no"]
        : request.criteria
          ? Object.keys(request.criteria)
          : [];
    if (offeredIds.length === 0) return undefined;
    const endpoint =
      options?.endpoint ?? resolveSystemOneDecideEndpoint(env);
    const timeoutMs = options?.timeoutMs ?? SYSTEMONE_DECIDE_TIMEOUT_MS;
    const fetchImpl = options?.fetchImpl ?? globalThis.fetch.bind(globalThis);
    const body: Record<string, unknown> = {
      state: state.slice(0, SYSTEMONE_DECIDE_STATE_CHARS),
      instructions,
      type,
    };
    if (request.criteria) body["criteria"] = request.criteria;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetchImpl(endpoint, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      if (!response.ok) return undefined;
      const answer =
        validateDecideAnswer(await response.json(), type, offeredIds) ??
        undefined;
      logDecideAnswer(type, offeredIds, answer, options?.goldLabel, env);
      return answer;
    } catch {
      return undefined;
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return undefined;
  }
}

/**
 * Best-effort decision-record logging for the calibration battery
 * (see ./systemone-decision-log.ts). The record matches the
 * fit_types.py shape (labels/probs aligned, gold = outcome index when
 * knowable). Never throws; a logging failure must not affect decide.
 */
function logDecideAnswer(
  type: SystemOneDecideType,
  offeredIds: readonly string[],
  answer: SystemOneDecideAnswer | undefined,
  goldLabel: string | undefined,
  env: NodeJS.ProcessEnv,
): void {
  try {
    if (!answer) return;
    const probs = offeredIds.map(
      (id) => answer.probabilities[id] ?? Number.NaN,
    );
    if (probs.some((p) => !Number.isFinite(p))) return;
    const gold =
      typeof goldLabel === "string" ? offeredIds.indexOf(goldLabel) : -1;
    appendDecisionRecord(
      {
      kind: "decide",
      ts: new Date().toISOString(),
      type,
      labels: [...offeredIds],
      probs,
      label: answer.label,
      ...(gold >= 0 ? { gold } : {}),
      confidence: answer.confidence,
      ...(answer.backend ? { backend: answer.backend } : {}),
      ...(answer.latencyMs !== undefined
        ? { latencyMs: answer.latencyMs }
        : {}),
      },
      env,
    );
  } catch {
    // logging never breaks decide
  }
}
