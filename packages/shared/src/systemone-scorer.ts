/**
 * Jev-compatible "System 1" scoring protocol for optional local ML routing.
 * Pure string/object logic (no Node imports) so UI and backend share it.
 *
 * Any backend that answers the `POST /v1/systemone` shape — GestaltLabs
 * Jeff-1 (`scripts.jev_clf_server`), Franzferdinan51 SystemOne (`shim.py`),
 * or a Laya-backed bridge speaking the same JSON — can re-rank the
 * heuristic survivors from `heuristicAutoRouteScorer`. Hard capability
 * filters always stay in TS; the ML tier only re-ranks survivors and the
 * caller fails open to the heuristic suggestion on any error, timeout, or
 * validation failure.
 *
 * Label design follows the jev-ultrafast fast path: single-letter option
 * ids ("A".."Z") have distinct first tokens so first-token readout needs
 * one forward pass; the human-readable "providerId/modelId + capabilities"
 * text travels in the criteria descriptions and maps back in TS.
 */

import type {
  AutoRouteCandidate,
  AutoRouteScoredCandidate,
  AutoRouteSignals,
  AutoRouteSuggestion,
} from "./auto-router.js";

/** Backends that speak the route-choice protocol (HTTP or stdio bridge). */
export type MlRouteBackendId = "jeff-1" | "systemone" | "laya" | "custom";

export interface MlRouteBackendConfig {
  readonly backend: MlRouteBackendId;
  /**
   * POST endpoint answering the SystemOne shape. Defaults: SystemOne
   * shim `http://127.0.0.1:8765/v1/systemone` (primary), Jeff-1
   * `http://127.0.0.1:8079/v1/systemone` (legacy `jev_clf_server`
   * passthrough — the modern :8079 decision sidecar is decide-only
   * and never answers route-choice). Loopback only.
   */
  readonly endpoint?: string;
  /** stdio bridge argv (NDJSON request line in, response line out). */
  readonly command?: readonly string[];
  /** Python executable for the bundled Laya bridge. Default "python3". */
  readonly pythonPath?: string;
  /** Extra env for a spawned bridge (e.g. ZCODE_LAYA_MODEL). */
  readonly env?: Readonly<Record<string, string>>;
  /** Per-request timeout. Default 20s (model load happens out of band). */
  readonly timeoutMs?: number;
  readonly minWinnerProbability?: number;
}

export const ML_ROUTE_DEFAULT_ENDPOINTS: Readonly<Record<string, string>> = {
  "jeff-1": "http://127.0.0.1:8079/v1/systemone",
  systemone: "http://127.0.0.1:8765/v1/systemone",
};

/**
 * Decision backends the SystemOne shim's typed-decide endpoint reports:
 * `decider` (the decision sidecar answered, e.g. Mapika/decider-4b) or
 * `fallback` (the shim answered locally with its GLiClass engine).
 * Observability only — callers fail open on any value.
 */
export const SYSTEMONE_DECIDE_BACKENDS = ["decider", "fallback"] as const;

/** One of the known `SYSTEMONE_DECIDE_BACKENDS` values. */
export type SystemOneDecideBackend = (typeof SYSTEMONE_DECIDE_BACKENDS)[number];

export const ML_ROUTE_DEFAULT_TIMEOUT_MS = 20_000;

/** Max options per choice question (single-letter alias budget). */
export const SYSTEMONE_MAX_OPTIONS = 26;

/** State char budget (SystemOne-class servers cap state near 6000 chars). */
export const SYSTEMONE_STATE_CHARS = 6000;

/** Minimum winner probability before the ML pick beats the heuristic. */
export const SYSTEMONE_MIN_WINNER_PROBABILITY = 0.35;

/** Probability mass tolerance for jev-style response validation. */
const PROBABILITY_SUM_TOLERANCE = 0.02;

export interface SystemOneCriterion {
  readonly id: string;
  readonly description: string;
}

export interface SystemOneChoiceQuestion {
  /** Stable question key so multi-question transports map answers back. */
  readonly id: string;
  readonly type: "choice";
  readonly instructions: string;
  readonly criteria: readonly SystemOneCriterion[];
}

export interface SystemOneRouteRequest {
  readonly state: string;
  readonly questions: readonly [SystemOneChoiceQuestion];
}

export interface SystemOneChoiceAnswer {
  readonly choice: string;
  readonly probabilities: Readonly<Record<string, number>>;
  readonly confidence: number;
}

export interface SystemOneRouteResponse {
  readonly answers: readonly [SystemOneChoiceAnswer];
}

export interface SystemOneRoutePrompt {
  readonly request: SystemOneRouteRequest;
  /** Alias ("A") -> survivor index in the input order. */
  readonly aliasToIndex: Readonly<Record<string, number>>;
}

function aliasForIndex(index: number): string {
  return String.fromCharCode("A".charCodeAt(0) + index);
}

function describeCandidate(candidate: AutoRouteCandidate): string {
  const capabilities: string[] = [];
  if (candidate.supportsToolCall) capabilities.push("tools");
  if (candidate.supportsJsonSchemaOutput) capabilities.push("json-schema");
  if (candidate.supportsImage) capabilities.push("images");
  if (candidate.supportsVideo) capabilities.push("video");
  if (candidate.supportsAudio) capabilities.push("audio");
  if (candidate.supportsPdf) capabilities.push("pdf");
  if (candidate.contextWindow > 0) {
    capabilities.push(candidate.contextWindow >= 128_000 ? "large-context" : "standard-context");
  }
  if (candidate.accessType === "external-harness") capabilities.push("external-harness");
  const detail = capabilities.length > 0 ? ` (${capabilities.join(", ")})` : "";
  return `${candidate.providerId}/${candidate.modelId}${detail}`;
}

/**
 * Build the route-choice question over heuristic survivors. Returns null
 * when there is nothing worth asking (fewer than 2 options or over budget).
 */
export function buildRouteChoicePrompt(
  survivors: readonly AutoRouteCandidate[],
  signals: AutoRouteSignals,
): SystemOneRoutePrompt | null {
  if (survivors.length < 2 || survivors.length > SYSTEMONE_MAX_OPTIONS) return null;
  const criteria: SystemOneCriterion[] = [];
  const aliasToIndex: Record<string, number> = {};
  survivors.forEach((candidate, index) => {
    const id = aliasForIndex(index);
    aliasToIndex[id] = index;
    criteria.push({ id, description: describeCandidate(candidate) });
  });
  const hints: string[] = [];
  if (signals.needsTools === true) hints.push("the request needs tool calls");
  if (signals.needsJsonSchema === true) hints.push("the request needs JSON-schema output");
  if ((signals.attachmentKinds ?? []).length > 0) {
    hints.push(`the request carries ${signals.attachmentKinds!.join(", ")} attachments`);
  }
  return {
    request: {
      state: (signals.textSample ?? "").slice(0, SYSTEMONE_STATE_CHARS),
      questions: [
        {
          id: "route",
          type: "choice",
          instructions:
            "Pick the single best model route for this chat request. " +
            "Reply with exactly one option id." +
            (hints.length > 0 ? ` Known facts: ${hints.join("; ")}.` : ""),
          criteria,
        },
      ],
    },
    aliasToIndex,
  };
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Parse + validate an untrusted sidecar response, jev-ultrafast style:
 * choice must be an offered id, probabilities must cover exactly the
 * offered ids, sum to 1 within tolerance, and the choice must be the
 * argmax. Returns the validated answer or null (caller fails open).
 */
export function validateRouteChoiceAnswer(
  prompt: SystemOneRoutePrompt,
  raw: unknown,
): SystemOneChoiceAnswer | null {
  if (!isRecord(raw)) return null;
  const answers = raw.answers;
  if (!Array.isArray(answers) || answers.length < 1) return null;
  const answer = answers[0];
  if (!isRecord(answer)) return null;
  const { choice, probabilities, confidence } = answer;
  if (typeof choice !== "string" || !(choice in prompt.aliasToIndex)) return null;
  if (!isRecord(probabilities)) return null;
  const offered = Object.keys(prompt.aliasToIndex);
  const covered = Object.keys(probabilities);
  if (covered.length !== offered.length) return null;
  let sum = 0;
  let argmax = "";
  let argmaxProb = -Infinity;
  for (const id of offered) {
    const prob = probabilities[id];
    if (typeof prob !== "number" || !Number.isFinite(prob) || prob < 0) return null;
    sum += prob;
    if (prob > argmaxProb) {
      argmaxProb = prob;
      argmax = id;
    }
  }
  if (Math.abs(sum - 1) > PROBABILITY_SUM_TOLERANCE) return null;
  if (argmax !== choice) return null;
  if (typeof confidence !== "number" || !Number.isFinite(confidence)) return null;
  const clean: Record<string, number> = {};
  for (const id of offered) clean[id] = probabilities[id] as number;
  return { choice, probabilities: clean, confidence };
}

/**
 * Convert a validated answer to scored survivors. Scores are probability
 * x10 so ML margins stay comparable to the heuristic integer scale used
 * by `suggestRouteAsync` (high confidence needs a margin >= 3). Returns
 * null when the winner is too uncertain — the caller keeps the heuristic.
 */
export function routeAnswerToScored(
  survivors: readonly AutoRouteCandidate[],
  prompt: SystemOneRoutePrompt,
  answer: SystemOneChoiceAnswer,
  minWinnerProbability: number = SYSTEMONE_MIN_WINNER_PROBABILITY,
): readonly AutoRouteScoredCandidate[] | null {
  const winnerProb = answer.probabilities[answer.choice] ?? 0;
  if (winnerProb < minWinnerProbability) return null;
  const scored: AutoRouteScoredCandidate[] = [];
  for (const [alias, index] of Object.entries(prompt.aliasToIndex)) {
    const candidate = survivors[index];
    if (!candidate) return null;
    const prob = answer.probabilities[alias] ?? 0;
    scored.push({
      candidate,
      score: Math.round(prob * 100) / 10,
      reasons:
        alias === answer.choice
          ? [`local ML route pick (${Math.round(prob * 100)}% confidence)`]
          : [],
    });
  }
  return scored;
}

export interface SystemOneRouteAgreement {
  readonly agree: boolean;
  readonly heuristicPick: string | null;
  readonly mlPick: string;
  readonly mlConfidence: number;
}

/** Compare ML vs heuristic picks for agreement logging (accuracy telemetry). */
export function buildRouteAgreement(
  heuristicPick: { providerId: string; modelId: string } | null,
  answer: SystemOneChoiceAnswer,
  survivors: readonly AutoRouteCandidate[],
  prompt: SystemOneRoutePrompt,
): SystemOneRouteAgreement | null {
  const mlCandidate = survivors[prompt.aliasToIndex[answer.choice] ?? -1];
  if (!mlCandidate) return null;
  const mlKey = `${mlCandidate.providerId}/${mlCandidate.modelId}`;
  const heuristicKey = heuristicPick
    ? `${heuristicPick.providerId}/${heuristicPick.modelId}`
    : null;
  return {
    agree: heuristicKey === mlKey,
    heuristicPick: heuristicKey,
    mlPick: mlKey,
    mlConfidence: answer.confidence,
  };
}

/** RPC payload: UI asks the host to re-rank via a local ML backend. */
export interface MlRouteServiceRequest {
  readonly candidates: readonly AutoRouteCandidate[];
  readonly signals: AutoRouteSignals;
  readonly backend: MlRouteBackendConfig;
}

export interface MlRouteServiceResponse {
  /** ML pick, or null when the caller should keep its heuristic result. */
  readonly suggestion: AutoRouteSuggestion | null;
  readonly backend: MlRouteBackendId;
  readonly agreement: SystemOneRouteAgreement | null;
  /** Machine-readable outcome: "ok" | "no-survivors" | "low-confidence" | "backend-error" | ... */
  readonly reason: string;
  readonly detail?: string;
}

/** Per-task reasoning effort tier a route backend may hint at. */
export type SystemOneRouteEffort = "low" | "medium" | "high" | "xhigh" | "ultra";

const SYSTEMONE_ROUTE_EFFORTS: readonly string[] = ["low", "medium", "high", "xhigh", "ultra"];

/**
 * Fail-open extraction of an optional `effort` hint from a SystemOne-style
 * route response object. Returns undefined for anything unexpected, so the
 * caller keeps its current routing behavior. Use with
 * resolveEffectiveEffortTier on the session side.
 */
export function extractRouteEffortHint(raw: unknown): SystemOneRouteEffort | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const record = raw as Record<string, unknown>;
  const effort = record["effort"];
  if (typeof effort !== "string") return undefined;
  const normalized = effort.trim().toLowerCase();
  return (SYSTEMONE_ROUTE_EFFORTS as readonly string[]).includes(normalized)
    ? (normalized as SystemOneRouteEffort)
    : undefined;
}
