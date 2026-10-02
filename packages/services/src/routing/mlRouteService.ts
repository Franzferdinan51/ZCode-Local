/**
 * ML route orchestration: heuristic hard filters first, then one local-ML
 * re-rank round trip. The transport is injectable so unit tests never
 * spawn a sidecar; production passes `queryMlRouteBackend`.
 *
 * The re-rank round trip is route-choice only, served by the SystemOne
 * shim on :8765 (`/v1/systemone` Jev dialect). Rank-plans and typed
 * decide questions are separate shim endpoints on the same host
 * (`/v1/systemone/{rank-plans,decide}`); the :8079 decision sidecar is
 * decide-only and is never used as a route fallback.
 */

import {
  heuristicAutoRouteScorer,
  suggestRoute,
  suggestRouteAsync,
  type AutoRouteSignals,
} from "@zcode/shared/auto-router";
import {
  buildRouteAgreement,
  buildRouteChoicePrompt,
  routeAnswerToScored,
  SYSTEMONE_MAX_OPTIONS,
  validateRouteChoiceAnswer,
  type MlRouteBackendConfig,
  type MlRouteServiceRequest,
  type MlRouteServiceResponse,
  type SystemOneRouteRequest,
} from "@zcode/shared/systemone-scorer";
import { queryMlRouteBackend } from "./mlRouteSidecar.js";

export interface MlRouteServiceDeps {
  queryBackend?: (
    request: SystemOneRouteRequest,
    backend: MlRouteBackendConfig,
  ) => Promise<unknown>;
  onAgreement?: (info: {
    backend: string;
    agree: boolean;
    heuristicPick: string | null;
    mlPick: string;
    mlConfidence: number;
  }) => void;
}

/** Hard cap on candidates accepted over RPC (prompt budget is far lower). */
const MAX_RPC_CANDIDATES = 64;
/** RPC text budget; the prompt builder slices state again to its own cap. */
const MAX_RPC_TEXT_CHARS = 8000;

function normalizeSignals(signals: AutoRouteSignals | undefined): AutoRouteSignals {
  if (!signals) return {};
  return {
    ...signals,
    textSample: signals.textSample?.slice(0, MAX_RPC_TEXT_CHARS),
  };
}

export async function suggestMlRoute(
  request: MlRouteServiceRequest,
  deps: MlRouteServiceDeps = {},
): Promise<MlRouteServiceResponse> {
  const backendId = request.backend.backend;
  const candidates = (request.candidates ?? []).slice(0, MAX_RPC_CANDIDATES);
  const signals = normalizeSignals(request.signals);
  // Hard filters (capabilities, consent, context fit) always stay in TS.
  const survivors = heuristicAutoRouteScorer(candidates, signals).map((entry) => entry.candidate);
  if (survivors.length < 2) {
    return { suggestion: null, backend: backendId, agreement: null, reason: "no-survivors" };
  }
  if (survivors.length > SYSTEMONE_MAX_OPTIONS) {
    return { suggestion: null, backend: backendId, agreement: null, reason: "too-many-options" };
  }
  const heuristic = suggestRoute(candidates, signals);
  const prompt = buildRouteChoicePrompt(survivors, signals);
  if (!prompt) {
    return { suggestion: null, backend: backendId, agreement: null, reason: "no-prompt" };
  }
  const query = deps.queryBackend ?? queryMlRouteBackend;
  let raw: unknown;
  try {
    raw = await query(prompt.request, request.backend);
  } catch (error) {
    return {
      suggestion: null,
      backend: backendId,
      agreement: null,
      reason: "backend-error",
      detail: error instanceof Error ? error.message : String(error),
    };
  }
  const answer = validateRouteChoiceAnswer(prompt, raw);
  if (!answer) {
    return { suggestion: null, backend: backendId, agreement: null, reason: "invalid-response" };
  }
  const agreement = buildRouteAgreement(
    heuristic ? { providerId: heuristic.providerId, modelId: heuristic.modelId } : null,
    answer,
    survivors,
    prompt,
  );
  if (agreement) {
    deps.onAgreement?.({
      backend: backendId,
      agree: agreement.agree,
      heuristicPick: agreement.heuristicPick,
      mlPick: agreement.mlPick,
      mlConfidence: agreement.mlConfidence,
    });
  }
  const scored = routeAnswerToScored(
    survivors,
    prompt,
    answer,
    request.backend.minWinnerProbability,
  );
  if (!scored) {
    return { suggestion: null, backend: backendId, agreement, reason: "low-confidence" };
  }
  const suggestion = await suggestRouteAsync(candidates, signals, async () => scored);
  return { suggestion, backend: backendId, agreement, reason: "ok" };
}
