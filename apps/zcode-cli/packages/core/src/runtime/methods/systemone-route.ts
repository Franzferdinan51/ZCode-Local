// ============================================================
// Speed Stack: per-task SystemOne route decision (Z0/Z1/Z2)
// ============================================================
//
// Fetches the route decision once per task (3.24.0: no more per-session
// caching — every task gets its own route call) and exposes the latest
// settled value to the turn loop via `systemOneRouteValue`.
// Fail-open: the fetch never rejects — a missing shim, timeout, or bad
// payload just yields undefined and the session behaves exactly like today.

import type { AgentRuntimeInternal } from "../internal.js";
import {
  fetchSystemOneRouteDecision,
  isSystemOneDisabled,
  type SystemOneRouteDecision,
} from "../../speedstack/systemone-route.js";

export async function ensureSystemOneRouteDecision(
  this: AgentRuntimeInternal,
  task: string,
): Promise<SystemOneRouteDecision | undefined> {
  // One route call per task: fetch fresh every time instead of caching the
  // first task's decision for the whole session. The latest value stays on
  // `systemOneRouteValue` for the Z1/Z2 policies consumed later in the turn.
  const decision = await fetchSystemOneRouteDecision(task).then(
    (resolved) => resolved,
    () => {
      // Unreachable: the client never rejects, but stay fail-open anyway.
      return undefined;
    },
  );
  this.systemOneRouteValue = decision;
  this.systemOneRouteTaskText = task;
  if (decision) {
    this.logger?.debug("SystemOne route decision fetched", {
      confidence: decision.confidence,
      effort: decision.effort,
      event: "systemone.route.fetched",
      modelId: decision.modelId,
      module: "core.runtime",
      taskLabels: decision.taskLabels,
      tier: decision.tier,
    });
    surfaceSecondOpinion(this.logger, decision);
  } else {
    this.logger?.debug("SystemOne route unavailable; fail-open", {
      event: "systemone.route.unavailable",
      module: "core.runtime",
    });
    warnShimDownOnce(this.logger);
  }
  return decision;
}

/**
 * User-visible warning when the shim is unreachable at runtime.
 * Throttled to once per process — routing already failed open, this just
 * makes the degraded state visible instead of debug-log-only.
 */
let warnedShimDown = false;
function warnShimDownOnce(logger: AgentRuntimeInternal["logger"]): void {
  try {
    if (warnedShimDown || isSystemOneDisabled()) return;
    warnedShimDown = true;
    logger?.warn(
      "SystemOne shim unreachable; continuing without routing (fail-open)",
      {
        event: "systemone.route.unavailable",
        module: "core.runtime",
      },
    );
  } catch {
    // logging never breaks the turn
  }
}

/**
 * Surface the shim's advisory tier second opinion on uncertain routes
 * (wire name `jeff1_second_opinion`; decider-backed now). Disagreement
 * is a warn, agreement an info — the routed tier never changes.
 */
function surfaceSecondOpinion(
  logger: AgentRuntimeInternal["logger"],
  decision: SystemOneRouteDecision,
): void {
  try {
    const opinion = decision.secondOpinion;
    if (!opinion) return;
    const context = {
      agree: opinion.agree,
      confidence: opinion.confidence,
      event: "systemone.route.second_opinion",
      module: "core.runtime",
      rationale: opinion.rationale,
      tier: opinion.tier,
    };
    if (opinion.agree) {
      logger?.info("SystemOne second opinion agrees with route", context);
    } else {
      logger?.warn(
        `SystemOne second opinion disagrees with route (advisory; tier unchanged): ${opinion.rationale ?? opinion.tier}`,
        context,
      );
    }
  } catch {
    // logging never breaks the turn
  }
}
