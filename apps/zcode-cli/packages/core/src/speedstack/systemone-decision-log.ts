// ============================================================
// Speed Stack: SystemOne decision-record logging
// ============================================================
//
// Appends JSONL decision records (one object per line) for every
// SystemOne route/decide call ZCode makes. The decide records use the
// shape SystemOne's calibration battery expects
// (systemone/battery/fit_types.py):
//   {"kind": "decide", "type": "choice"|"noul"|"score",
//    "labels": [...], "probs": [...], "gold": <index>,
//    "label": "<winning label>", ...}
// `gold` is the outcome index and is only present when the outcome is
// knowable at log time; fit_types.py consumes records that carry it
// (filter: jq 'select(.kind=="decide" and has("gold"))').
// Route records use {"kind": "route", ...} and are never fed to the
// fitter — they exist for auditing and future route-calibration work.
//
// Zero setup: the default path is ~/.config/zcode/
// systemone-decision-records.jsonl ($XDG_CONFIG_HOME respected).
// $ZCODE_SYSTEMONE_DECISION_LOG overrides the path; "0" disables
// logging. An unwritable path degrades silently — logging never throws
// and never breaks the calling flow.
//
// This module is intentionally pure (node builtins only) so it stays
// runnable under plain `node --test` type-stripping like its speedstack
// siblings. No model IDs appear anywhere here.

import { appendFileSync, mkdirSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";

/** Env var overriding the decision-record log path ("0" disables). */
export const SYSTEMONE_DECISION_LOG_ENV = "ZCODE_SYSTEMONE_DECISION_LOG";

const DECISION_LOG_FILENAME = "systemone-decision-records.jsonl";

/** One logged route decision. */
export interface SystemOneRouteLogRecord {
  readonly kind: "route";
  /** ISO timestamp of the call. */
  readonly ts: string;
  readonly tier: string;
  readonly confidence: number;
  readonly uncertain?: boolean | undefined;
  readonly margin?: number | undefined;
  readonly modelId?: string | undefined;
  readonly rankedModels?: ReadonlyArray<{
    readonly modelId: string;
    readonly tier: string;
    readonly utility: number;
  }> | undefined;
  /** Task text length (never the text itself — keeps the log compact). */
  readonly taskChars?: number | undefined;
}

/** One logged decide answer (fit_types.py-compatible when `gold` is set). */
export interface SystemOneDecideLogRecord {
  readonly kind: "decide";
  /** ISO timestamp of the call. */
  readonly ts: string;
  readonly type: "choice" | "score" | "noul";
  /** Label ids in probability order. */
  readonly labels: readonly string[];
  /** Winning probabilities aligned with `labels`. */
  readonly probs: readonly number[];
  /** Winning label id. */
  readonly label: string;
  /**
   * Index into `labels` of the correct answer — present only when the
   * outcome is knowable at log time. fit_types.py needs this field.
   */
  readonly gold?: number | undefined;
  readonly confidence: number;
  readonly backend?: string | undefined;
  readonly latencyMs?: number | undefined;
}

export type SystemOneDecisionLogRecord =
  | SystemOneRouteLogRecord
  | SystemOneDecideLogRecord;

/**
 * Resolve the decision-record log path. Returns undefined when logging
 * is disabled ($ZCODE_SYSTEMONE_DECISION_LOG="0"). Never throws.
 */
export function resolveDecisionLogPath(
  env: NodeJS.ProcessEnv = process.env,
): string | undefined {
  try {
    const override = (env[SYSTEMONE_DECISION_LOG_ENV] ?? "").trim();
    if (override === "0") return undefined;
    if (override !== "") return override;
    const configHome =
      (env["XDG_CONFIG_HOME"] ?? "").trim() || join(homedir(), ".config");
    return join(configHome, "zcode", DECISION_LOG_FILENAME);
  } catch {
    try {
      return join(tmpdir(), DECISION_LOG_FILENAME);
    } catch {
      return undefined;
    }
  }
}

/**
 * Append one decision record as a JSON line. Best-effort and silent:
 * creates the parent dir when needed, skips when the path is not
 * writable, and never throws — logging must never break the main flow.
 */
export function appendDecisionRecord(
  record: SystemOneDecisionLogRecord,
  env: NodeJS.ProcessEnv = process.env,
): void {
  try {
    const path = resolveDecisionLogPath(env);
    if (!path) return;
    const line = `${JSON.stringify(record)}\n`;
    try {
      mkdirSync(dirname(path), { recursive: true });
    } catch {
      // Parent may already exist or be creatable on write; keep going.
    }
    appendFileSync(path, line, { encoding: "utf8" });
  } catch {
    // Silent by design: an unwritable log path degrades to no logging.
  }
}
