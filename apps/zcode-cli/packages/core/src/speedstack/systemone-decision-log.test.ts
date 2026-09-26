// Tests for speedstack/systemone-decision-log.ts — node:test, no external deps.
//
// Covers:
// - path resolution: XDG_CONFIG_HOME respected, $ZCODE_SYSTEMONE_DECISION_LOG
//   override, "0" disables, never throws
// - append: JSONL line written, parent dirs created, disabled -> no file,
//   unwritable path -> silent (never throws)
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  appendDecisionRecord,
  resolveDecisionLogPath,
  SYSTEMONE_DECISION_LOG_ENV,
  type SystemOneDecideLogRecord,
} from "./systemone-decision-log.ts";

test("default path lives under the user config dir", () => {
  const path = resolveDecisionLogPath({
    XDG_CONFIG_HOME: "/tmp/fake-xdg",
    [SYSTEMONE_DECISION_LOG_ENV]: "",
  } as NodeJS.ProcessEnv);
  assert.equal(path, "/tmp/fake-xdg/zcode/systemone-decision-records.jsonl");
});

test("default path falls back to ~/.config when XDG is unset", () => {
  const path = resolveDecisionLogPath({} as NodeJS.ProcessEnv);
  assert.ok(path, "expected a path");
  assert.ok(path.endsWith(join(".config", "zcode", "systemone-decision-records.jsonl")));
});

test("env override wins over the default", () => {
  const path = resolveDecisionLogPath({
    [SYSTEMONE_DECISION_LOG_ENV]: "/tmp/custom-records.jsonl",
  } as NodeJS.ProcessEnv);
  assert.equal(path, "/tmp/custom-records.jsonl");
});

test('"0" disables logging', () => {
  const path = resolveDecisionLogPath({
    [SYSTEMONE_DECISION_LOG_ENV]: "0",
  } as NodeJS.ProcessEnv);
  assert.equal(path, undefined);
});

test("append writes one JSON line per record and creates parent dirs", () => {
  const dir = mkdtempSync(join(tmpdir(), "zcode-decision-log-"));
  const logPath = join(dir, "nested", "records.jsonl");
  const env = { [SYSTEMONE_DECISION_LOG_ENV]: logPath } as NodeJS.ProcessEnv;
  const record: SystemOneDecideLogRecord = {
    kind: "decide",
    ts: "2026-09-26T18:00:00.000Z",
    type: "choice",
    labels: ["a", "b"],
    probs: [0.7, 0.3],
    label: "a",
    gold: 0,
    confidence: 0.7,
    backend: "decider",
  };
  appendDecisionRecord(record, env);
  appendDecisionRecord(
    { kind: "route", ts: "2026-09-26T18:01:00.000Z", tier: "economy", confidence: 0.9 },
    env,
  );
  const lines = readFileSync(logPath, "utf8").trim().split("\n");
  assert.equal(lines.length, 2);
  const first = JSON.parse(lines[0]);
  assert.equal(first.kind, "decide");
  assert.equal(first.gold, 0);
  assert.deepEqual(first.probs, [0.7, 0.3]);
  const second = JSON.parse(lines[1]);
  assert.equal(second.kind, "route");
  assert.equal(second.tier, "economy");
});

test("append is silent when disabled", () => {
  const dir = mkdtempSync(join(tmpdir(), "zcode-decision-log-"));
  const logPath = join(dir, "records.jsonl");
  const env = { [SYSTEMONE_DECISION_LOG_ENV]: "0" } as NodeJS.ProcessEnv;
  appendDecisionRecord(
    { kind: "route", ts: "x", tier: "economy", confidence: 1 },
    env,
  );
  assert.equal(existsSync(logPath), false);
});

test("append never throws on an unwritable path", () => {
  const env = {
    [SYSTEMONE_DECISION_LOG_ENV]: "/proc/definitely-not-here/records.jsonl",
  } as NodeJS.ProcessEnv;
  assert.doesNotThrow(() =>
    appendDecisionRecord(
      { kind: "route", ts: "x", tier: "economy", confidence: 1 },
      env,
    ),
  );
});
