// Tests for speedstack/plan-execute.ts — node:test, no external deps.
//
// Covers the Rank-3 rewrite: router-advisory-first model resolution with NO
// hard-coded model ids, kill switch, explicit user config, planner budgets,
// and the plan artifact lifecycle.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import { join, sep } from "node:path";
import {
  DEFAULT_PLANNER_MAX_STEPS,
  DEFAULT_PLANNER_MAX_TOOL_CALLS,
  PLAN_ARTIFACT_FILENAME,
  buildExecutorPrompt,
  buildPlannerPrompt,
  isPlanThenExecuteDisabled,
  parseModelSizeBillions,
  readPlanArtifact,
  readPlanExecuteModelPreferences,
  resolvePlanExecuteModels,
  resolvePlanExecuteSessionDir,
  resolvePlannerBudgets,
  writePlanArtifact,
} from "./plan-execute.ts";

const AVAILABLE = [
  "ornith-1.5-35b-a3b",
  "ornith-1.5-9b",
  "google/gemma-4-12b-qat",
  "text-embedding-nomic-embed-text-v1.5",
];

test("executor falls back to the smallest usable local model", () => {
  const roles = resolvePlanExecuteModels(AVAILABLE);
  // No router advisory, no prefs: planner = first usable, executor =
  // smallest usable. ornith-1.5-9b (9b) wins over gemma-4-12b-qat (12b)
  // and ornith-1.5-35b-a3b (35b).
  assert.equal(roles.plannerModelId, "ornith-1.5-35b-a3b");
  assert.equal(roles.executorModelId, "ornith-1.5-9b");
  assert.ok(!/embed/i.test(roles.executorModelId));
});

test("explicit preferences win when available", () => {
  const roles = resolvePlanExecuteModels(
    AVAILABLE,
    { routeTier: "balanced" },
    {
      plannerModelId: "google/gemma-4-12b-qat",
      executorModelPreferences: ["ornith-1.5-9b", "google/gemma-4-12b-qat"],
    },
  );
  assert.equal(roles.plannerModelId, "google/gemma-4-12b-qat");
  assert.equal(roles.executorModelId, "ornith-1.5-9b");
});

test("unavailable preferences fall through to local fallbacks", () => {
  const roles = resolvePlanExecuteModels(
    AVAILABLE,
    { routeTier: "balanced" },
    { plannerModelId: "not-loaded-model", executorModelId: "also-missing" },
  );
  assert.equal(roles.plannerModelId, "ornith-1.5-35b-a3b");
  assert.equal(roles.executorModelId, "ornith-1.5-9b");
});

test("one model serves both roles when it is the only one available", () => {
  const roles = resolvePlanExecuteModels(["only-model"]);
  assert.equal(roles.plannerModelId, "only-model");
  assert.equal(roles.executorModelId, "only-model");
});

test("no hard-coded model defaults: order is stable without size hints", () => {
  // Two models with no parseable size — the resolver must NOT prefer a
  // baked-in id; it takes the first usable one in list order.
  const roles = resolvePlanExecuteModels(["model-b", "model-a"]);
  assert.equal(roles.plannerModelId, "model-b");
  assert.equal(roles.executorModelId, "model-b");
});

test("resolvePlanExecuteModels throws with no usable model", () => {
  assert.throws(
    () => resolvePlanExecuteModels(["text-embedding-nomic-embed-text-v1.5"]),
    /at least one locally available model/,
  );
  assert.throws(() => resolvePlanExecuteModels([]), /at least one/);
});

test("parseModelSizeBillions reads parameter counts from model ids", () => {
  assert.equal(parseModelSizeBillions("ornith-1.5-35b-a3b"), 35);
  assert.equal(parseModelSizeBillions("ornith-1.5-9b"), 9);
  assert.equal(parseModelSizeBillions("google/gemma-4-12b-qat"), 12);
  assert.equal(parseModelSizeBillions("minicpm5-2b"), 2);
  assert.equal(parseModelSizeBillions("mystery-model"), undefined);
});

test("kill switch defaults to enabled and honors ZCODE_PLAN_EXECUTE=0", () => {
  assert.equal(isPlanThenExecuteDisabled({}), false);
  assert.equal(isPlanThenExecuteDisabled({ ZCODE_PLAN_EXECUTE: "0" }), true);
  assert.equal(isPlanThenExecuteDisabled({ ZCODE_PLAN_EXECUTE: "false" }), true);
  assert.equal(isPlanThenExecuteDisabled({ ZCODE_PLAN_EXECUTE: "1" }), false);
});

test("readPlanExecuteModelPreferences parses explicit user config", () => {
  assert.deepEqual(readPlanExecuteModelPreferences({}), {});
  assert.deepEqual(
    readPlanExecuteModelPreferences({
      ZCODE_PLAN_EXECUTE_PLANNER_MODEL: "big-local",
      ZCODE_PLAN_EXECUTE_EXECUTOR_MODEL: "small-local",
      ZCODE_PLAN_EXECUTE_EXECUTOR_PREFS: "tiny-local, small-local",
    }),
    {
      plannerModelId: "big-local",
      executorModelId: "small-local",
      executorModelPreferences: ["tiny-local", "small-local"],
    },
  );
});

test("resolvePlannerBudgets defaults small and honors env overrides", () => {
  assert.deepEqual(resolvePlannerBudgets({}), {
    maxSteps: DEFAULT_PLANNER_MAX_STEPS,
    maxToolCalls: DEFAULT_PLANNER_MAX_TOOL_CALLS,
  });
  assert.deepEqual(
    resolvePlannerBudgets({
      ZCODE_PLAN_EXECUTE_PLANNER_MAX_STEPS: "20",
      ZCODE_PLAN_EXECUTE_PLANNER_MAX_TOOL_CALLS: "50",
    }),
    { maxSteps: 20, maxToolCalls: 50 },
  );
  // Invalid values fall back to the defaults, never throw.
  assert.deepEqual(
    resolvePlannerBudgets({
      ZCODE_PLAN_EXECUTE_PLANNER_MAX_STEPS: "nope",
      ZCODE_PLAN_EXECUTE_PLANNER_MAX_TOOL_CALLS: "-3",
    }),
    { maxSteps: DEFAULT_PLANNER_MAX_STEPS, maxToolCalls: DEFAULT_PLANNER_MAX_TOOL_CALLS },
  );
});

test("resolvePlanExecuteSessionDir scopes plans under the home session dir", () => {
  const dir = resolvePlanExecuteSessionDir(homedir(), "sess-1");
  assert.ok(dir.includes([".zcode-local", "speedstack", "plans", "sess-1"].join(sep)));
  const evil = resolvePlanExecuteSessionDir(homedir(), "../evil");
  assert.ok(!evil.includes(".."));
});

test("planner/executor prompts carry the task and plan", () => {
  const planner = buildPlannerPrompt("add retries");
  assert.ok(planner.includes("add retries"));
  assert.ok(planner.toLowerCase().includes("planner"));
  const executor = buildExecutorPrompt("# Plan\n1. step");
  assert.ok(executor.includes("# Plan"));
  assert.ok(executor.toLowerCase().includes("executor"));
});

test("plan artifact round-trips through the session dir", async () => {
  const dir = await mkdtemp(join(tmpdir(), "plan-artifact-"));
  try {
    assert.equal(await readPlanArtifact(dir), undefined);
    const path = await writePlanArtifact(dir, "# Plan\n1. do it");
    assert.ok(path.endsWith(PLAN_ARTIFACT_FILENAME));
    assert.equal(await readPlanArtifact(dir), "# Plan\n1. do it");
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
