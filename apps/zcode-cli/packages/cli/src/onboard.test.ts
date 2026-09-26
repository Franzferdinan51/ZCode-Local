/**
 * Tests for the `zcode onboard` first-run wizard.
 *
 * - non-TTY stdin exits 0 without prompting (never hangs)
 * - config writes are idempotent across runs (fixture home)
 * - no model IDs are hard-coded in the wizard source
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  applyOnboardingStateToEnv,
  checkRequirements,
  isOnboardingSkipped,
  probeLmStudioModels,
  probeSystemOneDecide,
  probeSystemOneShim,
  readOnboardingState,
  resolveOnboardingStatePath,
  resolveZCodeStorageDir,
  runOnboardCommand,
  runOnboardingWizard,
  shouldTriggerFirstRunOnboarding,
  writeOnboardingState,
  type WizardIO,
} from "./onboard.js";

// ---------------------------------------------------------------------------
// Fakes
// ---------------------------------------------------------------------------

interface FakeCtx {
  argv: string[];
  stdin: { isTTY: boolean };
  stdout: { write: (text: string) => void };
  stderr: { write: (text: string) => void };
  out: string[];
  err: string[];
}

function makeFakeCtx(isTTY: boolean): FakeCtx {
  const out: string[] = [];
  const err: string[] = [];
  return {
    argv: ["onboard"],
    stdin: { isTTY },
    stdout: {
      write: (text: string) => {
        out.push(text);
      },
    },
    stderr: {
      write: (text: string) => {
        err.push(text);
      },
    },
    out,
    err,
  };
}

function makeScriptedIO(
  answers: string[],
  opts: { isTTY?: boolean } = {},
): WizardIO & { log: string[] } {
  const log: string[] = [];
  let index = 0;
  return {
    stdinIsTTY: opts.isTTY ?? true,
    write: (text: string) => {
      log.push(text);
    },
    writeError: (text: string) => {
      log.push(text);
    },
    question: async (prompt: string) => {
      log.push(`Q:${prompt}`);
      const answer = answers[index++] ?? "";
      log.push(`A:${answer}`);
      return answer;
    },
    questionHidden: async (prompt: string) => {
      log.push(`Q(hidden):${prompt}`);
      const answer = answers[index++] ?? "";
      log.push("A(hidden):<redacted>");
      return answer;
    },
    close: () => {},
    log,
  };
}

const healthyProbes = {
  nodeVersion: "v24.14.0",
  statfs: () => ({ bavail: 100 * 1024 ** 3, bsize: 1 }),
  execPython: () => "Python 3.11.9",
  totalmemBytes: 16 * 1024 ** 3,
};

function stubFetch(
  models: string[] | null,
  shimUp: boolean,
  decideUp: boolean = shimUp,
  decideBackend: string = "jeff1",
): typeof fetch {
  return (async (url: unknown) => {
    const target = String(url);
    if (target.includes(":1234/v1/models")) {
      if (models === null) throw new Error("connect ECONNREFUSED");
      return { ok: true, json: async () => ({ data: models.map((id) => ({ id })) }) };
    }
    if (target.endsWith("/healthz")) {
      return { ok: shimUp, status: shimUp ? 200 : 503 };
    }
    if (target.endsWith("/v1/systemone/decide")) {
      if (!decideUp) throw new Error("connect ECONNREFUSED");
      return {
        ok: true,
        json: async () => ({
          type: "noul",
          label: "yes",
          probabilities: { yes: 0.9, no: 0.1 },
          confidence: 0.9,
          backend: decideBackend,
        }),
      };
    }
    throw new Error(`unexpected probe: ${target}`);
  }) as unknown as typeof fetch;
}

function fixtureHome(): string {
  return mkdtempSync(join(tmpdir(), "zcode-onboard-test-"));
}

// ---------------------------------------------------------------------------
// Headless behavior
// ---------------------------------------------------------------------------

test("runOnboardCommand: non-TTY stdin exits 0 without prompting", async () => {
  const ctx = makeFakeCtx(false);
  const code = await runOnboardCommand(ctx as never, {} as never, [], {
    env: {},
    home: fixtureHome(),
  });
  assert.equal(code, 0);
  assert.match(ctx.out.join(""), /non-interactive/);
});

test("runOnboardCommand: --yes on non-TTY accepts all defaults without prompting", async () => {
  const ctx = makeFakeCtx(false);
  const home = fixtureHome();
  const code = await runOnboardCommand(ctx as never, {} as never, ["--yes"], {
    env: {},
    home,
    probes: healthyProbes,
  });
  assert.equal(code, 0);
  const state = readOnboardingState({}, home);
  assert.equal(state?.onboarded, true);
  assert.deepEqual(state?.systemone, {
    enabled: true,
    jeff1: true,
    shimUrl: "http://127.0.0.1:8765",
    decideFallback: true,
  });
  assert.equal(state?.inference?.mode, "auto");
});

// ---------------------------------------------------------------------------
// First-run trigger
// ---------------------------------------------------------------------------

test("runOnboardCommand: honors deps.yes from the CLI arg parser", async () => {
  const ctx = makeFakeCtx(false);
  const home = fixtureHome();
  const code = await runOnboardCommand(ctx as never, {} as never, [], {
    env: {},
    home,
    yes: true,
    probes: healthyProbes,
  });
  assert.equal(code, 0);
  const state = readOnboardingState({}, home);
  assert.equal(state?.onboarded, true);
});

test("shouldTriggerFirstRunOnboarding: fresh TTY install triggers", () => {
  assert.equal(
    shouldTriggerFirstRunOnboarding({
      argv: [],
      env: {},
      stdinIsTTY: true,
      home: fixtureHome(),
    }),
    true,
  );
});

test("shouldTriggerFirstRunOnboarding: never triggers without a TTY", () => {
  assert.equal(
    shouldTriggerFirstRunOnboarding({
      argv: [],
      env: {},
      stdinIsTTY: false,
      home: fixtureHome(),
    }),
    false,
  );
});

test("shouldTriggerFirstRunOnboarding: skips via env var and flag", () => {
  const home = fixtureHome();
  assert.equal(
    shouldTriggerFirstRunOnboarding({
      argv: [],
      env: { ZCODE_SKIP_ONBOARDING: "1" },
      stdinIsTTY: true,
      home,
    }),
    false,
  );
  assert.equal(
    shouldTriggerFirstRunOnboarding({
      argv: ["tui", "--skip-onboarding"],
      env: {},
      stdinIsTTY: true,
      home,
    }),
    false,
  );
  assert.equal(isOnboardingSkipped(["--skip-onboarding"], {}), true);
  assert.equal(isOnboardingSkipped([], { ZCODE_SKIP_ONBOARDING: "1" }), true);
  assert.equal(isOnboardingSkipped([], {}), false);
});

test("shouldTriggerFirstRunOnboarding: no trigger after a completed run", async () => {
  const home = fixtureHome();
  const io = makeScriptedIO(["", "", "n", "y", "y"]);
  let startCalls = 0;
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, true),
    startShim: async () => {
      startCalls += 1;
    },
  });
  assert.equal(code, 0);
  assert.equal(startCalls, 0);
  assert.equal(
    shouldTriggerFirstRunOnboarding({ argv: [], env: {}, stdinIsTTY: true, home }),
    false,
  );
});

// ---------------------------------------------------------------------------
// Requirements
// ---------------------------------------------------------------------------

test("checkRequirements: Node < 24 is the only failure", () => {
  const checks = checkRequirements({ ...healthyProbes, nodeVersion: "v22.22.3" });
  const node = checks.find((c) => c.id === "node");
  assert.equal(node?.status, "fail");
  assert.equal(checks.filter((c) => c.status === "fail").length, 1);
});

test("checkRequirements: healthy machine passes; warnings never fail", () => {
  const checks = checkRequirements(healthyProbes);
  assert.ok(checks.every((c) => c.status !== "fail"));
  const degraded = checkRequirements({
    ...healthyProbes,
    execPython: () => {
      throw new Error("not found");
    },
    totalmemBytes: 2 * 1024 ** 3,
    statfs: () => ({ bavail: 512 * 1024 ** 2, bsize: 1 }),
  });
  assert.ok(degraded.every((c) => c.status !== "fail"));
  assert.ok(degraded.some((c) => c.status === "warn"));
});

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

test("probeLmStudioModels: lists models; null when unreachable", async () => {
  const models = await probeLmStudioModels(stubFetch(["alpha-model", "beta-model"], true));
  assert.deepEqual(models, ["alpha-model", "beta-model"]);
  assert.equal(await probeLmStudioModels(stubFetch(null, true)), null);
});

test("probeSystemOneShim: true/false from healthz", async () => {
  assert.equal(await probeSystemOneShim(stubFetch(null, true)), true);
  assert.equal(await probeSystemOneShim(stubFetch(null, false)), false);
  assert.equal(
    await probeSystemOneShim(stubFetch(null, true), "http://example:9999"),
    true,
  );
});

test("probeSystemOneDecide: ok carries the backend; false when unreachable", async () => {
  const probe = await probeSystemOneDecide(
    stubFetch(null, true, true, "decider"),
  );
  assert.equal(probe.ok, true);
  assert.equal(probe.backend, "decider");
  const down = await probeSystemOneDecide(stubFetch(null, true, false));
  assert.equal(down.ok, false);
  assert.ok(down.error);
});

// ---------------------------------------------------------------------------
// Idempotency
// ---------------------------------------------------------------------------

function stripVolatile(statePath: string): string {
  const parsed = JSON.parse(readFileSync(statePath, "utf8")) as Record<string, unknown>;
  delete parsed.completedAt;
  return JSON.stringify(parsed);
}

test("wizard is idempotent: two runs against a fixture home produce the same config", async () => {
  const home = fixtureHome();
  const storageDir = resolveZCodeStorageDir({}, home);
  const answers = ["1", "fixture-api-key", "", "y", "y", "y"];
  let startCalls = 0;
  const runOpts = {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(["picked-model-x", "picked-model-y"], true),
    startShim: async () => {
      startCalls += 1;
    },
  };

  // First run: pick model 1 + provide an API key.
  const io1 = makeScriptedIO(["1", "fixture-api-key", "", "y", "y", "y"]);
  assert.equal(await runOnboardingWizard(io1, runOpts), 0);
  const statePath = resolveOnboardingStatePath({}, home);
  const providerPath = join(storageDir, "v2", "provider_config.json");
  const state1 = stripVolatile(statePath);
  const provider1 = readFileSync(providerPath, "utf8");

  // Second run: identical answers must not change anything.
  const io2 = makeScriptedIO(["1", "fixture-api-key", "", "y", "y", "y"]);
  assert.equal(await runOnboardingWizard(io2, runOpts), 0);
  assert.equal(startCalls, 0);
  assert.equal(stripVolatile(statePath), state1);
  assert.equal(readFileSync(providerPath, "utf8"), provider1);
  void answers;
});

test("wizard is idempotent: re-running with --yes keeps the same config", async () => {
  const home = fixtureHome();
  const storageDir = resolveZCodeStorageDir({}, home);
  const runOpts = {
    env: {},
    home,
    yes: true,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, false),
  };
  const io1 = makeScriptedIO([]);
  assert.equal(await runOnboardingWizard(io1, runOpts), 0);
  const statePath = resolveOnboardingStatePath({}, home);
  const state1 = stripVolatile(statePath);

  const io2 = makeScriptedIO([]);
  assert.equal(await runOnboardingWizard(io2, runOpts), 0);
  assert.equal(stripVolatile(statePath), state1);
  // --yes never writes an API key or provider config on its own.
  assert.equal(
    (() => {
      try {
        readFileSync(join(storageDir, "v2", "provider_config.json"), "utf8");
        return true;
      } catch {
        return false;
      }
    })(),
    false,
  );
});

test("wizard: answering yes to the shim start offer calls the starter", async () => {
  const home = fixtureHome();
  // LM Studio is stubbed unreachable, so the model-pick question is skipped:
  // answers are [apiKey, shimUrl, startShim, enableSystemOne, decideFallback, jeff1].
  const io = makeScriptedIO(["", "", "y", "y", "y", "y"]);
  let startCalls = 0;
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, false),
    startShim: async () => {
      startCalls += 1;
    },
  });
  assert.equal(code, 0);
  assert.equal(startCalls, 1);
  assert.match(io.log.join("\n"), /does not auto-start/);
});

test("wizard: custom shim URL is probed and persisted", async () => {
  const home = fixtureHome();
  const seen: string[] = [];
  const inner = stubFetch(null, true);
  const fetchImpl = (async (url: unknown) => {
    seen.push(String(url));
    return (inner as (url: unknown) => Promise<unknown>)(url);
  }) as unknown as typeof fetch;
  // Answers: [apiKey, shimUrl, enableSystemOne, decideFallback, jeff1].
  const io = makeScriptedIO(["", "http://192.168.1.50:8765", "y", "y", "y"]);
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl,
  });
  assert.equal(code, 0);
  assert.ok(
    seen.some((u) => u === "http://192.168.1.50:8765/healthz"),
    `healthz probed at the custom URL: ${seen.join(", ")}`,
  );
  assert.ok(
    seen.some((u) => u === "http://192.168.1.50:8765/v1/systemone/decide"),
    `decide probed at the custom URL: ${seen.join(", ")}`,
  );
  const state = readOnboardingState({}, home);
  assert.equal(state?.systemone?.shimUrl, "http://192.168.1.50:8765");
  assert.match(io.log.join("\n"), /applied at runtime automatically/);
  assert.match(
    io.log.join("\n"),
    /SYSTEMONE_SHIM_URL=http:\/\/192\.168\.1\.50:8765/,
  );
  // And the runtime overlay picks it up without any shell export.
  const env: NodeJS.ProcessEnv = {};
  applyOnboardingStateToEnv(env, home);
  assert.equal(env.SYSTEMONE_SHIM_URL, "http://192.168.1.50:8765");
});

test("wizard: remote shim URL down never offers the local shim starter", async () => {
  const home = fixtureHome();
  // Remote URL, shim down: answers are [apiKey, shimUrl, enableSystemOne,
  // decideFallback, jeff1] — there is NO startShim question for remotes.
  const io = makeScriptedIO(["", "http://192.168.1.50:8765", "y", "y", "y"]);
  let startCalls = 0;
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, false),
    startShim: async () => {
      startCalls += 1;
    },
  });
  assert.equal(code, 0);
  assert.equal(startCalls, 0);
  const log = io.log.join("\n");
  assert.match(log, /remote/);
  assert.doesNotMatch(log, /Start the bundled shim now/);
  assert.match(log, /degraded mode/);
});

test("wizard: invalid shim URL falls back to the default with a warning", async () => {
  const home = fixtureHome();
  const io = makeScriptedIO(["", "not a url", "y", "y", "y"]);
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, true),
  });
  assert.equal(code, 0);
  assert.match(io.log.join("\n"), /doesn't look like a shim URL/);
  const state = readOnboardingState({}, home);
  assert.equal(state?.systemone?.shimUrl, "http://127.0.0.1:8765");
});

test("wizard: shim down warns and completes in degraded mode", async () => {
  const home = fixtureHome();
  // Answers: [apiKey, shimUrl, startShim(no), enableSystemOne, decideFallback, jeff1].
  const io = makeScriptedIO(["", "", "n", "y", "y", "y"]);
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, false),
    startShim: async () => {},
  });
  assert.equal(code, 0);
  const log = io.log.join("\n");
  assert.match(log, /degraded mode/);
  assert.match(log, /continuing in degraded mode/);
  const state = readOnboardingState({}, home);
  assert.equal(state?.systemone?.enabled, true);
});

test("wizard: decide engine down stays fail-open", async () => {
  const home = fixtureHome();
  const io = makeScriptedIO(["", "", "y", "y", "y"]);
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, true, false),
  });
  assert.equal(code, 0);
  assert.match(
    io.log.join("\n"),
    /Decide engine not answering — plan ranking stays fail-open/,
  );
});

test("wizard: disabling the decide fallback is saved and auto-applied", async () => {
  const home = fixtureHome();
  const io = makeScriptedIO(["", "", "y", "n", "y"]);
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, true),
  });
  assert.equal(code, 0);
  assert.match(io.log.join("\n"), /Decide engine off for this machine/);
  const state = readOnboardingState({}, home);
  assert.equal(state?.systemone?.decideFallback, false);
  // The kill switch is honored at runtime without a shell export.
  const env: NodeJS.ProcessEnv = {};
  applyOnboardingStateToEnv(env, home);
  assert.equal(env.ZCODE_SYSTEMONE_DECIDE, "0");
});

test("wizard --yes: shim URL defaults and the decide fallback stays on", async () => {
  const home = fixtureHome();
  const io = makeScriptedIO([]);
  const code = await runOnboardingWizard(io, {
    env: {},
    home,
    yes: true,
    probes: healthyProbes,
    fetchImpl: stubFetch(null, true),
  });
  assert.equal(code, 0);
  const state = readOnboardingState({}, home);
  assert.equal(state?.systemone?.shimUrl, "http://127.0.0.1:8765");
  assert.equal(state?.systemone?.decideFallback, true);
});

// ---------------------------------------------------------------------------
// No hard-coded model IDs
// ---------------------------------------------------------------------------

test("onboard.ts contains no hard-coded model IDs", () => {
  const source = readFileSync(new URL("./onboard.ts", import.meta.url), "utf8");
  const blocklist = [
    "ornith",
    "muse-spark",
    "glm-",
    "minicpm",
    "qwen",
    "deepseek",
    "local-chat",
    "llama",
    "mistral",
    "gemma",
    "phi-",
    "gpt-",
  ];
  for (const token of blocklist) {
    assert.ok(
      !source.toLowerCase().includes(token),
      `onboard.ts must not hard-code a model ID (found "${token}")`,
    );
  }
});

test("applyOnboardingStateToEnv: wizard choices apply without shell exports", () => {
  const home = fixtureHome();
  const storageDir = resolveZCodeStorageDir({}, home);
  writeOnboardingState(storageDir, {
    version: 1,
    onboarded: true,
    systemone: {
      enabled: true,
      jeff1: true,
      shimUrl: "http://192.168.1.50:8765",
      decideFallback: false,
    },
  });
  const env: NodeJS.ProcessEnv = {};
  applyOnboardingStateToEnv(env, home);
  assert.equal(env.SYSTEMONE_SHIM_URL, "http://192.168.1.50:8765");
  assert.equal(env.ZCODE_SYSTEMONE_DECIDE, "0");
});

test("applyOnboardingStateToEnv: explicit env vars win; decide-on sets nothing", () => {
  const home = fixtureHome();
  const storageDir = resolveZCodeStorageDir({}, home);
  writeOnboardingState(storageDir, {
    version: 1,
    onboarded: true,
    systemone: {
      enabled: true,
      jeff1: true,
      shimUrl: "http://192.168.1.50:8765",
      decideFallback: true,
    },
  });
  const env: NodeJS.ProcessEnv = {
    SYSTEMONE_SHIM_URL: "http://explicit:9999",
  };
  applyOnboardingStateToEnv(env, home);
  assert.equal(env.SYSTEMONE_SHIM_URL, "http://explicit:9999");
  assert.equal(env.ZCODE_SYSTEMONE_DECIDE, undefined);
});

test("applyOnboardingStateToEnv: no onboarding state, no-op", () => {
  const home = fixtureHome();
  const env: NodeJS.ProcessEnv = {};
  applyOnboardingStateToEnv(env, home);
  assert.equal(env.SYSTEMONE_SHIM_URL, undefined);
  assert.equal(env.ZCODE_SYSTEMONE_DECIDE, undefined);
});
