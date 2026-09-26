// ============================================================
// zcode onboard — first-run setup wizard
// ============================================================
//
// Interactive first-run wizard: requirements check, inference setup
// (LM Studio detection or Meta Model API key), and SystemOne routing
// preferences. Writes idempotently to ~/.zcode-local/v2/onboarding.json
// (plus provider_config.json for the optional Meta API key, via the
// existing NodePersonalProviderConfigRepository).
//
// Rules baked in:
// - No model IDs are hard-coded anywhere. The model question defaults to
//   "let SystemOne decide"; any specific pick is the user's own choice.
// - The wizard never loads, unloads, or switches LM Studio models —
//   it only detects what is already loaded.
// - Everything SystemOne is advisory and fail-open; the kill switches
//   (ZCODE_SYSTEMONE=0, ZCODE_SPEEDSTACK_PRUNE=0, ZCODE_SYSTEMONE_DECIDE=0,
//   SYSTEMONE_JEFF1=0) always win. The shim URL defaults to
//   $SYSTEMONE_SHIM_URL (else localhost:8765) and is overridable.
// - Headless-friendly: non-TTY stdin prints a note and exits 0;
//   --yes accepts every default non-interactively.

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statfsSync, writeFileSync } from "node:fs";
import { homedir, totalmem } from "node:os";
import { dirname, join } from "node:path";
import * as readline from "node:readline";
import {
  NodePersonalProviderConfigRepository,
  PERSONAL_PROVIDER_CONFIG_FILE_NAME,
} from "@zcode/provider-node";
import { parseProviderConfig } from "@zcode/provider";
import {
  DEFAULT_SYSTEMONE_SHIM_URL,
  SYSTEMONE_DECIDE_KILL_SWITCH_ENV,
  SYSTEMONE_SHIM_URL_ENV,
  isLoopbackShimUrl,
  isPlausibleShimUrl,
  resolveSystemOneShimUrl,
} from "@zcode/core";
import { ensureSystemOneShim } from "./systemone-shim-bootstrap.js";
import type { GlobalOptions, RunContext } from "@zcode/shared-types";

export const ONBOARDING_STATE_FILE_NAME = "onboarding.json";
export const ONBOARDING_STATE_VERSION = 1;
export const ONBOARDING_SKIP_ENV_VAR = "ZCODE_SKIP_ONBOARDING";
export const ONBOARDING_SKIP_FLAG = "--skip-onboarding";

const LM_STUDIO_BASE_URL = "http://127.0.0.1:1234";
const LM_STUDIO_MODELS_URL = `${LM_STUDIO_BASE_URL}/v1/models`;
const PROBE_TIMEOUT_MS = 2_500;

const META_PROVIDER_ID = "meta";
const META_TEMPLATE_ID = "meta";
const META_PROVIDER_NAME = "Meta";
const META_API_KEY_MANAGEMENT_URL = "https://dev.meta.ai/";
const LM_STUDIO_PROVIDER_ID = "local:lm-studio";

// ---------------------------------------------------------------------------
// Paths & state
// ---------------------------------------------------------------------------

/** Mirrors systemone-shim-bootstrap's storage-dir resolution. */
export function resolveZCodeStorageDir(
  env: NodeJS.ProcessEnv = process.env,
  home: string = homedir(),
): string {
  return env.ZCODE_STORAGE_DIR?.trim() || join(home, ".zcode-local");
}

export function resolveOnboardingStatePath(
  env: NodeJS.ProcessEnv = process.env,
  home: string = homedir(),
): string {
  return join(resolveZCodeStorageDir(env, home), "v2", ONBOARDING_STATE_FILE_NAME);
}

export function resolvePersonalProviderConfigPath(
  env: NodeJS.ProcessEnv = process.env,
  home: string = homedir(),
): string {
  return join(resolveZCodeStorageDir(env, home), "v2", PERSONAL_PROVIDER_CONFIG_FILE_NAME);
}

export interface OnboardingInferenceState {
  mode: "auto" | "lm-studio" | "meta-api-key";
  modelId?: string;
  metaApiKeyConfigured?: boolean;
}

export interface OnboardingState {
  version: number;
  onboarded: boolean;
  completedAt?: string;
  systemone?: {
    enabled: boolean;
    jeff1: boolean;
    /** Shim base URL chosen in the wizard (default: $SYSTEMONE_SHIM_URL or localhost:8765). */
    shimUrl?: string;
    /** Whether the decide fallback for plan ranking stays enabled. */
    decideFallback?: boolean;
  };
  inference?: OnboardingInferenceState;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function readOnboardingState(
  env: NodeJS.ProcessEnv = process.env,
  home: string = homedir(),
): OnboardingState | null {
  let raw: string;
  try {
    raw = readFileSync(resolveOnboardingStatePath(env, home), "utf8");
  } catch {
    return null;
  }
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) return null;
    return parsed as unknown as OnboardingState;
  } catch {
    return null;
  }
}

/**
 * Overlay the wizard's SystemOne choices onto the runtime env when the
 * corresponding env vars are unset. Explicit env vars always win. This is
 * what makes "your choices are honored at runtime" true: the CLI calls
 * this at startup (see env.ts), so a custom shim URL or a disabled decide
 * engine applies even when the user never exported anything. Never throws.
 */
export function applyOnboardingStateToEnv(
  env: NodeJS.ProcessEnv = process.env,
  home: string = homedir(),
): void {
  let state: OnboardingState | null = null;
  try {
    state = readOnboardingState(env, home);
  } catch {
    return;
  }
  const systemone = state?.systemone;
  if (!systemone) return;
  if (
    (env[SYSTEMONE_SHIM_URL_ENV] ?? "").trim() === "" &&
    (systemone.shimUrl ?? "").trim() !== ""
  ) {
    env[SYSTEMONE_SHIM_URL_ENV] = systemone.shimUrl!.trim();
  }
  if (
    (env[SYSTEMONE_DECIDE_KILL_SWITCH_ENV] ?? "").trim() === "" &&
    systemone.decideFallback === false
  ) {
    env[SYSTEMONE_DECIDE_KILL_SWITCH_ENV] = "0";
  }
}

/** Idempotent: only touches the file when the content actually changed. */
export function writeOnboardingState(
  storageDir: string,
  state: OnboardingState,
): "written" | "unchanged" {
  const dir = join(storageDir, "v2");
  mkdirSync(dir, { recursive: true });
  const path = join(dir, ONBOARDING_STATE_FILE_NAME);
  const content = `${JSON.stringify(state, null, 2)}\n`;
  let existing: string | null = null;
  try {
    existing = readFileSync(path, "utf8");
  } catch {
    existing = null;
  }
  if (existing === content) return "unchanged";
  writeFileSync(path, content, { mode: 0o600 });
  return "written";
}

// ---------------------------------------------------------------------------
// First-run trigger
// ---------------------------------------------------------------------------

export function isOnboardingSkipped(
  argv: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
): boolean {
  if (env[ONBOARDING_SKIP_ENV_VAR] === "1") return true;
  return argv.includes(ONBOARDING_SKIP_FLAG);
}

export interface FirstRunTriggerInput {
  argv: readonly string[];
  env?: NodeJS.ProcessEnv;
  stdinIsTTY: boolean;
  home?: string;
}

/**
 * True when an interactive TUI launch should run the wizard first:
 * a TTY, not skipped, and no sign of a previous setup (neither the
 * onboarding marker nor a hand-configured provider file exists).
 */
export function shouldTriggerFirstRunOnboarding(input: FirstRunTriggerInput): boolean {
  if (!input.stdinIsTTY) return false;
  const env = input.env ?? process.env;
  if (isOnboardingSkipped(input.argv, env)) return false;
  const state = readOnboardingState(env, input.home);
  if (state?.onboarded === true) return false;
  if (existsSync(resolvePersonalProviderConfigPath(env, input.home))) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Requirements check
// ---------------------------------------------------------------------------

export type RequirementStatus = "pass" | "warn" | "fail";

export interface RequirementCheck {
  id: string;
  label: string;
  status: RequirementStatus;
  detail: string;
  hint?: string;
}

export interface RequirementProbes {
  nodeVersion?: string;
  statfs?: (path: string) => { bavail: number | bigint; bsize: number | bigint };
  execPython?: (exe: string, args: readonly string[]) => string;
  totalmemBytes?: number;
  storageDir?: string;
}

function nearestExistingDir(dir: string): string {
  let current = dir;
  for (;;) {
    if (existsSync(current)) return current;
    const parent = dirname(current);
    if (parent === current) return current;
    current = parent;
  }
}

function defaultStatfs(path: string): { bavail: number; bsize: number } {
  const stats = statfsSync(path);
  return { bavail: Number(stats.bavail), bsize: Number(stats.bsize) };
}

function defaultExecPython(exe: string, args: readonly string[]): string {
  return execFileSync(exe, [...args], {
    timeout: 10_000,
    stdio: ["ignore", "pipe", "pipe"],
  }).toString();
}

function probePythonVersion(
  execPython: (exe: string, args: readonly string[]) => string,
): string | null {
  for (const exe of ["python3.11", "python3"]) {
    try {
      const output = execPython(exe, ["--version"]);
      const match = /Python\s+(\d+)\.(\d+)/.exec(output);
      if (match) return `${match[1]}.${match[2]}`;
    } catch {
      // try the next candidate
    }
  }
  return null;
}

export function checkRequirements(probes: RequirementProbes = {}): RequirementCheck[] {
  const checks: RequirementCheck[] = [];

  const rawVersion = (probes.nodeVersion ?? process.version).trim().replace(/^v/i, "");
  const major = Number.parseInt(rawVersion.split(".")[0] ?? "", 10);
  if (!Number.isFinite(major) || major < 24) {
    checks.push({
      id: "node",
      label: "Node.js",
      status: "fail",
      detail: `found ${probes.nodeVersion ?? process.version}, need >= 24`,
      hint: "Install Node.js 24+ (https://nodejs.org), then re-run `zcode onboard`.",
    });
  } else {
    checks.push({
      id: "node",
      label: "Node.js",
      status: "pass",
      detail: `v${rawVersion} (>= 24)`,
    });
  }

  const dir = probes.storageDir ?? resolveZCodeStorageDir();
  try {
    const target = nearestExistingDir(dir);
    const stats = (probes.statfs ?? defaultStatfs)(target);
    const freeGB = (Number(stats.bavail) * Number(stats.bsize)) / 1024 ** 3;
    if (freeGB < 2) {
      checks.push({
        id: "disk",
        label: "Disk space",
        status: "warn",
        detail: `${freeGB.toFixed(1)} GB free at ${target}`,
        hint: "Free up some space — the runtime plus the bundled SystemOne shim want ~2 GB.",
      });
    } else {
      checks.push({
        id: "disk",
        label: "Disk space",
        status: "pass",
        detail: `${freeGB.toFixed(1)} GB free at ${target}`,
      });
    }
  } catch {
    checks.push({
      id: "disk",
      label: "Disk space",
      status: "warn",
      detail: "could not measure",
      hint: "Make sure the storage directory is writable.",
    });
  }

  const python = probePythonVersion(probes.execPython ?? defaultExecPython);
  if (python === null) {
    checks.push({
      id: "python",
      label: "Python",
      status: "warn",
      detail: "not found",
      hint: "Install Python >= 3.10 — ZCode's bundled SystemOne shim runs on it.",
    });
  } else {
    const [majorStr, minorStr] = python.split(".");
    const ok =
      Number.parseInt(majorStr ?? "", 10) > 3 ||
      (Number.parseInt(majorStr ?? "", 10) === 3 && Number.parseInt(minorStr ?? "", 10) >= 10);
    checks.push({
      id: "python",
      label: "Python",
      status: ok ? "pass" : "warn",
      detail: ok ? `${python} (>= 3.10, for the SystemOne shim)` : `${python} (< 3.10)`,
      ...(ok ? {} : { hint: "Upgrade to Python >= 3.10 for the SystemOne shim." }),
    });
  }

  const memGB = (probes.totalmemBytes ?? totalmem()) / 1024 ** 3;
  if (memGB < 4) {
    checks.push({
      id: "ram",
      label: "Memory",
      status: "warn",
      detail: `${memGB.toFixed(1)} GB total`,
      hint: "8 GB+ is recommended for local inference; ZCode will still run.",
    });
  } else {
    checks.push({
      id: "ram",
      label: "Memory",
      status: "pass",
      detail: `${memGB.toFixed(1)} GB total`,
    });
  }

  return checks;
}

// ---------------------------------------------------------------------------
// Probes (detect-only; nothing is ever started, loaded, or switched here)
// ---------------------------------------------------------------------------

export async function probeLmStudioModels(
  fetchImpl: typeof fetch = fetch,
): Promise<string[] | null> {
  try {
    const response = await fetchImpl(LM_STUDIO_MODELS_URL, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!response.ok) return null;
    const json = (await response.json()) as { data?: Array<{ id?: unknown }> };
    const ids = (json.data ?? [])
      .map((entry) => (typeof entry?.id === "string" ? entry.id : ""))
      .filter((id) => id.length > 0);
    return ids;
  } catch {
    return null;
  }
}

export async function probeSystemOneShim(
  fetchImpl: typeof fetch = fetch,
  shimUrl: string = DEFAULT_SYSTEMONE_SHIM_URL,
): Promise<boolean> {
  try {
    const response = await fetchImpl(`${shimUrl.replace(/\/+$/, "")}/healthz`, {
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    return response.ok;
  } catch {
    return false;
  }
}

export interface SystemOneDecideProbe {
  ok: boolean;
  backend?: string;
  error?: string;
}

/**
 * POST a tiny noul probe to the shim's decide endpoint. Fail-open by
 * design: any failure returns { ok: false } and the wizard reports it
 * and moves on — a down decide engine never fails setup.
 */
export async function probeSystemOneDecide(
  fetchImpl: typeof fetch = fetch,
  shimUrl: string = DEFAULT_SYSTEMONE_SHIM_URL,
): Promise<SystemOneDecideProbe> {
  const url = `${shimUrl.replace(/\/+$/, "")}/v1/systemone/decide`;
  try {
    const response = await fetchImpl(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        state: "onboarding connectivity probe",
        instructions: "Is this an onboarding connectivity probe? Answer yes.",
        type: "noul",
      }),
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
    if (!response.ok) return { ok: false, error: `HTTP ${response.status}` };
    const payload = (await response.json()) as Record<string, unknown>;
    if (payload?.["type"] !== "noul") {
      return { ok: false, error: "unexpected decide reply shape" };
    }
    const backend = payload["backend"];
    return { ok: true, ...(typeof backend === "string" ? { backend } : {}) };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

// ---------------------------------------------------------------------------
// Config writes (idempotent, via the existing personal-config repository)
// ---------------------------------------------------------------------------

function openPersonalConfigRepository(storageDir: string): NodePersonalProviderConfigRepository {
  const filePath = join(storageDir, "v2", PERSONAL_PROVIDER_CONFIG_FILE_NAME);
  mkdirSync(join(storageDir, "v2"), { recursive: true });
  return new NodePersonalProviderConfigRepository({
    filePath,
    pollingIntervalMs: false,
  });
}

/** Stores the Meta Model API key as a personal provider overlay (template "meta"). */
export async function saveMetaApiKey(input: { storageDir: string; apiKey: string }): Promise<void> {
  const apiKey = input.apiKey.trim();
  if (!apiKey) throw new Error("API key must not be empty");
  const repo = openPersonalConfigRepository(input.storageDir);
  try {
    await repo.update((current) => {
      const existing = current.providers.getRule(META_PROVIDER_ID);
      const access = {
        type: "api-key",
        apiKey,
        apiKeyManagementUrl: META_API_KEY_MANAGEMENT_URL,
      } as const;
      const nextConfig = existing
        ? existing.config.overlay(parseProviderConfig({ access }))
        : parseProviderConfig({
            group: "standard-personal",
            access,
            personalModelIds: [],
            modelOrder: [],
          });
      return {
        providers: current.providers.setRule({
          providerId: META_PROVIDER_ID,
          templateId: META_TEMPLATE_ID,
          providerName: existing?.providerName ?? META_PROVIDER_NAME,
          enabled: existing?.enabled ?? true,
          config: nextConfig,
        }),
        models: current.models,
        providerOrder: current.providerOrder,
        defaultModelSelection: current.defaultModelSelection,
      };
    });
  } finally {
    repo.dispose();
  }
}

/** Records the user's explicit LM Studio model pick as the default model selection. */
export async function saveDefaultModelSelection(input: {
  storageDir: string;
  providerId: string;
  modelId: string;
}): Promise<void> {
  const repo = openPersonalConfigRepository(input.storageDir);
  try {
    await repo.update((current) => ({
      providers: current.providers,
      models: current.models,
      providerOrder: current.providerOrder,
      defaultModelSelection: {
        providerId: input.providerId,
        modelId: input.modelId,
      },
    }));
  } finally {
    repo.dispose();
  }
}

// ---------------------------------------------------------------------------
// Interactive IO
// ---------------------------------------------------------------------------

export interface WizardIO {
  readonly stdinIsTTY: boolean;
  write(text: string): void;
  writeError(text: string): void;
  question(prompt: string): Promise<string>;
  questionHidden(prompt: string): Promise<string>;
  close(): void;
}

/**
 * Non-interactive WizardIO: no readline, every prompt falls back to its
 * default. Used when stdin is not a TTY so the command can never block.
 */
function createNonInteractiveIO(ctx: RunContext): WizardIO {
  return {
    stdinIsTTY: false,
    write: (text: string) => {
      ctx.stdout.write(text);
    },
    writeError: (text: string) => {
      ctx.stderr.write(text);
    },
    question: async () => "",
    questionHidden: async () => "",
    close: () => {},
  };
}

function createWizardIO(ctx: RunContext): WizardIO {
  const rl = readline.createInterface({
    input: ctx.stdin,
    output: ctx.stdout,
    terminal: true,
  });
  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => {
      rl.question(prompt, (answer) => resolve(answer));
    });
  const questionHidden = (prompt: string): Promise<string> =>
    new Promise((resolve) => {
      const mutable = rl as unknown as { _writeToOutput(text: string): void };
      const original = mutable._writeToOutput.bind(mutable);
      mutable._writeToOutput = (text: string) => {
        // Let the prompt itself print normally; mask everything typed after it.
        if (text === prompt || text.startsWith(prompt)) {
          original(text);
        } else {
          const output = (rl as unknown as { output?: NodeJS.WritableStream }).output;
          output?.write(text.replace(/[^\r\n]/g, "*"));
        }
      };
      rl.question(prompt, (answer) => {
        mutable._writeToOutput = original;
        resolve(answer.trim());
      });
    });
  return {
    stdinIsTTY: ctx.stdin.isTTY === true,
    write: (text) => {
      ctx.stdout.write(text);
    },
    writeError: (text) => {
      ctx.stderr.write(text);
    },
    question,
    questionHidden,
    close: () => {
      rl.close();
    },
  };
}

// ---------------------------------------------------------------------------
// The wizard
// ---------------------------------------------------------------------------

export interface WizardRunOptions {
  env?: NodeJS.ProcessEnv;
  home?: string;
  yes?: boolean;
  fetchImpl?: typeof fetch;
  probes?: RequirementProbes;
  /** Test seam: override the shim starter (defaults to ensureSystemOneShim). */
  startShim?: () => Promise<void>;
}

export async function runOnboardingWizard(
  io: WizardIO,
  opts: WizardRunOptions = {},
): Promise<number> {
  const env = opts.env ?? process.env;
  const home = opts.home ?? homedir();
  const yes = opts.yes === true;
  const storageDir = resolveZCodeStorageDir(env, home);
  const out = (text: string): void => io.write(`${text}\n`);

  const ask = async (prompt: string, defaultAnswer: string): Promise<string> => {
    if (yes) {
      out(`${prompt}[auto: ${defaultAnswer === "" ? "(skip)" : defaultAnswer}]`);
      return defaultAnswer;
    }
    const answer = (await io.question(prompt)).trim();
    return answer === "" ? defaultAnswer : answer;
  };

  out("");
  out("ZCode Local setup");
  out("-----------------");
  out("");

  // Step 1 — requirements. Node < 24 is the only hard failure.
  out("Step 1 of 3 — Requirements");
  const checks = checkRequirements({ ...(opts.probes ?? {}), storageDir });
  let failed = false;
  for (const check of checks) {
    const mark = check.status === "pass" ? "ok" : check.status === "warn" ? "!!" : "XX";
    out(`  [${mark}] ${check.label}: ${check.detail}`);
    if (check.hint) out(`       ${check.hint}`);
    if (check.status === "fail") failed = true;
  }
  if (failed) {
    out("");
    out("Setup cannot continue until the failed check above is fixed.");
    return 1;
  }
  out("");

  // Step 2 — inference. Detect only: never load, unload, or switch models.
  out("Step 2 of 3 — Inference");
  let inference: OnboardingInferenceState = { mode: "auto" };

  out(`Looking for LM Studio at ${LM_STUDIO_BASE_URL} ...`);
  const models = await probeLmStudioModels(opts.fetchImpl);
  if (models === null) {
    out("  LM Studio is not answering there.");
    out("  Start LM Studio and load a model, or continue with an API key below.");
  } else if (models.length === 0) {
    out("  LM Studio is running, but no models are loaded.");
    out("  (Detect-only: ZCode never loads or switches models on its own.");
    out("   Load one in LM Studio, then re-run `zcode onboard`.)");
  } else {
    out("  LM Studio is running. Models currently loaded:");
    models.forEach((id, index) => out(`    ${index + 1}) ${id}`));
    out("  (Detect-only: ZCode never loads, unloads, or switches models by itself.)");
    const pick = await ask(
      `Which model should ZCode use? [1-${models.length}, Enter = let SystemOne decide]: `,
      "",
    );
    const n = Number.parseInt(pick, 10);
    if (pick !== "" && Number.isInteger(n) && n >= 1 && n <= models.length) {
      const modelId = models[n - 1] as string;
      inference = { mode: "lm-studio", modelId };
      await saveDefaultModelSelection({
        storageDir,
        providerId: LM_STUDIO_PROVIDER_ID,
        modelId,
      });
      out("  Saved — ZCode will default to that LM Studio model.");
      out("  (Change it later with /model in the TUI.)");
    } else {
      out("  Saved — SystemOne will decide per task (recommended).");
    }
  }

  out("");
  out("Meta Model API key (optional) — enables the meta provider (https://api.meta.ai/v1).");
  out("Get a key at https://dev.meta.ai/. Press Enter to skip.");
  const apiKey = yes ? "" : (await io.questionHidden("API key (hidden input): ")).trim();
  if (apiKey !== "") {
    await saveMetaApiKey({ storageDir, apiKey });
    inference = { ...inference, metaApiKeyConfigured: true };
    if (inference.mode === "auto") {
      inference = { mode: "meta-api-key", metaApiKeyConfigured: true };
    }
    out(
      `  Saved to ${join(storageDir, "v2", PERSONAL_PROVIDER_CONFIG_FILE_NAME)} (provider "meta").`,
    );
  } else {
    out("  Skipped.");
  }
  out("");

  // Step 3 — SystemOne routing + decide engine. Advisory + fail-open, always.
  // ZCode does NOT auto-start the bundled shim (grok-local does) — the
  // wizard probes for it and offers to start it, but never starts it
  // silently. A down shim never fails setup: the wizard warns and keeps
  // going, and everything runs degraded (fail-open) afterwards.
  out("Step 3 of 3 — SystemOne routing + decide engine");

  // Shim URL: default from $SYSTEMONE_SHIM_URL (else localhost), overridable.
  const defaultShimUrl = resolveSystemOneShimUrl(env);
  const shimUrlAnswer = await ask(
    `SystemOne shim URL? [Enter = ${defaultShimUrl}]: `,
    defaultShimUrl,
  );
  let shimUrl = shimUrlAnswer.trim().replace(/\/+$/, "");
  if (!isPlausibleShimUrl(shimUrl)) {
    out(`  "${shimUrlAnswer}" doesn't look like a shim URL — keeping ${defaultShimUrl}.`);
    shimUrl = defaultShimUrl;
  }
  if (shimUrl !== defaultShimUrl) {
    out("  Custom shim URL — saved, and applied at runtime automatically");
    out(`  (override anytime with ${SYSTEMONE_SHIM_URL_ENV}=${shimUrl} in your shell).`);
  }

  out(`Checking the SystemOne shim at ${shimUrl} ...`);
  let shimUp = await probeSystemOneShim(opts.fetchImpl, shimUrl);
  if (shimUp) {
    out("  The shim is answering — routing is ready.");
    const decideProbe = await probeSystemOneDecide(opts.fetchImpl, shimUrl);
    if (decideProbe.ok) {
      out(
        `  Decide engine answering${decideProbe.backend ? ` (backend: ${decideProbe.backend})` : ""} — plan ranking can use it.`,
      );
    } else {
      out("  Decide engine not answering — plan ranking stays fail-open.");
    }
  } else {
    out("  The shim is not answering right now — continuing in degraded mode.");
    out("  ZCode does not auto-start the bundled shim — grok-local does.");
    if (!isLoopbackShimUrl(shimUrl)) {
      out("  That shim URL is remote — start the shim on its own host;");
      out("  ZCode won't start a local shim for a remote URL.");
    } else {
      out("  To run it yourself: python3.11 -m systemone.shim --port 8765");
      if (!yes) {
        const startAnswer = await ask("Start the bundled shim now? [y/N]: ", "n");
        if (/^(y|yes)$/i.test(startAnswer)) {
          out("  Starting the bundled shim ...");
          try {
            await (opts.startShim ?? ensureSystemOneShim)();
          } catch (error) {
            out(
              `  Could not start the shim: ${error instanceof Error ? error.message : String(error)}`,
            );
          }
          shimUp = await probeSystemOneShim(opts.fetchImpl, shimUrl);
          out(
            shimUp
              ? "  The shim is answering now."
              : "  Still not answering — routing stays fail-open.",
          );
        }
      }
    }
  }
  out("");
  out("SystemOne routing is advisory and fail-open: it suggests the cheapest");
  out("sufficient model tier, effort level, and tool budget per task, but it never");
  out("blocks a run. If the router is missing or unsure, ZCode just runs normally.");
  out("Kill switches: ZCODE_SYSTEMONE=0 (routing + auto-start off),");
  out("ZCODE_SPEEDSTACK_PRUNE=0 (tool pruning off),");
  out(`${SYSTEMONE_DECIDE_KILL_SWITCH_ENV}=0 (decide engine off).`);
  out("");

  const enableAnswer = await ask("Enable SystemOne routing? [Y/n]: ", "y");
  const systemoneEnabled = !/^(n|no)$/i.test(enableAnswer);
  const decideAnswer = await ask(
    "Enable the decide engine for plan ranking? (advisory fallback when rank-plans is down; default on) [Y/n]: ",
    "y",
  );
  const decideFallback = !/^(n|no)$/i.test(decideAnswer);
  if (!decideFallback) {
    out("  Decide engine off for this machine — saved, applied at runtime automatically.");
  }
  const jeffAnswer = await ask(
    "Enable Jeff-1 second-opinion routing? (optional decision head; default on) [Y/n]: ",
    "y",
  );
  const jeff1 = !/^(n|no)$/i.test(jeffAnswer);
  if (!systemoneEnabled) {
    out("");
    out("SystemOne disabled for this machine — saved, applied at runtime automatically.");
    out("  (Override anytime with ZCODE_SYSTEMONE=0 in your shell.)");
  }
  out("The shim's decide engine is selected server-side by SYSTEMONE_DECISION_BACKEND");
  out("(jeff1 | decider); decider is Mapika/decider-4b (Apache 2.0).");
  out("Your choices are saved now and honored at runtime. Kill switch: SYSTEMONE_JEFF1=0.");

  const state: OnboardingState = {
    version: ONBOARDING_STATE_VERSION,
    onboarded: true,
    completedAt: new Date().toISOString(),
    systemone: { enabled: systemoneEnabled, jeff1, shimUrl, decideFallback },
    inference,
  };
  writeOnboardingState(storageDir, state);
  out("");
  out(`Setup complete. Answers saved to ${resolveOnboardingStatePath(env, home)}.`);
  out("Re-run `zcode onboard` any time to change them.");
  return 0;
}

// ---------------------------------------------------------------------------
// `zcode onboard` command entry + first-run trigger for the TUI
// ---------------------------------------------------------------------------

export async function runOnboardCommand(
  ctx: RunContext,
  _options: GlobalOptions,
  positionals: readonly string[],
  deps: {
    env?: NodeJS.ProcessEnv;
    home?: string;
    probes?: RequirementProbes;
    /** Explicit --yes from the CLI arg parser (global flag is consumed there). */
    yes?: boolean;
  } = {},
): Promise<number> {
  const env = deps.env ?? process.env;
  const home = deps.home ?? homedir();
  const yes = deps.yes === true || positionals.includes("--yes");
  const stdinIsTTY = ctx.stdin.isTTY === true;
  if (!stdinIsTTY && !yes) {
    ctx.stdout.write(
      "zcode onboard needs an interactive terminal. Nothing to do in non-interactive mode.\n",
    );
    ctx.stdout.write("Re-run it in a terminal, or pass --yes to accept all defaults.\n");
    return 0;
  }
  const io = ctx.stdin.isTTY === true ? createWizardIO(ctx) : createNonInteractiveIO(ctx);
  const runOpts: WizardRunOptions = { env, home, yes };
  if (deps.probes) runOpts.probes = deps.probes;
  try {
    return await runOnboardingWizard(io, runOpts);
  } finally {
    io.close();
  }
}

/**
 * First-run trigger for `zcode` / `zcode tui`: when launching interactively
 * with no existing user config (and not skipped), run the wizard first,
 * then continue into the TUI. Returns an exit code to stop, or undefined
 * to continue.
 */
export async function maybeRunFirstTimeOnboarding(
  ctx: RunContext,
  input: {
    argv: readonly string[];
    env?: NodeJS.ProcessEnv;
    stdinIsTTY: boolean;
    home?: string;
  },
): Promise<number | undefined> {
  const env = input.env ?? process.env;
  if (
    !shouldTriggerFirstRunOnboarding({
      argv: input.argv,
      env,
      stdinIsTTY: input.stdinIsTTY,
      home: input.home,
    })
  ) {
    return undefined;
  }
  ctx.stdout.write("\nFirst run detected — starting the ZCode Local setup wizard.\n");
  ctx.stdout.write(
    `(Skip any time with ${ONBOARDING_SKIP_FLAG} or ${ONBOARDING_SKIP_ENV_VAR}=1.)\n\n`,
  );
  const code = await runOnboardCommand(ctx, {} as GlobalOptions, [], {
    env,
    home: input.home,
  });
  return code === 0 ? undefined : code;
}
