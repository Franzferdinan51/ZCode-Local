// ============================================================
// SystemOne shim auto-start (zero-setup routing)
// ============================================================
//
// At ZCode startup (CLI, desktop-GUI agent sessions, and server-spawned
// agents all run through the CLI bundle's main()), probe the SystemOne
// shim (GET $SYSTEMONE_SHIM_URL/healthz, default
// http://127.0.0.1:8765/healthz). When nothing answers on a LOCAL shim
// URL, spawn the shim bundled with the ZCode release
// (`<release>/systemone`, the `systemone` Python package) as a detached
// background process:
//
//   systemone serve --port 8765 [--engine auto|local|sglang|jevk5|onnx|jev|kev|clef]
// (preferred when the `systemone` console script is on PATH; the engine
// flag defaults to $SYSTEMONE_ENGINE, else auto), else:
//
//   python3.11 -m systemone.shim --port 8765 [--daemonize on Windows]
// (+ --daemonize on win32: the shim self-detaches from sshd's
// KILL_ON_JOB_CLOSE job object; see systemone/shim.py)
//
// A shim already answering the probe (e.g. a manually managed one) is
// used as-is — never duplicated. A REMOTE $SYSTEMONE_SHIM_URL is probed
// but never auto-started locally: spawning a localhost shim for a remote
// URL would leave the runtime pointed at the wrong place.
//
// Everything here is fail-open: a missing Python, a missing bundled shim,
// or a failed spawn logs one line to stderr and startup continues with
// routing disabled for the process. A session is never broken because the
// router is missing.
//
// Kill-switches / overrides (env):
//   ZCODE_SYSTEMONE=0        disables auto-start AND all route lookups
//   ZCODE_SYSTEMONE_DIR      bundled `systemone` package dir (has shim.py)
//   ZCODE_SYSTEMONE_PYTHON   python executable used to run the shim
//   ZCODE_SYSTEMONE_WAIT_MS  max ms to wait for healthz after spawning
//                            (default 30000)
//   SYSTEMONE_SHIM_URL       shim base URL to probe (default
//                            http://127.0.0.1:8765); remote URLs are
//                            never auto-started
//   SYSTEMONE_ENGINE         engine the spawned shim serves
//                            (auto|local|sglang|jevk5|onnx|jev|kev|clef)

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, open, rm, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, isAbsolute, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import {
  DEFAULT_SYSTEMONE_SHIM_URL,
  isLoopbackShimUrl,
  isSystemOneDisabled,
  resolveSystemOneShimUrl,
} from "@zcode/core";

export const SYSTEMONE_SHIM_PORT = 8765;
export const SYSTEMONE_HEALTHZ_URL = `http://127.0.0.1:${SYSTEMONE_SHIM_PORT}/healthz`;

/**
 * Resolve the shim base URL under test (honors $SYSTEMONE_SHIM_URL).
 * Exported for tests.
 */
export function resolveSystemOneShimBase(
  env: NodeJS.ProcessEnv = process.env,
): string {
  try {
    return resolveSystemOneShimUrl(env);
  } catch {
    return DEFAULT_SYSTEMONE_SHIM_URL;
  }
}

/** Healthz URL for the resolved shim base. Exported for tests. */
export function resolveSystemOneHealthzUrl(
  env: NodeJS.ProcessEnv = process.env,
): string {
  return `${resolveSystemOneShimBase(env)}/healthz`;
}

/**
 * Port a locally spawned shim should listen on: the resolved shim base's
 * explicit port, else 8765. Unparseable bases fail open to 8765.
 * Exported for tests.
 */
export function resolveSystemOneSpawnPort(
  env: NodeJS.ProcessEnv = process.env,
): number {
  try {
    const port = new URL(resolveSystemOneShimBase(env)).port;
    const parsed = Number(port);
    if (port !== "" && Number.isInteger(parsed) && parsed > 0 && parsed < 65536) {
      return parsed;
    }
  } catch {
    // fall through to the default
  }
  return SYSTEMONE_SHIM_PORT;
}

const HEALTHZ_PROBE_TIMEOUT_MS = 1_500;
const PYTHON_PROBE_TIMEOUT_MS = 25_000;
const GLICLASS_IMPORT_CHECK_TIMEOUT_MS = 20_000;
const SPAWN_LOCK_TTL_MS = 120_000;
const DEFAULT_WAIT_MS = 30_000;

export interface SystemOneBootstrapOptions {
  env?: NodeJS.ProcessEnv;
  stderr?: Pick<NodeJS.WriteStream, "write">;
}

function resolveEnv(options?: SystemOneBootstrapOptions): NodeJS.ProcessEnv {
  return options?.env ?? process.env;
}

function zcodeHome(env: NodeJS.ProcessEnv): string {
  return env.ZCODE_STORAGE_DIR?.trim() || join(homedir(), ".zcode-local");
}

/** Skip the bootstrap for trivial or nested invocations. Exported for tests. */
export function shouldEnsureSystemOneShim(
  argv: readonly string[],
  options?: { env?: NodeJS.ProcessEnv; isPluginHost?: boolean },
): boolean {
  const env = options?.env ?? process.env;
  if (isSystemOneDisabled(env)) return false;
  if (argv.includes("--prepare-storage")) return false;
  if (options?.isPluginHost) return false;
  for (const arg of argv) {
    if (
      arg === "--version" ||
      arg === "-v" ||
      arg === "--help" ||
      arg === "-h" ||
      arg === "--licenses"
    ) {
      return false;
    }
  }
  return true;
}

async function healthzOk(
  timeoutMs: number,
  url: string = SYSTEMONE_HEALTHZ_URL,
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, {
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

function bundleDirectory(): string | undefined {
  // The CLI ships as a single CJS bundle (zcode.cjs); esbuild preserves
  // __dirname as the bundle's directory.
  if (typeof __dirname === "string" && __dirname) return __dirname;
  const argv1 = process.argv[1];
  if (argv1) return dirname(argv1);
  return undefined;
}

/**
 * Locate the bundled `systemone` Python package directory (contains
 * shim.py). Release layout is `<release>/systemone` next to
 * `<release>/agent/zcode.cjs`; desktop/Electron and dev setups fall back
 * to the installed runtime's `current` symlink. Exported for tests.
 */
export function findSystemOneDir(
  options?: { env?: NodeJS.ProcessEnv; fromDir?: string },
): string | undefined {
  const env = options?.env ?? process.env;
  const explicit = env.ZCODE_SYSTEMONE_DIR?.trim();
  if (explicit && existsSync(join(explicit, "shim.py"))) return explicit;
  const roots: Array<string | undefined> = [
    options?.fromDir,
    bundleDirectory(),
    join(zcodeHome(env), "runtime", "current"),
  ];
  const seen = new Set<string>();
  for (const root of roots) {
    if (!root || seen.has(root)) continue;
    seen.add(root);
    for (const candidate of [
      join(root, "systemone"),
      join(root, "..", "systemone"),
    ]) {
      try {
        if (existsSync(join(candidate, "shim.py"))) return candidate;
      } catch {
        // ignore
      }
    }
  }
  return undefined;
}

async function resolveOnPath(
  name: string,
  timeoutMs: number,
): Promise<string | undefined> {
  const command = process.platform === "win32" ? "where" : "sh";
  const args =
    process.platform === "win32" ? [name] : ["-c", `command -v ${name}`];
  return new Promise((resolvePromise) => {
    const child = spawn(command, args, {
      stdio: ["ignore", "pipe", "ignore"],
      windowsHide: true,
    });
    let output = "";
    const timer = setTimeout(() => {
      child.kill();
      resolvePromise(undefined);
    }, timeoutMs);
    child.stdout?.on("data", (chunk: Buffer) => {
      output += chunk.toString();
    });
    child.on("error", () => {
      clearTimeout(timer);
      resolvePromise(undefined);
    });
    child.on("close", () => {
      clearTimeout(timer);
      const first = output.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)[0];
      resolvePromise(first);
    });
  });
}

/**
 * Python imports the spawned shim needs for the configured engine.
 * Slim floor is always `numpy`; the torch/GLiClass stack (`gliclass`)
 * is required only when a local engine will actually load weights
 * (explicit `local`, or `auto` with no remote engine configured);
 * `onnx` needs `onnxruntime` instead, and `clef` needs the
 * torch/transformers stack (which pulls huggingface_hub/safetensors;
 * pillow is only needed for image inputs). Exported for tests.
 */
export function requiredShimImports(
  env: NodeJS.ProcessEnv = process.env,
): readonly string[] {
  try {
    const engine = (env.SYSTEMONE_ENGINE ?? "auto").trim().toLowerCase();
    if (engine === "onnx") return ["numpy", "onnxruntime"];
    if (engine === "clef") return ["numpy", "torch", "transformers"];
    if (
      engine === "sglang" ||
      engine === "jevk5" ||
      engine === "jev" ||
      engine === "kev"
    ) {
      return ["numpy"];
    }
    if (engine === "local") return ["numpy", "gliclass"];
    // auto (or unknown — the shim validates names itself): remote when a
    // remote engine URL is configured, else local weights.
    const remoteConfigured =
      (env.JEV_URL ?? "").trim() !== "" ||
      (env.SGLANG_BASE_URL ?? "").trim() !== "" ||
      (env.JEVK5_BASE_URL ?? "").trim() !== "" ||
      (env.KEV_BASE_URL ?? "").trim() !== "";
    return remoteConfigured ? ["numpy"] : ["numpy", "gliclass"];
  } catch {
    return ["numpy", "gliclass"];
  }
}

async function pythonCanRunShim(
  python: string,
  timeoutMs: number,
  imports: readonly string[],
): Promise<boolean> {
  const snippet = imports.map((name) => `import ${name}`).join("; ");
  return new Promise((resolvePromise) => {
    const child = spawn(python, ["-c", snippet], {
      stdio: "ignore",
      windowsHide: true,
    });
    const timer = setTimeout(() => {
      child.kill();
      resolvePromise(false);
    }, timeoutMs);
    child.on("error", () => {
      clearTimeout(timer);
      resolvePromise(false);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      resolvePromise(code === 0);
    });
  });
}

/**
 * Find a Python interpreter able to run the bundled shim (i.e. one where
 * the engine's required imports succeed — see requiredShimImports).
 * Returns undefined when none is usable — the caller then logs once and
 * continues fail-open. Exported for tests.
 */
export async function findSuitablePython(
  options?: { env?: NodeJS.ProcessEnv },
): Promise<string | undefined> {
  const env = options?.env ?? process.env;
  const imports = requiredShimImports(env);
  const explicit = env.ZCODE_SYSTEMONE_PYTHON?.trim();
  const names = [
    ...(explicit ? [explicit] : []),
    "python3.11",
    "python3",
    "python",
  ];
  const absoluteFallbacks: string[] =
    process.platform === "darwin"
      ? [
          join(homedir(), ".local", "bin", "python3.11"),
          "/opt/homebrew/bin/python3.11",
          "/usr/local/bin/python3.11",
        ]
      : [];
  const seen = new Set<string>();
  for (const name of [...names, ...absoluteFallbacks]) {
    if (!name || seen.has(name)) continue;
    seen.add(name);
    let resolved: string | undefined;
    if (isAbsolute(name)) {
      resolved = existsSync(name) ? name : undefined;
    } else {
      resolved = await resolveOnPath(name, PYTHON_PROBE_TIMEOUT_MS);
    }
    if (!resolved) continue;
    if (
      await pythonCanRunShim(resolved, GLICLASS_IMPORT_CHECK_TIMEOUT_MS, imports)
    ) {
      return resolved;
    }
  }
  return undefined;
}

function parseWaitMs(env: NodeJS.ProcessEnv): number {
  const raw = Number(env.ZCODE_SYSTEMONE_WAIT_MS);
  if (Number.isFinite(raw) && raw >= 0) return Math.floor(raw);
  return DEFAULT_WAIT_MS;
}

function spawnLockPath(env: NodeJS.ProcessEnv): string {
  return join(zcodeHome(env), "systemone-shim.lock");
}

/**
 * Best-effort mutex so two ZCode processes starting at once don't both
 * spawn the shim. True when this process owns the lock. Stale locks
 * (older than SPAWN_LOCK_TTL_MS) are taken over.
 */
async function acquireSpawnLock(env: NodeJS.ProcessEnv): Promise<boolean> {
  const lockPath = spawnLockPath(env);
  try {
    const handle = await open(lockPath, "wx");
    await handle.writeFile(
      JSON.stringify({ pid: process.pid, ts: Date.now() }),
    );
    await handle.close();
    return true;
  } catch (error: unknown) {
    if ((error as NodeJS.ErrnoException)?.code !== "EEXIST") return false;
    try {
      const lockStat = await stat(lockPath);
      if (Date.now() - lockStat.mtimeMs < SPAWN_LOCK_TTL_MS) return false;
      await rm(lockPath, { force: true });
      return acquireSpawnLock(env);
    } catch {
      return false;
    }
  }
}

async function releaseSpawnLock(env: NodeJS.ProcessEnv): Promise<void> {
  await rm(spawnLockPath(env), { force: true });
}

/**
 * Resolve the argv used to spawn the shim: prefer the installed
 * `systemone serve` console script when it is on PATH (modern entry
 * point; serves the $SYSTEMONE_ENGINE engine, default auto), else run
 * the bundled package directly with the probed Python. Exported for
 * tests (the PATH probe is injectable).
 */
export async function resolveShimSpawnCommand(
  python: string,
  port: number,
  options?: {
    env?: NodeJS.ProcessEnv;
    resolveOnPathImpl?: (
      name: string,
      timeoutMs: number,
    ) => Promise<string | undefined>;
  },
): Promise<{ command: string; args: string[]; viaServe: boolean }> {
  const probe = options?.resolveOnPathImpl ?? resolveOnPath;
  const serve = await probe("systemone", PYTHON_PROBE_TIMEOUT_MS).catch(
    () => undefined,
  );
  if (serve) {
    return {
      command: serve,
      args: ["serve", "--port", String(port)],
      viaServe: true,
    };
  }
  return {
    command: python,
    args: [
      "-m",
      "systemone.shim",
      "--port",
      String(port),
      // Windows: sshd runs the session in a KILL_ON_JOB_CLOSE job object
      // that Node's detached:true cannot escape (Node can't set creation
      // flags). The shim re-spawns itself detached
      // (CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS) so it survives the
      // SSH session close. No-op on other platforms; fail-open everywhere.
      ...(process.platform === "win32" ? ["--daemonize"] : []),
    ],
    viaServe: false,
  };
}

async function spawnShim(det: {
  python: string;
  systemoneDir: string;
  port: number;
  env: NodeJS.ProcessEnv;
}): Promise<boolean> {
  const { python, systemoneDir, port, env } = det;
  const logDir = join(zcodeHome(env), "logs");
  await mkdir(logDir, { recursive: true });
  const outLog = join(logDir, "systemone-shim.out.log");
  const outHandle = await open(outLog, "a");
  try {
    const { command, args, viaServe } = await resolveShimSpawnCommand(
      python,
      port,
      { env },
    );
    const child = spawn(command, args, {
      // `-m systemone.shim` needs the package's parent dir on sys.path;
      // Python puts the cwd there for -m invocations. `systemone serve`
      // resolves its own install, so the cwd is harmless there.
      cwd: dirname(systemoneDir),
      detached: true,
      stdio: ["ignore", outHandle.fd, outHandle.fd],
      windowsHide: true,
      env: {
        ...env,
        // Keep the shim's own rotating request log out of the release dir.
        SYSTEMONE_LOG_FILE: join(logDir, "systemone-shim.log"),
      },
    });
    child.on("error", () => {
      // The wait loop below observes the failure via healthz; nothing to do.
    });
    child.unref();
    return viaServe;
  } finally {
    await outHandle.close();
  }
}

async function waitForHealthz(
  deadlineMs: number,
  url: string,
): Promise<boolean> {
  const start = Date.now();
  for (;;) {
    if (await healthzOk(HEALTHZ_PROBE_TIMEOUT_MS, url)) return true;
    if (Date.now() - start >= deadlineMs) return false;
    await delay(500);
  }
}

/**
 * Probe for the SystemOne shim and auto-start the bundled copy when
 * nothing answers on a LOCAL shim URL. Never throws: any failure logs
 * one line and startup continues with routing disabled (fail-open).
 */
export async function ensureSystemOneShim(
  options?: SystemOneBootstrapOptions,
): Promise<void> {
  const env = resolveEnv(options);
  const log = (message: string): void => {
    try {
      options?.stderr?.write(`[zcode] systemone: ${message}\n`);
    } catch {
      // logging must never break startup
    }
  };
  try {
    if (isSystemOneDisabled(env)) return;
    const shimBase = resolveSystemOneShimBase(env);
    const healthzUrl = resolveSystemOneHealthzUrl(env);
    if (await healthzOk(HEALTHZ_PROBE_TIMEOUT_MS, healthzUrl)) return;
    if (!isLoopbackShimUrl(shimBase)) {
      log(
        `shim at ${shimBase} is unreachable and the URL is remote, so no local shim was started; continuing without routing (fail-open)`,
      );
      return;
    }
    const port = resolveSystemOneSpawnPort(env);
    const systemoneDir = findSystemOneDir({ env });
    if (!systemoneDir) {
      log(
        "no bundled shim found (looked next to the agent bundle and under ~/.zcode-local/runtime/current); routing disabled for this process (fail-open)",
      );
      return;
    }
    const python = await findSuitablePython({ env });
    if (!python) {
      const missing = requiredShimImports(env).join(", ");
      log(
        `no Python able to 'import ${missing}' found; install the shim deps (pip install -r <release>/systemone/requirements.txt) or set ZCODE_SYSTEMONE_PYTHON; routing disabled for this process (fail-open)`,
      );
      return;
    }
    const lockHeld = await acquireSpawnLock(env);
    if (lockHeld) {
      const viaServe = await spawnShim({ python, systemoneDir, port, env });
      log(
        viaServe
          ? `started shim ('systemone serve --port ${port}') for ${shimBase}; waiting for it to answer healthz`
          : `started bundled shim (python ${python}) on 127.0.0.1:${port}; waiting for it to answer healthz`,
      );
    }
    const healthy = await waitForHealthz(parseWaitMs(env), healthzUrl);
    if (lockHeld) await releaseSpawnLock(env).catch(() => {});
    log(
      healthy
        ? `shim healthy at ${shimBase}; per-task routing enabled`
        : `shim did not answer within ${parseWaitMs(env)}ms; continuing without routing (fail-open)`,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`auto-start failed (${message}); continuing without routing (fail-open)`);
  }
}
