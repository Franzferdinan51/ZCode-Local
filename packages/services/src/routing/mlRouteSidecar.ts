/**
 * Local ML route backends (Jeff-1 / SystemOne / Laya / custom).
 *
 * Two transports, one protocol (`SystemOneRouteRequest`, the Jev-style
 * route-choice shape served by the SystemOne shim at
 * `http://127.0.0.1:8765/v1/systemone`):
 * - HTTP: POST JSON to a loopback `/v1/systemone` endpoint. Covers the
 *   SystemOne `shim.py` (127.0.0.1:8765, primary) and legacy Jeff-1
 *   `jev_clf_server` instances (127.0.0.1:8079 passthrough), which the
 *   user runs themselves. The modern :8079 decision sidecar is
 *   decide-only (`/v1/jeff1/*`) — it never answers route-choice, so it
 *   is never a route fallback; route/rank/decide all live on :8765
 *   (`/v1/systemone/{route,rank-plans,decide}` plus the Jev `/v1/systemone`).
 * - stdio: spawn a persistent NDJSON bridge process. Covers Laya via the
 *   embedded `layaBridgeSource.ts` (`python3 <bridge>`), kept alive so the
 *   checkpoint loads once (~7-10s cold) and later calls cost milliseconds.
 *
 * Everything fails open: any spawn/network/timeout/protocol error throws
 * `MlRouteBackendError` and the caller keeps its heuristic suggestion.
 * HTTP endpoints are restricted to loopback hosts so chat text can never
 * be routed to a LAN/internet URL by a mis-edited config.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdir, writeFile } from "node:fs/promises";
import {
  ML_ROUTE_DEFAULT_ENDPOINTS,
  ML_ROUTE_DEFAULT_TIMEOUT_MS,
  type MlRouteBackendConfig,
  type SystemOneRouteRequest,
} from "@zcode/shared/systemone-scorer";
import { LAYA_BRIDGE_SOURCE } from "./layaBridgeSource.js";

export class MlRouteBackendError extends Error {
  readonly detail?: string;
  constructor(message: string, detail?: string) {
    super(message);
    this.name = "MlRouteBackendError";
    this.detail = detail;
  }
}

/** First-request budget covering checkpoint cold load (cached after). */
const BRIDGE_READY_TIMEOUT_MS = 180_000;
const MAX_STDERR_CHARS = 2000;

function timeoutMsOf(backend: MlRouteBackendConfig): number {
  const value = backend.timeoutMs ?? ML_ROUTE_DEFAULT_TIMEOUT_MS;
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : ML_ROUTE_DEFAULT_TIMEOUT_MS;
}

function assertLoopbackEndpoint(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new MlRouteBackendError(`invalid endpoint URL: ${raw}`);
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new MlRouteBackendError(`endpoint must be http(s): ${raw}`);
  }
  const host = url.hostname.toLowerCase();
  if (host !== "localhost" && host !== "127.0.0.1" && host !== "::1" && host !== "[::1]") {
    throw new MlRouteBackendError(`endpoint must be loopback (got ${url.hostname})`);
  }
  return url;
}

async function queryHttpBackend(
  request: SystemOneRouteRequest,
  endpoint: string,
  timeoutMs: number,
): Promise<unknown> {
  const url = assertLoopbackEndpoint(endpoint);
  let response: Response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(timeoutMs),
    });
  } catch (error) {
    throw new MlRouteBackendError(
      `endpoint unreachable: ${url.host}`,
      error instanceof Error ? error.message : String(error),
    );
  }
  if (!response.ok) {
    throw new MlRouteBackendError(
      `endpoint HTTP ${response.status}`,
      (await response.text().catch(() => "")).slice(0, 500),
    );
  }
  try {
    return (await response.json()) as unknown;
  } catch {
    throw new MlRouteBackendError("endpoint returned invalid JSON");
  }
}

async function layaBridgeFile(): Promise<string> {
  const dir = join(tmpdir(), "zcode-ml-route");
  await mkdir(dir, { recursive: true });
  const digest = createHash("sha256").update(LAYA_BRIDGE_SOURCE).digest("hex").slice(0, 16);
  const file = join(dir, `laya-bridge-${digest}.py`);
  await writeFile(file, LAYA_BRIDGE_SOURCE, "utf8");
  return file;
}

interface BridgeProcess {
  child: ChildProcess;
  /** Complete lines awaiting a waiter. */
  lines: string[];
  /** Incomplete tail carried between chunks (never a complete line). */
  tail: string;
  waiters: Array<{ resolve: (line: string) => void; reject: (error: Error) => void }>;
  ready: Promise<void>;
  stderrTail: string;
  queue: Promise<unknown>;
  exited: boolean;
}

function failWaiters(bridge: BridgeProcess, error: Error): void {
  const waiters = bridge.waiters.splice(0);
  for (const waiter of waiters) waiter.reject(error);
}

function pumpStdout(bridge: BridgeProcess, chunk: Buffer): void {
  const parts = `${bridge.tail}${chunk.toString("utf8")}`.split("\n");
  // Hold the last element back: it may be an incomplete line.
  bridge.tail = parts.pop() ?? "";
  for (const line of parts) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const waiter = bridge.waiters.shift();
    if (waiter) waiter.resolve(trimmed);
    else bridge.lines.push(trimmed);
  }
}

function nextBridgeLine(bridge: BridgeProcess, timeoutMs: number): Promise<string> {
  const queued = bridge.lines.shift();
  if (queued !== undefined) return Promise.resolve(queued);
  return new Promise<string>((resolve, reject) => {
    const timer = setTimeout(() => {
      const index = bridge.waiters.findIndex((waiter) => waiter.resolve === resolveLine);
      if (index >= 0) bridge.waiters.splice(index, 1);
      reject(new Error("bridge response timeout"));
    }, timeoutMs);
    const resolveLine = (line: string): void => {
      clearTimeout(timer);
      resolve(line);
    };
    bridge.waiters.push({
      resolve: resolveLine,
      reject: (error: Error) => {
        clearTimeout(timer);
        reject(error);
      },
    });
  });
}

const bridges = new Map<string, BridgeProcess>();

async function spawnBridge(
  key: string,
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
): Promise<BridgeProcess> {
  const existing = bridges.get(key);
  if (existing && !existing.exited) return existing;
  const [command, ...args] = argv;
  if (!command) throw new MlRouteBackendError("empty bridge command");
  let child: ChildProcess;
  try {
    child = spawn(command, args, { env, stdio: ["pipe", "pipe", "pipe"] });
  } catch (error) {
    throw new MlRouteBackendError(
      `bridge spawn failed: ${command}`,
      error instanceof Error ? error.message : String(error),
    );
  }
  const bridge: BridgeProcess = {
    child,
    lines: [],
    tail: "",
    waiters: [],
    ready: Promise.resolve(),
    stderrTail: "",
    queue: Promise.resolve(),
    exited: false,
  };
  bridges.set(key, bridge);
  child.stdout?.on("data", (chunk: Buffer) => pumpStdout(bridge, chunk));
  child.stderr?.on("data", (chunk: Buffer) => {
    bridge.stderrTail = `${bridge.stderrTail}${chunk.toString("utf8")}`.slice(-MAX_STDERR_CHARS);
  });
  child.on("error", (error: Error) => {
    bridge.exited = true;
    bridges.delete(key);
    failWaiters(bridge, error);
  });
  child.on("exit", (code: number | null) => {
    bridge.exited = true;
    bridges.delete(key);
    failWaiters(bridge, new Error(`bridge exited (code ${code ?? "?"})`));
  });
  bridge.ready = (async () => {
    const line = await nextBridgeLine(bridge, BRIDGE_READY_TIMEOUT_MS).catch((error: Error) => {
      killBridge(key, bridge);
      throw new MlRouteBackendError("bridge failed to become ready", error.message);
    });
    let hello: { ready?: boolean; fatal?: string } = {};
    try {
      hello = JSON.parse(line) as { ready?: boolean; fatal?: string };
    } catch {
      killBridge(key, bridge);
      throw new MlRouteBackendError("bridge sent invalid handshake", line.slice(0, 200));
    }
    if (hello.fatal) {
      killBridge(key, bridge);
      throw new MlRouteBackendError("bridge fatal", hello.fatal);
    }
    if (hello.ready !== true) {
      killBridge(key, bridge);
      throw new MlRouteBackendError("bridge sent invalid handshake", line.slice(0, 200));
    }
  })();
  await bridge.ready;
  return bridge;
}

function killBridge(key: string, bridge: BridgeProcess): void {
  bridges.delete(key);
  bridge.exited = true;
  try {
    bridge.child.kill();
  } catch {
    // Already dead; the exit handler cleans up the rest.
  }
}

/** Stop all cached bridge processes (shutdown/tests). */
export function disposeMlRouteSidecars(): void {
  // Deleting during Map iteration is safe: removed entries are not visited.
  for (const [key, bridge] of bridges) killBridge(key, bridge);
}

async function queryStdioBackend(
  request: SystemOneRouteRequest,
  argv: readonly string[],
  env: NodeJS.ProcessEnv,
  timeoutMs: number,
): Promise<unknown> {
  const key = JSON.stringify({ argv, env });
  const bridge = await spawnBridge(key, argv, env);
  // Serialize requests: one in-flight line per process.
  const run = bridge.queue.then(async () => {
    if (bridge.exited) throw new MlRouteBackendError("bridge exited before request");
    if (!bridge.child.stdin?.writable) throw new MlRouteBackendError("bridge stdin is closed");
    bridge.child.stdin.write(`${JSON.stringify(request)}\n`);
    const line = await nextBridgeLine(bridge, timeoutMs).catch((error: Error) => {
      killBridge(key, bridge);
      throw new MlRouteBackendError("bridge response timeout", error.message);
    });
    let payload: { error?: string } | unknown = null;
    try {
      payload = JSON.parse(line) as unknown;
    } catch {
      throw new MlRouteBackendError("bridge returned invalid JSON", line.slice(0, 200));
    }
    if (payload !== null && typeof payload === "object" && "error" in payload) {
      const detail = (payload as { error?: unknown }).error;
      throw new MlRouteBackendError(
        "bridge request failed",
        typeof detail === "string" ? detail : bridge.stderrTail || undefined,
      );
    }
    return payload;
  });
  // Keep the chain alive across rejections; the caller still sees this one.
  bridge.queue = run.catch(() => undefined);
  return run;
}

/**
 * Send one route-choice request to the configured backend. Resolves with
 * the raw (untrusted) payload — the caller validates it jev-style.
 */
export async function queryMlRouteBackend(
  request: SystemOneRouteRequest,
  backend: MlRouteBackendConfig,
  spawnEnv: NodeJS.ProcessEnv = process.env,
): Promise<unknown> {
  const timeoutMs = timeoutMsOf(backend);
  if (backend.command && backend.command.length > 0) {
    return queryStdioBackend(request, backend.command, { ...spawnEnv, ...backend.env }, timeoutMs);
  }
  if (backend.backend === "laya") {
    const file = await layaBridgeFile();
    const argv = [backend.pythonPath || "python3", file];
    return queryStdioBackend(request, argv, { ...spawnEnv, ...backend.env }, timeoutMs);
  }
  const endpoint = backend.endpoint ?? ML_ROUTE_DEFAULT_ENDPOINTS[backend.backend];
  if (!endpoint) {
    throw new MlRouteBackendError(`backend "${backend.backend}" needs an endpoint or command`);
  }
  return queryHttpBackend(request, endpoint, timeoutMs);
}
