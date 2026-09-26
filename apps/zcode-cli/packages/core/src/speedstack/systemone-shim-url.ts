// ============================================================
// Speed Stack: SystemOne shim URL resolution
// ============================================================
//
// Single source of truth for the SystemOne shim's base address. Route,
// decide, rank-plans, and the onboard wizard all resolve through here so
// $SYSTEMONE_SHIM_URL overrides the localhost default everywhere at once.
//
// Intentionally pure (no runtime imports): runnable under plain
// `node --test` type-stripping like the other speedstack modules.
// No model IDs appear anywhere here.

/** Env var overriding the SystemOne shim base URL. */
export const SYSTEMONE_SHIM_URL_ENV = "SYSTEMONE_SHIM_URL";

/** Default shim address when $SYSTEMONE_SHIM_URL is unset. */
export const DEFAULT_SYSTEMONE_SHIM_URL = "http://127.0.0.1:8765";

/**
 * Resolve the shim base URL: $SYSTEMONE_SHIM_URL (trimmed, trailing
 * slashes stripped), else the localhost default. Never throws; an
 * empty or whitespace-only override falls back to the default.
 */
export function resolveSystemOneShimUrl(
  env: NodeJS.ProcessEnv = process.env,
): string {
  const raw = (env[SYSTEMONE_SHIM_URL_ENV] ?? "").trim().replace(/\/+$/, "");
  return raw === "" ? DEFAULT_SYSTEMONE_SHIM_URL : raw;
}

/**
 * Join a shim path (e.g. "/v1/systemone/route") onto the resolved base
 * URL. The path must start with "/".
 */
export function systemOneShimEndpoint(
  path: string,
  env: NodeJS.ProcessEnv = process.env,
): string {
  return `${resolveSystemOneShimUrl(env)}${path}`;
}

/**
 * Lightweight sanity check for wizard-entered URLs: an http(s) scheme
 * plus a host. Not a full URL validator — the probe is the real check.
 */
export function isPlausibleShimUrl(value: string): boolean {
  return /^https?:\/\/[^/\s]+(:\d+)?(\/.*)?$/.test(value.trim());
}

/**
 * True when the shim URL points at this machine. The wizard only offers
 * to start the bundled local shim for loopback URLs — starting a local
 * shim for a remote URL would leave the runtime pointed at the wrong
 * place.
 */
export function isLoopbackShimUrl(value: string): boolean {
  const match = /^https?:\/\/([^/:\s\[]+|\[[^\]]+\])/.exec(value.trim());
  if (!match) return false;
  const host = match[1].toLowerCase().replace(/^\[|\]$/g, "");
  return host === "localhost" || host === "127.0.0.1" || host === "::1";
}
