import type { ModelSource, ModelTextResult, TraceContext } from "@zcode/contracts";
import type { ToolExecutionContext } from "../types.js";

const WEBSEARCH_TOOL_NAME = "WebSearch";

/** Local SearXNG base URL (same default as grok-local's search tool). */
export const SEARXNG_DEFAULT_BASE_URL = "http://127.0.0.1:8888";
export const SEARXNG_BASE_URL_ENV = "ZCODE_SEARXNG_URL";
const SEARXNG_TIMEOUT_MS = 10_000;
const SEARXNG_MAX_RESULTS = 10;

export function resolveSearXngBaseUrl(env: NodeJS.ProcessEnv = process.env): string {
  const raw = env[SEARXNG_BASE_URL_ENV]?.trim();
  return raw && raw.length > 0 ? raw.replace(/\/+$/, "") : SEARXNG_DEFAULT_BASE_URL;
}

interface SearXngJsonResult {
  url?: unknown;
  title?: unknown;
  content?: unknown;
}

/**
 * Local-first search fallback: query SearXNG's JSON API and shape the
 * hits as a ModelTextResult (sources + a link summary) so the normal
 * output builder works unchanged. Fail-open: any failure (SearXNG down,
 * timeout, malformed body, zero usable hits) returns undefined and the
 * caller keeps its original behavior. Never throws.
 */
export async function fetchSearXngResult(
  query: string,
  options?: { baseUrl?: string; timeoutMs?: number; env?: NodeJS.ProcessEnv },
): Promise<ModelTextResult | undefined> {
  try {
    if (!query || !query.trim()) return undefined;
    const env = options?.env ?? process.env;
    const baseUrl = options?.baseUrl ?? resolveSearXngBaseUrl(env);
    const timeoutMs = options?.timeoutMs ?? SEARXNG_TIMEOUT_MS;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(
        `${baseUrl}/search?q=${encodeURIComponent(query.trim())}&format=json`,
        { signal: controller.signal },
      );
      if (!response.ok) return undefined;
      const payload = (await response.json()) as { results?: unknown };
      if (!payload || !Array.isArray(payload.results)) return undefined;
      const sources: ModelSource[] = [];
      const lines: string[] = [];
      for (const item of payload.results.slice(0, SEARXNG_MAX_RESULTS)) {
        const hit = (item ?? {}) as SearXngJsonResult;
        if (typeof hit.url !== "string" || hit.url.trim().length === 0) continue;
        const url = hit.url.trim();
        const title = typeof hit.title === "string" && hit.title.trim().length > 0
          ? hit.title.trim()
          : url;
        sources.push({ type: "source", sourceType: "url", url, title });
        const snippet = typeof hit.content === "string" ? hit.content.trim().slice(0, 300) : "";
        lines.push(`- [${title}](${url})${snippet ? ` — ${snippet}` : ""}`);
      }
      if (sources.length === 0) return undefined;
      return {
        text: lines.join("\n"),
        finishReason: "stop",
        usage: {},
        sources,
      };
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return undefined;
  }
}

export function webSearchTraceFromContext(context: ToolExecutionContext): TraceContext {
  return {
    traceId: context.traceId,
    spanId: context.spanId,
    parentSpanId: context.parentSpanId,
    sessionId: context.sessionId,
    turnId: context.turnId,
    attributes: {
      toolCallId: context.toolCallId,
      toolName: WEBSEARCH_TOOL_NAME,
    },
  };
}
