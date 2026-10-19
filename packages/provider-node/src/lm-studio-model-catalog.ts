import {
  DEFAULT_LM_STUDIO_API_KEY,
  lmStudioV0ModelsUrl,
  lmStudioV1ModelsUrl,
  mergeLmStudioModelCatalogEntries,
  resolveLmStudioBaseUrl,
  type LmStudioModelListEntry,
} from "@zcode/provider";

export interface FetchLmStudioModelCatalogOptions {
  readonly baseUrl?: string;
  readonly apiKey?: string | null;
  readonly request?: typeof fetch;
  readonly timeoutMs?: number;
}

const DEFAULT_TIMEOUT_MS = 1500;
// 模型列表是小 JSON；超过此上限视为异常服务端，直接丢弃，避免无界 response.json() 撑爆内存。
const MAX_CATALOG_BYTES = 4 * 1024 * 1024;

function authorizationHeader(apiKey?: string | null): Record<string, string> {
  const token =
    apiKey?.trim() || process.env.LM_STUDIO_API_KEY?.trim() || DEFAULT_LM_STUDIO_API_KEY;
  return { Authorization: `Bearer ${token}` };
}

async function readJson(
  url: string,
  request: typeof fetch,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<unknown | undefined> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await request(url, { headers, signal: controller.signal });
    if (!response.ok) return undefined;
    const declared = Number(response.headers?.get?.("content-length") ?? Number.NaN);
    if (Number.isFinite(declared) && declared > MAX_CATALOG_BYTES) return undefined;
    const text = await response.text();
    if (text.length > MAX_CATALOG_BYTES) return undefined;
    return JSON.parse(text) as unknown;
  } catch {
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Load the LM Studio picker catalog from `/api/v0/models` when present, then `/v1/models`.
 * Does not start or load models; missing endpoints fall through to the next.
 * 两个端点并行请求：本地服务任一慢响应不再拖慢整体（约省一半等待时间）。
 */
export async function fetchLmStudioModelCatalogEntries(
  options: FetchLmStudioModelCatalogOptions = {},
): Promise<readonly LmStudioModelListEntry[]> {
  const baseUrl = resolveLmStudioBaseUrl(options.baseUrl);
  const request = options.request ?? globalThis.fetch;
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const headers = authorizationHeader(options.apiKey);
  const [v0, v1] = await Promise.all([
    readJson(lmStudioV0ModelsUrl(baseUrl), request, headers, timeoutMs),
    readJson(lmStudioV1ModelsUrl(baseUrl), request, headers, timeoutMs),
  ]);
  return mergeLmStudioModelCatalogEntries({
    ...(v0 === undefined ? {} : { v0 }),
    ...(v1 === undefined ? {} : { v1 }),
  });
}

export async function fetchLmStudioModelCatalog(
  options: FetchLmStudioModelCatalogOptions = {},
): Promise<readonly string[]> {
  const entries = await fetchLmStudioModelCatalogEntries(options);
  return Object.freeze(entries.map((entry) => entry.id));
}
