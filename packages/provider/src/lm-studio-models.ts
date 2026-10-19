import { DEFAULT_LM_STUDIO_BASE_URL, LM_STUDIO_PROVIDER_ID } from "@zcode/shared";
import type { ProviderConfigLayerSnapshot } from "./config-service.js";
import { ModelConfig, ModelConfigRules } from "./config/model-config.js";

export {
  DEFAULT_LM_STUDIO_API_KEY,
  DEFAULT_LM_STUDIO_BASE_URL,
  LM_STUDIO_FALLBACK_MODEL_ID,
  LM_STUDIO_PROVIDER_ID,
  LM_STUDIO_TEMPLATE_ID,
} from "@zcode/shared";

export interface LmStudioModelListEntry {
  readonly id: string;
  readonly loaded: boolean;
  readonly embedding: boolean;
  readonly chat: boolean;
  /** True when LM Studio classifies the model as a VLM (vision-language). */
  readonly vision: boolean;
}

const EMBEDDING_TYPE = /embed/i;
const CHAT_TYPE = /^(llm|vlm|llm-model|chat)$/i;
const VLM_TYPE = /^vlm$/i;

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readModelArray(payload: unknown): unknown[] {
  if (Array.isArray(payload)) return payload;
  const record = asRecord(payload);
  if (!record) return [];
  if (Array.isArray(record.data)) return record.data;
  if (Array.isArray(record.models)) return record.models;
  return [];
}

function isEmbeddingEntry(id: string, type: string): boolean {
  return EMBEDDING_TYPE.test(type) || EMBEDDING_TYPE.test(id);
}

function isChatEntry(type: string, embedding: boolean): boolean {
  if (embedding) return false;
  if (!type) return true;
  return CHAT_TYPE.test(type);
}

function isLoadedEntry(record: Record<string, unknown>): boolean {
  if (record.loaded === true) return true;
  if (record.state === "loaded") return true;
  if (typeof record.status === "string" && record.status.toLowerCase() === "loaded") return true;
  return false;
}

/** Parse one LM Studio `/api/v0/models` or OpenAI-compatible `/v1/models` JSON body. */
export function parseLmStudioModelsJson(payload: unknown): LmStudioModelListEntry[] {
  const seen = new Set<string>();
  const entries: LmStudioModelListEntry[] = [];
  for (const item of readModelArray(payload)) {
    const record = asRecord(item);
    const id = typeof record?.id === "string" ? record.id.trim() : "";
    if (!id || seen.has(id)) continue;
    seen.add(id);
    const type =
      typeof record?.type === "string"
        ? record.type
        : typeof record?.model_type === "string"
          ? record.model_type
          : "";
    const embedding = isEmbeddingEntry(id, type);
    entries.push(
      Object.freeze({
        id,
        loaded: isLoadedEntry(record ?? {}),
        embedding,
        chat: isChatEntry(type, embedding),
        vision: VLM_TYPE.test(type),
      }),
    );
  }
  return entries;
}

/**
 * Merge LM Studio catalogs: prefer `/api/v0/models` metadata, then `/v1/models`.
 * Loaded chat models first; embeddings are excluded. The v0 `type` field is
 * the only place vision capability is reported, so it is carried through the
 * merge (v1 entries have no type and must not erase a v0 `vlm` marking).
 */
export function mergeLmStudioModelCatalogEntries(input: {
  readonly v0?: unknown;
  readonly v1?: unknown;
}): readonly LmStudioModelListEntry[] {
  const byId = new Map<string, LmStudioModelListEntry>();
  for (const entry of parseLmStudioModelsJson(input.v0)) byId.set(entry.id, entry);
  for (const entry of parseLmStudioModelsJson(input.v1)) {
    const current = byId.get(entry.id);
    if (!current) {
      byId.set(entry.id, entry);
      continue;
    }
    byId.set(entry.id, {
      id: entry.id,
      loaded: current.loaded || entry.loaded,
      embedding: current.embedding || entry.embedding,
      chat: current.chat || entry.chat,
      vision: current.vision || entry.vision,
    });
  }
  return Object.freeze(
    [...byId.values()]
      .filter((entry) => !entry.embedding && entry.chat)
      .sort(
        (left, right) =>
          Number(right.loaded) - Number(left.loaded) || left.id.localeCompare(right.id),
      ),
  );
}

export function mergeLmStudioModelCatalog(input: {
  readonly v0?: unknown;
  readonly v1?: unknown;
}): readonly string[] {
  return Object.freeze(mergeLmStudioModelCatalogEntries(input).map((entry) => entry.id));
}

export function applyLmStudioCatalogToProviderLayer(
  snapshot: ProviderConfigLayerSnapshot,
  modelIds: readonly string[],
  visionModelIds: readonly string[] = [],
): ProviderConfigLayerSnapshot {
  const ids = modelIds.filter((modelId) => modelId.trim().length > 0);
  if (ids.length === 0) return snapshot;
  const current = snapshot.providers.get(LM_STUDIO_PROVIDER_ID);
  if (!current) return snapshot;
  // LM Studio already told us which of these models are VLMs; without this
  // overlay they resolve with the text-only default input format and the
  // runtime silently replaces every image (screenshots included) with an
  // "unsupported media" stub. A provider-model rule at this (builtin) layer
  // declares just the one leaf; personal rules still overlay on top.
  const visionIds = [...new Set(visionModelIds.filter((modelId) => ids.includes(modelId)))];
  const models =
    visionIds.length === 0
      ? snapshot.models
      : new ModelConfigRules([
          ...(snapshot.models?.rules() ?? []),
          ...visionIds.map((modelId) => ({
            type: "provider-model" as const,
            providerId: LM_STUDIO_PROVIDER_ID,
            modelId,
            config: ModelConfig.fromData({
              properties: { inputFormat: { supportsImage: true } },
            }),
          })),
        ]);
  return Object.freeze({
    ...snapshot,
    revision:
      `${snapshot.revision}:lm-studio-catalog:${ids.join("\0")}` +
      (visionIds.length === 0 ? "" : `:lm-vision:${visionIds.join("\0")}`),
    providers: snapshot.providers.set(LM_STUDIO_PROVIDER_ID, current.withBuiltinModelIds(ids)),
    ...(models === snapshot.models ? {} : { models }),
  });
}

/**
 * Move one configured model id to the front of a merged catalog. Exact match
 * only, and the id must already be listed: a typo in LM_STUDIO_MODEL never
 * invents a model the server does not serve, it just keeps catalog order.
 */
export function preferLmStudioModelId(
  modelIds: readonly string[],
  preferred?: string | null,
): readonly string[] {
  const want = preferred?.trim();
  if (!want || !modelIds.includes(want)) return modelIds;
  return Object.freeze([want, ...modelIds.filter((modelId) => modelId !== want)]);
}

export function resolveLmStudioBaseUrl(baseUrl?: string | null): string {
  const trimmed = baseUrl?.trim();
  return trimmed && trimmed.length > 0 ? trimmed.replace(/\/+$/, "") : DEFAULT_LM_STUDIO_BASE_URL;
}

export function lmStudioV0ModelsUrl(baseUrl?: string | null): string {
  return new URL("/api/v0/models", resolveLmStudioBaseUrl(baseUrl)).toString();
}

export function lmStudioV1ModelsUrl(baseUrl?: string | null): string {
  return `${resolveLmStudioBaseUrl(baseUrl)}/models`;
}
