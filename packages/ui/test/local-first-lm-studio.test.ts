import assert from "node:assert/strict";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  createFailClosedAccountProviderConfigSnapshot,
  DEFAULT_LM_STUDIO_BASE_URL,
  LM_STUDIO_PROVIDER_ID,
  mergeLmStudioModelCatalog,
  mergeLmStudioModelCatalogEntries,
  ModelConfigRules,
  parseLmStudioModelsJson,
  preferLmStudioModelId,
  ProviderConfigMap,
  ProviderConfigResolver,
  resolveInitialModelSelection,
  type LmStudioModelListEntry,
} from "../../provider/src/index.js";
import { decodeZCodeBuiltinRelease } from "../../provider-node/src/zcode-builtin-release.js";
import { fetchLmStudioModelCatalog } from "../../provider-node/src/lm-studio-model-catalog.js";
import { LmStudioCatalogOverlaySource } from "../../provider-node/src/lm-studio-catalog-source.js";

const builtinPath = fileURLToPath(
  new URL("../../../config/provider/zcode-builtin.json", import.meta.url),
);

const v0Fixture = {
  data: [
    { id: "embed-nomic", type: "embeddings", state: "loaded" },
    { id: "qwen-unloaded", type: "llm", state: "not-loaded" },
    { id: "qwen-loaded", type: "llm", state: "loaded" },
    { id: "vision-loaded", type: "vlm", state: "loaded" },
  ],
};

const v1Fixture = {
  object: "list",
  data: [
    { id: "qwen-loaded", object: "model" },
    { id: "qwen-unloaded", object: "model" },
    { id: "vision-loaded", object: "model" },
    { id: "text-embedding-foo", object: "model" },
    { id: "only-in-v1", object: "model" },
  ],
};

const expectedPickerIds = ["qwen-loaded", "vision-loaded", "only-in-v1", "qwen-unloaded"];

function loadBuiltinRelease() {
  return decodeZCodeBuiltinRelease(JSON.parse(readFileSync(builtinPath, "utf8")));
}

function resolveBuiltinRegistry(accountFailClosed: boolean) {
  const release = loadBuiltinRelease();
  const config = Object.freeze({
    revision: "test-builtin",
    zcodeBuiltinRevision: String(release.revision),
    personalRevision: "none",
    zcodeBuiltinProviders: release.config.providers,
    zcodeBuiltinProviderTemplates: release.config.providerTemplates,
    personalProviders: ProviderConfigMap.empty(),
    zcodeBuiltinModelRules: release.config.modelConfigRules,
    personalModels: ModelConfigRules.empty(),
  });
  const account = accountFailClosed
    ? createFailClosedAccountProviderConfigSnapshot(config)
    : {
        providers: ProviderConfigMap.empty(),
        basedOnZCodeBuiltinRevision: config.zcodeBuiltinRevision,
        revision: "empty",
      };
  return {
    config,
    account,
    resolution: new ProviderConfigResolver().resolve({
      zcodeBuiltinProviders: config.zcodeBuiltinProviders,
      zcodeBuiltinProviderTemplates: config.zcodeBuiltinProviderTemplates,
      personalProviders: config.personalProviders,
      zcodeBuiltinModelRules: config.zcodeBuiltinModelRules,
      personalModels: config.personalModels,
      accountProviders: account.providers,
    }),
  };
}

test("builtin default provider is LM Studio at the local OpenAI-compatible base URL", () => {
  const release = loadBuiltinRelease();
  const lm = release.config.providers.get(LM_STUDIO_PROVIDER_ID);
  assert.ok(lm, "local:lm-studio must be a builtin provider");
  assert.equal(lm.api?.baseUrl, DEFAULT_LM_STUDIO_BASE_URL);
  assert.equal(DEFAULT_LM_STUDIO_BASE_URL, "http://127.0.0.1:1234/v1");
  assert.equal(lm.access?.type, "api-key");
  assert.notEqual(lm.access?.type, "zhipu-account");
  assert.equal(release.config.providers.keys()[0], LM_STUDIO_PROVIDER_ID);
  assert.equal(LM_STUDIO_PROVIDER_ID.startsWith("account:zai-"), false);
  assert.equal(LM_STUDIO_PROVIDER_ID.startsWith("account:bigmodel-"), false);
});

test("fail-closed zhipu overlay does not hide or disable the LM Studio default", () => {
  const { account, resolution } = resolveBuiltinRegistry(true);
  const overlay = account.providers.get(LM_STUDIO_PROVIDER_ID);
  assert.equal(overlay, undefined);
  const zai = account.providers.get("account:zai-individual-coding-plan");
  assert.equal(zai?.access?.type, "zhipu-account");
  assert.equal(zai?.access && "entitled" in zai.access ? zai.access.entitled : undefined, false);

  const lm = resolution.registryProviders.find(
    (provider) => provider.providerId === LM_STUDIO_PROVIDER_ID,
  );
  assert.ok(lm);
  assert.equal(lm.config.api.baseUrl, DEFAULT_LM_STUDIO_BASE_URL);
  assert.equal(lm.config.access.type, "api-key");
  assert.ok(lm.models.length > 0);
  assert.equal(
    resolution.registryProviders.some((provider) => provider.providerId.startsWith("account:zai-")),
    false,
  );
  assert.equal(
    resolution.registryProviders.some((provider) =>
      provider.providerId.startsWith("account:bigmodel-"),
    ),
    false,
  );

  const initial = resolveInitialModelSelection({
    registry: { revision: 1, providers: resolution.registryProviders },
  });
  assert.notEqual(initial.source, "none");
  if (initial.source === "none") throw new Error("expected a default selection");
  assert.equal(initial.selection.providerId, LM_STUDIO_PROVIDER_ID);
  assert.equal(initial.selection.providerId.startsWith("account:zai-"), false);
});

test("LM Studio model list parser puts loaded chat models first and excludes embeddings", () => {
  const parsedV0 = parseLmStudioModelsJson(v0Fixture);
  assert.equal(
    parsedV0.some((entry) => entry.id === "embed-nomic" && entry.embedding),
    true,
  );
  const ids = mergeLmStudioModelCatalog({ v0: v0Fixture, v1: v1Fixture });
  assert.deepEqual([...ids], expectedPickerIds);
  assert.equal(ids.includes("embed-nomic"), false);
  assert.equal(ids.includes("text-embedding-foo"), false);
});

test("fixture LM Studio HTTP catalog is fetched and parsed by the shipped catalog function", async () => {
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    if (req.url === "/api/v0/models") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(v0Fixture));
      return;
    }
    if (req.url === "/v1/models") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify(v1Fixture));
      return;
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  assert.ok(address && typeof address === "object");
  const baseUrl = `http://127.0.0.1:${address.port}/v1`;
  try {
    const ids = await fetchLmStudioModelCatalog({ baseUrl, timeoutMs: 2000 });
    assert.deepEqual([...ids], expectedPickerIds);
  } finally {
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  }
});

test("LM_STUDIO_MODEL preference moves only a listed id to the front", () => {
  const ids = ["a", "b", "c"];
  assert.deepEqual([...preferLmStudioModelId(ids, "b")], ["b", "a", "c"]);
  // Unknown, empty, or missing preference keeps catalog order: never invent an id.
  assert.deepEqual([...preferLmStudioModelId(ids, "zzz")], ["a", "b", "c"]);
  assert.deepEqual([...preferLmStudioModelId(ids, undefined)], ["a", "b", "c"]);
  assert.deepEqual([...preferLmStudioModelId(ids, "  ")], ["a", "b", "c"]);
  assert.deepEqual([...preferLmStudioModelId(ids, " b ")], ["b", "a", "c"]);
});

test("catalog fetch starts v1 before v0 resolves (parallel, not sequential)", async () => {
  let v1Started = false;
  let v0ResolvedSawV1 = false;
  const request = (async (url: string) => {
    if (String(url).includes("/api/v0/models")) {
      await new Promise((resolve) => setImmediate(resolve));
      v0ResolvedSawV1 = v1Started;
      return jsonResponse({ data: [] });
    }
    v1Started = true;
    return jsonResponse({ data: [{ id: "m", object: "model" }] });
  }) as typeof fetch;
  const ids = await fetchLmStudioModelCatalog({
    baseUrl: "http://127.0.0.1:9/v1",
    request,
    timeoutMs: 2000,
  });
  assert.equal(v0ResolvedSawV1, true);
  assert.deepEqual([...ids], ["m"]);
});

test("oversized catalog bodies are discarded instead of parsed", async () => {
  const big = "x".repeat(4 * 1024 * 1024 + 1);
  const request = (async () => ({
    ok: true,
    headers: { get: () => null },
    text: async () => big,
  })) as unknown as typeof fetch;
  const ids = await fetchLmStudioModelCatalog({
    baseUrl: "http://127.0.0.1:9/v1",
    request,
    timeoutMs: 2000,
  });
  assert.deepEqual([...ids], []);
});

test("a declared huge content-length short-circuits before reading the body", async () => {
  let textCalls = 0;
  const request = (async () => ({
    ok: true,
    headers: { get: (name: string) => (name === "content-length" ? "999999999" : null) },
    text: async () => {
      textCalls += 1;
      return "{}";
    },
  })) as unknown as typeof fetch;
  const ids = await fetchLmStudioModelCatalog({
    baseUrl: "http://127.0.0.1:9/v1",
    request,
    timeoutMs: 2000,
  });
  assert.deepEqual([...ids], []);
  assert.equal(textCalls, 0);
});

function jsonResponse(body: unknown) {
  const text = JSON.stringify(body);
  return {
    ok: true,
    headers: { get: () => null },
    text: async () => text,
  };
}

function overlaySnapshot(providerFields: Record<string, unknown> = {}) {
  const providers = new Map();
  providers.set(LM_STUDIO_PROVIDER_ID, {
    withBuiltinModelIds: (ids: readonly string[]) => ({ modelIds: [...ids] }),
    ...providerFields,
  });
  return { revision: "r", providers };
}

function catalogEntry(id: string, vision = false): LmStudioModelListEntry {
  return { id, loaded: true, embedding: false, chat: true, vision };
}

function overlaySource(
  fetchCatalog: () => Promise<readonly (string | LmStudioModelListEntry)[]>,
  options: Record<string, unknown> = {},
) {
  // Single-id catalogs make preferLmStudioModelId a no-op, so these tests are
  // immune to whatever LM_STUDIO_MODEL the ambient environment carries.
  return new LmStudioCatalogOverlaySource({
    inner: {
      read: async () => overlaySnapshot(),
      onDidChange: () => () => {},
    } as never,
    fetchCatalog: (async () =>
      (await fetchCatalog()).map((entry) =>
        typeof entry === "string" ? catalogEntry(entry) : entry,
      )) as never,
    ...(options as Record<string, never>),
  });
}

function overlaidIds(snapshot: { providers: Map<string, { modelIds: string[] }> }): string[] {
  return snapshot.providers.get(LM_STUDIO_PROVIDER_ID)?.modelIds ?? [];
}

test("catalog overlay caches reads within the TTL and refetches after it", async () => {
  let calls = 0;
  let now = 1_000_000;
  const source = overlaySource(
    async () => {
      calls += 1;
      return ["m1"];
    },
    { cacheTtlMs: 30_000, now: () => now },
  );
  const first = (await source.read()) as unknown as Parameters<typeof overlaidIds>[0];
  const second = (await source.read()) as unknown as Parameters<typeof overlaidIds>[0];
  assert.equal(calls, 1);
  assert.deepEqual(overlaidIds(first), ["m1"]);
  assert.deepEqual(overlaidIds(second), ["m1"]);
  now += 30_001;
  await source.read();
  assert.equal(calls, 2);
});

test("concurrent overlay reads share one in-flight catalog request", async () => {
  let calls = 0;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  const source = overlaySource(async () => {
    calls += 1;
    await gate;
    return ["m1"];
  });
  const pending = [source.read(), source.read()];
  release();
  await Promise.all(pending);
  assert.equal(calls, 1);
});

test("LM_STUDIO_MODEL moves the configured id to the front of the overlaid catalog", async () => {
  const savedModel = process.env.LM_STUDIO_MODEL;
  process.env.LM_STUDIO_MODEL = "want";
  try {
    const source = overlaySource(async () => ["a", "want", "b"], { cacheTtlMs: 0 });
    const snapshot = (await source.read()) as unknown as Parameters<typeof overlaidIds>[0];
    assert.deepEqual(overlaidIds(snapshot), ["want", "a", "b"]);
  } finally {
    if (savedModel === undefined) delete process.env.LM_STUDIO_MODEL;
    else process.env.LM_STUDIO_MODEL = savedModel;
  }
});

test("usable LM Studio registry resolves with no login concept", () => {
  const { resolution } = resolveBuiltinRegistry(true);
  const usable = resolution.registryProviders.filter(
    (provider) => provider.models.length > 0,
  );
  assert.ok(usable.length > 0);
  assert.equal(usable[0]?.providerId, LM_STUDIO_PROVIDER_ID);
});

test("catalog parse marks vlm entries as vision-capable and the merge keeps it", () => {
  const parsed = parseLmStudioModelsJson(v0Fixture);
  assert.equal(parsed.find((entry) => entry.id === "vision-loaded")?.vision, true);
  assert.equal(parsed.find((entry) => entry.id === "qwen-loaded")?.vision, false);
  const entries = mergeLmStudioModelCatalogEntries({ v0: v0Fixture, v1: v1Fixture });
  assert.deepEqual(
    entries.map((entry) => entry.id),
    expectedPickerIds,
  );
  // v1 carries no type info; the v0 vlm marking must survive the merge.
  assert.equal(entries.find((entry) => entry.id === "vision-loaded")?.vision, true);
  assert.equal(entries.find((entry) => entry.id === "qwen-loaded")?.vision, false);
});

test("catalog overlay makes vlm models resolve with image input support", async () => {
  const release = loadBuiltinRelease();
  const source = new LmStudioCatalogOverlaySource({
    inner: {
      read: async () => ({
        revision: "test-builtin",
        providers: release.config.providers,
        models: release.config.modelConfigRules,
      }),
      onDidChange: () => () => {},
    } as never,
    fetchCatalog: (async () =>
      mergeLmStudioModelCatalogEntries({ v0: v0Fixture, v1: v1Fixture })) as never,
    cacheTtlMs: 0,
  });
  const overlaid = await source.read();
  const resolution = new ProviderConfigResolver().resolve({
    zcodeBuiltinProviders: overlaid.providers,
    zcodeBuiltinProviderTemplates: release.config.providerTemplates,
    personalProviders: ProviderConfigMap.empty(),
    zcodeBuiltinModelRules: overlaid.models,
    personalModels: ModelConfigRules.empty(),
    accountProviders: ProviderConfigMap.empty(),
  });
  const lm = resolution.registryProviders.find(
    (provider) => provider.providerId === LM_STUDIO_PROVIDER_ID,
  );
  assert.ok(lm);
  const vision = lm.models.find((model) => model.modelId === "vision-loaded");
  assert.ok(vision, "vlm fixture model must be in the registry");
  assert.equal(vision.config.properties.inputFormat?.supportsImage, true);
  const textOnly = lm.models.find((model) => model.modelId === "qwen-loaded");
  assert.ok(textOnly);
  assert.equal(textOnly.config.properties.inputFormat?.supportsImage, false);
});
