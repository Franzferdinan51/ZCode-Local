/**
 * Unit tests for the SystemOne shim auto-start bootstrap's pure helpers.
 * Process-spawning paths (findSuitablePython, ensureSystemOneShim) are
 * exercised end-to-end in the zero-setup verification instead.
 */
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
  findSystemOneDir,
  requiredShimImports,
  resolveShimSpawnCommand,
  resolveSystemOneHealthzUrl,
  resolveSystemOneShimBase,
  resolveSystemOneSpawnPort,
  shouldEnsureSystemOneShim,
} from "./systemone-shim-bootstrap.js";

test("shouldEnsureSystemOneShim: kill-switch disables the bootstrap", () => {
  assert.equal(
    shouldEnsureSystemOneShim([], { env: { ZCODE_SYSTEMONE: "0" } }),
    false,
  );
});

test("shouldEnsureSystemOneShim: trivial and nested invocations are skipped", () => {
  for (const argv of [
    ["--version"],
    ["-v"],
    ["--help"],
    ["-h"],
    ["--licenses"],
    ["--prepare-storage"],
  ]) {
    assert.equal(shouldEnsureSystemOneShim(argv, { env: {} }), false, argv.join(" "));
  }
  assert.equal(
    shouldEnsureSystemOneShim(["__zcode-plugin-host", "x"], {
      env: {},
      isPluginHost: true,
    }),
    false,
  );
});

test("shouldEnsureSystemOneShim: normal invocations bootstrap", () => {
  assert.equal(shouldEnsureSystemOneShim([], { env: {} }), true);
  assert.equal(shouldEnsureSystemOneShim(["-p", "hi"], { env: {} }), true);
  assert.equal(shouldEnsureSystemOneShim(["app-server", "--stdio"], { env: {} }), true);
});

test("findSystemOneDir: explicit ZCODE_SYSTEMONE_DIR wins", () => {
  const dir = mkdtempSync(join(tmpdir(), "s1dir-"));
  writeFileSync(join(dir, "shim.py"), "# test");
  assert.equal(findSystemOneDir({ env: { ZCODE_SYSTEMONE_DIR: dir } }), dir);
});

test("findSystemOneDir: release layout <root>/systemone next to the bundle", () => {
  const root = mkdtempSync(join(tmpdir(), "s1rel-"));
  const agentDir = join(root, "agent");
  mkdirSync(agentDir, { recursive: true });
  mkdirSync(join(root, "systemone"), { recursive: true });
  writeFileSync(join(root, "systemone", "shim.py"), "# test");
  assert.equal(
    findSystemOneDir({ env: {}, fromDir: agentDir }),
    join(root, "systemone"),
  );
});

test("findSystemOneDir: returns undefined when nothing is bundled", () => {
  const dir = mkdtempSync(join(tmpdir(), "s1none-"));
  assert.equal(findSystemOneDir({ env: {}, fromDir: dir }), undefined);
});

test("resolveSystemOneShimBase honors $SYSTEMONE_SHIM_URL", () => {
  assert.equal(resolveSystemOneShimBase({}), "http://127.0.0.1:8765");
  assert.equal(
    resolveSystemOneShimBase({ SYSTEMONE_SHIM_URL: "http://macmini:8765/" }),
    "http://macmini:8765",
  );
  assert.equal(
    resolveSystemOneHealthzUrl({ SYSTEMONE_SHIM_URL: "http://macmini:8765" }),
    "http://macmini:8765/healthz",
  );
});

test("resolveSystemOneSpawnPort takes the shim URL port, else 8765", () => {
  assert.equal(resolveSystemOneSpawnPort({}), 8765);
  assert.equal(
    resolveSystemOneSpawnPort({ SYSTEMONE_SHIM_URL: "http://127.0.0.1:9999" }),
    9999,
  );
  assert.equal(
    resolveSystemOneSpawnPort({ SYSTEMONE_SHIM_URL: "not a url" }),
    8765,
  );
});

test("requiredShimImports: slim floor + local weights only when needed", () => {
  assert.deepEqual(requiredShimImports({}), ["numpy", "gliclass"]);
  assert.deepEqual(requiredShimImports({ SYSTEMONE_ENGINE: "local" }), [
    "numpy",
    "gliclass",
  ]);
  assert.deepEqual(requiredShimImports({ SYSTEMONE_ENGINE: "sglang" }), ["numpy"]);
  assert.deepEqual(requiredShimImports({ SYSTEMONE_ENGINE: "kev" }), ["numpy"]);
  assert.deepEqual(
    requiredShimImports({
      SYSTEMONE_ENGINE: "auto",
      SGLANG_BASE_URL: "http://x:30000",
    }),
    ["numpy"],
  );
  assert.deepEqual(
    requiredShimImports({
      SYSTEMONE_ENGINE: "auto",
      KEV_BASE_URL: "http://x:8008",
    }),
    ["numpy"],
  );
  assert.deepEqual(requiredShimImports({ SYSTEMONE_ENGINE: "onnx" }), [
    "numpy",
    "onnxruntime",
  ]);
});

test("resolveShimSpawnCommand prefers `systemone serve`, falls back to -m", async () => {
  const viaServe = await resolveShimSpawnCommand("/usr/bin/python3", 8765, {
    resolveOnPathImpl: async () => "/usr/local/bin/systemone",
  });
  assert.equal(viaServe.viaServe, true);
  assert.deepEqual(viaServe.args, ["serve", "--port", "8765"]);

  const viaModule = await resolveShimSpawnCommand("/usr/bin/python3", 9999, {
    resolveOnPathImpl: async () => undefined,
  });
  assert.equal(viaModule.viaServe, false);
  assert.equal(viaModule.command, "/usr/bin/python3");
  assert.deepEqual(viaModule.args.slice(0, 4), [
    "-m",
    "systemone.shim",
    "--port",
    "9999",
  ]);
});
