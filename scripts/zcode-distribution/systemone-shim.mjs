#!/usr/bin/env node
// Bundles the SystemOne Python shim into the ZCode release so routing
// works out of the box with zero manual setup.
//
// Copies the `systemone` Python package (shim.py + its local imports +
// the model registry) from the SystemOne release checkout into
// `<release>/systemone`, plus ZCode-specific docs. The CLI auto-starts it
// (`systemone serve --port 8765`, else `python3 -m systemone.shim`)
// when nothing answers on 127.0.0.1:8765.
//
// Source resolution: ZCODE_SYSTEMONE_SRC env, else the vendored in-repo
// copy (systemone-shim-files/systemone). A missing source is a warning,
// not a build failure — the release still works, routing just stays
// fail-open.

import { cp, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const bundledDocsDir = join(scriptDir, "systemone-shim-files");

// 3.24.0: the SystemOne shim is vendored in-repo
// (systemone-shim-files/systemone/) so releases are hermetic — no dependency
// on a Mac-local checkout. ZCODE_SYSTEMONE_SRC still overrides for dev.
const DEFAULT_SOURCE_DIR = join(bundledDocsDir, "systemone");

// The shim's runtime closure (systemone 0.1.0): shim.py imports .patterns,
// .api (guarded: absent on slim installs), the engine backends
// (jev/jevk5/kev/rerank/sglang), .scoring, .jeff1, .calibration, and
// .metrics; __init__ additionally imports .loop and .rotation;
// cli.py/client.py/jeff1_sidecar.py back `systemone serve`
// (+ --with-jeff1); cli.py lazily imports .jevbench (jevbench command),
// .battery (battery command), and .mcp_server (mcp helpers).
// mcp_server.py/acp_server.py are the stdio MCP/ACP servers. Data files
// ride alongside (model registry, kev tier pack, calibration, tool
// registry, tuning, openapi). Everything else in the checkout
// (tune/distill/bench_2048, examples, tests, logs) is dev tooling and
// stays out of the release.
const SHIM_FILES = [
  "__init__.py",
  "acp_server.py",
  "api.py",
  "calibration.py",
  "cli.py",
  "client.py",
  "jeff1.py",
  "jeff1_sidecar.py",
  "jev_backend.py",
  "jevbench.py",
  "jevk5_backend.py",
  "kev_backend.py",
  "loop.py",
  "mcp_server.py",
  "metrics.py",
  "patterns.py",
  "rerank_backend.py",
  "rotation.py",
  "scoring.py",
  "sglang_backend.py",
  "shim.py",
  "battery/run.py",
  "battery/fit.py",
  "battery/fit_types.py",
  "battery/tasks.jsonl",
  "calibration.json",
  "kev_registry.json",
  "model_registry.json",
  "openapi.json",
  "tool_registry.json",
  "tuning.json",
  "README.md",
];

const DOC_FILES = ["requirements.txt", "README-ZCODE.md"];

export function resolveSystemOneSourceDir(env = process.env) {
  return env.ZCODE_SYSTEMONE_SRC?.trim() || DEFAULT_SOURCE_DIR;
}

/**
 * Copy the SystemOne shim package into `<packageRoot>/systemone`.
 * Returns true when bundled, false when the source was missing (warned).
 */
export async function stageSystemOneShim(packageRoot, options = {}) {
  const env = options.env ?? process.env;
  const sourceDir = resolve(options.sourceDir ?? resolveSystemOneSourceDir(env));
  const sourceStat = await stat(sourceDir).catch(() => null);
  if (!sourceStat?.isDirectory()) {
    console.warn(
      `[zcode] SystemOne shim source not found at ${sourceDir} ` +
        `(set ZCODE_SYSTEMONE_SRC to override); release ships without a ` +
        `bundled shim — auto-start disabled, routing stays fail-open.`,
    );
    return false;
  }
  const destDir = join(packageRoot, "systemone");
  await mkdir(destDir, { recursive: true });
  for (const file of SHIM_FILES) {
    const from = join(sourceDir, file);
    const fromStat = await stat(from).catch(() => null);
    if (!fromStat?.isFile()) {
      throw new Error(`SystemOne shim source is missing ${file}: ${from}`);
    }
    const to = join(destDir, file);
    await mkdir(dirname(to), { recursive: true });
    await cp(from, to);
  }
  for (const file of DOC_FILES) {
    const from = join(bundledDocsDir, file);
    const fromStat = await stat(from).catch(() => null);
    if (!fromStat?.isFile()) {
      throw new Error(`Missing bundled shim doc: ${from}`);
    }
    await cp(from, join(destDir, file));
  }
  const manifest = {
    name: "systemone-shim",
    bundledAt: new Date().toISOString(),
    source: sourceDir,
    files: [...SHIM_FILES, ...DOC_FILES],
    entry:
      "systemone serve --port 8765 (or: python3.11 -m systemone.shim --port 8765; cwd: release root)",
  };
  await writeFile(join(destDir, ".zcode-bundle.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`[zcode] bundled SystemOne shim from ${sourceDir}`);
  return true;
}
