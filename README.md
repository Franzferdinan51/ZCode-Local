# ZCode Local

<div align="center">
  <img src="public/logo/icons-local/1024x1024.png" alt="ZCode Local" width="128" height="128" />
</div>
<p align="center">
  <a href="https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=47ag983c-8fcb-4d6d-814b-5395193a712c&amp;qr_code=true">Feishu community</a> ·
  <a href="https://discord.gg/z9aBcQXZQ3">Discord</a>
</p>
<p align="center">
  <a href="README.zh-CN.md">简体中文</a> | English
</p>

ZCode is an AI coding workspace with desktop, browser, and terminal interfaces. This repository contains the clients, backend services, shared UI, and Agent CLI and runtime source code.

**Local-first:** the default inference path is a local [LM Studio](https://lmstudio.ai) OpenAI-compatible server at `http://127.0.0.1:1234/v1`. Load a chat model in LM Studio yourself, then send prompts — no Z.ai / BigModel login, API key, Coding Plan, or trial quota is required. ZCode does not start or load models. Optional cloud templates remain in settings but do not block the core path.

**Side-by-side with official ZCode:** this fork ships as **ZCode Local** and never touches official installs — data lives in `~/.zcode-local` (override with `ZCODE_DATA_BASE_DIR`), the terminal is `zcode-local`, deep links use `zcode-local://`, and official auto-update is disabled. API keys for a protected LM Studio server go in `LM_STUDIO_API_KEY` (see `.env.example`).

| Interface                    | Purpose                                                                                         | Development command            |
| ---------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------ |
| Desktop                      | Electron desktop application                                                                    | `pnpm dev:desktop`             |
| Web / ZCode CLI distribution | Terminal and browser workspace; packages the TUI, Web client, backend, and Agent together       | `pnpm dev:web`                 |
| Agent CLI                    | The `zcode-local` terminal interface, which also provides the Agent runtime for Desktop and Web | `pnpm --filter @zcode/cli dev` |

## Fork changes (ZCode Local)

Everything below is what this fork changed relative to upstream ZCode. The UI, components, and features are otherwise untouched.

**LM Studio is the main provider**

- Default inference is the OpenAI-compatible server at `http://127.0.0.1:1234/v1` (`local:lm-studio` is the first builtin provider).
- The model picker loads its catalog live from LM Studio (`/api/v0/models`, then `/v1/models`): loaded chat models first, embeddings excluded.
- `LM_STUDIO_API_KEY` supplies the Bearer [REDACTED] servers with auth enabled; unset uses the `lm-studio` default that works with unprotected instances (see `.env.example`).

**Z.ai paywalls removed from the core path**

- Sending a prompt no longer requires Z.ai / BigModel login, API key, Coding Plan, or trial quota.
- The fail-closed account overlay no longer hides or disables the LM Studio default, and the startup login entry only opens when no usable provider exists.
- The bundled provider config no longer refreshes from the official remote endpoint.
- Purchase/plan panels remain visible in settings but are inert without an account.

**Meta Model API (Muse) via the official Responses integration**

- The bundled `meta` provider template points at `https://api.meta.ai/v1` and drives
  Muse Spark over the Responses API (the integration Meta documents for coding
  agents), not raw chat-completions. That keeps tool calling, streaming, and
  structured output working, and avoids the HTTP 400s chat-completions returns
  for reasoning-model params (`stop`, `logit_bias`, `logprobs`).
- Add the provider from the Meta template and paste a Model API key (created at
  https://dev.meta.ai/) in provider settings. Muse models are never a default;
  routing still flows router -> registry/config -> your explicit selection.

**Side-by-side identity (no conflicts with official installs)**

| Surface               | Official                | This fork                           |
| --------------------- | ----------------------- | ----------------------------------- |
| Desktop app           | ZCode (`dev.zcode.app`) | ZCode Local (`dev.zcode.app.local`) |
| App icon              | Black tile, white Z     | Light tile, black Z                 |
| Data directory        | `~/.zcode`              | `~/.zcode-local`                    |
| Terminal / SEA binary | `zcode`                 | `zcode-local`                       |
| Deep links            | `zcode://`              | `zcode-local://`                    |
| Auto-update           | Official channel        | Disabled                            |

- Linux executable, package names, `.desktop` file, MIME type, icon, Windows AUMID, and CUA helper variant all follow the fork identity.
- No fixed ports exist to collide (ephemeral `listen` only); the Electron single-instance lock follows the separate userData directory.
- Builds default to the `local` flavor; `ZCODE_OFFICIAL_IDENTITY=1` restores legacy production/preview resolution.


**SystemOne agent-flow control (3.25.0)**

SystemOne is this fork's local model router — and as of 3.25.0 it drives
the full agent behavior plan, not just model selection. Each task asks the
router for a tier, an effort level, and task labels, and the runtime
conditions the turn on that decision: effort maps to a behavior policy (max
model steps and tool calls, subagent allowance, read/search breadth,
verification passes after edits, compaction aggressiveness), the labels
drive per-task tool-pack pruning (irrelevant MCP servers and tools are
suppressed before the request is built), heavy/ambiguous work gets
plan-then-execute (planner writes `PLAN.md` with read-only tools, executor
implements it), and doom-loop fingerprinting escalates repeated-call
patterns from warning to strategy-change nudge to pause-and-retry.

What SystemOne decides, in one line: **how much effort the task gets, which
tools it sees, how long it may run, whether it plans first, how deeply it
verifies, and how aggressively it compacts.**

The fail-open contract: a down, slow, or malformed router changes nothing
about what the agent *can* do — the turn runs exactly as it would without
SystemOne, with the full tool surface attached. Low-confidence routes never
prune. If a pruned tool turns out to be needed, the turn retries once with
the full set. Every behavior number (budgets, policy dimensions) lives in
config or env, tunable without a release, and model choice always flows
from the router, the registry, or explicit user config — no hard-coded
model IDs.

**Decision engine:** the router is SystemOne — a **Jev-style**
typed-decision layer over local GLiClass checkpoints ("tiny decides, big
works"). Each call returns a scored decision in ~100 ms: per-tier
probabilities, top-1/top-2 margins, and an uncertainty flag; the scores are
calibrated, then drive tier, effort, expected-utility model ranking,
tool/MCP relevance ranking, and plan ranking. It is advisory-only and
fail-open, and lives in the Python shim (`python -m systemone.shim`,
`http://127.0.0.1:8765`). Details: https://github.com/Franzferdinan51/SystemOne.

Jeff-1-backed plan ranking: `plan-execute` asks the shim for N
candidate plans and posts them to `/v1/systemone/rank-plans`, executing the
winner. The ranking blends the GLiClass score 50/50 with **Jeff-1**
([GestaltLabs/Jeff-1](https://huggingface.co/GestaltLabs/Jeff-1) — Apache 2.0,
LoRA on [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)),
an open-weight model trained for Jev-compatible typed decisions
(`choice`, `score`, `noul`);
the full ranking lands in run diagnostics. On uncertain routes the shim's
`jeff1_second_opinion` is advisory-only — and the agent never prunes,
bumps effort one level, and records why. Jeff-1 runs as its own sidecar
process (on by default; `SYSTEMONE_JEFF1=0` runs GLiClass-only) and fails
open with no behavior change when down or slow. It never loads, unloads,
switches, or competes with the loaded worker model — it lives on the Mac
mini, never on the inference host.

Kill switches (each disables only its own surface; all default on):

- `ZCODE_SYSTEMONE=0` — all SystemOne integration
- `ZCODE_SYSTEMONE_DECIDE=0` — the decide engine only (routing untouched)
- `ZCODE_SPEEDSTACK_PRUNE=0` — per-task tool-pack / MCP pruning
- `ZCODE_BUDGET_ENFORCE=0` — turn/tool-call budget enforcement
- `ZCODE_PLAN_EXECUTE=0` — route-driven plan-then-execute
- `ZCODE_ANCHORED_COMPACT=0` — anchored compaction (falls back to plain)
- `ZCODE_DOOMLOOP=0` — doom-loop escalation (plain warnings stay)
- `mcpPruning=false` (session config) — MCP pruning via config instead of env

Independent controls: model selection stays with the pinned LM Studio
model, the model picker, or explicit user config — SystemOne never loads,
unloads, or switches models. The thinking level (`--thinking`) is set
independently; the route's effort mapping only applies when the user hasn't
set one, and explicit flags always win.

Full decision table, fail-open contract, and tuning knobs:
`apps/zcode-cli/specs/speed-stack.md` ("SystemOne decision surface").

## Setup

Install Git, Node.js **24.14.0**, and pnpm **10.33.2**. [mise.toml](mise.toml) is the source of truth for tool versions. Run all development and packaging commands below from the repository root.

```bash
pnpm bootstrap
```

`pnpm bootstrap` installs workspace dependencies, prepares local desktop runtime assets, and runs `build:bootstrap`.

The Agent CLI and runtime source code lives in [apps/zcode-cli/](apps/zcode-cli/) as a regular directory included when you clone this repository. No separate checkout or Git submodule initialization is required.

Additional setup and build commands:

| Command                        | Purpose                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `pnpm install`                 | Install dependencies                                                                                                                |
| `pnpm prepare:desktop-runtime` | Prepare desktop runtime assets, including remote assets by default                                                                  |
| `pnpm prepare:remote-assets`   | Prepare remote runtime assets separately                                                                                            |
| `pnpm bootstrap:with-remote`   | Set up dependencies and local and remote assets, then build the relevant packages sequentially; skip the desktop application bundle |
| `pnpm build`                   | Recursively run each workspace package's build script, including its asset preparation steps                                        |

The default `bootstrap` skips remote asset preparation and is suitable for local desktop development. Run the corresponding preparation command when working with remote workspaces or validating remote distribution assets.

## Minimum requirements

For running the **released runtime** (the `zcode-<version>.tar.gz`
tarballs / `~/.zcode-local/runtime/releases/` installs). Dev and
packaging requirements (Git, Node 24.14.0, pnpm 10.33.2 via mise) are
under Setup above.

**First time?** Run `zcode onboard` — the guided first-run wizard checks
these requirements, probes LM Studio and the SystemOne router, optionally
stores a Meta API key, and writes your initial config. It also runs
automatically on first interactive launch of `zcode tui` (skip with
`--skip-onboarding` or `ZCODE_SKIP_ONBOARDING=1`).

The wizard probes the shim URL you give it (default
`http://127.0.0.1:8765`; override anytime with `SYSTEMONE_SHIM_URL`) for
both `/healthz` and the decide engine. If the shim is down and the URL is
local, it asks whether to start the bundled shim (ZCode never auto-starts
it); for a remote URL it never offers a local start. A down shim means
routing and the decide engine go fail-open and ZCode runs normally —
setup never fails because SystemOne is unreachable. Your answers are
saved to `~/.zcode-local/v2/onboarding.json` and honored at runtime
automatically (explicit env vars still win): `ZCODE_SYSTEMONE=0` /
`ZCODE_SYSTEMONE_DECIDE=0` turn routing and the decide fallback off.

- **Node.js** — >= 24.0.0 (per `package.json` `engines`; verified).
  The runtime tarball does **not** bundle Node — `bin/zcode.mjs` runs
  on your system Node, so install Node 24+ first.
- **OS / CPU** — the runtime is platform-agnostic JavaScript and runs
  anywhere Node 24 runs: macOS, Windows, Linux (x64 or arm64). No
  official platform restriction is stated.
- **Disk** — ~300 MB for the installed runtime (tarball 64 MB,
  extracted ~283 MB — both measured), plus ~1 GB for the bundled
  SystemOne shim if you run it locally (torch ~0.6 GB, GLiClass edge
  checkpoint 256 MB — both measured), plus ~15 GB if you enable the
  Jeff-1 second head (its model weights in the HuggingFace cache —
  measured; skip with `SYSTEMONE_JEFF1=0`).
- **Inference (the real requirement)** — ZCode needs somewhere to run
  models: LM Studio locally (OpenAI-compatible) or configured API
  providers. Model requirements are the *model's*, not the tool's: a
  35B-class local model wants tens of GB of RAM/VRAM — check the model
  card / LM Studio for the specific model before loading it. Reference
  setup: Windows 11 PC with 64 GB RAM serving models through LM Studio;
  Mac mini (M4 Pro, 24 GB) reaching them over LM Link.
- **SystemOne agent-flow routing (on by default)** — ZCode queries the
  router at `http://127.0.0.1:8765` (bundled under the release's
  `systemone/` directory: `pip install -r systemone/requirements.txt`,
  the GLiClass checkpoint downloads on first start; needs
  Python >= 3.10). ZCode does **not** auto-start the shim — grok-local
  does, or start it manually
  (`python3.11 -m systemone.shim --port 8765`). Routing is advisory and
  fail-open. The shim base URL defaults to `http://127.0.0.1:8765` and is
  overridable with `SYSTEMONE_SHIM_URL`. Kill switches:
  `ZCODE_SYSTEMONE=0` (routing off), `ZCODE_SPEEDSTACK_PRUNE=0`
  (no MCP pruning), `ZCODE_SYSTEMONE_DECIDE=0` (decide engine off),
  `SYSTEMONE_JEFF1=0` (GLiClass only, no Jeff-1 second head).

## Development and Usage

### Desktop

```bash
pnpm dev:desktop

# Use the test environment
pnpm dev:desktop:test
```

`pnpm dev:desktop` defaults to `pnpm dev:desktop:prod` and uses production service configuration. The startup script prepares local runtime assets, builds the desktop Agent, then starts Electron and source watchers.

Set `ZCODE_DATA_BASE_DIR` to use a separate development data directory. For example, on macOS / Linux:

```bash
ZCODE_DATA_BASE_DIR="$HOME/.zcode-dev-home" pnpm dev:desktop:test
```

### Web Development

Use development mode when editing Web or backend source code:

```bash
pnpm dev:web

# Set the backend workspace (macOS / Linux)
ZCODE_SERVER_WORKSPACE=/path/to/project pnpm dev:web
```

This starts both the Web development server (default: `http://localhost:5173`) and the backend (default: `http://localhost:3030`). Open the Web development server in your browser. `/ws` and general `/api` requests are proxied to the local backend; `/api/v1/oauth/token` is proxied separately to the configured product service.

After changing Agent source code, run `pnpm --filter @zcode/cli... build` and restart the service. To validate the complete distribution, extract and run it as described under Packaging → ZCode CLI distribution below.

### ZCode CLI distribution

The command-line distribution includes the TUI, Web client, and Agent behind one `zcode-local` command. With no arguments it starts the TUI; a leading `--web` starts Web mode; all other arguments go to the existing Agent CLI. Both modes run locally without Electron.

```bash
# Start the terminal UI by default
zcode-local

# Start the Web interface
zcode-local --web

# Set the project and port without opening a browser automatically
zcode-local --web --workspace /path/to/project --port 3030 --no-open

# Show CLI or Web options
zcode-local --help
zcode-local --web --help
```

In Web mode, it uses the current directory as the workspace, listens on `127.0.0.1` without token authentication by default, selects an available port, and opens a browser. Use the URL printed in the terminal and press `Ctrl+C` to stop the service. For LAN access, use `--host 0.0.0.0`; listening on a non-local address generates an access token by default. Use the token-bearing URL printed in the terminal. Set a token with `--token`, or disable token authentication with `--no-token`.

When starting the general Web service's HTTP entry directly, configure API/WebSocket authentication with `ZCODE_SERVER_AUTH_TOKEN`. When creating the service programmatically, use the `authToken` option.

See Packaging below for build instructions. `pnpm build:zcode` only creates the distribution; it does not replace an existing `zcode-local` on `PATH`. If the command still points to an older installation or another checkout, check it with `command -v zcode-local` on macOS / Linux or `where.exe zcode-local` on Windows.

### CLI Source Development

Use the source entry when developing the TUI or Agent:

```bash
pnpm --filter @zcode/cli dev --help
pnpm --filter @zcode/cli dev

# Build the CLI and its workspace dependencies
pnpm --filter @zcode/cli... build
node apps/zcode-cli/packages/cli/dist/zcode.cjs --help
```

This entry runs the Agent CLI directly and does not handle the distribution's `--web` switch. Use `pnpm dev:web` for Web development, or the extracted `bin/zcode.mjs` shown below to test the unified command.

## Configuration

The root [.env.example](.env.example) provides sample service URLs and build configuration. Copy it to `.env` as needed and place local overrides in `.env.local`. Select the Desktop development environment with `dev:desktop:test` or `dev:desktop:prod`.

| Setting                              | Purpose                                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| `ZCODE_DATA_BASE_DIR`                | Base directory for application data, stored under its `.zcode-local/` subdirectory                     |
| `ZCODE_SERVER_WORKSPACE`             | Workspace path for the Web backend                                                                     |
| `ZCODE_BUILTIN_PROVIDER_CONFIG_FILE` | Path to a local provider configuration file; uses the built-in configuration when unset                |
| `ZCODE_DIST_BASE_URL`                | Download base URL used by the CLI distribution installer                                               |
| `LM_STUDIO_MODEL`                    | Pin one LM Studio model id to the front of the picker catalog (exact match; unset keeps catalog order) |

Runtime variables can be set explicitly in the environment of the startup command. See [config/README.md](config/README.md) for the default configuration shipped with the client.

## Packaging

See [third-party/README.md](third-party/README.md) for notice generation, distribution checks, and where the notices are included in each distribution.

### Desktop

```bash
pnpm bundle:desktop

# Set the target platform and CPU architecture
pnpm bundle:desktop -- --os win --arch x64

pnpm bundle:desktop -- --help
```

The default target is macOS arm64, and the default output directory is `packages/desktop/dist/`. `--os` accepts `mac`, `win`, or `linux`; `--arch` accepts `x64` or `arm64`. Packaging and signing require the tools and configuration for the target platform.

### ZCode CLI distribution

Run `pnpm build:zcode` to build the CLI/TUI, backend, and Web client, collect the TUI native libraries, workers, and runtime dependencies, then assemble the distribution. Running the distribution still requires Node.js; use the version specified in `mise.toml`.

Builds default to our GitHub releases as the download base URL, so installs and updates always pull our binaries. Override with `ZCODE_DIST_BASE_URL` in `.env`, `.env.local`, or the process environment, or pass `--base-url` (explicit mirrors only; official Z.ai does not publish this layout, so never point it at official hosts).

```bash
# Release build, pinned to its own tag URL for reproducibility
pnpm build:zcode --base-url https://github.com/Franzferdinan51/ZCode/releases/download/zcode-local-v3.15.0/

# Floating build (installs/updates resolve our latest release)
pnpm build:zcode

# Repackage existing Agent, backend, and Web build outputs
pnpm build:zcode --skip-build

# Show options for the version, output directory, and more
pnpm build:zcode --help
```

The version defaults to the root `package.json` version. Output is written to `dist/zcode/`:

- `releases/<version>/zcode-<version>.tar.gz`: runtime package.
- `releases/<version>/sha256.txt`: checksum file.
- `latest.json` and `install.sh`: version index and installer.

Attach these files to the GitHub release. To ship desktop auto-updates from the same release, also generate and attach the Electron manifest (desktop file URLs must be absolute):

```bash
pnpm build:electron-manifest --version 3.15.0 \
  --asset-base-url https://github.com/Franzferdinan51/ZCode/releases/download/zcode-local-v3.15.0/ \
  --asset "ZCode-Local-3.15.0.dmg:<sha512>"
```

Install or update from our latest release:

```bash
curl -fsSL https://github.com/Franzferdinan51/ZCode/releases/latest/download/install.sh | sh
```

The installer downloads the runtime package from that release, installs it to `~/.zcode-local/runtime` by default, and creates the `zcode-local` command in `~/.local/bin`. Override these directories with `ZCODE_DIST_HOME` and `ZCODE_DIST_BIN_DIR`, respectively. Every install verifies the `latest.json` sha256 checksum before unpacking and refuses to install when it is missing or mismatched.

Existing Lite users should switch to the new build command, environment variables, and installer. Installation does not remove old Lite directories or migrate/delete session data.

To test a packaged build locally, extract and run it directly without uploading or installing it:

```bash
zcode_version=$(node -p "require('./dist/zcode/latest.json').version")
mkdir -p dist/zcode/debug
tar -xzf "dist/zcode/releases/$zcode_version/zcode-$zcode_version.tar.gz" \
  -C dist/zcode/debug
# Start the TUI by default
node dist/zcode/debug/zcode/bin/zcode.mjs

# Start Web mode
node dist/zcode/debug/zcode/bin/zcode.mjs --web \
  --workspace "$PWD" --port 3030 --no-open
```

Open `http://127.0.0.1:3030` to validate the complete flow, with one backend serving the Web pages and running the Agent. The port must be available; if `pnpm dev:web` is already running, choose another `--port`.

## Repository Structure

| Directory                                            | Responsibility                                                                          |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `packages/desktop`                                   | Electron Main, Host, Renderer, and desktop packaging                                    |
| `packages/web`                                       | Web client                                                                              |
| `packages/server`                                    | HTTP / WebSocket services and remote connections                                        |
| `packages/zcode-server-cli`                          | Standalone server startup and process management                                        |
| `packages/ui`                                        | Shared React components, hooks, and Zustand state                                       |
| `packages/services`                                  | Business services and persistence                                                       |
| `packages/shared`, `packages/rpc`, `packages/client` | Shared protocols and types, RPC framework, and Agent client SDK                         |
| `packages/provider`, `packages/provider-node`        | Common provider capabilities and Node implementations                                   |
| `apps/zcode-cli`                                     | Agent CLI, TUI, runtime, and tools                                                      |
| `scripts`, `config`, `third-party`                   | Build and maintenance scripts, built-in configuration, and third-party notice materials |

## Project Notice

See [NOTICE.md](NOTICE.md) for feature and promotion scope, maintenance policy, execution and data risks, licensing, and third-party copyright information.
