# Changelog

## 3.27.0 (2026-09-26)

### Features

* SystemOne decision engine integration
  * New typed-decision client (`systemone-decide.ts`): choice/score/noul decisions with calibrated probabilities, strict response validation, fail-open on anything unexpected
  * Plan ranking falls back to the decider when `/v1/systemone/rank-plans` is unreachable — picks the winning plan by decision instead of silently running the first candidate
  * Configurable shim URL (`systemone-shim-url.ts`); `ZCODE_SYSTEMONE_DECIDE=0` kill switch
* Onboarding wizard: shim URL prompt (default localhost:8765, env-overridable), live decide-endpoint probe, fail-open degraded setup when the shim is down; choices persist to `~/.zcode-local/v2/onboarding.json`

## 3.26.0 (2026-09-24)

### Features

* First-run onboarding wizard: `zcode onboard`
  * Guided first-run setup for new users — walks through the essentials (providers/models, SystemOne routing defaults) so ZCode Local is usable out of the box with nothing to chase down

### Documentation

* README: Decision engine section — Jev-style classifier + Jeff-1 sidecar facts
* README: Minimum requirements section (Node 24+, measured disk, inference requirements)


## 3.24.0 (2026-09-23)

### Features

* SystemOne speed-stack UI + routing transparency
  * Desktop model picker: "Auto (SystemOne)" top entry — per-task model routing toggle; the pinned model stays as fallback (no fake provider/model)
  * Thought control: seven SystemOne modes (Off/Low/Medium/High/XHigh/Ultra/Auto), independent of provider reasoning levels; stored as speedStack.thinkingMode, frozen submission keeps a provider-valid reasoningLevel
  * Per-response routing chip (⚡ model · effort) from the v4 snapshot's systemOneLastRouting; pinned-model turns now emit routing facts too (effort-only routing)
  * Draft persistence for speedStack knobs (modelRouting + thinkingMode); defaults unchanged, no silent flips
* Vendored SystemOne shim in-repo (hermetic releases)
  * scripts/zcode-distribution/systemone-shim-files/systemone/ — shim.py (Windows self-daemonization), api.py, calibration.py (ultra-trivial Q&A), model_registry.json
  * Builder defaults to the vendored copy; ZCODE_SYSTEMONE_SRC still overrides for dev
* Windows lifecycle: CLI spawns the bundled shim with --daemonize (escapes sshd KILL_ON_JOB_CLOSE); fail-open everywhere
* Effort tiers: low=4k / medium=12k / high=32k / xhigh=64k / ultra=model max; Z1 override maps applied tier to model levels per task
* CLI: --model auto, --thinking off|auto|low|medium|high|xhigh|ultra


## 3.23.0 (2026-09-23)

### Features

* zero-setup SystemOne routing: shim bundled + auto-started
  * The desktop GUI runs the same agent bundle as the CLI, so per-task SystemOne routing (route call, effort mapping, conservative MCP pruning) already applied to both — verified at the shared turn-loop layer, no separate desktop agent path exists
  * The SystemOne Python shim is now bundled in the release (`systemone/`); ZCode probes `127.0.0.1:8765/healthz` at startup and auto-starts the bundled shim when nothing answers. A manually managed shim on :8765 is reused, never duplicated
  * New master kill-switch `ZCODE_SYSTEMONE=0` disables auto-start and routing; existing `ZCODE_SPEEDSTACK_PRUNE=0` / `mcpPruning=false` unchanged; everything fail-open

## 3.20.0 (2026-09-22)

### Features

* provider-grouped model picker, memory settings UI, optional duckbot-rag-memory ([86a97bc](https://github.com/Franzferdinan51/ZCode/commit/86a97bca287aabe7796826ac6b8b912cc6dd7ed6))
  * /model picker grouped by provider with current-model marker, context-window and reasoning badges, descriptions and disabled reasons
  * Memory settings UI with optional duckbot-rag-memory toggle (embedding mode, custom paths)
  * ragMemory plumbing through startup preferences, protocol records, inherited sessions, and host runtime

## 3.19.0 (2026-09-22)

### Features

* local-first ML routing (Jeff-1/SystemOne/Laya) + Hermes speed ports (FTS5, failover) ([c5478e5](https://github.com/Franzferdinan51/ZCode/commit/c5478e5bec34faa12fb6724e0c28812d6c85aced))
  * Async AutoRouteScorer seam + Jev-compatible protocol (fail-open)
  * Local sidecar: Jeff-1/SystemOne endpoints + persistent Laya bridge
  * Settings > General opt-in card (default OFF) + async draft upgrade
  * Hermes: FTS5 task search + 429/5xx same-provider failover
  * Tests 62/62, typecheck/lint/architecture green

## 3.18.0 (2026-09-22)

### Features

* open source ([872ad96](https://github.com/Franzferdinan51/ZCode/commit/872ad960de7ec172591f7e1952f7849229f94521))

* remove Z.ai login, OAuth, and plan upsells ([a8bc5d7](https://github.com/Franzferdinan51/ZCode/commit/a8bc5d757d0c9330f72a1d708d04d78a5a7cce1b))
  * Delete app login wall: WelcomeScreen, OAuth session lifecycle,
  * Delete provider-connect-via-login and purchase/upgrade/quota
  * Delete CLI browser OAuth (zcode login, TUI OAuth); keep API-key
  * Delete orphaned OAuth/purchase helpers, tests updated (20 green)
  * Bump version to 3.15.0

* true background routing to external harness CLIs, smart auto-router, Meta provider ([cfedfaa](https://github.com/Franzferdinan51/ZCode/commit/cfedfaae1017eacbda4bd938499633f9af13c749))


### Other Changes

* Faster, safer LM Studio catalog with model pin; English README ([d58f335](https://github.com/Franzferdinan51/ZCode/commit/d58f33528787794985a70b5e9493097c07826912))

* Give shared/services/client packages build scripts (tsc emit) ([6d55347](https://github.com/Franzferdinan51/ZCode/commit/6d55347134c08b329b2a37cfec43deebe7b27842))

* Hide all paid-plan upsells in the local fork ([9028cd6](https://github.com/Franzferdinan51/ZCode/commit/9028cd68136ccc721b7f4036ea4e38d57435e1dc))

* Initial commit ([77432b6](https://github.com/Franzferdinan51/ZCode/commit/77432b6dbf9f70176ced3f4dcdc25f851c3acb2d))

* Installer falls back to flat release-asset layout ([6751395](https://github.com/Franzferdinan51/ZCode/commit/6751395b30a0cabc0494b8930fb8ed579e627b08))

* Make LM Studio the local-first default provider. ([8356b21](https://github.com/Franzferdinan51/ZCode/commit/8356b218fd3e47b4dee26ba04f477ea825ecf6e9))

* Remove stray local files from 3.17.2 commit (playwright snapshots, generated index.html/.zcodeignore) ([a2d3671](https://github.com/Franzferdinan51/ZCode/commit/a2d3671be869dccce78fc945fa4acb54bdf3e659))

* Ship as ZCode Local: side-by-side identity so the fork never fights official installs ([1dd0df5](https://github.com/Franzferdinan51/ZCode/commit/1dd0df566af1be82118c2719461d5aed5e6e1823))
  * New local desktop flavor (ZCode Local, dev.zcode.app.local, zcode-local
  * Non-production flavor automatically disables official auto-update,
  * Deep links use zcode-local:// for local builds (packaged protocol,
  * Data dir renamed ~/.zcode -> ~/.zcode-local across services, desktop,
  * CLI bin renamed zcode -> zcode-local; SEA artifacts renamed to
  * README (EN/CN): side-by-side notes, LM_STUDIO_API_KEY pointer

* Verify installer checksum; write synced settings atomically ([83d8d2a](https://github.com/Franzferdinan51/ZCode/commit/83d8d2a71fbc71d28272e126c7b8cb2021c35c7c))

* ZCode Local 3.16.0: our-releases update system, Harness Router tab, English defaults ([a238dcc](https://github.com/Franzferdinan51/ZCode/commit/a238dccb0f975f6d0637660f66044741a7814faa))
  * Updates: CLI installs and desktop local-flavor updater pull from our
  * Harness Router side tab: pick the default model route (provider/model)
  * English: default locale en-US, translated comments/logs in touched

* ZCode Local 3.17.0: external harness picker, CLI+Web ship, zero high vulns ([6569dff](https://github.com/Franzferdinan51/ZCode/commit/6569dff382405c83fc57f4fda20a0b77e5b1b170))
  * Harness Router tab: external harness catalog (codex, pi, grok,
  * Agent CLI + Web: verified distributable (agent/server/web), fixed dead
  * Router retargets the open session (sessionRouteRequests pub/sub).
  * Perf: lazy-split heavy previews/mermaid/vendor, web sourcemaps off.
  * Security: overrides clear all 12 high audit findings (0 high left);
  * Tests 47/47, typecheck/lint/architecture green.

* ZCode Local 3.17.1: fix desktop startup crash, English packaging comments ([d7c659d](https://github.com/Franzferdinan51/ZCode/commit/d7c659d42e41705b067d8544b63c3424b309f70b))
  * Main process crashed at startup with Dynamic require of events:
  * Translate packaging comments/messages to English in tsup.config.ts,

* ZCode Local 3.17.2: Harness Router tab open by default ([a515fb2](https://github.com/Franzferdinan51/ZCode/commit/a515fb2b123671c2fd2de306fbad605a84b75beb))
  * Fresh workspaces seed the side pane with the Harness Router tab and
  * Verified in headless browser: fresh load shows provider routes plus

* ZCode Local 3.17.3: Harness Router sidebar no longer forced open ([07de7dd](https://github.com/Franzferdinan51/ZCode/commit/07de7dd962167bee3b7220a6841a2804c9ba9c0f))

* ZCode Local 3.17.4: fix Rules-of-Hooks crash in workspace service hooks ([7557e71](https://github.com/Franzferdinan51/ZCode/commit/7557e7147b3a225e4b6c43f0ad0dc1c22009ba9d))

* ZCode Local gets its own icon: light tile, black Z ([10db0cf](https://github.com/Franzferdinan51/ZCode/commit/10db0cf43e7c496dcbff08fef47c73a24f5cd630))
