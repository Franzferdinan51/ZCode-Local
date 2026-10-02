# systemone package

Jev-style typed decisions (`choice` / `score` / `noul`) on a local GLiClass
engine or an SGLang-served decision model. The full documentation lives in
the **[root README](../README.md)** — install, quickstart, endpoints,
SGLang interop, calibration, and design notes. This file is only a map of
the modules.

## Map

| module | needs | role |
|---|---|---|
| `patterns.py` | stdlib + numpy | shared decision patterns: `SystemOneError`, validators, `StallGuard`, `LatencyStats`, `make_questions`, TypeSafe confidence, prompt rows |
| `api.py` | `systemone[local]` | `SystemOne`: batched GLiClass inference over the patterns above |
| `sglang_backend.py` | stdlib | `SGLangBackend` (`/v1/decisions` client) + `HybridBackend` (local-first, SGLang escalation) + `decide_fn_for` (loop adapter) |
| `jev_backend.py` | stdlib | `JevDecideBackend`: JEV decision models over `POST /v1/decide` + System 2 chat |
| `loop.py` | stdlib | `DecisionLoop`: the one See > Decide > Act agent loop |
| `shim.py` | stdlib + numpy (+ local for GLiClass mode) | HTTP server: `POST /v1/systemone`, `/v1/decisions`, `/v1/decide`, `/v1/systemone/route`, `/v1/systemone/rank-plans`, `/v1/systemone/decide`, `/v1/systemone/permute`; `GET /healthz`, `/metrics`, `/openapi.json`, `/v1/decide/info`. Engine via `SYSTEMONE_ENGINE=auto\|local\|sglang\|jevk5\|onnx\|jev\|kev` |
| `client.py` | stdlib | `SystemOneClient`: HTTP client for the shim |
| `cli.py` | stdlib + numpy (+ local for `local`/`ask`) | `systemone` command: `route` / `decide` / `status` / `battery` via shim, `local` / `ask` / `serve` on-box |
| `jeff1_sidecar.py` | decider-ai (+ torch via it) | decision sidecar serving the decider-4b backend on `:8079` |
| `jeff1.py` | stdlib | sidecar HTTP helpers (`decide_via_jeff1`, blending, second opinions) |
| `scoring.py` | stdlib | route scoring, calibration application, model/tool ranking, LM Studio inventory |
| `calibration.py` | numpy | temperature / Platt / isotonic calibrators, per-type maps |
| `metrics.py` | numpy | `ece` / `brier` / `nll` / `aurc` / selective accuracy + tables |
| `mcp_server.py` | `systemone[mcp]` (shim tools) + local for engine tools | MCP tools over stdio (`typesafe_ask`, `verify_claims`, `screen_content`, `rank_candidates`, shim tools); engine tools lazy-load `systemone[local]` |
| `acp_server.py` | stdlib | ACP agent adapter over stdio |
| `distill.py` | stdlib | teacher labeling (`SyntheticTeacher`, `HFTeacher`, `LMStudioTeacher`) → GLiClass training JSON → `train.py` |
| `tune.py` | stdlib | `make_training_json()` + `tune()` domain fine-tuning wrappers |
| `battery/` | stdlib | regression battery runner + temperature fitting |
| `jevbench.py` | stdlib + numpy | JevBench-split scoring adapter (`score_item`, `run_file`, `DecisionResult`); CLI: `systemone jevbench --items` |
| `jevk5_backend.py` | stdlib + numpy | `JevK5ServerBackend`: judge via `jevk5-serve`'s `/v1/systemone` (`JEVK5_BASE_URL`) |
| `rerank_backend.py` | stdlib + numpy (+ onnxruntime for `OnnxCrossEncoder`) | `RerankBackend`: cross-encoder judge over any score fn; `OnnxCrossEncoder`: CPU ONNX cross-encoder |
| `bench_2048.py` | varies | headless 2048 decision-loop benchmark (see docstring for limits) |
| `tests/` + `test_route.py` | `pytest` (+ local for heavy tests) | suite: `pytest -m "not slow"` runs green on a slim install; torch-only tests skip with a reason |
| `openapi.json` | — | machine-readable API spec, served at `GET /openapi.json` |
| `examples/` | varies | runnable demos; SGLang/stub-capable ones import clean on slim |

Install shapes: `pip install -e .` (slim: SGLang + shim client),
`pip install -e '.[local]'` (+ torch/GLiClass engine),
`pip install -e '.[local,mcp,dev]'` (everything).
