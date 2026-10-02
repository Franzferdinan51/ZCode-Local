"""MCP server exposing the System One API as agent tools (stdio).

Two tool families:

Shim tools (default; talk to the live shim over HTTP — no model is loaded
by this process):
- systemone_route:   cheapest sufficient tier for a task (+confidence/margin)
- systemone_decide:  one typed decision (choice | noul | score)
- systemone_status:  shim + sidecar health, decision backend in use
- systemone_rank_plans: order plans by P(success | task)

Legacy local tools (load a GLiClass checkpoint in-process, one at a time):
- typesafe_ask: Jev-compatible typed judgments (choice / score / noul) over
  application state. Same question interface as Loki's `typesafe_ask` tool,
  but fully local — NO TYPESAFE_API_KEY needed.
- verify_claims: check each claim against provided evidence text ->
                 verdict per claim (supported / contradicted / unverifiable)
                 with confidence.
- screen_content: classify text as safe / needs_review / blocked with
                 confidence (policy text optional).
- rank_candidates: score + order candidates by relevance to a query.

Run:  python -m systemone.mcp_server
or:   systemone-mcp   (if installed)

Configure with env vars:
  SYSTEMONE_SHIM_URL  shim base URL for the systemone_* tools
                      (default: http://127.0.0.1:8765)
  SYSTEMONE_MODEL     HF model id for the legacy local tools
                      (default: smallest available, edge first)
  SYSTEMONE_DEVICE    cuda | mps | cpu  (default: auto — CUDA, else Apple MPS, else CPU)
  SYSTEMONE_CALIBRATOR  path to a pickled TemperatureCalibrator (optional)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

from .calibration import PerTypeTemperatureCalibrator, TemperatureCalibrator
from .client import ShimError, SystemOneClient, default_shim_url

mcp = FastMCP("systemone")

_engine: SystemOne | None = None
_engine_model: str | None = None

_client: SystemOneClient | None = None
_client_url: str | None = None


def get_client() -> SystemOneClient:
    """Return the shared shim client (rebuilt if $SYSTEMONE_SHIM_URL changed)."""
    global _client, _client_url
    url = default_shim_url()
    if _client is None or _client_url != url:
        _client = SystemOneClient(url)
        _client_url = url
    return _client


def get_engine(model_name: str | None = None) -> SystemOne:
    """Return the shared engine.

    Only ONE local model is ever loaded at a time: requesting a model
    different from the loaded one rebuilds the engine, evicting the
    previous model first.
    """
    global _engine, _engine_model
    from .api import SystemOne  # local engine; needs systemone[local]

    wanted = (model_name or os.environ.get("SYSTEMONE_MODEL") or "").strip() or None
    if _engine is None or (wanted is not None and wanted != _engine_model):
        _engine = SystemOne(
            model_name=wanted,
            device=os.environ.get("SYSTEMONE_DEVICE"),
        )
        _engine_model = _engine.model_name
        # Optional: point SYSTEMONE_CALIBRATOR at a JSON calibrator file
        # (TemperatureCalibrator.save() format; per-type maps also accepted).
        cal_path = os.environ.get("SYSTEMONE_CALIBRATOR")
        if cal_path and os.path.exists(cal_path):
            from .calibration import load_calibrator_file

            try:
                cal = load_calibrator_file(cal_path)
            except Exception:
                cal = None  # corrupt file: serve raw, never crash the engine
            if isinstance(cal, (TemperatureCalibrator, PerTypeTemperatureCalibrator)):
                _engine.set_calibrator(cal)
    return _engine


# -- typesafe_ask: Jev-compatible interface, local engine -------------------

_ALLOWED_QUESTION_TYPES = {"choice", "score", "noul"}


def _stringify(value: Any) -> str:
    """Render an instructions value (str | object | array) as text."""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, default=str).strip()


def _convert_questions(
    questions: Any,
) -> tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Validate the Jev-style question list and map it onto engine questions.

    Returns (engine_questions, score_level_order).
    Raises ValueError with a Loki-style message on bad input.
    """
    if not isinstance(questions, list) or not questions:
        raise ValueError("questions must be a non-empty list")

    converted: List[Dict[str, Any]] = []
    level_orders: Dict[str, List[str]] = {}
    seen: set = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"questions[{index}] must be an object")

        question_id = str(question.get("id") or "").strip()
        if not question_id:
            raise ValueError(f"questions[{index}].id is required")
        if question_id in seen:
            raise ValueError(f"duplicate question id: {question_id}")
        seen.add(question_id)

        question_type = str(question.get("type") or "").strip().lower()
        if question_type not in _ALLOWED_QUESTION_TYPES:
            raise ValueError(
                f"questions[{index}].type must be one of: "
                f"{', '.join(sorted(_ALLOWED_QUESTION_TYPES))}"
            )

        instructions = _stringify(question.get("instructions"))
        if not instructions:
            raise ValueError(f"questions[{index}].instructions must not be empty")

        criteria = question.get("criteria")
        if question_type == "choice":
            # criteria: {option_name: optional description} — mirrors Jev/Loki
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(
                    f"questions[{index}].criteria must be an object with at "
                    "least two named options for a choice question"
                )
            options = [str(k) for k in criteria.keys()]
            desc_lines = [
                f"- {opt}: {_stringify(criteria[opt])}"
                for opt in options
                if _stringify(criteria[opt])
            ]
            prompt = instructions
            if desc_lines:
                prompt += "\nOptions:\n" + "\n".join(desc_lines)
            converted.append(
                {"name": question_id, "type": "choice", "options": options,
                 "prompt": prompt}
            )
        elif question_type == "score":
            # criteria: ordered list of level descriptions (str | object)
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(
                    f"questions[{index}].criteria must be an ordered list with "
                    "at least two described levels for a score question"
                )
            levels = [_stringify(c) for c in criteria]
            if any(not lv for lv in levels):
                raise ValueError(f"questions[{index}].criteria levels must not be empty")
            level_orders[question_id] = levels
            converted.append(
                {"name": question_id, "type": "score", "levels": levels,
                 "prompt": instructions}
            )
        else:  # noul
            # criteria: optional {true: "...", false: "..."} descriptions
            statement = instructions
            if isinstance(criteria, dict):
                true_d = _stringify(criteria.get("true"))
                false_d = _stringify(criteria.get("false"))
                unknown = set(criteria) - {"true", "false"}
                if unknown:
                    raise ValueError(
                        f"questions[{index}].criteria for noul may only define "
                        "true/false descriptions"
                    )
                if true_d or false_d:
                    statement += (
                        f" ('yes' means: {true_d or 'the condition holds'}; "
                        f"'no' means: {false_d or 'it does not'}.)"
                    )
            converted.append(
                {"name": question_id, "type": "noul", "statement": statement}
            )
    return converted, level_orders


def _confidence(a: Dict[str, Any], dist: Dict[str, float]) -> float:
    """Confidence with a Loki-style fallback: max(probabilities) when absent."""
    conf = a.get("confidence")
    try:
        return round(float(conf), 4)
    except (TypeError, ValueError):
        pass
    if dist:
        return round(float(max(dist.values())), 4)
    return 0.0


def _to_jev_answers(
    out: Dict[str, Any], questions: List[Dict[str, Any]],
    level_orders: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Map engine answers onto TypeSafe's answer shape: {"answers": {id: ...}}."""
    answers: Dict[str, Any] = {}
    for q in questions:
        qid = str(q.get("id") or "")
        a = out[qid]
        qtype = str(q.get("type") or "").lower()
        if qtype == "choice":
            probs = {k: round(float(v), 4) for k, v in a["probabilities"].items()}
            answers[qid] = {
                "choice": a["choice"],
                "probabilities": probs,
                "confidence": _confidence(a, probs),
            }
        elif qtype == "score":
            levels = level_orders[qid]
            # NOTE: the engine names the score distribution "distribution"
            # (choice uses "probabilities"); both map onto Jev's "probabilities".
            dist = a.get("probabilities", a.get("distribution", {}))
            probs = [float(dist[lv]) for lv in levels]
            # Jev's score is a fractional position on the ordered levels; our
            # local equivalent is the probability-weighted expected position.
            position = sum(i * p for i, p in enumerate(probs))
            prob_map = {k: round(float(v), 4) for k, v in dist.items()}
            answers[qid] = {
                "score": round(position, 4),
                "level": a["level"],
                "probabilities": prob_map,
                "confidence": _confidence(a, prob_map),
            }
        else:  # noul — Jev's value is P(yes), 0..1
            p_yes = float(a["probability"])
            answers[qid] = {
                "noul": round(p_yes, 4),
                "confidence": _confidence(a, {"yes": p_yes, "no": 1.0 - p_yes}),
            }
    meta = dict(out.get("_meta", {}))
    meta.update({"engine": "local-gliclass", "api_key_required": False})
    return {"answers": answers, "_meta": meta}


@mcp.tool()
def typesafe_ask(
    state: Any,
    questions: List[Dict[str, Any]],
    model: str | None = None,
) -> Dict[str, Any]:
    """Jev-compatible typed judgments, fully local — no API key needed.

    Same question interface as Loki's `typesafe_ask` tool (and TypeSafe's
    System One API): pass application `state` (string, object, or array)
    plus a list of typed questions; get `{"answers": {id: ...}}` back in
    one batched local pass. Jev does not generate prose — use this for
    bounded semantic decisions and your normal LLM for writing/reasoning.

    Question format (each): {"id", "type", "instructions", "criteria"}
      - id: caller-chosen stable id, used to key the answer in the response.
      - type: "choice" | "score" | "noul".
          choice = pick one defined option;
          score  = place state on ordered described levels;
          noul   = probability that a condition holds.
      - instructions: one narrow, self-contained judgment to make from the
          state (string, object, or array).
      - criteria: choice -> object mapping option names to descriptions
          (min 2); score -> ordered list of level descriptions (min 2);
          noul -> optional {"true": "...", "false": "..."}.

    Answer shape per question:
      - choice: {"choice", "probabilities", "confidence"}
      - score:  {"score" (fractional position), "level",
                 "probabilities", "confidence"}
      - noul:   {"noul" (P(yes) 0..1), "confidence"}

    `model` optionally names a GLiClass checkpoint to judge with
    (default: SYSTEMONE_MODEL / smallest available). Batching independent
    questions over the same state in one call is much cheaper than one
    call per question.
    """
    state_text = state if isinstance(state, str) else json.dumps(
        state, ensure_ascii=False, default=str
    )
    converted, level_orders = _convert_questions(questions)
    engine = get_engine(model_name=model)
    out = engine.systemone(state_text, converted)
    return _to_jev_answers(out, questions, level_orders)


@mcp.tool()
def verify_claims(claims: List[str], evidence: str) -> List[dict]:
    """Check each claim against the evidence text.

    Returns per claim: verdict (supported | contradicted | unverifiable),
    confidence, and per-verdict probabilities. All claims are evaluated in
    one batched pass.
    """
    eng = get_engine()
    questions = [
        {
            "name": f"claim_{i}",
            "type": "choice",
            "options": ["supported", "contradicted", "unverifiable"],
            "prompt": (
                "Given the evidence, is the following claim supported, "
                f"contradicted, or unverifiable? Claim: {claim}"
            ),
        }
        for i, claim in enumerate(claims)
    ]
    out = eng.systemone(evidence, questions)
    results = []
    for i in range(len(claims)):
        a = out[f"claim_{i}"]
        results.append(
            {
                "claim": claims[i],
                "verdict": a["choice"],
                "confidence": round(a["confidence"], 4),
                "probabilities": {k: round(v, 4) for k, v in a["probabilities"].items()},
            }
        )
    return results


@mcp.tool()
def screen_content(text: str, policy: str = "") -> dict:
    """Screen text against a content policy.

    Returns decision (safe | needs_review | blocked) with confidence and
    per-decision probabilities.
    """
    eng = get_engine()
    prompt = (
        "Classify this content against the policy. "
        f"Policy: {policy or 'general safety: no disallowed, harmful, or deceptive content'}. "
        "Decide: safe, needs_review, or blocked."
    )
    out = eng.systemone(
        text,
        [
            {
                "name": "screen",
                "type": "choice",
                "options": ["safe", "needs_review", "blocked"],
                "prompt": prompt,
            }
        ],
    )
    a = out["screen"]
    return {
        "decision": a["choice"],
        "confidence": round(a["confidence"], 4),
        "probabilities": {k: round(v, 4) for k, v in a["probabilities"].items()},
    }


@mcp.tool()
def rank_candidates(query: str, candidates: List[str]) -> List[dict]:
    """Score and order candidates by relevance to the query.

    Returns candidates sorted by relevance probability (desc), each with
    its score. All candidates are evaluated in one batched pass.
    """
    eng = get_engine()
    questions = [
        {
            "name": f"cand_{i}",
            "type": "noul",
            "statement": f"Is this candidate relevant to the query: {query!r}? Candidate: {c}",
        }
        for i, c in enumerate(candidates)
    ]
    out = eng.systemone("Rank by relevance.", questions)
    ranked = [
        {
            "candidate": c,
            "relevance": round(out[f"cand_{i}"]["probability"], 4),
        }
        for i, c in enumerate(candidates)
    ]
    ranked.sort(key=lambda r: r["relevance"], reverse=True)
    return ranked


# -- systemone_* shim tools: thin HTTP clients over the live shim ----------

_ERROR_KEY = "error"


def _shim_error(exc: ShimError) -> Dict[str, Any]:
    return {_ERROR_KEY: str(exc)}


def _route_impl(
    task: str,
    cost_bias: str = "balanced",
    tiers: List[str] | None = None,
) -> Dict[str, Any]:
    try:
        payload = get_client().route(task, cost_bias=cost_bias, tiers=tiers)
    except ShimError as exc:
        return _shim_error(exc)
    return payload.get("route", {})


@mcp.tool()
def systemone_route(
    task: str,
    cost_bias: str = "balanced",
    tiers: List[str] | None = None,
) -> Dict[str, Any]:
    """Route a task to the cheapest sufficient model tier (via the live shim).

    Returns the route dict: tier, confidence, margin, calibrated
    probabilities, rationale, model_id, effort, and the uncertain flag.
    cost_bias is one of economy | balanced | quality. No model is loaded by
    this tool — the shim owns its registry.
    """
    return _route_impl(task, cost_bias=cost_bias, tiers=tiers)


def _decide_impl(
    state: Any,
    instructions: str,
    criteria: Any = None,
    type: str = "choice",
) -> Dict[str, Any]:
    try:
        return get_client().decide(
            state, instructions, criteria=criteria, type=type
        )
    except ShimError as exc:
        return _shim_error(exc)


@mcp.tool()
def systemone_decide(
    state: Any,
    instructions: str,
    criteria: Any = None,
    type: str = "choice",
) -> Dict[str, Any]:
    """One typed decision via the live shim's decide endpoint.

    type: "choice" | "noul" | "score". criteria: choice -> {label: desc},
    noul -> {"yes": ..., "no": ...} or {"true": ..., "false": ...}
    (optional), score -> ordered list of level descriptions. Returns
    label/level + probabilities/distribution + confidence + backend
    ("decider" sidecar or "fallback"). No model is loaded by this tool.
    """
    return _decide_impl(state, instructions, criteria=criteria, type=type)


def _status_impl() -> Dict[str, Any]:
    try:
        return get_client().status()
    except ShimError as exc:  # status() is defensive; belt-and-braces
        return _shim_error(exc)


@mcp.tool()
def systemone_status() -> Dict[str, Any]:
    """Shim + sidecar health: liveness, engine model, decision backend in use."""
    return _status_impl()


def _rank_plans_impl(
    task: str, plans: List[Dict[str, Any]]
) -> Dict[str, Any]:
    try:
        return get_client().rank_plans(task, plans)
    except ShimError as exc:
        return _shim_error(exc)


@mcp.tool()
def systemone_rank_plans(
    task: str, plans: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Order plans by P(success | task), with the routed tier's cost penalty.

    plans: [{"id": ..., "text": ...}, ...]. Returns the shim's ranking
    (ordered best-first) plus the tier it was scored under.
    """
    return _rank_plans_impl(task, plans)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
