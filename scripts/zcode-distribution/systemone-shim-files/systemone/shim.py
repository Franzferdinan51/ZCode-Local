"""Local drop-in for TypeSafe's hosted `/v1/systemone` endpoint.

Ryan's `jev-ultrafast` and `mobile-jev` agents POST Jev-shaped bodies to
`https://api.typesafe.ai/v1/systemone` with an API key. This module serves
the same request/response dialect from a local GLiClass engine — no key,
no cloud, no per-call cost.

Run:
    python -m systemone.shim [--port 8765]
    # env: SYSTEMONE_MODEL, SYSTEMONE_DEVICE

Then point the agent at it. In jev-ultrafast's `model.py`, the only change
is the endpoint passed to `post_json`:

    # before
    post_json("https://api.typesafe.ai/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    # after
    post_json("http://127.0.0.1:8765/v1/systemone", "local", body)

Endpoints:
    POST /v1/systemone        TypeSafe dialect (see below); also accepts
                              SGLang's options-list question shape and
                              yes_no questions, plus list-form questions;
                              Clef's images/videos/media_kwargs, score
                              legends, the noul key, and zero-token usage
    POST /v1/decisions        SGLang's /v1/decisions dialect (choice / score /
                              yes_no, label_mass), served by the local engine
                              — one call, N typed questions, single batched
                              pass (see below)
    POST /v1/decide           JEV System 1 dialect (kind / state / question /
                              options, images in state); native on the jev
                              engine, text projection elsewhere (see below)
    GET  /v1/decide/info      option limit, kinds, image support, backend
    GET  /metrics            Prometheus request counters + latency sums
    POST /v1/systemone/route  model router: pick the cheapest sufficient
                              local tier for a task (see below)
    POST /v1/systemone/rank-plans
                              rank candidate plans for a task (see below);
                              consults the Jeff-1 sidecar when enabled and
                              blends its P(plan succeeds | task) 50/50 with
                              the GLiClass scores (fail-open)
    POST /v1/systemone/decide
                              typed decision: {"state", "instructions",
                              "criteria", "type"} — proxy to the decision
                              sidecar when reachable (backend "decider"),
                              else answer locally with the GLiClass engine
                              (backend "fallback", fail-open)
    POST /v1/systemone/permute
                              permutation-robustness probe (Kev-style):
                              re-run one choice question under n_perm
                              option orders; reports per-order answers,
                              argmax stability, and per-option spread
    POST /v1/systemone/batch  judge up to 32 TypeSafe bodies in one call;
                              per-item {"status", ...} results (Cohere
                              Classify-style bulk judging)
    GET  /healthz, /           liveness

Request body for /v1/systemone (TypeSafe dialect):
    {"model": "jev-latest",
     "state": {"page": {"url","title","text"}, "elements": [...],
               "recent_actions": [...]} | "plain text...",
     "questions": {"operation": {"type": "choice",
                                "criteria": {"CLICK": "Click ...", ...},
                                "instructions": {"goal": ..., "rules": ...}},
                   "click_target": {"type": "choice",
                                   "criteria": {"1": {"element": "[1] ...",
                                                      "current_value": ...}, ...},
                                   "instructions": ...}}}

Response:
    {"answers": {"operation": {"type": "choice", "choice": "CLICK",
                              "probabilities": {"CLICK": 0.7, ...},
                              "confidence": 0.7}, ...},
     "model": "<local checkpoint>", "usage": {}}

Request body for /v1/systemone/route:
    {"task": "triage this support ticket for urgency",
     "cost_bias": "economy" | "balanced" | "quality",   # optional, default balanced
     "tiers": ["edge", "base"],                          # optional subset
     "registry": {"edge": {"model_id": ..., "description": ...}, ...}}
        # optional; defaults to the bundled model_registry.json.
        # Tiers with model_id=null are not routable.

Response:
    {"route": {"model_id": "knowledgator/gliclass-edge-v3.0",
               "tier": "edge",
               "rationale": "Task '...' — 'edge' (...) is the cheapest tier
                             rated sufficient under the 'balanced' policy
                             (confidence 0.81).",
               "confidence": 0.81,
               "probabilities": {"edge": 0.81, "base": 0.19},
               "cost_bias": "balanced"},
     "model": "<local checkpoint>", "usage": {},
     "latency_ms": 123.4}

The routing judgment itself is made by the already-loaded local engine
(one batched choice call); the registry is only ever a *catalog* — the
router never loads the routed models. Tiers are capability/cost metadata,
mirroring the Loki autorouter pattern (see examples/demo_autorouter.py).

Every POST decision (both endpoints) is appended as one JSON line to
logs/systemone-shim.log (rotating, 1MB x 4): {"ts", "endpoint",
"latency_ms", "status", "model", ...}. Set SYSTEMONE_LOG_FILE to override
the path (tests use this) or SYSTEMONE_LOG_DISABLE=1 to silence.

Only one local model is ever loaded, same as the rest of the package.
"""

from __future__ import annotations

import argparse
from collections import deque
import datetime
import json
import logging
import os
import random
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .patterns import (
    MAX_BATCH_ITEMS_PER_REQUEST,
    MAX_PLANS_PER_REQUEST,
    MAX_QUESTIONS_PER_REQUEST,
    MAX_STATE_CHARS,
    BodyTooLarge,
    api_token_ok,
    check_body_length,
    date_facts,
    scrub_pii,
    validate_choice,
)


def maybe_date_facts(state_text: str) -> str:
    """Append Kev-style date_facts when SYSTEMONE_DATE_FACTS=1.

    Opt-in preprocessing (ported from Kev's with_date_facts): when the
    state mentions two or more absolute dates, a "date_facts: ..." line
    with the pairwise day counts is appended so the judge can use
    stated day counts instead of subtracting dates itself.
    """
    if os.environ.get("SYSTEMONE_DATE_FACTS") != "1":
        return state_text
    facts = date_facts(state_text)
    if not facts:
        return state_text
    return f"{state_text}\n\ndate_facts: {facts}"


def maybe_scrub_pii(state_text: str) -> tuple:
    """Redact identifiers when SYSTEMONE_SCRUB_PII=1 -> (text, kinds).

    Returns the input untouched (and no kinds) when the env var is off,
    so the hot path pays one getenv per request and nothing else.
    """
    if os.environ.get("SYSTEMONE_SCRUB_PII") != "1":
        return state_text, []
    return scrub_pii(state_text)

try:
    from .api import SystemOne
except ImportError:  # slim install (no torch/gliclass): SGLang-engine or
    # injected-engine mode only; create_engine raises a helpful error for
    # local-engine construction.
    SystemOne = None  # type: ignore[assignment,misc]


from .jev_backend import JevDecideBackend
from .jevk5_backend import JevK5ServerBackend
from .kev_backend import KevBackend
from .clef_backend import ClefBackend
from .rerank_backend import OnnxCrossEncoder, RerankBackend
from .sglang_backend import HybridBackend, SGLangBackend
from .scoring import (
    apply_calibration,
    apply_inventory,
    disabled as scoring_disabled,
    fetch_lmstudio_models,
    load_calibration,
    load_tool_registry,
    model_top_n,
    rank_models,
    rank_plans,
    score_tools,
    start_inventory_refresher,
    tool_floor,
    tool_topk,
)
from .jeff1 import (
    blend_rankings,
    decide_via_jeff1,
    jeff1_enabled,
    rank_plans_via_jeff1,
    second_opinion as jeff1_second_opinion,
)
from .calibration import load_type_calibration
from .metrics import summarize as summarize_metrics

# -- model router -----------------------------------------------------------

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "model_registry.json")
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
TOOL_REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "tool_registry.json")
OPENAPI_PATH = os.path.join(os.path.dirname(__file__), "openapi.json")

COST_BIAS_POLICIES = {
    "economy": "Aggressively prefer the cheapest tier that is still sufficiently capable.",
    "balanced": "Prefer lower cost when capability is sufficient; pay more only for material task needs.",
    "quality": "Prefer capability and reliability, using cost as the tie-breaker among sufficient tiers.",
}


def load_registry(path: str | None = None) -> Dict[str, Dict[str, Any]]:
    """Load the tier registry; returns {tier_name: tier_entry}."""
    with open(path or REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    tiers = data.get("tiers", data)
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("model registry must define a non-empty 'tiers' mapping")
    # Pre-seed availability flags so the background inventory refresher only
    # ever rebinds existing keys (atomic under the GIL) instead of growing
    # model dicts while request threads iterate them (RuntimeError).
    for entry in tiers.values():
        if not isinstance(entry, dict):
            continue
        for m in entry.get("models", []) or []:
            if isinstance(m, dict) and m.get("model_id"):
                m.setdefault("available", False)
    return tiers


def candidates_from_registry(
    registry: Dict[str, Dict[str, Any]],
    tiers: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Filter the registry to routable candidates.

    A tier is routable when its entry is a dict with a non-empty string
    `model_id`. Tiers with `model_id: null` (unconfigured, e.g. the 35B
    tier before the user fills in their checkpoint) are skipped.
    Raises ValueError on unknown tier names or zero candidates.
    """
    reg = registry.get("tiers", registry)  # accept both file shape and bare mapping
    if not isinstance(reg, dict):
        raise ValueError("registry must map tier names to tier entries")
    names = list(tiers) if tiers else list(reg.keys())
    unknown = [t for t in names if t not in reg]
    if unknown:
        raise ValueError(f"unknown tier(s): {', '.join(unknown)}")
    candidates: List[Dict[str, str]] = []
    skipped: List[str] = []
    for name in names:
        entry = reg[name]
        if not isinstance(entry, dict):
            raise ValueError(f"registry tier {name!r} must be a mapping")
        model_id = entry.get("model_id")
        if not model_id or not isinstance(model_id, str):
            skipped.append(name)
            continue
        candidates.append(
            {
                "tier": name,
                "model_id": model_id,
                "description": str(entry.get("description", "")),
            }
        )
    if not candidates:
        raise ValueError(
            "no routable tiers: every requested tier has model_id=null "
            "(configure the tier in model_registry.json or pass a registry "
            "with real model ids)"
            + (f"; skipped: {', '.join(skipped)}" if skipped else "")
        )
    return candidates


def build_route_question(
    task: str, candidates: List[Dict[str, str]], cost_bias: str
) -> Dict[str, Any]:
    """A single choice question: which tier should handle this task?"""
    lines = [
        COST_BIAS_POLICIES[cost_bias],
        "",
        "Task:",
        task,
        "",
        "Candidate tiers — choose the cheapest tier sufficient for the task:",
    ]
    for c in candidates:
        lines.append(f"- {c['tier']}: {c['model_id']} — {c['description']}")
    return {
        "name": "route",
        "type": "choice",
        "options": [c["tier"] for c in candidates],
        "prompt": "\n".join(lines),
    }


# ---------------------------------------------------------------------------
# Hybrid routing policy (deterministic signals + GLiClass judge)
# ---------------------------------------------------------------------------
#
# The zero-shot choice head is a weak signal for capability-tier routing: on
# the bundled tiers it returns near-uniform probabilities (~0.44-0.47) no
# matter how the tier descriptions are worded. So the deterministic layer
# below carries the decision, and the classifier acts as a cheap Jev-style
# second opinion that can only *raise* the tier when it is confident
# (>= 0.65). When the two judges confidently disagree, routing confidence
# drops; below 0.60 the router escalates one tier rather than risk
# under-provisioning the task.

_TIER_ORDER = ("economy", "balanced", "heavy")
_ESCALATION_THRESHOLD = 0.60
_CLASSIFIER_RAISE_CONFIDENCE = 0.65
_DISAGREEMENT_PENALTY = 0.12
_LONG_INPUT_CHARS = 2000

# Coarse reasoning-effort hint derived from the routed tier, consumed by
# downstream agent loops (ZCode CLI) to size reasoningLevel/maxOutputTokens.
# economy -> low, balanced -> medium, heavy -> high.
_TIER_EFFORT = {"economy": "low", "balanced": "medium", "heavy": "high"}

# Deterministic keyword labels for the routed task. Deliberately short and
# documented: downstream tool routing matches on these labels, not free text.
# (keyword-group, label) pairs; groups are independent so a task can carry
# several labels.
_TASK_LABEL_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("calendar", "meeting", "schedule", "appointment"), "calendar"),
    (("email", "inbox", "gmail"), "email"),
    (("debug", "bug", "stack trace", "race condition", "refactor"), "code"),
    (("summar", "tldr", "recap"), "summarize"),
    (("write", "draft", "compose"), "writing"),
    (("file", "folder", "directory"), "files"),
    (("search", "find", "lookup"), "search"),
    (("image", "photo", "picture", "video"), "media"),
    (("deploy", "server", "docker"), "ops"),
)


def task_labels_for(task: str) -> list:
    """Small deterministic keyword labels for *task* (case-insensitive)."""
    lowered = (task or "").lower()
    return [
        label
        for keywords, label in _TASK_LABEL_KEYWORDS
        if any(keyword in lowered for keyword in keywords)
    ]

# Obvious "this needs the strong model" markers (regexes over lowercased text).
_HEAVY_PATTERNS = [
    r"debug(ging|ger)?s?\b",
    r"deadlock",
    r"\brac(e condition|ing)\b",
    r"thread[- ]?safe\b",
    r"stack ?trace",
    r"traceback",
    r"segfault",
    r"memory leak",
    r"multithread",
    r"concurren\w*",
    r"distributed",
    r"\bcrdt\b",
    r"refactor",
    r"restructur\w*",
    r"\b\d+\s*-line\b",
    r"architect(ure)?",
    r"theorem",
    r"\bproof\b",
    r"\bprov(e|ing)\b",
    r"calculus",
    r"\bintegral\b",
    r"\bdifferential\b",
    r"\bdy/dx\b",
    r"linear algebra",
    r"cryptograph",
    r"compiler",
    r"\bkernel\b",
    r"contract",
    r"\blegal\b",
    r"lawsuit",
    r"compliance",
    r"medical",
    r"diagnos(is|ed|ing|tic)\b",
    r"financial (report|filing)",
    r"annual report",
    r"\b\d+\s*-page\b",
    r"\bsolvency\b",
    r"security audit",
    r"\baudit\b.*\bsecur",
    r"\bsecur\w* flaw",
    r"vulnerab",
    r"\bexploit\b",
    r"penetration test",
    r"\bai agent\b",
    r"tool[- ]?use\b",
    r"\btool[- ]?call\w*\b",
    r"\bmcp\b",
    r"\bnavigat\w*\b",
    r"\bscrol\w*\b",
    r"\bbrowser (automation|navigat\w*|tool\w*|agent\w*)\b",
    r"page through",
    r"\bautomat\w* this\b",
    r"system design",
    r"design a (system|distributed)",
    r"roadmap",
    r"performance tun",
]

# Obvious "the tiny model is plenty" markers. The summariz pattern carries a
# negative lookahead: extractive/brevity-scoped summarization is trivial,
# but analytical summarization ("causes of", "compare", ...) is not.
_ECONOMY_PATTERNS = [
    r"summariz\w*\b(?!.*\b(causes|compare|contrast|versus|pros and cons|explain why)\b)",
    r"\bsummary\b",
    r"tl;?dr\b",
    r"one[- ]sentence",
    r"\bextract\b",
    r"classif(y|ication)",
    r"rewrite",
    r"translat",
    r"spell(ing| ?check)?\b",
    r"\bgrammar\b",
    r"proofread",
    r"\bhaiku\b",
    r"\bpoem\b",
    r"capital of",
    r"bullet points",
]

# Ultra-trivial Q&A: bare arithmetic and short factual questions. Single-lookup
# tasks where the cheapest tier is plenty. Kept separate from _ECONOMY_PATTERNS
# because the short-question rule also needs a length + shape check (below).
# Single-lookup shapes only — the imperative forms ("Name the capital...",
# "Multiply 17 by 23.", "Convert 3 km to miles.") are just as trivial as the
# interrogative ones. The short-question rule below excludes open-ended
# advice/explanation questions (see _NONTRIVIAL_QUESTION_MARKERS).
_TRIVIAL_ARITHMETIC_PATTERNS = [
    r"\d+\s*[+\-*/^]\s*\d+",  # bare arithmetic expression: 2+2, 3 * 4
    r"\bwhat is [\d][\d\s+\-*/().^%]*\??",  # "what is 2+2?"
    r"\bcalculat\w*\b",
    r"\bcompute\b",
    r"\b(multiply|divide)\b",
    r"\b(find|compute|calculate)\b.*\bpercent of\b",
    r"\bconvert\b.*\b(miles|kilometers|km|ounces|pounds|kg|grams|celsius|fahrenheit|inches|feet|meters)\b",
    r"\bhow much is\b",
    r"\bhow many\b",
    r"^name the\b",
]

# Markers that disqualify a short "What/Who/...?" from the trivial rule: the
# question asks for advice, recommendations, comparison, or an explanation —
# not a single lookup fact.
_NONTRIVIAL_QUESTION_MARKERS = [
    r"\bstrategies\b",
    r"\badvice\b",
    r"\btips\b",
    r"\bshould i\b",
    r"\bpros and cons\b",
    r"\badvantages and disadvantages\b",
    r"\bwhat makes\b",
    r"\bwhat happens when\b",
    r"\bmust-see\b",
    r"\bgifts?\b",
    r"\bwhat gear\b",
]

# Short factual questions ("What/Who/When/Where/Which ...?") under this length
# are single-lookup tasks -> economy. Heavy patterns still win on conflict
# (substance over form), and the classifier can only raise from here.
_TRIVIAL_QUESTION_MAX_CHARS = 140
_TRIVIAL_QUESTION_STARTERS = ("what", "who", "when", "where", "which")


def analyze_task(task: str) -> Dict[str, Any]:
    """Deterministic complexity analysis: map task text to a suggested tier.

    Returns a dict with the suggested tier name ("economy" | "balanced" |
    "heavy"), a confidence in [0, 1], human-readable reasons, and whether any
    real signal fired (vs. the default middle-tier guess).
    """
    text = (task or "").lower()

    def _hits(patterns: Sequence[str]) -> List[str]:
        found: List[str] = []
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                found.append(match.group(0))
        return found

    heavy_hits = _hits(_HEAVY_PATTERNS)
    economy_hits = _hits(_ECONOMY_PATTERNS)
    if len(task or "") > _LONG_INPUT_CHARS:
        heavy_hits.append(f"long input (>{_LONG_INPUT_CHARS} chars)")

    # Ultra-trivial Q&A shapes: bare arithmetic + short factual questions.
    # Heavy patterns still win on conflict (substance over form), and the
    # classifier can only raise from here. Advice/explanation/recommendation
    # questions are NOT single lookups, even when short and What-led.
    trivial_hits = _hits(_TRIVIAL_ARITHMETIC_PATTERNS)
    stripped = (task or "").strip()
    words = stripped.split()
    if (
        len(stripped) <= _TRIVIAL_QUESTION_MAX_CHARS
        and stripped.endswith("?")
        and words
        and words[0].lower().rstrip(",") in _TRIVIAL_QUESTION_STARTERS
        and not _hits(_NONTRIVIAL_QUESTION_MARKERS)
    ):
        trivial_hits.append(f"short {words[0].lower()}-question")
    if stripped.lower().startswith("define ") and len(stripped) <= _TRIVIAL_QUESTION_MAX_CHARS:
        trivial_hits.append("define-X")
    economy_hits += trivial_hits

    reasons = [f"complexity signal: '{h}'" for h in heavy_hits]
    reasons += [f"trivial-task signal: '{h}'" for h in economy_hits]

    if heavy_hits and economy_hits:
        # Substance wins over form: a legal/medical/technical document that
        # needs summarizing still needs the strong model to get it right.
        reasons.append("conflicting signals; erring toward heavy")
        return {"tier": "heavy", "confidence": 0.55, "reasons": reasons,
                "has_signal": True}
    if len(heavy_hits) >= 2:
        return {"tier": "heavy", "confidence": 0.85, "reasons": reasons,
                "has_signal": True}
    if heavy_hits:
        return {"tier": "heavy", "confidence": 0.62, "reasons": reasons,
                "has_signal": True}
    if economy_hits:
        return {"tier": "economy", "confidence": 0.80, "reasons": reasons,
                "has_signal": True}
    reasons.append("no strong complexity signals; default middle tier")
    return {"tier": "balanced", "confidence": 0.68, "reasons": reasons,
            "has_signal": False}


def parse_route_body(
    body: Dict[str, Any], default_registry: Dict[str, Dict[str, Any]]
) -> tuple[str, str, List[Dict[str, str]]]:
    """Validate a /v1/systemone/route body -> (task, cost_bias, candidates)."""
    task = body.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("request must include a non-empty 'task' string")
    cost_bias = body.get("cost_bias", "balanced")
    if cost_bias not in COST_BIAS_POLICIES:
        raise ValueError(
            f"unknown cost_bias {cost_bias!r}; "
            f"expected one of: {', '.join(COST_BIAS_POLICIES)}"
        )
    registry = body.get("registry") or default_registry
    tiers = body.get("tiers")
    if tiers is not None and (
        not isinstance(tiers, list) or not all(isinstance(t, str) for t in tiers)
    ):
        raise ValueError("'tiers' must be a list of tier-name strings")
    candidates = candidates_from_registry(registry, tiers)
    return task.strip(), cost_bias, candidates


_ROUTE_SORTS = ("utility", "quality", "cost", "latency")


def route_decision(
    engine: Any,
    task: str,
    candidates: List[Dict[str, str]],
    cost_bias: str,
    scoring: Optional[Dict[str, Any]] = None,
    sort: str = "utility",
    fallbacks: int = 2,
    explore: float = 0.0,
    rng: Any = None,
) -> Dict[str, Any]:
    """Pick the cheapest sufficient tier for *task*.

    Hybrid policy:
      1. Deterministic complexity analysis sets the base tier (and a floor
         when real signals fired).
      2. The GLiClass choice head is a second opinion: it may *raise* the
         tier when confident (>= 0.65), never lower it.
      3. cost_bias nudges one tier toward cheap ("economy") or capable
         ("quality"); "economy" never drops below the deterministic floor.
      4. If final confidence is below 0.60, escalate one tier toward
         capability rather than risk under-provisioning.

    Candidate order is capability order (cheapest first); the bundled
    registry lists tiers economy -> balanced -> heavy.

    When `scoring` is provided ({"calibration": ...|None, "tools": [...],
    "registry": {...}}), the route dict additionally gains the calibrated
    decision surface: calibrated_probabilities, margin, uncertain,
    ranked_models, ranked_tools/tool_scoring. confidence becomes the
    (calibrated) probability of the routed tier. Without `scoring` the
    legacy shape is returned unchanged (backward compatible).

    Returns the {"model_id", "tier", "rationale", "confidence",
    "probabilities", "cost_bias", "deterministic_tier", "signals", "effort",
    "task_labels", ...} route dict.

    OpenRouter/LiteLLM-style per-request controls (additive; defaults
    preserve the legacy route exactly):

    - sort: ranked_models order — "utility" (default), "quality",
      "cost", or "latency".
    - fallbacks: how many ranked model ids (excluding the winner) ride
      along in route["fallbacks"] for ordered failover. Needs `scoring`
      (the ranked list); 0 disables.
    - explore: epsilon-greedy exploration rate in [0, 1] (default 0 =
      pure exploitation). With probability `explore` the winner is
      replaced by a uniform pick among the top-3 ranked models and
      route["explored"] is set; effort still follows the routed tier,
      not the exploration pick. `rng` (default the random module)
      supplies .random()/.choice() — pass random.Random(seed) for
      deterministic tests.
    """
    if sort not in _ROUTE_SORTS:
        raise ValueError(
            f"unknown sort {sort!r}; want {'|'.join(_ROUTE_SORTS)}")
    if isinstance(fallbacks, bool) or not isinstance(fallbacks, int):
        raise ValueError(f"'fallbacks' must be an int, got {fallbacks!r}")
    if not 0 <= fallbacks <= 8:
        raise ValueError(f"'fallbacks' must be 0..8, got {fallbacks!r}")
    if isinstance(explore, bool) or not isinstance(explore, (int, float)):
        raise ValueError(f"'explore' must be a number, got {explore!r}")
    if not 0.0 <= float(explore) <= 1.0:
        raise ValueError(f"'explore' must be in [0, 1], got {explore!r}")
    explore = float(explore)
    # The registry is a name->entry mapping with no guaranteed key order;
    # sort candidates cheapest-first so the index math below is sound.
    _order = {t: i for i, t in enumerate(_TIER_ORDER)}
    candidates = sorted(candidates, key=lambda c: _order.get(c["tier"], 99))
    tiers = [c["tier"] for c in candidates]
    n = len(tiers)
    if n == 0:
        raise ValueError("no candidate tiers to route over")

    det = analyze_task(task)
    det_rank = _TIER_ORDER.index(det["tier"])
    det_idx = {0: 0, 1: n // 2, 2: n - 1}[det_rank]

    question = build_route_question(task, candidates, cost_bias)
    answers = engine.systemone(task, [question])
    answer = validate_choice(answers["route"], question["options"])
    clf_probs = {tier: float(prob) for tier, prob in answer["probabilities"].items()}
    clf_tier = answer["choice"]
    clf_conf = float(answer["confidence"])
    clf_idx = tiers.index(clf_tier)

    idx = det_idx
    notes = list(det["reasons"])
    if clf_conf >= _CLASSIFIER_RAISE_CONFIDENCE and clf_idx > idx:
        idx = clf_idx
        conf = clf_conf
        notes.append(
            f"classifier raised tier to '{clf_tier}' (confidence {clf_conf:.2f})"
        )
    elif clf_conf >= _CLASSIFIER_RAISE_CONFIDENCE and clf_idx < det_idx:
        conf = max(0.0, det["confidence"] - _DISAGREEMENT_PENALTY)
        notes.append(
            f"classifier disagreed downward ('{clf_tier}', {clf_conf:.2f}); "
            "confidence reduced"
        )
    else:
        conf = det["confidence"] + (0.05 if clf_idx == idx else 0.0)
        conf = min(0.95, conf)
        notes.append(f"classifier chose '{clf_tier}' (confidence {clf_conf:.2f})")

    floor = det_idx if det["has_signal"] else 0
    if cost_bias == "economy":
        new_idx = max(floor, idx - 1)
        if new_idx != idx:
            notes.append(f"'economy' bias shifted tier down to '{tiers[new_idx]}'")
        idx = new_idx
    elif cost_bias == "quality":
        new_idx = min(n - 1, idx + 1)
        if new_idx != idx:
            notes.append(f"'quality' bias shifted tier up to '{tiers[new_idx]}'")
        idx = new_idx

    if conf < _ESCALATION_THRESHOLD and idx < n - 1:
        idx += 1
        conf = _ESCALATION_THRESHOLD
        notes.append("low routing confidence; escalated one tier toward capability")

    # Blended probability distribution for the response: deterministic
    # one-hot (smoothed) carries 0.65, the classifier's head 0.35.
    det_dist = {
        tier: (0.70 if i == det_idx else (0.30 / (n - 1) if n > 1 else 0.0))
        for i, tier in enumerate(tiers)
    }
    blended = {
        tier: 0.65 * det_dist[tier] + 0.35 * clf_probs.get(tier, 0.0)
        for tier in tiers
    }
    total = sum(blended.values()) or 1.0
    blended = {tier: prob / total for tier, prob in blended.items()}

    winner = candidates[idx]
    task_snip = task if len(task) <= 80 else task[:77] + "..."
    rationale = (
        f"Task '{task_snip}' -> '{winner['tier']}' ({winner['model_id']}): "
        + "; ".join(notes)
        + f" (final confidence {conf:.2f})."
    )
    effort = _TIER_EFFORT.get(winner["tier"], "medium")
    route: Dict[str, Any] = {
        "model_id": winner["model_id"],
        "tier": winner["tier"],
        "rationale": rationale,
        "confidence": round(conf, 4),
        "probabilities": blended,
        "cost_bias": cost_bias,
        "deterministic_tier": det["tier"],
        "signals": det["reasons"],
        # Effort hint for agent loops: coarse reasoning budget for this task.
        "effort": effort,
        # Deterministic keyword labels for tool routing (see _TASK_LABEL_KEYWORDS).
        "task_labels": task_labels_for(task),
    }
    if scoring is not None:
        _apply_scoring(engine, task, route, blended,
                       {**scoring, "sort": sort})
        ranked = route.get("ranked_models") or []
        if fallbacks:
            route["fallbacks"] = [
                m["model_id"] for m in ranked
                if isinstance(m, dict)
                and m.get("model_id") != route["model_id"]
            ][:fallbacks]
        else:
            route["fallbacks"] = []
        route["explored"] = False
        if explore > 0 and ranked:
            r = rng if rng is not None else random
            if r.random() < explore:
                pool = [m for m in ranked[:3]
                        if isinstance(m, dict) and m.get("model_id")]
                if pool:
                    pick = r.choice(pool)
                    if pick["model_id"] != route["model_id"]:
                        route["rationale"] += (
                            f" Exploration roll (p={explore:.2f}) picked "
                            f"'{pick['model_id']}' over '{route['model_id']}'."
                        )
                        route["model_id"] = pick["model_id"]
                        if pick.get("tier"):
                            route["tier"] = pick["tier"]
                        route["explored"] = True
                    else:
                        route["rationale"] += (
                            f" Exploration roll (p={explore:.2f}) kept "
                            f"'{route['model_id']}'."
                        )
    return route


_EFFORT_LEVELS = ("low", "medium", "high")


def _apply_scoring(
    engine: Any,
    task: str,
    route: Dict[str, Any],
    blended: Dict[str, float],
    scoring: Dict[str, Any],
) -> None:
    """Additive decision surface; mutates `route`. Never raises.

    - calibrated_probabilities / margin / uncertain / calibrated; confidence
      becomes the (calibrated) probability of the routed tier — not the
      distribution top-1, which can differ when rules raise/escalate.
    - uncertain -> effort bumps one level (low->medium->high).
    - ranked_models: top-3 available by expected utility (advisory).
    - ranked_tools: relevance-ranked tools, top-k with relevance >= floor;
      skipped (cheap path) when uncertain or effort is low.
    """
    try:
        cal = apply_calibration(blended, scoring.get("calibration"))
        route["calibrated"] = cal["calibrated"]
        route["calibrated_probabilities"] = cal["calibrated_probabilities"]
        route["margin"] = cal["margin"]
        route["uncertain"] = cal["uncertain"]
        try:
            winner_p = float(
                (cal["calibrated_probabilities"] or {}).get(
                    route.get("tier"), cal["confidence"]))
        except (TypeError, ValueError):
            winner_p = cal["confidence"]
        route["confidence"] = round(winner_p, 4)

        effort = route.get("effort", "medium")
        if cal["uncertain"] and effort in _EFFORT_LEVELS:
            bumped = _EFFORT_LEVELS[min(2, _EFFORT_LEVELS.index(effort) + 1)]
            if bumped != effort:
                route["effort"] = bumped
                route["rationale"] += (
                    f" Uncertain (margin {cal['margin']:.2f} < floor); "
                    f"effort bumped to {bumped}, no tool pruning."
                )

        # Jeff-1 second head (uncertain routes only): record an advisory
        # tier second opinion on the route. Never changes the routed tier.
        if cal["uncertain"] and jeff1_enabled():
            opinion = _jeff1_second_opinion(task, route, cal, scoring)
            if opinion is not None:
                route["jeff1_second_opinion"] = opinion
                if not opinion.get("agree", True):
                    route["rationale"] += (
                        " Jeff-1 second opinion disagrees (advisory; tier "
                        f"unchanged): {opinion.get('rationale', '')}"
                    )

        registry = scoring.get("registry") or {}
        try:
            route["ranked_models"] = rank_models(
                registry, cal["calibrated_probabilities"],
                topn=model_top_n(),
                sort=scoring.get("sort") or "utility")
        except Exception:
            route["ranked_models"] = []

        # Cheap path: don't burn an engine call deciding tools for a task
        # we're unsure about or that needs barely any reasoning.
        if cal["uncertain"] or route.get("effort") == "low":
            route["ranked_tools"] = []
            route["tool_scoring"] = "skipped"
            return
        tools = scoring.get("tools") or []
        try:
            ranked = score_tools(engine, task, tools)
        except Exception:
            ranked = []
        floor, topk = tool_floor(), tool_topk()
        # No floor/topk configured -> unfiltered (fail-open, never prune blind).
        ranked_tools = ranked if floor is None else [
            t for t in ranked if t["relevance"] >= floor
        ]
        route["ranked_tools"] = ranked_tools if topk is None else ranked_tools[:topk]
        route["tool_scoring"] = "full"
    except Exception:
        # Fail open: scoring must never break the route consumers rely on.
        route.setdefault("ranked_models", [])
        route.setdefault("ranked_tools", [])
        route.setdefault("tool_scoring", "skipped")


def _jeff1_second_opinion(
    task: str,
    route: Dict[str, Any],
    cal: Dict[str, Any],
    scoring: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Fetch Jeff-1's advisory tier second opinion; None on any failure.

    Builds the candidate list from the scoring registry and hands the
    uncertain route summary to the sidecar. Pure fail-open: every error
    path returns None so the route stands on the GLiClass judgment alone.
    """
    try:
        registry = scoring.get("registry") or {}
        tiers = registry.get("tiers", registry) \
            if isinstance(registry, dict) else {}
        candidates = [
            {"tier": name, "description": str(entry.get("description", ""))}
            for name, entry in tiers.items()
            if isinstance(name, str) and isinstance(entry, dict)
        ]
        if not candidates:
            return None
        result = jeff1_second_opinion(
            task,
            {
                "tier": route.get("tier"),
                "confidence": cal.get("confidence"),
                "margin": cal.get("margin"),
                "candidates": candidates,
            },
        )
        if not isinstance(result, dict) or not result.get("tier"):
            return None
        return {
            "tier": result["tier"],
            "confidence": result.get("confidence"),
            "agree": bool(result.get("agree")),
            "rationale": str(result.get("rationale", "")),
        }
    except Exception:
        return None


# -- latency logging --------------------------------------------------------

_log_lock = threading.Lock()
_logger: Optional[logging.Logger] = None


def get_logger() -> Optional[logging.Logger]:
    """Rotating JSON-lines decision log. Env: SYSTEMONE_LOG_FILE override,
    SYSTEMONE_LOG_DISABLE=1 to silence."""
    global _logger
    if os.environ.get("SYSTEMONE_LOG_DISABLE") == "1":
        return None
    with _log_lock:
        if _logger is not None:
            return _logger
        path = os.environ.get("SYSTEMONE_LOG_FILE") or os.path.join(
            os.path.dirname(__file__), "logs", "systemone-shim.log"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger = logging.getLogger("systemone.shim.decisions")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        _logger = logger
        return logger


class Metrics:
    """Thread-safe per-endpoint request counters and latency sums.

    One instance lives on the server (``server.metrics``); every request
    records exactly one observation. Renders Prometheus text exposition
    for GET /metrics. Never raises: observation failures are swallowed so
    metrics can never break serving.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: Dict[tuple, int] = {}
        self._latency_ms: Dict[tuple, float] = {}

    def observe(self, endpoint: str, status: int, latency_ms: float) -> None:
        try:
            key = (str(endpoint), int(status))
            with self._lock:
                self._counts[key] = self._counts.get(key, 0) + 1
                self._latency_ms[key] = (
                    self._latency_ms.get(key, 0.0) + float(latency_ms))
        except Exception:
            pass

    def render_prometheus(self) -> str:
        with self._lock:
            items = sorted(self._counts.items())
            lat = dict(self._latency_ms)
        lines = [
            "# HELP systemone_requests_total Requests served by endpoint and status.",
            "# TYPE systemone_requests_total counter",
        ]
        for (endpoint, status), count in items:
            lines.append(
                f'systemone_requests_total{{endpoint="{endpoint}",'
                f'status="{status}"}} {count}')
        lines += [
            "# HELP systemone_request_latency_ms_sum Total request latency by endpoint and status.",
            "# TYPE systemone_request_latency_ms_sum counter",
        ]
        for (endpoint, status), count in items:
            lines.append(
                f'systemone_request_latency_ms_sum{{endpoint="{endpoint}",'
                f'status="{status}"}} {lat.get((endpoint, status), 0.0):.1f}')
        return "\n".join(lines) + "\n"


class DecisionCache:
    """Thread-safe exact-match decision cache (FrugalGPT completion cache).

    Keys are sha256 over the canonical request JSON plus the engine's
    model name, so identical state+questions judge once per TTL window.
    Bounded (LRU eviction) and TTL-expiring; disabled (ttl<=0 or
    max_entries<=0) caches nothing. Hits are served without touching
    the engine. Stats ride along for operators (hits/misses/size).
    """

    def __init__(self, ttl_seconds: float = 0.0, max_entries: int = 512) -> None:
        self.ttl = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._lock = threading.Lock()
        self._entries: Dict[str, tuple] = {}
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.ttl > 0 and self.max_entries > 0

    @staticmethod
    def cache_key(model_name: str, canonical_body: str) -> str:
        import hashlib

        return hashlib.sha256(
            f"{model_name}\n{canonical_body}".encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            hit = self._entries.get(key)
            if hit is None or hit[0] <= now:
                self._entries.pop(key, None)
                self.misses += 1
                return None
            self.hits += 1
            self._entries[key] = self._entries.pop(key)  # MRU refresh
            return json.loads(json.dumps(hit[1]))  # deep copy out

    def put(self, key: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            while len(self._entries) >= self.max_entries:
                self._entries.pop(next(iter(self._entries)))
            self._entries[key] = (
                time.monotonic() + self.ttl,
                json.loads(json.dumps(payload)),  # deep copy in
            )

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"enabled": self.enabled, "hits": self.hits,
                    "misses": self.misses, "size": len(self._entries)}


def decision_cache_from_env() -> DecisionCache:
    """Build the /v1/systemone decision cache from env (default off).

    SYSTEMONE_CACHE_TTL seconds (<=0 disables), SYSTEMONE_CACHE_MAX
    entries (default 512). Unparseable values fail open to disabled.
    """
    try:
        ttl = float(os.environ.get("SYSTEMONE_CACHE_TTL") or "0")
    except (TypeError, ValueError):
        ttl = 0.0
    try:
        maximum = int(os.environ.get("SYSTEMONE_CACHE_MAX") or "512")
    except (TypeError, ValueError):
        maximum = 0
    return DecisionCache(ttl_seconds=ttl, max_entries=maximum)


_audit_lock = threading.Lock()
_audit_seq = 0
_audit_prev = "genesis"


def reset_audit_chain() -> None:
    """Reset the audit chain (tests only; production chains never reset)."""
    global _audit_seq, _audit_prev
    with _audit_lock:
        _audit_seq = 0
        _audit_prev = "genesis"


def log_decision(record: Dict[str, Any]) -> None:
    """Append one JSON decision record; never raises.

    With SYSTEMONE_AUDIT_CHAIN=1 each record also gains audit_seq /
    audit_prev / audit_hash (sha256 over prev-hash + canonical record),
    making the log tamper-evident: verify_audit_chain() replays it.
    """
    try:
        logger = get_logger()
        if logger is None:
            return
        rec = {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        rec.update(record)
        if os.environ.get("SYSTEMONE_AUDIT_CHAIN") == "1":
            import hashlib

            global _audit_seq, _audit_prev
            with _audit_lock:
                _audit_seq += 1
                seq = _audit_seq
                prev = _audit_prev
                body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
                digest = hashlib.sha256(
                    f"{prev}\n{body}".encode("utf-8")).hexdigest()
                _audit_prev = digest
            rec["audit_seq"] = seq
            rec["audit_prev"] = prev
            rec["audit_hash"] = digest
        logger.info(json.dumps(rec))
    except Exception:
        pass  # logging must never break serving


def verify_audit_chain(path: str) -> Dict[str, Any]:
    """Replay a chained decision log; returns {"ok", "records", "error"}.

    Checks sequence continuity, prev-hash linkage, and recomputed
    digests over every line carrying audit_* fields. Lines without
    audit fields are skipped (mixed chained/unchained logs verify the
    chained subsequence). {"ok": False} names the first bad line.
    """
    import hashlib

    checked = 0
    prev: Optional[str] = None  # unknown until the first chained record
    last_seq: Optional[int] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as exc:
        return {"ok": False, "records": 0, "error": f"unreadable: {exc}"}
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            return {"ok": False, "records": checked,
                    "error": f"line {lineno}: not JSON"}
        if not isinstance(rec, dict) or "audit_hash" not in rec:
            continue
        if prev is None:
            # A rotated log starts mid-chain: trust the file's first
            # prev as the linkage root, enforce everything after it.
            prev = rec.get("audit_prev")
        elif rec.get("audit_prev") != prev:
            return {"ok": False, "records": checked,
                    "error": f"line {lineno}: prev-hash linkage broken"}
        body_rec = {k: v for k, v in rec.items()
                    if k not in ("audit_seq", "audit_prev", "audit_hash")}
        body = json.dumps(body_rec, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(
            f"{prev}\n{body}".encode("utf-8")).hexdigest()
        if digest != rec.get("audit_hash"):
            return {"ok": False, "records": checked,
                    "error": f"line {lineno}: digest mismatch (tampered?)"}
        seq = rec.get("audit_seq")
        if isinstance(seq, int):
            if last_seq is not None and seq != last_seq + 1:
                return {"ok": False, "records": checked,
                        "error": f"line {lineno}: sequence gap"}
            last_seq = seq
        prev = rec["audit_hash"]
        checked += 1
    return {"ok": True, "records": checked, "error": ""}


# -- TypeSafe dialect translation (unchanged) --------------------------------

def _render_instructions(instructions: Any) -> str:
    """TypeSafe instructions may be a string or a dict (goal/rules)."""
    if instructions is None:
        return ""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, dict):
        parts = []
        if instructions.get("goal"):
            parts.append(f"Goal: {instructions['goal']}")
        rules = instructions.get("rules")
        if rules:
            if isinstance(rules, (list, tuple)):
                parts.append("Rules:\n" + "\n".join(f"- {r}" for r in rules))
            else:
                parts.append(f"Rules: {rules}")
        for k, v in instructions.items():
            if k not in ("goal", "rules"):
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(instructions)


def _render_criterion(key: str, value: Any) -> str:
    """A criterion may be a plain description or a structured dict."""
    if isinstance(value, str):
        return f"{key}: {value}"
    if isinstance(value, dict):
        bits = [str(value.get("element", key))]
        for field in ("current_value", "value", "role", "checked", "selected", "expanded", "label"):
            if value.get(field) not in (None, ""):
                bits.append(f"{field}={value[field]}")
        return f"{key}: " + " | ".join(bits)
    return f"{key}: {value}"


def translate_question(name: str, q: Dict[str, Any]) -> Dict[str, Any]:
    """TypeSafe question -> systemone question.

    {"type": "choice", "criteria": {opt: desc|{...}}, "instructions": ...}
    becomes {"name", "type": "choice", "options": [...], "prompt": ...}.

    Also accepts SGLang's /v1/systemone question shape (wire-compat):
    {"type": "choice", "question": "...", "options": [{"name": ...}, ...]}
    and {"type": "yes_no", "question": "..."} -> noul. The two shapes are
    distinguished by the presence of "criteria" (TypeSafe) vs a list-valued
    "options" (SGLang).

    instructions is optional (Clef convention): when it is absent the
    question ID stands in, so judges always see what is being decided.
    """
    qtype = q.get("type", "choice")
    if qtype not in ("choice", "score", "noul", "yes_no"):
        raise ValueError(
            f"question {name!r} has unknown type {qtype!r}; "
            "expected one of: choice, score, noul, yes_no")
    if qtype == "yes_no":
        # SGLang boolean question -> local noul; handled before the shape
        # dispatch below since there are no options to speak of.
        statement = (q.get("question")
                     or _render_instructions(q.get("instructions"))
                     or str(q.get("criteria", ""))
                     or name)
        return {"name": name, "type": "noul", "statement": statement}
    criteria = q.get("criteria", {}) or {}
    if "criteria" not in q and isinstance(q.get("options"), list):
        # SGLang shape: options are [{"name": ...}] or bare strings.
        criteria = {}
        for opt in q["options"]:
            if isinstance(opt, dict):
                label, desc = opt.get("name"), opt.get("description")
            else:
                label, desc = opt, None
            if isinstance(label, str) and label and label not in criteria:
                criteria[label] = desc
    if isinstance(criteria, list):
        # TypeSafe score shape: criteria IS the ordered level list.
        criteria = {str(x): None for x in criteria}
    if not isinstance(criteria, dict):
        raise ValueError("question criteria must be an object or a list")
    options = list(criteria.keys())
    if qtype in ("choice", "score") and len(options) < 2:
        raise ValueError(
            f"question {name!r} needs >= 2 options/levels, got {len(options)}")
    sglang_question = str(q.get("question") or "") if "criteria" not in q else ""
    prompt_bits = [
        _render_instructions(q.get("instructions"))
        or sglang_question
        or f"Question '{name}'."
    ]
    if sglang_question and sglang_question not in prompt_bits[0]:
        # SGLang shape carries the prompt as "question".
        prompt_bits.append(sglang_question)
    prompt_bits.append(
        "Options:\n" + "\n".join(_render_criterion(k, v) for k, v in criteria.items())
    )
    prompt = "\n".join(b for b in prompt_bits if b).strip()
    if qtype == "score":
        return {"name": name, "type": "score", "levels": options,
                "prompt": prompt,
                "legend": {lv: (criteria[lv] if isinstance(criteria[lv], str)
                                       and criteria[lv].strip() else lv)
                           for lv in options}}
    if qtype == "noul":
        statement = prompt or str(criteria)
        return {"name": name, "type": "noul", "statement": statement}
    out: Dict[str, Any] = {"name": name, "type": "choice", "options": options,
                           "prompt": prompt}
    descs = {k: v for k, v in criteria.items()
             if isinstance(v, str) and v.strip()}
    if descs:
        # Per-option descriptions for judges that score option text
        # (RerankBackend); GLiClass/SGLang paths ignore this key.
        out["descriptions"] = descs
    return out


def state_to_text(state: Any) -> str:
    """TypeSafe state may be a rich dict or plain text; flatten to text."""
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        parts: List[str] = []
        page = state.get("page") or {}
        if isinstance(page, dict):
            if page.get("url"):
                parts.append(f"URL: {page['url']}")
            if page.get("title"):
                parts.append(f"Title: {page['title']}")
            if page.get("text"):
                parts.append(f"Page text: {page['text']}")
        else:
            parts.append(str(page))
        elements = state.get("elements") or []
        if elements:
            lines = []
            for el in elements:
                if isinstance(el, dict):
                    label = el.get("label", el.get("index", "?"))
                    role = el.get("role", "")
                    lines.append(f"[{el.get('index', '?')}] {label} ({role})".strip())
                else:
                    lines.append(str(el))
            parts.append("Elements:\n" + "\n".join(lines))
        actions = state.get("recent_actions") or state.get("history") or []
        if actions:
            lines = []
            for a in actions[-10:]:
                if isinstance(a, dict):
                    lines.append(
                        f"- {a.get('action', a.get('kind', '?'))}: {a.get('text', '')}".strip()
                    )
                else:
                    lines.append(f"- {a}")
            parts.append("Recent actions:\n" + "\n".join(lines))
        return "\n\n".join(parts)
    return str(state)


def translate_media(body: Dict[str, Any]) -> tuple[List[Any], List[Any]]:
    """Clef media fields -> (images, videos).

    Accepts Clef's top-level "images" / "videos" lists: over HTTP the items
    are image/video URLs or data URLs (strings) or {"image"|"url": ...}
    mappings; in-process callers may also pass frame arrays (lists).
    "media_kwargs", when present, must be a mapping (reserved for
    processor-backed engines; validated here, consumed downstream).
    """
    for key in ("images", "videos"):
        raw = body.get(key, [])
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ValueError(f"'{key}' must be a list")
        for i, item in enumerate(raw):
            if isinstance(item, str) and item.strip():
                continue  # URL / data URL
            if isinstance(item, list):
                continue  # frame array (in-process callers)
            if isinstance(item, dict):
                ref = item.get("image", item.get("url", ""))
                if isinstance(ref, str) and ref.strip():
                    continue
                raise ValueError(
                    f"'{key}[{i}]' mapping needs an 'image'/'url' string")
            raise ValueError(
                f"'{key}[{i}]' must be a URL/data-URL string, "
                "an {'image'|'url': ...} mapping, or a frame array")
    kwargs = body.get("media_kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("'media_kwargs' must be a mapping")
    return list(body.get("images") or []), list(body.get("videos") or [])


def _engine_supports(engine: Any, param: str) -> bool:
    """True when engine.systemone() accepts the `param` keyword."""
    try:
        import inspect

        return param in inspect.signature(engine.systemone).parameters
    except (TypeError, ValueError):
        return False


def translate_body(body: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    """Split a TypeSafe request into (state_text, systemone questions).

    Accepts questions as a dict keyed by id (TypeSafe / SGLang style) or as
    a list of question mappings carrying "id" (or "name").
    """
    state_text = state_to_text(body.get("state", ""))
    if len(state_text) > MAX_STATE_CHARS:
        state_text = state_text[:MAX_STATE_CHARS]
    state_text = maybe_date_facts(state_text)
    raw = body.get("questions") or {}
    if isinstance(raw, list):
        items = [
            (str(q.get("id") or q.get("name") or f"q{i}"), q)
            for i, q in enumerate(raw)
            if isinstance(q, dict)
        ]
    else:
        items = list(raw.items())
    questions = [
        translate_question(name, q)
        for name, q in items
    ]
    if not questions:
        raise ValueError("request must include at least one question")
    if len(questions) > MAX_QUESTIONS_PER_REQUEST:
        raise ValueError(
            f"'questions' exceeds the {MAX_QUESTIONS_PER_REQUEST}-question cap")
    return state_text, questions


def translate_answers(answers: Dict[str, Any]) -> Dict[str, Any]:
    """systemone answers -> TypeSafe {"answers": ...} response body."""
    out: Dict[str, Any] = {}
    for name, ans in answers.items():
        if name == "_meta" or not isinstance(ans, dict):
            continue
        atype = ans.get("type")
        # "x_label_mass" mirrors SGLang's /v1/systemone answers (null:
        # the local engine has no candidate-scoring signal).
        if atype == "choice":
            out[name] = {
                "type": "choice",
                "choice": ans["choice"],
                "probabilities": ans["probabilities"],
                "confidence": ans["confidence"],
                "x_label_mass": ans.get("label_mass"),
            }
        elif atype == "score":
            dist = ans["distribution"]
            out[name] = {
                "type": "score",
                "level": ans["level"],
                "distribution": dist,
                "confidence": ans["confidence"],
                # Clef legend: level -> description (identity when the
                # engine was not given descriptions).
                "legend": dict(ans.get("legend") or {lv: lv for lv in dist}),
                "x_label_mass": ans.get("label_mass"),
            }
        elif atype == "noul":
            out[name] = {
                "type": "noul",
                "probability": ans["probability"],
                # "noul" = P(true): the JevK5/Clef key; jevk5_backend
                # already reads it, so round-trips preserve the value.
                "noul": ans["probability"],
                "answer": ans["answer"],
                "confidence": ans["confidence"],
                "x_label_mass": ans.get("label_mass"),
            }
    return out


# -- SGLang /v1/decisions compatibility --------------------------------------
#
# SGLang (nightly, post-2026-09-29 main) serves POST /v1/decisions: batched
# typed questions (choice / score / yes_no) answered with zero completion
# tokens. This shim answers the same dialect with the local engine, so any
# client written against SGLang works unchanged against this box.
#
# Version status (verified 2026-10-02 against sglang main): the endpoints
# are main-branch only, not in any tagged SGLang release — pin a nightly
# build until a release contains them. Image input is confirmed
# unsupported upstream (input is string | object | array, rendered as
# compact JSON), so image parts are noted and skipped here, and
# SGLangBackend drops images= (reported in _meta["media_dropped"]).


def _option_names(options: Any) -> List[str]:
    """SGLang option items: [{"name": str}] or bare strings -> [str]."""
    names: List[str] = []
    for o in options or []:
        if isinstance(o, dict):
            names.append(str(o.get("name", o)))
        else:
            names.append(str(o))
    return names


class Unprocessable(ValueError):
    """422: the request was well-formed JSON but violates endpoint limits
    (mirrors SGLang's /v1/decisions, which answers 422 on invalid bodies)."""


def decisions_input_to_text(state_input: Any) -> str:
    """SGLang /v1/decisions `input` -> plain text for the local engine.

    Accepts a string, or OpenAI-style content parts
    ({"type": "text", "text": ...}, {"type": "image_url", ...}). Image
    parts are noted and skipped — the local GLiClass engine is text-only,
    and SGLang's decisions docs describe no image path (experimental in
    SGLangBackend; verify against a live nightly server).
    """
    if state_input is None:
        return ""
    if isinstance(state_input, str):
        return state_input
    if isinstance(state_input, list):
        texts = []
        skipped_images = 0
        for part in state_input:
            if not isinstance(part, dict):
                texts.append(str(part))
                continue
            ptype = part.get("type")
            if ptype == "text":
                texts.append(str(part.get("text", "")))
            elif ptype == "image_url":
                skipped_images += 1
            else:
                texts.append(str(part))
        text = "\n".join(t for t in texts if t).strip()
        if skipped_images and not text:
            raise Unprocessable(
                "input contained only image parts; the local engine is "
                "text-only (use SGLangBackend with a VLM for images)"
            )
        return text
    return state_to_text(state_input)


# SGLang's documented /v1/decisions limits; the local endpoint mirrors them
# so clients get the same contract whichever server they point at.
_DECISIONS_MAX_CHOICE_OPTIONS = 26
_DECISIONS_MIN_CHOICE_OPTIONS = 2
_DECISIONS_MAX_SCORE_LEVELS = 10
_DECISIONS_MIN_SCORE_LEVELS = 2


def translate_decisions_body(
    body: Dict[str, Any],
) -> tuple[str, List[Dict[str, Any]], List[str]]:
    """SGLang /v1/decisions body -> (state_text, engine questions, id order).

    {"input": str|parts,
     "questions": [{"id", "type": "choice"|"score"|"yes_no",
                    "question": str,
                    "options": [{"name"}...] | [str...] (choice),
                    "levels": [{"name"}...] | [str...] (score)}]}
    """
    state_text = decisions_input_to_text(body.get("input", ""))
    if len(state_text) > MAX_STATE_CHARS:
        state_text = state_text[:MAX_STATE_CHARS]
    state_text = maybe_date_facts(state_text)
    raw_questions = body.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise Unprocessable("'questions' must be a non-empty list")
    if len(raw_questions) > MAX_QUESTIONS_PER_REQUEST:
        raise Unprocessable(
            f"'questions' exceeds the {MAX_QUESTIONS_PER_REQUEST}-question cap")
    questions: List[Dict[str, Any]] = []
    ids: List[str] = []
    for i, rq in enumerate(raw_questions):
        if not isinstance(rq, dict):
            raise Unprocessable(f"question #{i} must be a mapping")
        qid = str(rq.get("id") or rq.get("name") or f"q{i}")
        if qid in ids:
            raise Unprocessable(f"duplicate question id: {qid!r}")
        ids.append(qid)
        qtype = rq.get("type", "choice")
        prompt = rq.get("question") or ""
        if qtype == "choice":
            options = _option_names(rq.get("options"))
            if not (_DECISIONS_MIN_CHOICE_OPTIONS
                    <= len(options) <= _DECISIONS_MAX_CHOICE_OPTIONS):
                raise Unprocessable(
                    f"choice question {qid!r} has {len(options)} options; "
                    f"expected {_DECISIONS_MIN_CHOICE_OPTIONS}-"
                    f"{_DECISIONS_MAX_CHOICE_OPTIONS}"
                )
            questions.append({"name": qid, "type": "choice",
                              "options": options, "prompt": prompt})
        elif qtype == "score":
            levels = _option_names(rq.get("levels"))
            if not (_DECISIONS_MIN_SCORE_LEVELS
                    <= len(levels) <= _DECISIONS_MAX_SCORE_LEVELS):
                raise Unprocessable(
                    f"score question {qid!r} has {len(levels)} levels; "
                    f"expected {_DECISIONS_MIN_SCORE_LEVELS}-"
                    f"{_DECISIONS_MAX_SCORE_LEVELS}"
                )
            questions.append({"name": qid, "type": "score",
                              "levels": levels, "prompt": prompt})
        elif qtype == "yes_no":
            if not prompt:
                raise Unprocessable(
                    f"yes_no question {qid!r} needs a 'question' string")
            questions.append({"name": qid, "type": "noul", "statement": prompt})
        else:
            raise Unprocessable(
                f"question {qid!r} has unknown type {qtype!r}; "
                "expected one of: choice, score, yes_no")
    return state_text, questions, ids


def translate_decisions_answers(
    answers: Dict[str, Any], ids: List[str]
) -> Dict[str, Any]:
    """Engine answers -> SGLang /v1/decisions {"answers": {id: {...}}}.

    Shapes mirror SGLang's DecisionAnswer exactly (verified against
    sglang main serving_decisions.py): choice probabilities keyed by
    option name; score probabilities keyed by level INDEX ("0"-"9")
    with the index-weighted mean as "score"; yes_no with
    probabilities {"yes", "no"}. The local engine has no label-mass
    signal (that is an SGLang serving concept), so "label_mass" is
    null here — the key stays for shape compatibility with SGLang
    clients. ("probability"/"answer" ride along on yes_no as local
    extensions; upstream readers use "probabilities".)
    """
    out: Dict[str, Any] = {}
    for qid in ids:
        ans = answers.get(qid)
        if not isinstance(ans, dict):
            continue
        atype = ans.get("type")
        if atype == "choice":
            out[qid] = {
                "type": "choice",
                "choice": ans["choice"],
                "probabilities": ans["probabilities"],
                "label_mass": ans.get("label_mass"),
            }
        elif atype == "score":
            dist = ans["distribution"]
            levels = list(dist.keys())
            wmean = sum(i * float(dist[lv]) for i, lv in enumerate(levels))
            out[qid] = {
                "type": "score",
                "score": wmean,
                # Index-keyed, like upstream: the client sent the levels
                # in order and maps positions back to names itself.
                "probabilities": {
                    str(i): float(dist[lv]) for i, lv in enumerate(levels)
                },
                "label_mass": ans.get("label_mass"),
            }
        elif atype == "noul":
            p_yes = float(ans["probability"])
            out[qid] = {
                "type": "yes_no",
                "probabilities": {"yes": p_yes, "no": 1.0 - p_yes},
                "probability": ans["probability"],
                "answer": ans["answer"],
                "label_mass": ans.get("label_mass"),
            }
    return out


# -- JEV /v1/decide compatibility (AutoTrust JEV-27B-VL wire format) ----------
#
# JEV decision models (JEV-27B-VL model card,
# https://huggingface.co/autotrust/JEV-27B-VL) serve System 1 over
# POST /v1/decide: {kind, state, question, options?} with kind = noul |
# choice | score, state = string | JSON | list mixing text and images, and
# a calibrated probability for every option in the reply. This shim
# answers the same dialect: clients written against serve_decide.py or a
# hosted Jev API work unchanged against this box.
#
# With SYSTEMONE_ENGINE=jev the request is forwarded natively (images
# preserved); every other engine answers the text projection (image parts
# noted and skipped, reported in "warnings").

_JEV_DECIDE_KINDS = ("noul", "choice", "score")
_JEV_DECIDE_MAX_OPTIONS = 256


def decide_option_limit(engine: Any, backend_name: str) -> int:
    """Max choice options /v1/decide serves on this engine.

    Native JEV servers take the wire-format 256; the SGLang engine caps at
    26 (SGLang's /v1/decisions limit); the local GLiClass pass width fits
    255. Anything above the serving engine's limit is a 422, mirroring
    /v1/decisions behavior.
    """
    if isinstance(engine, JevDecideBackend):
        return 256
    if backend_name == "sglang":
        return 26
    return 255


def decide_state_parts(state: Any) -> tuple[str, List[str]]:
    """JEV /v1/decide `state` -> (text, image_refs).

    Accepts a string, a JSON value, or a list mixing text with images in
    the model card's {"image": ...} form or OpenAI image_url parts.
    """
    if state is None:
        return "", []
    if isinstance(state, str):
        return state, []
    if isinstance(state, list):
        texts: List[str] = []
        images: List[str] = []
        for part in state:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and "image" in part:
                images.append(str(part["image"]))
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                inner = part.get("image_url") or {}
                images.append(
                    str(inner.get("url", "") if isinstance(inner, dict) else inner)
                )
            else:
                texts.append(state_to_text(part))
        return "\n".join(t for t in texts if t).strip(), [i for i in images if i]
    return state_to_text(state), []


def translate_decide_body(
    body: Dict[str, Any],
) -> tuple[str, str, List[str], str, List[str]]:
    """JEV /v1/decide body -> (kind, state_text, images, question, options)."""
    kind = body.get("kind")
    if kind not in _JEV_DECIDE_KINDS:
        raise Unprocessable(
            f"'kind' must be one of {', '.join(_JEV_DECIDE_KINDS)}"
        )
    if "state" not in body:
        raise Unprocessable("request must include 'state'")
    state_text, images = decide_state_parts(body.get("state"))
    if len(state_text) > MAX_STATE_CHARS:
        state_text = state_text[:MAX_STATE_CHARS]
    state_text = maybe_date_facts(state_text)
    question = body.get("question")
    if not isinstance(question, str) or not question.strip():
        raise Unprocessable("request must include a non-empty 'question' string")
    options: List[str] = []
    if kind == "choice":
        raw = body.get("options")
        if not isinstance(raw, list) or not (
            2 <= len(raw) <= _JEV_DECIDE_MAX_OPTIONS
        ):
            raise Unprocessable(
                "'options' must be a list of 2-"
                f"{_JEV_DECIDE_MAX_OPTIONS} strings for kind 'choice'"
            )
        options = [str(o) for o in raw]
    elif kind == "noul":
        options = ["false", "true"]
    else:
        options = [str(i) for i in range(6)]
    return kind, state_text, images, question.strip(), options


def translate_decide_answer(
    kind: str,
    options: List[str],
    answer: Dict[str, Any],
    model: str,
    latency_ms: Any,
    adaptation: str = "native",
) -> Dict[str, Any]:
    """One engine answer -> JEV /v1/decide response mapping."""
    if kind == "noul":
        probs = [1.0 - float(answer["probability"]), float(answer["probability"])]
    elif kind == "score":
        dist = answer.get("distribution") or {}
        probs = [float(dist.get(o, 0.0)) for o in options]
    else:
        dist = answer.get("probabilities") or {}
        probs = [float(dist.get(o, 0.0)) for o in options]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1.0 / len(options)] * len(options)
    idx = max(range(len(probs)), key=probs.__getitem__)
    try:
        elapsed = float(latency_ms) / 1000.0 if latency_ms is not None else 0.0
    except (TypeError, ValueError):
        elapsed = 0.0
    return {
        "kind": kind,
        "effective_kind": kind,
        "options": options,
        "probabilities": probs,
        "choice_index": idx,
        "choice": options[idx],
        "adaptation": adaptation,
        "protocol": "jev27-bare-v1",
        "model": model,
        "usage": {},
        "num_model_requests": 1,
        "elapsed_seconds": elapsed,
    }


# -- typed decision endpoint (/v1/systemone/decide) --------------------------
#
# Request/response schema mirrors the Jeff-1 sidecar's POST /v1/jeff1/decide
# so the shim can proxy (primary) or answer locally (fail-open fallback)
# with identical response shapes. Request validation mirrors the sidecar's
# decide validators; the fallback runs the very same GLiClass machinery as
# /v1/systemone — state-first prompt rows via build_decision_prompts(),
# the question's own per-type temperature when the engine carries a fitted
# per-answer-type map, and the TypeSafe-compatible confidence helpers.
# Confidence/temperature conventions are adapted from Mapika/decider
# (Apache-2.0); the implementation here is original.

DECIDE_TYPES = ("choice", "noul", "score")
"""Answer types accepted by /v1/systemone/decide."""

_DECIDE_METRICS_LOG_NAME = "decide-metrics.log"
_DECIDE_RECORDS_MAX = 5000


def _decide_instructions(body: Dict[str, Any]) -> str:
    instructions = body.get("instructions")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("request must include a non-empty 'instructions' string")
    return instructions.strip()


def _decide_choice_criteria(raw: Any) -> Dict[str, Optional[str]]:
    """choice criteria -> ordered {label: description} (mirrors the sidecar)."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "'criteria' must be a non-empty mapping of label -> description")
    criteria: Dict[str, Optional[str]] = {}
    for label, desc in raw.items():
        if not isinstance(label, str) or not label:
            raise ValueError("criteria labels must be non-empty strings")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"description for label '{label}' must be a string or null")
        criteria[label] = desc
    return criteria


def _decide_score_levels(raw: Any) -> List[Tuple[str, Optional[str]]]:
    """score criteria -> [(level_label, description)] in level order.

    Accepts a dict keyed by contiguous level indexes "0".."n-1" or an
    ordered list of level descriptions (mirrors the sidecar).
    """
    indexed: Dict[int, Any] = {}
    if isinstance(raw, list) and raw:
        indexed = dict(enumerate(raw))
    elif isinstance(raw, dict) and raw:
        for key in raw:
            try:
                i = int(key)
            except (TypeError, ValueError):
                raise ValueError(
                    "score 'criteria' keys must be level indexes '0'..'n-1'")
            if i < 0 or i in indexed:
                raise ValueError(
                    "score 'criteria' keys must be level indexes '0'..'n-1'")
            indexed[i] = raw[key]
    else:
        raise ValueError(
            "score 'criteria' must be a non-empty list of level descriptions "
            "or a mapping '0'..'n-1' -> description")
    n = len(indexed)
    if sorted(indexed) != list(range(n)):
        raise ValueError(
            "score 'criteria' keys must be contiguous level indexes '0'..'n-1'")
    levels: List[Tuple[str, Optional[str]]] = []
    for i in range(n):
        desc = indexed[i]
        if desc is not None and not isinstance(desc, str):
            raise ValueError(
                f"description for level '{i}' must be a string or null")
        levels.append((str(i), desc))
    return levels


def _decide_noul_descriptions(raw: Any) -> Tuple[Optional[str], Optional[str]]:
    """Optional noul criteria -> (yes_desc, no_desc) (mirrors the sidecar)."""
    if raw is None:
        return None, None
    if isinstance(raw, dict):
        unknown = set(raw) - {"yes", "no", "true", "false"}
        if unknown:
            raise ValueError(
                f"unknown noul criteria keys: {sorted(unknown)}; use "
                "{yes, no} or {true, false}")
        if "yes" in raw or "no" in raw:
            yes_desc, no_desc = raw.get("yes"), raw.get("no")
        else:
            yes_desc, no_desc = raw.get("true"), raw.get("false")
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        yes_desc, no_desc = raw[0], raw[1]
    else:
        raise ValueError(
            "'criteria' for noul must be null, a {yes, no} or {true, false} "
            "mapping, or a 2-item [yes, no] list")
    for name, desc in (("yes", yes_desc), ("no", no_desc)):
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"'criteria.{name}' must be a string or null")
    return yes_desc, no_desc


def _decide_labels_block(qtype: str, instructions: str,
                         criteria: Any) -> Tuple[str, List[str], str]:
    """Validate decide criteria -> (qtype, ordered labels, prompt text).

    The prompt text embeds the label descriptions so the local GLiClass
    fallback sees the same criterion definitions the sidecar would.
    """
    if qtype == "choice":
        crit = _decide_choice_criteria(criteria)
        labels = list(crit.keys())
        lines = [instructions, "", "Labels:"]
        for lab, desc in crit.items():
            lines.append(f"- {lab}" + (f": {desc}" if desc else ""))
        return qtype, labels, "\n".join(lines).strip()
    if qtype == "score":
        levels = _decide_score_levels(criteria)
        labels = [lab for lab, _ in levels]
        lines = [instructions, "", "Levels (in order):"]
        for lab, desc in levels:
            lines.append(f"- {lab}" + (f": {desc}" if desc else ""))
        return qtype, labels, "\n".join(lines).strip()
    yes_desc, no_desc = _decide_noul_descriptions(criteria)
    lines = [instructions]
    if yes_desc:
        lines.append(f"'yes' means: {yes_desc}")
    if no_desc:
        lines.append(f"'no' means: {no_desc}")
    return qtype, ["yes", "no"], "\n".join(lines).strip()


def _fallback_decide(engine: Any, state_text: str, qtype: str,
                     labels: List[str], prompt: str
                     ) -> Tuple[Dict[str, Any], List[float]]:
    """Answer one typed decision with the local GLiClass engine.

    The fail-open path behind /v1/systemone/decide: the very same
    machinery as /v1/systemone — engine.systemone() builds state-first
    prompt rows (build_decision_prompts), applies the question's own
    per-type temperature when the engine carries a fitted
    PerTypeTemperatureCalibrator, and reports the TypeSafe-compatible
    confidence helpers (Mapika/decider semantics, Apache-2.0).

    Returns (sidecar-shaped decision dict, ordered probability vector).
    """
    if qtype == "noul":
        question: Dict[str, Any] = {"name": "decision", "type": "noul",
                                    "statement": prompt}
    elif qtype == "score":
        question = {"name": "decision", "type": "score", "levels": labels,
                    "prompt": prompt}
    else:
        question = {"name": "decision", "type": "choice", "options": labels,
                    "prompt": prompt}
    answers = engine.systemone(state_text, [question])
    ans = answers["decision"]
    if qtype == "choice":
        probs = [float(ans["probabilities"][lab]) for lab in labels]
        return {
            "type": "choice",
            "label": ans["choice"],
            "probabilities": {lab: round(p, 4) for lab, p in zip(labels, probs)},
            "confidence": round(float(ans["confidence"]), 4),
        }, probs
    if qtype == "score":
        probs = [float(ans["distribution"][lab]) for lab in labels]
        return {
            "type": "score",
            "level": ans["level"],
            "distribution": {lab: round(p, 4) for lab, p in zip(labels, probs)},
            "confidence": round(float(ans["confidence"]), 4),
        }, probs
    p_yes = float(ans["probability"])
    probs = [p_yes, 1.0 - p_yes]
    return {
        "type": "noul",
        "label": "yes" if ans["answer"] else "no",
        "probabilities": {"yes": round(p_yes, 4), "no": round(1.0 - p_yes, 4)},
        "confidence": round(float(ans["confidence"]), 4),
    }, probs


def _decide_temperature_info(engine: Any,
                             qtype: str) -> Optional[Dict[str, Any]]:
    """Per-type temperature info for the decide response.

    Reported only when the engine actually carries a fitted
    per-answer-type calibrator (decider's >=1.4.0 applied-per-type
    semantics); otherwise the key is omitted from the response.
    """
    cal = getattr(engine, "calibrator", None)
    if cal is None or not getattr(cal, "fitted_", False):
        return None
    if not hasattr(cal, "temperature_for"):
        return None
    try:
        per_type = {
            str(t): float(v)
            for t, v in dict(getattr(cal, "temperature_by_type_", {}) or {}).items()
        }
        return {
            "applied": float(cal.temperature_for(qtype)),
            "pooled": float(cal.temperature_),
            "per_type": per_type,
        }
    except Exception:
        return None


def record_decide_metric(server: Any, record: Dict[str, Any]) -> None:
    """Record one fallback decision for offline calibration and metrics.

    Kept in a bounded in-memory deque on the server and appended to a
    JSONL log (logs/decide-metrics.log; SYSTEMONE_DECIDE_LOG_FILE
    overrides; SYSTEMONE_LOG_DISABLE=1 silences). Records carry no gold
    labels at serve time, so ECE/Brier/NLL are computed later by
    decide_metrics_summary() over gold-annotated rows (e.g. backfilled by
    a calibration battery). Never raises.
    """
    try:
        rec = dict(record)
        rec["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        buf = getattr(server, "decide_records", None)
        if buf is not None:
            buf.append(rec)
        if os.environ.get("SYSTEMONE_LOG_DISABLE") == "1":
            return
        path = os.environ.get("SYSTEMONE_DECIDE_LOG_FILE") or os.path.join(
            os.path.dirname(__file__), "logs", _DECIDE_METRICS_LOG_NAME)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def decide_metrics_summary(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Per-type ECE/Brier/NLL via systemone.metrics for gold-annotated rows.

    Pure function: groups records carrying a valid integer "gold" label by
    "type" and returns {type: metrics.summarize(...)} — decider's summarize
    convention (accuracy, ece_15, brier, nll, aurc, selective accuracies;
    metric definitions adapted from Mapika/decider, Apache-2.0). Records
    without gold are skipped; types with no gold rows are absent.
    """
    groups: Dict[str, Dict[str, List[Any]]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        try:
            qtype = str(r["type"]).lower()
            probs = [float(x) for x in r["probs"]]
            gold = int(r["gold"])
            if not (0 <= gold < len(probs)) or qtype not in DECIDE_TYPES:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        g = groups.setdefault(qtype, {"y": [], "P": []})
        g["y"].append(gold)
        g["P"].append(probs)
    out: Dict[str, Dict[str, float]] = {}
    for qtype, g in groups.items():
        try:
            out[qtype] = summarize_metrics(g["y"], g["P"], name=qtype)
        except Exception:
            continue
    return out


class ShimHandler(BaseHTTPRequestHandler):
    """HTTP handler; the engine is attached as `server.engine`,
    the tier registry as `server.registry`."""

    server_version = "SystemOneShim/0.2"

    def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-ID", getattr(self, "_request_id", "-"))
        self.end_headers()
        self.wfile.write(data)
        self._last_status = code

    def _send_text(self, code: int, text: str, ctype: str = "text/plain") -> None:
        data = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-ID", getattr(self, "_request_id", "-"))
        self.end_headers()
        self.wfile.write(data)
        self._last_status = code

    def _begin_request(self) -> float:
        """Stamp the request ID (client-supplied or fresh) and start timing."""
        incoming = (self.headers.get("X-Request-ID") or "").strip()
        self._request_id = incoming[:64] if incoming else uuid.uuid4().hex[:16]
        self._last_status = 0
        return time.perf_counter()

    def _record_metrics(self, t0: float) -> None:
        metrics = getattr(self.server, "metrics", None)
        if metrics is None:
            return
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        metrics.observe(self.path, getattr(self, "_last_status", 0), latency_ms)

    def _read_body(self) -> Dict[str, Any]:
        length = check_body_length(self.headers)
        return json.loads(self.rfile.read(length) or b"{}")

    def _handle_systemone(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone -> (status, payload)."""
        return 200, self._judge_one(self._read_body())

    def _judge_one(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Judge one TypeSafe body -> response payload (batch shares this).

        Exact-match cache (SYSTEMONE_CACHE_TTL>0) serves repeats without
        touching the engine; PII scrubbing (SYSTEMONE_SCRUB_PII=1)
        redacts identifiers before judging and reports the kinds.
        """
        engine = self.server.engine
        cache = getattr(self.server, "decision_cache", None)
        cache_enabled = cache is not None and cache.enabled
        key = None
        if cache_enabled and cache is not None:
            try:
                canonical = json.dumps(body, sort_keys=True,
                                       separators=(",", ":"), default=str)
            except (TypeError, ValueError):
                canonical = None
            if canonical is not None:
                key = DecisionCache.cache_key(engine.model_name, canonical)
                hit = cache.get(key)
                if hit is not None:
                    hit["cached"] = True
                    hit["latency_ms"] = 0.0  # no engine call happened
                    return hit
        state_text, questions = translate_body(body)
        state_text, pii_kinds = maybe_scrub_pii(state_text)
        images, videos = translate_media(body)
        kwargs: Dict[str, Any] = {}
        if images and _engine_supports(engine, "images"):
            kwargs["images"] = images
        if videos and _engine_supports(engine, "videos"):
            kwargs["videos"] = videos
        answers = engine.systemone(state_text, questions, **kwargs)
        meta = answers.get("_meta", {}) if isinstance(answers, dict) else {}
        # Clef usage: decisions take zero completion tokens; input tokens
        # ride along when the engine counted them.
        usage: Dict[str, Any] = {"output_tokens": 0}
        if isinstance(meta.get("input_tokens"), int):
            usage["input_tokens"] = meta["input_tokens"]
        payload: Dict[str, Any] = {
            "answers": translate_answers(answers),
            "model": engine.model_name,
            "usage": usage,
            "latency_ms": meta.get("latency_ms"),
        }
        if images or videos:
            dropped = meta.get("media_dropped") or {}
            payload["media"] = {"images": len(images), "videos": len(videos)}
            dropped_imgs = int(dropped.get("images", 0) or 0)
            dropped_vids = int(dropped.get("videos", 0) or 0)
            if not _engine_supports(engine, "images"):
                dropped_imgs = len(images)
            if not _engine_supports(engine, "videos"):
                dropped_vids = len(videos)
            if dropped_imgs or dropped_vids:
                bits = []
                if dropped_imgs:
                    bits.append(f"{dropped_imgs} image(s)")
                if dropped_vids:
                    bits.append(f"{dropped_vids} video(s)")
                payload["warnings"] = [
                    f"{' and '.join(bits)} dropped: the "
                    f"{getattr(self.server, 'engine_backend', 'local')} "
                    "engine has no media path for them"
                ]
        if pii_kinds:
            payload["pii"] = {"redacted": True, "kinds": list(pii_kinds),
                              "count": state_text.count("[REDACTED_")}
        if cache_enabled:
            payload["cached"] = False
            if key is not None and cache is not None:
                cache.put(key, payload)
        return payload

    def _handle_batch(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone/batch -> (status, payload).

        Body: {"items": [TypeSafe bodies, 1..32]}. Each item judges like
        POST /v1/systemone (cache included); per-item failures come back
        as {"status", "error"} results instead of failing the batch.
        """
        body = self._read_body()
        items = body.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("'items' must be a non-empty list")
        if len(items) > MAX_BATCH_ITEMS_PER_REQUEST:
            raise ValueError(
                f"'items' exceeds the {MAX_BATCH_ITEMS_PER_REQUEST}-item cap")
        results = []
        for item in items:
            if not isinstance(item, dict):
                results.append({"status": 400,
                                "error": "batch item must be a mapping"})
                continue
            try:
                results.append({"status": 200, **self._judge_one(item)})
            except (ValueError, KeyError) as exc:
                results.append({"status": 400,
                                "error": f"bad request: {exc}"})
            except Exception as exc:  # never leak internals beyond the name
                results.append(
                    {"status": 500,
                     "error": f"engine failure: {type(exc).__name__}"})
        return 200, {"results": results,
                     "model": self.server.engine.model_name,
                     "n_items": len(items)}

    def _handle_decisions(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/decisions -> (status, payload).

        SGLang's /v1/decisions request/response dialect, served by the local
        engine: one POST carries the state plus N typed questions and they
        are answered in a single batched pass. Lets clients written against
        SGLang work unchanged against this box (label_mass is null here —
        that signal only exists on a real SGLang server; see
        systemone/sglang_backend.py).
        """
        body = self._read_body()
        try:
            state_text, questions, ids = translate_decisions_body(body)
        except Unprocessable as e:
            return 422, {"error": f"unprocessable: {e}"}
        answers = self.server.engine.systemone(state_text, questions)
        return 200, {
            # "object" mirrors SGLang's DecisionResponse; this box does no
            # SGLang prompt rendering, so it reports no prompt_format_version
            # rather than echo a version it does not implement.
            "object": "decisions",
            "answers": translate_decisions_answers(answers, ids),
            "model": self.server.engine.model_name,
            "usage": {},
            "latency_ms": answers.get("_meta", {}).get("latency_ms"),
        }

    def _handle_decide_v1(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/decide -> (status, payload).

        JEV System 1 dialect (kind/state/question/options), served natively
        by the jev engine (images preserved) or via the text projection on
        every other engine (image parts noted, skipped, and reported in
        "warnings"). Lets clients written against serve_decide.py or a
        hosted Jev API work unchanged against this box.
        """
        body = self._read_body()
        try:
            kind, state_text, images, question, options = translate_decide_body(body)
        except Unprocessable as e:
            return 422, {"error": f"unprocessable: {e}"}
        engine = self.server.engine
        backend_name = getattr(self.server, "engine_backend", "custom")
        limit = decide_option_limit(engine, backend_name)
        if kind == "choice" and len(options) > limit:
            return 422, {"error": (
                f"unprocessable: {len(options)} options exceeds the "
                f"{backend_name} engine's limit of {limit} "
                "(use SYSTEMONE_ENGINE=jev for the full 256)"
            )}
        if isinstance(engine, JevDecideBackend):
            resp = engine.decide(
                kind, state_text, question,
                options if kind == "choice" else None,
                images=images,
            )
            return 200, resp
        if images and not state_text:
            return 422, {"error": (
                "unprocessable: state contained only image parts; the "
                f"{getattr(self.server, 'engine_backend', 'local')} engine "
                "is text-only (use SYSTEMONE_ENGINE=jev with a VLM for images)"
            )}
        if kind == "choice":
            question_spec: Dict[str, Any] = {
                "name": "q", "type": "choice",
                "options": options, "prompt": question,
            }
        elif kind == "score":
            question_spec = {
                "name": "q", "type": "score",
                "levels": options, "prompt": question,
            }
        else:
            question_spec = {"name": "q", "type": "noul", "statement": question}
        answers = engine.systemone(state_text, [question_spec])
        payload = translate_decide_answer(
            kind, options, answers["q"], engine.model_name,
            answers.get("_meta", {}).get("latency_ms"),
            adaptation="text-projection",
        )
        if images:
            payload["warnings"] = [
                f"{len(images)} image(s) dropped: the "
                f"{getattr(self.server, 'engine_backend', 'local')} engine "
                "is text-only (use SYSTEMONE_ENGINE=jev with a VLM for images)"
            ]
        return 200, payload

    def _scoring_ctx(self) -> Dict[str, Any]:
        return {
            "calibration": getattr(self.server, "calibration", None),
            "tools": getattr(self.server, "tools", []),
            "registry": getattr(self.server, "registry", {}),
        }

    def _handle_route(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone/route -> (status, payload)."""
        body = self._read_body()
        task, cost_bias, candidates = parse_route_body(body, self.server.registry)
        sort = body.get("sort", "utility")
        if sort not in _ROUTE_SORTS:
            raise ValueError(
                f"unknown sort {sort!r}; want {'|'.join(_ROUTE_SORTS)}")
        fallbacks = body.get("fallbacks", 2)
        if (isinstance(fallbacks, bool) or not isinstance(fallbacks, int)
                or not 0 <= fallbacks <= 8):
            raise ValueError(f"'fallbacks' must be an int 0..8, got {fallbacks!r}")
        explore = body.get("explore", 0.0)
        if (isinstance(explore, bool) or not isinstance(explore, (int, float))
                or not 0.0 <= float(explore) <= 1.0):
            raise ValueError(f"'explore' must be in [0, 1], got {explore!r}")
        route = route_decision(
            self.server.engine, task, candidates, cost_bias,
            scoring=self._scoring_ctx(),
            sort=sort, fallbacks=fallbacks, explore=float(explore),
        )
        return 200, {
            "route": route,
            "model": self.server.engine.model_name,
            "usage": {},
        }

    def _handle_rank_plans(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone/rank-plans -> (status, payload).

        Body: {"task": "...", "plans": [{"id": "...", "text": "..."}, ...]}.
        Scores each plan's P(success | task) with the zero-shot head minus a
        cost penalty (est_steps x routed-tier cost, normalized). The ranking
        (not just the winner) is returned for the run log.
        """
        body = self._read_body()
        task = body.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("request must include a non-empty 'task' string")
        plans = body.get("plans")
        if not isinstance(plans, list) or not plans:
            raise ValueError("'plans' must be a non-empty list")
        if len(plans) > MAX_PLANS_PER_REQUEST:
            raise ValueError(
                f"'plans' exceeds the {MAX_PLANS_PER_REQUEST}-plan cap")
        for p in plans:
            if not isinstance(p, dict) or not isinstance(p.get("text"), str):
                raise ValueError("each plan must be a mapping with a 'text' string")
        _, _, candidates = parse_route_body(
            {"task": task.strip()}, self.server.registry)
        route = route_decision(
            self.server.engine, task.strip(), candidates, "balanced",
            scoring=self._scoring_ctx(),
        )
        tier_cost = None
        reg = self.server.registry or {}
        tiers = reg.get("tiers", reg) if isinstance(reg, dict) else {}
        entry = tiers.get(route["tier"]) if isinstance(tiers, dict) else None
        if isinstance(entry, dict):
            for m in entry.get("models", []) or []:
                if isinstance(m, dict) and m.get("model_id") == route["model_id"]:
                    try:
                        tier_cost = float(m["cost"])
                    except (KeyError, TypeError, ValueError):
                        pass
                    break
        ranking = rank_plans(self.server.engine, task.strip(), plans, tier_cost)
        jeff1: Dict[str, Any] = {"consulted": False, "latency_ms": None,
                                 "blended": False}
        if jeff1_enabled():
            t0 = time.perf_counter()
            stub_plans = [
                {"id": p.get("id", f"plan_{i}"), "text": p.get("text") or ""}
                for i, p in enumerate(plans)
            ]
            jeff_ranking = rank_plans_via_jeff1(task.strip(), stub_plans)
            jeff1["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            if jeff_ranking is not None:
                ranking = blend_rankings(ranking, jeff_ranking, tier_cost)
                jeff1["consulted"] = True
                jeff1["blended"] = True
        return 200, {
            "task": task.strip(),
            "tier": route["tier"],
            "ranking": ranking,
            "jeff1": jeff1,
            "model": self.server.engine.model_name,
            "usage": {},
        }


    def _handle_decide(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone/decide -> (status, payload).

        Body: {"state", "instructions", "criteria", "type"} — the decision
        sidecar's decide schema. The primary path proxies the request to
        the sidecar (backend "decider", the sole backend). When the sidecar
        is unreachable, 404s
        (endpoint not deployed yet), or returns a malformed reply, the
        request fails open to the local GLiClass engine
        (backend "fallback") via the same machinery as /v1/systemone.
        """
        body = self._read_body()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        if "state" not in body:
            raise ValueError("request must include 'state'")
        instructions = _decide_instructions(body)
        qtype = body.get("type")
        if qtype not in DECIDE_TYPES:
            raise ValueError("'type' must be one of 'choice', 'noul', 'score'")
        qtype, labels, prompt = _decide_labels_block(
            qtype, instructions, body.get("criteria"))
        state_text = state_to_text(body["state"])
        if len(state_text) > MAX_STATE_CHARS:
            state_text = state_text[:MAX_STATE_CHARS]
        state_text = maybe_date_facts(state_text)

        t0 = time.perf_counter()
        jeff1 = decide_via_jeff1({
            "state": body["state"],
            "instructions": instructions,
            "criteria": body.get("criteria"),
            "type": qtype,
        })
        if jeff1 is not None:
            payload = dict(jeff1)
            # Trust the sidecar's own backend report ("decider", the sole
            # backend); fall back to that label for sidecars that don't
            # report one.
            payload["backend"] = jeff1.get("backend") or "decider"
            payload["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            return 200, payload

        # Fail-open fallback: the local GLiClass decision path.
        decision, ordered_probs = _fallback_decide(
            self.server.engine, state_text, qtype, labels, prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        payload = dict(decision)
        payload["backend"] = "fallback"
        payload["latency_ms"] = latency_ms
        payload["model"] = getattr(self.server.engine, "model_name", "?")
        temp_info = _decide_temperature_info(self.server.engine, qtype)
        if temp_info is not None:
            payload["temperature"] = temp_info
        record_decide_metric(self.server, {
            "type": qtype,
            "labels": labels,
            "probs": ordered_probs,
            "confidence": payload.get("confidence"),
            "latency_ms": latency_ms,
            "backend": "fallback",
            "model": getattr(self.server.engine, "model_name", "?"),
        })
        return 200, payload

    def _handle_permute(self) -> tuple[int, Dict[str, Any]]:
        """POST /v1/systemone/permute -> (status, payload).

        Body: {"state", "question": {TypeSafe choice question},
               "n_perm" (2-32, default 8), "seed" (default 0)}.
        Re-runs the question under n_perm option orders (first the
        given order, then seeded shuffles) against the serving engine
        and reports per-order answers, argmax stability, and the
        per-option probability spread — Kev's permutation probe
        (kev.serve systemone_permute), engine-agnostic.
        """
        body = self._read_body()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        if "state" not in body:
            raise ValueError("request must include 'state'")
        raw_q = body.get("question")
        if not isinstance(raw_q, dict):
            raise ValueError("request must include a 'question' mapping")
        n_perm = body.get("n_perm", 8)
        if isinstance(n_perm, bool) or not isinstance(n_perm, int):
            raise ValueError("'n_perm' must be an integer")
        if not 2 <= n_perm <= 32:
            raise ValueError("'n_perm' must be in 2..32")
        seed = body.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("'seed' must be an integer")
        question = translate_question("permute", raw_q)
        if question.get("type") != "choice":
            raise ValueError("'question' must be a choice question")
        options = list(question.get("options") or [])
        if len(options) < 2:
            raise ValueError("'question' needs >= 2 options")
        state_text = state_to_text(body["state"])
        if len(state_text) > MAX_STATE_CHARS:
            state_text = state_text[:MAX_STATE_CHARS]
        state_text = maybe_date_facts(state_text)

        rng = random.Random(seed)
        orders: List[List[str]] = [list(options)]
        for _ in range(n_perm - 1):
            order = list(options)
            rng.shuffle(order)
            orders.append(order)
        runs: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for order in orders:
            answers = self.server.engine.systemone(
                state_text, [{**question, "options": order}])
            ans = answers.get("permute") or {}
            probs = dict(ans.get("probabilities") or {})
            choice = ans.get("choice") or (
                max(probs, key=probs.get) if probs else None)
            if choice is None:
                raise ValueError("engine returned no choice answer")
            runs.append({"order": order, "probabilities": probs,
                         "choice": choice})
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        spread = {opt: round(max(r["probabilities"].get(opt, 0.0)
                                 for r in runs)
                             - min(r["probabilities"].get(opt, 0.0)
                                   for r in runs), 4)
                  for opt in options}
        return 200, {
            "runs": runs,
            "argmax_stable": len({r["choice"] for r in runs}) == 1,
            "spread": spread,
            "n_perm": n_perm,
            "seed": seed,
            "model": getattr(self.server.engine, "model_name", "?"),
            "latency_ms": latency_ms,
        }

    def do_GET(self) -> None:  # noqa: N802
        t0 = self._begin_request()
        try:
            self._do_GET()
        finally:
            self._record_metrics(t0)

    def _do_GET(self) -> None:
        if self.path in ("/", "/healthz"):
            self._send_json(200, {
                "ok": True,
                "model": self.server.engine.model_name,
                "backend": getattr(self.server, "engine_backend", "custom"),
            })
        elif self.path == "/openapi.json":
            spec = getattr(self.server, "openapi_spec", None)
            if isinstance(spec, dict):
                self._send_json(200, spec)
            else:
                self._send_json(500, {"error": "openapi spec unavailable"})
        elif self.path == "/v1/decide/info":
            engine = self.server.engine
            backend_name = getattr(self.server, "engine_backend", "custom")
            self._send_json(200, {
                "option_limit": decide_option_limit(engine, backend_name),
                "kinds": ["noul", "choice", "score"],
                "image_support": isinstance(engine, JevDecideBackend),
                "backend": backend_name,
                "model": getattr(engine, "model_name", "?"),
                "temperatures": None,
            })
        elif self.path == "/metrics":
            metrics = getattr(self.server, "metrics", None)
            if metrics is None:
                self._send_json(500, {"error": "metrics unavailable"})
            else:
                self._send_text(200, metrics.render_prometheus())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        t0 = self._begin_request()
        status, payload, extra = 500, {"error": "internal"}, {}
        try:
            if not api_token_ok(self.headers.get("Authorization")):
                # Opt-in shared token ($SYSTEMONE_API_TOKEN); unset = open.
                status, payload = 401, {
                    "error": "unauthorized: missing or wrong bearer token"
                }
            elif scoring_disabled() and self.path in (
                "/v1/systemone/route", "/v1/systemone/rank-plans"
            ):
                # Kill switch: refuse routing; upstream consumers fail open.
                status, payload = 503, {
                    "error": "systemone routing disabled (SYSTEMONE_DISABLE=1)"
                }
            elif self.path == "/v1/systemone":
                status, payload = self._handle_systemone()
                extra = {"n_questions": len(payload.get("answers", {}))}
            elif self.path == "/v1/decisions":
                status, payload = self._handle_decisions()
                extra = {"n_questions": len(payload.get("answers", {}))}
            elif self.path == "/v1/decide":
                status, payload = self._handle_decide_v1()
                extra = {
                    "decide_kind": payload.get("kind"),
                    "decide_choice": payload.get("choice"),
                }
            elif self.path == "/v1/systemone/route":
                status, payload = self._handle_route()
                route = payload.get("route", {})
                extra = {
                    "route_tier": route.get("tier"),
                    "route_model": route.get("model_id"),
                    "route_confidence": route.get("confidence"),
                }
            elif self.path == "/v1/systemone/rank-plans":
                status, payload = self._handle_rank_plans()
                extra = {
                    "n_plans": len(payload.get("ranking", [])),
                    "top_plan": (payload.get("ranking") or [{}])[0].get("id"),
                }
            elif self.path == "/v1/systemone/decide":
                status, payload = self._handle_decide()
                extra = {
                    "decide_type": payload.get("type"),
                    "backend": payload.get("backend"),
                }
            elif self.path == "/v1/systemone/permute":
                status, payload = self._handle_permute()
                extra = {
                    "n_perm": payload.get("n_perm"),
                    "argmax_stable": payload.get("argmax_stable"),
                }
            elif self.path == "/v1/systemone/batch":
                status, payload = self._handle_batch()
                extra = {
                    "n_items": payload.get("n_items"),
                }
            else:
                status = 404
                payload = {
                    "error": "not found, POST /v1/decisions, /v1/decide, "
                             "/v1/systemone, /v1/systemone/batch, "
                             "/v1/systemone/route, /v1/systemone/rank-plans, "
                             "/v1/systemone/decide, or /v1/systemone/permute"
                }
        except BodyTooLarge as e:
            status, payload = 413, {"error": f"request too large: {e}"}
        except (ValueError, KeyError) as e:
            status, payload = 400, {"error": f"bad request: {e}"}
        except Exception as e:  # never leak internals beyond the class name
            status, payload = 500, {"error": f"engine failure: {type(e).__name__}"}
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if status == 200 and "latency_ms" not in payload:
            payload["latency_ms"] = latency_ms
        self._send_json(status, payload)
        self._record_metrics(t0)
        log_decision(
            {
                "endpoint": self.path,
                "latency_ms": latency_ms,
                "status": status,
                "model": getattr(self.server.engine, "model_name", "?"),
                "request_id": getattr(self, "_request_id", "-"),
                **extra,
            }
        )

    def log_message(self, fmt: str, *args: Any) -> None:
        pass  # quiet by default; the decision log records what matters


# -- self-daemonization (Windows sshd job-object escape) --------------------
#
# sshd on Windows runs each session inside a Job Object with
# KILL_ON_JOB_CLOSE. Node's `detached: true` cannot escape that job -- Node
# exposes no way to set process creation flags -- so a shim spawned by the
# ZCode CLI would die when the SSH session closes. The shim instead re-spawns
# *itself* with CREATE_BREAKAWAY_FROM_JOB | DETACHED_PROCESS; the original
# process exits immediately and the detached grandchild (outside the job)
# serves. Non-Windows platforms are unaffected (Node's setsid() already
# detaches there). Every failure path is fail-open: the shim simply keeps
# running in-process.

def _win32_detach(argv: list[str]) -> bool:
    """Re-launch this shim detached on Windows.

    Returns True when the caller must exit immediately (a detached copy was
    started); False when the current process should keep serving -- not on
    Windows, opted out, or the re-spawn failed (fail-open).
    """
    if sys.platform != "win32":
        return False
    try:
        import subprocess

        creationflags = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
        if not creationflags:
            return False
        # --no-daemonize goes last so the grandchild does not respawn again
        # (SYSTEMONE_DAEMONIZE is inherited through the environment).
        cmd = [sys.executable, "-m", "systemone.shim", *argv, "--no-daemonize"]
        subprocess.Popen(
            cmd,
            creationflags=creationflags,
            close_fds=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


# -- baked-in engine selection (local GLiClass vs SGLang) --------------------

ENGINE_ENV = "SYSTEMONE_ENGINE"
ENGINE_CHOICES = ("auto", "local", "sglang", "jevk5", "onnx", "jev", "kev", "clef")


def engine_backend_name(engine: Any) -> str:
    """Short backend label for an engine instance: sglang | hybrid | local.

    Anything else (injected stubs, test doubles) reports "custom". Used in
    the /healthz payload and the serve banner so operators can see which
    judge is actually answering.
    """
    if isinstance(engine, HybridBackend):
        return "hybrid"
    if isinstance(engine, JevDecideBackend):
        return "jev"
    if isinstance(engine, SGLangBackend):
        return "sglang"
    if isinstance(engine, JevK5ServerBackend):
        return "jevk5"
    if isinstance(engine, KevBackend):
        return "kev"
    if isinstance(engine, ClefBackend):
        return "clef"
    if isinstance(engine, RerankBackend):
        return "rerank"
    if SystemOne is not None and isinstance(engine, SystemOne):
        return "local"
    if type(engine).__name__ == "SystemOne":
        return "local"
    return "custom"


def create_engine(name: str | None = None) -> Any:
    """Build the decision engine the shim serves.

    Args:
        name: "auto" (default) | "local" | "sglang" | "jevk5" | "onnx" |
            "jev" | "kev" | "clef". Unset -> the SYSTEMONE_ENGINE env var,
            defaulting to "auto".

    - auto: JEV when JEV_URL is set and healthy, else SGLang when
      SGLANG_BASE_URL is set and healthy, else JevK5 when JEVK5_BASE_URL
      is set and healthy, else Kev when KEV_BASE_URL is set and healthy,
      else the local GLiClass engine. Probes only run
      for explicitly configured servers, so a default box never stalls at
      startup. The JEV decision model wins when configured — it is the
      flagship judge (calibrated System 1 + System 2 in one engine).
    - local: the GLiClass engine. Needs torch/transformers/gliclass
      (pip install 'systemone[local]').
    - sglang: SGLangBackend. When the server is unreachable it fails open
      to the local engine if one can be built, else raises SGLangError.
    - jevk5: JevK5ServerBackend (jevk5-serve's /v1/systemone). Same
      fail-open behavior as sglang.
    - onnx: RerankBackend over an ONNX cross-encoder (default
      Xenova/bge-reranker-base int8, override with RERANK_MODEL_ID /
      RERANK_ONNX_FILE). Needs onnxruntime + tokenizers +
      huggingface_hub. Never auto-selected (it downloads weights).
    - jev: JevDecideBackend (a JEV decision model's /v1/decide, e.g.
      serve_decide.py or a hosted Jev API). Same fail-open behavior as
      sglang. Only this engine serves images natively.
    - kev: KevBackend (kev.serve's /v1/systemone: Kev-0.8B/4B/9B/27B).
      Same fail-open behavior as sglang. Text-only, but the long-doc
      specialist (up to 65,536 tokens on Kev-27B).
    - clef: ClefBackend (Cloudflare clef/clef-flash weights, run locally).
      Multimodal (text, JSON, images, video) with joint schema scoring.
      Needs torch/transformers/huggingface_hub/safetensors/pillow (pip
      install 'systemone[clef]'). Never auto-selected (it downloads
      9-27B weights); failing open to local when unavailable.

    Raises:
        ValueError: unknown engine name.
        SGLangError / JevK5Error / KevError: remote requested but
            unreachable and no local fallback.
        ImportError: local requested but the heavy deps are not installed.
    """
    from .jev_backend import JevError
    from .jevk5_backend import JevK5Error
    from .kev_backend import KevError
    from .sglang_backend import SGLangError

    sel = (name or os.environ.get(ENGINE_ENV) or "auto").strip().lower()
    if sel not in ENGINE_CHOICES:
        raise ValueError(
            f"unknown engine {sel!r} (SYSTEMONE_ENGINE must be one of: "
            f"{', '.join(ENGINE_CHOICES)})"
        )

    def _local() -> Any:
        if SystemOne is None:
            raise ImportError(
                "the local GLiClass engine needs torch + transformers + "
                "gliclass, which are not installed. Either install them "
                "(pip install 'systemone[local]') or serve a remote engine "
                "instead (SYSTEMONE_ENGINE=sglang|jevk5|jev|kev with its "
                "base URL set) or the ONNX judge (SYSTEMONE_ENGINE=onnx "
                "with onnxruntime installed)."
            )
        return SystemOne(model_name=os.environ.get("SYSTEMONE_MODEL"))

    def _remote(kind: str, build: Any, err_cls: Any, env_var: str) -> Any:
        backend = build()
        try:
            healthy = backend.health()
        except Exception:
            healthy = False
        if healthy:
            return backend
        if sel == kind:
            if SystemOne is not None:
                logging.warning(
                    "%s server unreachable at %s; failing open to the "
                    "local engine",
                    kind, backend.base_url,
                )
                return _local()
            raise err_cls(
                f"could not reach the {kind} server and no local engine "
                "is available",
                hint=f"tried {backend.base_url} (set {env_var}); "
                "slim install has no local fallback — pip install "
                "'systemone[local]' to add one",
            )
        logging.warning(
            "%s server unreachable at %s; failing open onward",
            kind, backend.base_url,
        )
        return None

    if sel == "local":
        return _local()
    if sel == "onnx":
        try:
            enc = OnnxCrossEncoder(
                model_id=os.environ.get("RERANK_MODEL_ID") or "Xenova/bge-reranker-base",
                filename=os.environ.get("RERANK_ONNX_FILE") or "onnx/model_int8.onnx",
            )
        except Exception as exc:
            if SystemOne is not None:
                logging.warning(
                    "ONNX judge unavailable (%s); failing open to local", exc)
                return _local()
            raise
        return RerankBackend(enc.score, model_name=enc.model_id + " [onnx]")
    if sel == "clef":
        try:
            return ClefBackend()
        except Exception as exc:
            if SystemOne is not None:
                logging.warning(
                    "clef weights unavailable (%s); failing open to local", exc)
                return _local()
            raise
    if sel == "jev" or (os.environ.get("JEV_URL") or "").strip():
        found = _remote("jev", JevDecideBackend, JevError, "JEV_URL")
        if found is not None:
            return found
    if sel == "sglang" or (os.environ.get("SGLANG_BASE_URL") or "").strip():
        found = _remote("sglang", SGLangBackend, SGLangError, "SGLANG_BASE_URL")
        if found is not None:
            return found
    if sel == "jevk5" or (os.environ.get("JEVK5_BASE_URL") or "").strip():
        found = _remote("jevk5", JevK5ServerBackend, JevK5Error, "JEVK5_BASE_URL")
        if found is not None:
            return found
    if sel == "kev" or (os.environ.get("KEV_BASE_URL") or "").strip():
        found = _remote("kev", KevBackend, KevError, "KEV_BASE_URL")
        if found is not None:
            return found
    return _local()


def serve(
    port: int = 8765,
    engine: Any = None,
    registry: Dict[str, Dict[str, Any]] | None = None,
    engine_name: str | None = None,
) -> ThreadingHTTPServer:
    """Build (but do not block on) the shim server.

    Loads calibration.json (temperature for the blended tier distribution;
    absent -> serve raw, calibrated=false) and tool_registry.json. When the
    bundled calibration.json also carries a per-answer-type temperature map,
    it is attached to the engine so every decision endpoint serves each
    question type at its own fitted temperature (decider's applied-per-type
    semantics). Starts a daemon thread refreshing model availability from
    the LM Studio inventory (fail-open; registry values stand when LM
    Studio is unreachable).

    Args:
        engine: explicit engine instance (wins over engine_name; tests use
            this to inject stubs).
        engine_name: "auto" | "local" | "sglang" | "jevk5" | "onnx" |
            "jev" | "kev" | "clef" (see create_engine); unset ->
            $SYSTEMONE_ENGINE, default "auto".
    """
    engine = engine or create_engine(engine_name)
    server = ThreadingHTTPServer(("127.0.0.1", port), ShimHandler)
    server.engine = engine  # type: ignore[attr-defined]
    server.engine_backend = engine_backend_name(engine)  # type: ignore[attr-defined]
    server.metrics = Metrics()  # type: ignore[attr-defined]
    # Exact-match decision cache for /v1/systemone (+ batch items):
    # SYSTEMONE_CACHE_TTL>0 enables, SYSTEMONE_CACHE_MAX bounds.
    server.decision_cache = decision_cache_from_env()  # type: ignore[attr-defined]
    reg = registry if registry is not None else load_registry()
    server.registry = reg  # type: ignore[attr-defined]
    server.calibration = load_calibration(CALIBRATION_PATH)  # type: ignore[attr-defined]
    server.tools = load_tool_registry(TOOL_REGISTRY_PATH)  # type: ignore[attr-defined]
    # OpenAPI document served at GET /openapi.json (fail-open: a missing or
    # invalid spec degrades to a 500 on that path only, never breaks serve()).
    try:
        with open(OPENAPI_PATH, "r", encoding="utf-8") as f:
            server.openapi_spec = json.load(f)  # type: ignore[attr-defined]
    except Exception:
        server.openapi_spec = None  # type: ignore[attr-defined]
    # Per-answer-type temperature map for the decision path (fail-open:
    # absent or pooled-only calibration.json leaves the engine untouched).
    server.type_calibration = None  # type: ignore[attr-defined]
    try:
        type_cal = load_type_calibration(CALIBRATION_PATH)
        if type_cal is not None:
            if hasattr(engine, "set_calibrator"):
                engine.set_calibrator(type_cal)
            server.type_calibration = type_cal  # type: ignore[attr-defined]
    except Exception:
        pass
    # Bounded ring of fallback-decision records for offline calibration
    # and ECE/Brier/NLL metrics (see record_decide_metric /
    # decide_metrics_summary).
    server.decide_records = deque(maxlen=_DECIDE_RECORDS_MAX)  # type: ignore[attr-defined]
    # One quick inventory probe at startup (fail-open), then background refresh.
    try:
        ids = fetch_lmstudio_models()
        if ids is not None:
            apply_inventory(reg, ids)
    except Exception:
        pass
    try:
        start_inventory_refresher(reg)
    except Exception:
        pass
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Local /v1/systemone shim server")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--engine",
        choices=list(ENGINE_CHOICES),
        default=None,
        help=(
            "decision engine to serve: auto (JEV, then SGLang, then "
            "JevK5, then local), local (GLiClass), sglang, jevk5, onnx "
            "(local ONNX cross-encoder judge), or jev (JEV decision "
            "model). Remotes fail open to local when unreachable. "
            "Default: $SYSTEMONE_ENGINE or auto."
        ),
    )
    parser.add_argument(
        "--daemonize",
        action="store_true",
        default=os.environ.get("SYSTEMONE_DAEMONIZE", "").strip() == "1",
        help=(
            "Windows only: re-spawn detached (break away from the sshd job "
            "object) so the shim survives the parent SSH session. Fail-open; "
            "no-op on other platforms."
        ),
    )
    parser.add_argument(
        "--no-daemonize",
        dest="daemonize",
        action="store_false",
        help="Opt out of --daemonize / SYSTEMONE_DAEMONIZE.",
    )
    args = parser.parse_args()
    if args.daemonize and _win32_detach(sys.argv[1:]):
        print("systemone shim detached; parent exiting")
        return
    server = serve(args.port, engine_name=args.engine)
    print(
        f"systemone shim on http://127.0.0.1:{args.port}/v1/systemone "
        f"and /v1/systemone/route (model {server.engine.model_name}, "
        f"backend {server.engine_backend})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
