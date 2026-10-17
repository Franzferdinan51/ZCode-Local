"""D1 decision models (Liquid d1-3b / d1-omni-600m) via LM Studio.

LM Studio serves its loaded decision models on ``POST /v1/systemone``
— the SystemOne wire: a state plus named typed questions (choice /
noul / score), answered with calibrated probabilities from a single
readout (``output_tokens: 0``; there is no generated text). This module
is the shim's client for that endpoint, written in the same shape as
the jeff1.py sidecar helpers: every function fails open to ``None``
and never raises.

Duckets' rule (2026-10-15): the decision models are d1-3b and
d1-omni-600m, full stop. Clients name the decision model per request
(grok-local sends ``model: "d1-3b"`` with every SystemOne call); only
d1-family ids are honored — anything else falls back to the
configured default, so a chat model can never be named into the
decision seat.

Environment:
  SYSTEMONE_D1          kill switch (off-values disable; default on)
  SYSTEMONE_D1_URL      LM Studio base URL (default
                        http://127.0.0.1:1234; LMSTUDIO_URL fallback)
  SYSTEMONE_D1_MODEL    default decision model (default "d1-3b")
  SYSTEMONE_D1_TIMEOUT  per-request seconds (default 20)
  SYSTEMONE_D1_BLEND    D1 weight in the rank-plans blend (default 0.5)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

_OFF_VALUES = {"0", "false", "no", "off"}
_DEFAULT_MODEL = "d1-3b"
FAST_MODEL = "d1-omni-600m"


def d1_enabled() -> bool:
    """Is the D1 decision head on? Default ON; explicit off-values disable."""
    raw = os.environ.get("SYSTEMONE_D1", "1").strip().lower()
    return raw not in _OFF_VALUES


def d1_base_url() -> str:
    """LM Studio base URL (no trailing slash)."""
    from .patterns import require_http_url

    raw = (
        os.environ.get("SYSTEMONE_D1_URL", "").strip()
        or os.environ.get("LMSTUDIO_URL", "").strip()
        or "http://127.0.0.1:1234"
    )
    return require_http_url(raw, what="D1/LM Studio URL").rstrip("/")


def d1_timeout() -> float:
    """Per-request timeout in seconds (default 20)."""
    raw = os.environ.get("SYSTEMONE_D1_TIMEOUT", "").strip()
    try:
        t = float(raw)
        if t > 0:
            return t
    except (TypeError, ValueError):
        pass
    return 20.0


def d1_blend_weight() -> float:
    """D1 weight in the plan blend, clamped to [0, 1] (default 0.5)."""
    raw = os.environ.get("SYSTEMONE_D1_BLEND", "").strip()
    try:
        w = float(raw)
        return max(0.0, min(1.0, w))
    except (TypeError, ValueError):
        return 0.5


def resolve_model(name: Any) -> str:
    """The decision model to call: a client-named d1 id, else the default.

    Only ids starting with "d1" (d1-3b, d1-omni-600m, future d1-*) are
    honored; anything else resolves to SYSTEMONE_D1_MODEL / "d1-3b".
    """
    default = os.environ.get("SYSTEMONE_D1_MODEL", "").strip() or _DEFAULT_MODEL
    if isinstance(name, str):
        candidate = name.strip()
        if candidate.lower().startswith("d1"):
            return candidate
    return default


def _answers(
    model: str,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """POST /v1/systemone -> the answers mapping, or None (fail-open)."""
    if not d1_enabled():
        return None
    try:
        url = d1_base_url() + "/v1/systemone"
        body = {"model": model, "state": state, "questions": questions}
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(  # nosec B310 -- scheme enforced in d1_base_url via patterns.require_http_url; nosemgrep
            req, timeout=timeout if timeout is not None else d1_timeout()
        ) as resp:
            if resp.status != 200:
                return None
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None  # LM Studio down, model unloaded, malformed — fail open
    if not isinstance(payload, dict):
        return None
    answers = payload.get("answers")
    return answers if isinstance(answers, dict) else None


def _clamp01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def route_tier_via_d1(
    task: str,
    candidates: List[Dict[str, str]],
    model: Any = None,
) -> Optional[Dict[str, Any]]:
    """D1's tier choice for *task*; None on any failure.

    ``candidates``: [{"tier": name, "description": str}] in capability
    order. Returns {"tier", "confidence", "probabilities", "model"}.
    """
    tiers = [c["tier"] for c in candidates if c.get("tier")]
    if not tiers:
        return None
    criteria = {
        c["tier"]: (c.get("description") or c["tier"]) for c in candidates
    }
    used = resolve_model(model)
    answers = _answers(
        used,
        task,
        {
            "route": {
                "type": "choice",
                "instructions": (
                    "Route the task to the compute tier that fits it "
                    "best: the cheapest tier that can do the job well, "
                    "escalating only when the work genuinely needs the "
                    "capability."
                ),
                "criteria": criteria,
            }
        },
    )
    if not answers:
        return None
    ans = answers.get("route")
    if not isinstance(ans, dict):
        return None
    tier = ans.get("choice")
    if tier not in tiers:
        return None
    probs = ans.get("probabilities")
    probabilities = (
        {t: _clamp01(probs.get(t)) for t in tiers}
        if isinstance(probs, dict)
        else {}
    )
    try:
        confidence = float(ans.get("confidence"))
    except (TypeError, ValueError):
        confidence = probabilities.get(tier, 0.0)
    return {
        "tier": tier,
        "confidence": round(_clamp01(confidence), 4),
        "probabilities": probabilities,
        "model": used,
    }


def decide_via_d1(
    state_text: str,
    qtype: str,
    instructions: str,
    criteria: Any,
    model: Any = None,
) -> Optional[Dict[str, Any]]:
    """One typed decision from D1, shaped like the sidecar's reply.

    Returns {"type", "label"|"level", "probabilities"|"distribution",
    "confidence", "model"} or None (fail-open). ``criteria`` is the
    request's raw criteria (choice: {label: description} or [labels];
    score: [descriptions] or {"0": description, ...}; noul: optional
    {yes/no: description}).
    """
    used = resolve_model(model)
    question: Dict[str, Any] = {"type": qtype, "instructions": instructions}
    if qtype == "choice":
        crit: Dict[str, str] = {}
        if isinstance(criteria, dict):
            crit = {str(k): (str(v) if v is not None else str(k))
                    for k, v in criteria.items()}
        elif isinstance(criteria, (list, tuple)):
            crit = {str(v): str(v) for v in criteria}
        if len(crit) < 2:
            return None
        question["criteria"] = crit
    elif qtype == "score":
        levels: List[str] = []
        if isinstance(criteria, (list, tuple)):
            levels = [str(v) for v in criteria]
        elif isinstance(criteria, dict):
            try:
                levels = [str(criteria[str(i)]) for i in range(len(criteria))]
            except (KeyError, TypeError):
                return None
        if len(levels) < 2:
            return None
        question["criteria"] = levels
    elif qtype == "noul":
        # Fold any yes/no descriptions into the instructions; the D1
        # noul head takes no criteria (proven wire behavior).
        extra = []
        if isinstance(criteria, dict):
            if criteria.get("yes"):
                extra.append(f"'yes' means: {criteria['yes']}")
            if criteria.get("no"):
                extra.append(f"'no' means: {criteria['no']}")
        if extra:
            question["instructions"] = instructions + "\n" + "\n".join(extra)
    else:
        return None

    answers = _answers(used, state_text, {"q": question})
    if not answers:
        return None
    ans = answers.get("q")
    if not isinstance(ans, dict):
        return None

    if qtype == "choice":
        label = ans.get("choice")
        probs = ans.get("probabilities")
        if not isinstance(label, str) or not isinstance(probs, dict):
            return None
        probabilities = {str(k): round(_clamp01(v), 4)
                         for k, v in probs.items()}
        if label not in probabilities:
            return None
        try:
            confidence = float(ans.get("confidence"))
        except (TypeError, ValueError):
            confidence = probabilities.get(label, 0.0)
        return {
            "type": "choice",
            "label": label,
            "probabilities": probabilities,
            "confidence": round(_clamp01(confidence), 4),
            "model": used,
        }
    if qtype == "score":
        probs = ans.get("probabilities")
        if not isinstance(probs, dict) or not probs:
            return None
        distribution = {str(k): round(_clamp01(v), 4)
                        for k, v in probs.items()}
        # Argmax over level probabilities — never the rounded
        # expectation (the sidecar's serving doctrine).
        level = max(distribution, key=lambda k: distribution[k])
        try:
            confidence = float(ans.get("confidence"))
        except (TypeError, ValueError):
            confidence = distribution[level]
        return {
            "type": "score",
            "level": int(level),
            "distribution": distribution,
            "confidence": round(_clamp01(confidence), 4),
            "model": used,
        }
    # noul: the answer carries P(yes) only.
    p_yes = ans.get("noul")
    if p_yes is None:
        probs = ans.get("probabilities")
        if isinstance(probs, dict) and "yes" in probs:
            p_yes = probs["yes"]
    if p_yes is None:
        return None
    p = _clamp01(p_yes)
    return {
        "type": "noul",
        "label": "yes" if p >= 0.5 else "no",
        "probabilities": {"yes": round(p, 4), "no": round(1.0 - p, 4)},
        "confidence": round(max(p, 1.0 - p), 4),
        "model": used,
    }


def rank_plans_via_d1(
    task: str,
    plans: List[Dict[str, Any]],
    model: Any = None,
) -> Optional[List[Dict[str, Any]]]:
    """D1's P(plan succeeds | task) per plan, or None (fail-open).

    All plans ride in ONE /v1/systemone call as parallel noul
    questions. Returns [{"id": ..., "p_success": float}] — the same
    shape as jeff1.rank_plans_via_jeff1, so blend_rankings consumes
    it unchanged.
    """
    if not plans:
        return None
    questions: Dict[str, Dict[str, Any]] = {}
    for i, plan in enumerate(plans):
        text = str(plan.get("text") or "")
        questions[f"plan_{i}"] = {
            "type": "noul",
            "instructions": (
                f"Plan: {text}\n\nThis plan succeeds at the task."
            ),
        }
    answers = _answers(resolve_model(model), task, questions)
    if not answers:
        return None
    out: List[Dict[str, Any]] = []
    for i, plan in enumerate(plans):
        ans = answers.get(f"plan_{i}")
        if not isinstance(ans, dict) or ans.get("noul") is None:
            return None
        out.append({
            "id": plan.get("id", f"plan_{i}"),
            "p_success": _clamp01(ans["noul"]),
        })
    return out
