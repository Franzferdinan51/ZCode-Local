"""Jeff-1 second decision head — shim-side HTTP client.

GestaltLabs/Jeff-1 (Apache-2.0) is a LoRA adapter on Qwen3-4B-Instruct-2507
that answers Jev-style typed questions (choice / noul / score) from the
language model's own next-token distribution. SystemOne consults it as an
*advisory* second head behind a local sidecar (systemone/jeff1_sidecar.py):

- ``POST /v1/systemone/rank-plans``: blend the GLiClass plan scores with
  Jeff-1's P(plan succeeds | task) when the sidecar is reachable.
- ``POST /v1/systemone/route`` on *uncertain* routes: record Jeff-1's tier
  second opinion. Advisory only — it never changes the routed tier.

Hot-path rule: certain routes never touch Jeff-1.

Configuration — no hard-coded knobs:

- ``SYSTEMONE_JEFF1``         "1"/"0" — enable the head (default "1" = ON)
- ``SYSTEMONE_JEFF1_URL``      sidecar base URL
                              (default "http://127.0.0.1:8079")
- ``SYSTEMONE_JEFF1_TIMEOUT``  per-request seconds (default "2.5")
- ``SYSTEMONE_JEFF1_BLEND``    sidecar weight in the plan blend (default "0.5")

Fail-open everywhere: connection refused, timeout, malformed replies —
every public helper returns None / degrades instead of raising.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_URL = "http://127.0.0.1:8079"
DEFAULT_TIMEOUT = 2.5
DEFAULT_BLEND = 0.5
_RANK_PLANS_PATH = "/v1/jeff1/rank-plans"
_SECOND_OPINION_PATH = "/v1/jeff1/second-opinion"
_DECIDE_PATH = "/v1/jeff1/decide"

_OFF_VALUES = {"0", "false", "no", "off", "disabled"}


def jeff1_enabled() -> bool:
    """Is the Jeff-1 head on? Default ON; only explicit off-values disable."""
    raw = os.environ.get("SYSTEMONE_JEFF1", "1").strip().lower()
    return raw not in _OFF_VALUES


def jeff1_url() -> str:
    """Sidecar base URL (default http://127.0.0.1:8079).

    Raises ValueError on a non-http(s) URL (fail-closed scheme check;
    _post swallows it into the usual fail-open None).
    """
    from .patterns import require_http_url

    return require_http_url(
        os.environ.get("SYSTEMONE_JEFF1_URL", "").strip() or DEFAULT_URL,
        what="sidecar URL",
    )


def jeff1_timeout() -> float:
    """Per-request timeout in seconds (default 2.5)."""
    raw = os.environ.get("SYSTEMONE_JEFF1_TIMEOUT", "").strip()
    try:
        t = float(raw)
        if t > 0:
            return t
    except (TypeError, ValueError):
        pass
    return DEFAULT_TIMEOUT


def jeff1_blend_weight() -> float:
    """Jeff-1 weight in the plan blend, clamped to [0, 1] (default 0.5)."""
    raw = os.environ.get("SYSTEMONE_JEFF1_BLEND", "").strip()
    try:
        w = float(raw)
        return max(0.0, min(1.0, w))
    except (TypeError, ValueError):
        return DEFAULT_BLEND


def _post(path: str, payload: Dict[str, Any],
          timeout: Optional[float]) -> Optional[Dict[str, Any]]:
    """POST JSON to the sidecar; parsed body or None on any failure."""
    if not jeff1_enabled():
        return None
    try:
        url = jeff1_url().rstrip("/") + path
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(  # nosec B310 -- scheme enforced in jeff1_url() via patterns.require_http_url; nosemgrep
            req, timeout=timeout if timeout is not None else jeff1_timeout()
        ) as resp:
            if resp.status != 200:
                return None
            body = json.loads(resp.read().decode("utf-8"))
        return body if isinstance(body, dict) else None
    except Exception:
        return None  # fail-open: refused, timeout, malformed — all the same


def rank_plans_via_jeff1(
    task: str,
    plans: List[Dict[str, Any]],
    timeout: Optional[float] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Jeff-1's P(plan succeeds | task) per plan, or None (fail-open).

    Returns ``[{"id": ..., "p_success": float}]``. Validates the sidecar's
    reply shape; anything unexpected degrades to None.
    """
    body = _post(
        _RANK_PLANS_PATH,
        {"task": task, "plans": [{"id": p.get("id"), "text": p.get("text")}
                                for p in plans]},
        timeout,
    )
    if body is None:
        return None
    ranking = body.get("ranking")
    if not isinstance(ranking, list) or not ranking:
        return None
    out = []
    for entry in ranking:
        if not isinstance(entry, dict):
            return None
        try:
            p = float(entry["p_success"])
        except (KeyError, TypeError, ValueError):
            return None
        out.append({"id": entry.get("id"), "p_success": max(0.0, min(1.0, p))})
    return out


def second_opinion(
    task: str,
    route_summary: Dict[str, Any],
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Jeff-1's tier second opinion, or None (fail-open).

    ``route_summary``: {"tier", "confidence", "margin",
    "candidates": [{"tier", "description"}]}.
    Returns {"tier", "confidence", "agree", "rationale"}.
    """
    body = _post(
        _SECOND_OPINION_PATH,
        {
            "task": task,
            "tier": route_summary.get("tier"),
            "confidence": route_summary.get("confidence"),
            "margin": route_summary.get("margin"),
            "candidates": route_summary.get("candidates") or [],
        },
        timeout,
    )
    if body is None or not isinstance(body.get("tier"), str):
        return None
    try:
        conf = float(body.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {
        "tier": body["tier"],
        "confidence": max(0.0, min(1.0, conf)),
        "agree": bool(body.get("agree", False)),
        "rationale": str(body.get("rationale", "")),
    }


def blend_rankings(
    gliclass_ranking: List[Dict[str, Any]],
    jeff1_ranking: List[Dict[str, Any]],
    tier_cost: Optional[float] = None,
    weight: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Blend GLiClass and Jeff-1 plan scores (pure; never raises).

    Per plan present in both rankings:
        p = (1 - w) * gliclass.p_success + w * jeff1.p_success
    then the same cost penalty as scoring.rank_plans
    (est_steps * tier_cost / 100, 0.0 when tier_cost is None) is re-applied
    and the list is re-sorted by score desc. Plans missing from the Jeff-1
    reply keep their GLiClass score (fail-open). Output shape matches
    scoring.rank_plans.
    """
    w = jeff1_blend_weight() if weight is None else max(0.0, min(1.0, weight))
    jeff = {e["id"]: e["p_success"] for e in jeff1_ranking
            if isinstance(e, dict) and "id" in e}
    blended = []
    for g in gliclass_ranking:
        if not isinstance(g, dict):
            continue
        pid = g.get("id")
        try:
            gp = float(g.get("p_success", 0.5))
        except (TypeError, ValueError):
            gp = 0.5
        blended_w = pid in jeff
        try:
            jp = float(jeff[pid]) if blended_w else gp
        except (TypeError, ValueError):
            jp = gp
            blended_w = False
        p = (1.0 - w) * gp + w * jp
        try:
            steps = int(g.get("est_steps", 1))
        except (TypeError, ValueError):
            steps = 1
        try:
            penalty = (round(steps * float(tier_cost) / 100.0, 4)
                       if tier_cost is not None else 0.0)
        except (TypeError, ValueError, ArithmeticError):
            penalty = 0.0
        blended.append({
            "id": pid,
            "score": round(p - penalty, 4),
            "p_success": round(p, 4),
            "cost_penalty": penalty,
            "est_steps": steps,
            "jeff1_blended": blended_w,
        })
    blended.sort(key=lambda r: (r["score"] is not None, r["score"]),
                 reverse=True)
    return blended


def decide_via_jeff1(
    payload: Dict[str, Any],
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Forward a typed decision request to the sidecar's /v1/jeff1/decide.

    ``payload``: {"state", "instructions", "criteria", "type"} — the
    sidecar's decide schema (the same shape the shim's
    /v1/systemone/decide accepts). Returns the sidecar's reply dict,
    validated to carry a known "type", or None on any failure — sidecar
    down, timeout, HTTP 404 (endpoint not deployed yet), malformed body.
    Never raises: fail-open everywhere, like the other helpers here.
    """
    body = _post(_DECIDE_PATH, payload, timeout)
    if body is None:
        return None
    if body.get("type") not in ("choice", "noul", "score"):
        return None
    return body
