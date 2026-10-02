"""Scoring & ranking surfaces for the SystemOne shim.

Additive decision intelligence on top of the tier router in shim.py:

- calibration: temperature-scale the blended tier distribution, report
  top-1/top-2 margin and an `uncertain` flag.
- model ranking: expected-utility rank of the registry's models[] entries.
- tool ranking: GLiClass zero-shot relevance of registry tools/MCP servers.
- plan ranking: score candidate plans for /v1/systemone/rank-plans.

All tuning knobs come from tuning.json (this package's config), overridable
at runtime by the SYSTEMONE_* environment variables. No tuning defaults are
hard-coded here: if tuning.json is missing, accessors return None and the
call sites degrade gracefully (features off / unfiltered) instead of
inventing numbers.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

# -- tuning config -------------------------------------------------------------

_TUNING: Optional[Dict[str, Any]] = None


def tuning_path() -> str:
    return os.path.join(os.path.dirname(__file__), "tuning.json")


def load_tuning(path: Optional[str] = None) -> Dict[str, Any]:
    """Load tuning.json once; {} when missing/unreadable (fail-open)."""
    global _TUNING
    if _TUNING is not None:
        return _TUNING
    try:
        with open(path or tuning_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        _TUNING = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        _TUNING = {}
    return _TUNING


def reset_tuning() -> None:
    """Forget the cached tuning (tests only)."""
    global _TUNING
    _TUNING = None


def _knob(env_name: str, key: str) -> Optional[float]:
    """Env var wins; else tuning.json; else None (caller degrades)."""
    raw = os.environ.get(env_name)
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    val = load_tuning().get(key)
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def margin_floor() -> Optional[float]:
    return _knob("SYSTEMONE_MARGIN_FLOOR", "margin_floor")


def cost_lambda() -> Optional[float]:
    return _knob("SYSTEMONE_COST_LAMBDA", "cost_lambda")


def tool_floor() -> Optional[float]:
    return _knob("SYSTEMONE_TOOL_FLOOR", "tool_floor")


def tool_topk() -> Optional[int]:
    val = _knob("SYSTEMONE_TOOL_TOPK", "tool_top_k")
    return int(val) if val is not None else None


def inventory_ttl() -> Optional[float]:
    return _knob("SYSTEMONE_INVENTORY_TTL", "inventory_ttl_s")


def inventory_min_interval() -> Optional[float]:
    return _knob("SYSTEMONE_INVENTORY_MIN_INTERVAL",
                 "inventory_min_interval_s")


def disabled() -> bool:
    return os.environ.get("SYSTEMONE_DISABLE", "").strip() == "1"


# -- calibration --------------------------------------------------------------

def load_calibration(path: str) -> Optional[Dict[str, Any]]:
    """Load calibration.json; None when missing/unreadable (serve raw)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cal = json.load(f)
        T = float(cal.get("temperature", 1.0))
        if not (math.isfinite(T) and T > 0):
            return None
        return cal
    except (OSError, ValueError, TypeError):
        return None


def temperature_for(
    calibration: Optional[Dict[str, Any]], qtype: str = "choice"
) -> float:
    """Pick the serving temperature for an answer type from a calibration dict.

    Uses calibration["temperature_by_type"][qtype] when present and sane,
    else the pooled calibration["temperature"] (decider's <50-rows fallback
    is encoded by simply omitting the type from the map). Fail-open:
    anything missing or invalid -> 1.0 (identity).
    """
    def _sane(v):
        try:
            T = float(v)
        except (TypeError, ValueError):
            return None
        return T if (math.isfinite(T) and T > 0) else None

    if not calibration:
        return 1.0
    pooled = _sane(calibration.get("temperature", 1.0))
    by_type = calibration.get("temperature_by_type")
    per = _sane(by_type.get(qtype)) if isinstance(by_type, dict) else None
    T = per if per is not None else pooled
    return T if T is not None else 1.0


def calibrate_probs(probs: Dict[str, float], T: float) -> Dict[str, float]:
    """Temperature-scale an already-normalized distribution.

    softmax(log p / T): T > 1 softens overconfident distributions,
    T < 1 sharpens underconfident ones. T == 1 is the identity.
    """
    if abs(T - 1.0) < 1e-9:
        return dict(probs)
    items = list(probs.items())
    logits = [math.log(max(p, 1e-12)) / T for _, p in items]
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    total = sum(exps) or 1.0
    return {k: e / total for (k, _), e in zip(items, exps)}


def top2_margin(probs: Dict[str, float]) -> float:
    """P(top1) - P(top2) on a calibrated distribution (float)."""
    vals = sorted(probs.values(), reverse=True)
    p1 = vals[0] if vals else 0.0
    p2 = vals[1] if len(vals) > 1 else 0.0
    return max(0.0, p1 - p2)


def apply_calibration(
    probs: Dict[str, float],
    calibration: Optional[Dict[str, Any]],
    qtype: str = "choice",
) -> Dict[str, Any]:
    """Calibrated view of a blended tier distribution.

    Returns {calibrated, calibrated_probabilities, margin, uncertain,
    confidence} where confidence is the calibrated P(top1). With no
    calibration file, serves the raw distribution and marks calibrated=False.
    qtype selects the per-answer-type temperature from the calibration map
    (decider convention); the tier route is a "choice" question.
    """
    if not calibration:
        vals = sorted(probs.values(), reverse=True)
        p1 = vals[0] if vals else 0.0
        p2 = vals[1] if len(vals) > 1 else 0.0
        return {
            "calibrated": False,
            "calibrated_probabilities": dict(probs),
            "margin": round(max(0.0, p1 - p2), 4),
            "uncertain": False,  # unknown without calibration; don't guess
            "confidence": round(p1, 4),
        }
    T = temperature_for(calibration, qtype)
    cal = calibrate_probs(probs, T)
    margin = top2_margin(cal)
    floor = margin_floor()
    p1 = max(cal.values()) if cal else 0.0
    return {
        "calibrated": True,
        "calibrated_probabilities": {k: round(v, 4) for k, v in cal.items()},
        "margin": round(margin, 4),
        # No floor configured -> don't guess; fail-open (never uncertain).
        "uncertain": bool(floor is not None and margin < floor),
        "confidence": round(p1, 4),
    }


# -- tool registry ------------------------------------------------------------

def load_tool_registry(path: str) -> List[Dict[str, Any]]:
    """Load tool_registry.json -> list of tool entries (fail-open: [])."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tools = data.get("tools", [])
        return [t for t in tools if isinstance(t, dict) and t.get("id")]
    except (OSError, ValueError):
        return []


_TOOL_CHOICE_PROMPT = (
    "Which tool or MCP server would be most relevant and helpful "
    "for completing this task?"
)

# Capability-mediated relevance: deterministic keyword recall + model precision.
# The zero-shot head has strong label priors (bash/todowrite/read score ~0.9+
# on anything) and its per-label scores distort as the label count grows, so
# keywords do the high-recall candidate surfacing and the model refines order
# and vetoes keyword false positives. All weights/thresholds live in
# tuning.json; triggers live in tool_registry.json (config, not code).

_FAILS_CLAUSE_RE = re.compile(r" ?FAILS at.*$", re.IGNORECASE | re.DOTALL)
_WORD_RE = re.compile(r"[a-z0-9]+")
_KW_STOP = frozenset(
    "the a an and or for to of in on with is are was were be been this that "
    "from at by as it its they them their we you your i he she me my".split()
)

# Fixed calibration probes for measuring per-label priors ("how much does
# the head like this label when the task needs nothing"). Not tuning: they
# are just diverse ordinary tasks, and priors are remeasured per label set.
_PRIOR_PROBES = (
    "What is the capital of France?",
    "Tell me a joke about ducks.",
    "Explain how photosynthesis works.",
    "What is 15 percent of 240?",
    "Write a haiku about the ocean.",
)

_priors_cache: Dict[tuple, Dict[str, float]] = {}
_priors_lock = threading.Lock()
_inventory_lock = threading.Lock()


def _tool_option_text(tool: Dict[str, Any]) -> str:
    """Normalized option text: id + positive capability description.

    Strips the registry's "FAILS at ..." negative clauses (contrastive
    noise for the choice head), collapses whitespace, caps length.
    """
    desc = (tool.get("description") or "").strip()
    desc = _FAILS_CLAUSE_RE.sub("", desc).strip()
    desc = re.sub(r"\s+", " ", desc)
    if len(desc) > 220:
        desc = desc[:217].rstrip() + "..."
    tool_id = str(tool.get("id", ""))
    return f"{tool_id}: {desc}" if desc else tool_id


def _tool_keywords(tool: Dict[str, Any]) -> frozenset:
    """Keyword set for a tool: triggers + id + description + capabilities."""
    parts = [str(tool.get("id", "")).replace("-", " ")]
    desc = (tool.get("description") or "").split("FAILS")[0]
    parts.append(desc)
    parts.extend(str(c) for c in tool.get("capabilities", []))
    parts.extend(str(w) for w in tool.get("triggers", []))
    words = _WORD_RE.findall(" ".join(parts).lower())
    return frozenset(w for w in words if w not in _KW_STOP and len(w) > 2)


def _task_keywords(task: str) -> frozenset:
    words = _WORD_RE.findall(task.lower())
    return frozenset(w for w in words if w not in _KW_STOP and len(w) > 2)


def _measure_priors(engine: Any, options: List[str]) -> Dict[str, float]:
    """Per-label prior P(relevant) on task-neutral probes; cached per options."""
    key = tuple(options)
    with _priors_lock:
        hit = _priors_cache.get(key)
    if hit is not None:
        return hit
    priors = {o: 0.0 for o in options}
    try:
        for probe in _PRIOR_PROBES:
            sdicts = engine.raw_scores(
                [probe], [options], [_TOOL_CHOICE_PROMPT],
                classification_type="multi_label")
            scores = sdicts[0] if sdicts else {}
            for o in options:
                priors[o] += float(scores.get(o, 0.0)) / len(_PRIOR_PROBES)
    except Exception:
        return {}
    with _priors_lock:
        _priors_cache[key] = priors
    return priors


def reset_priors() -> None:
    """Forget cached label priors (tests only)."""
    with _priors_lock:
        _priors_cache.clear()


def tool_kw_weight() -> Optional[float]:
    return _knob("SYSTEMONE_TOOL_KW_WEIGHT", "tool_kw_weight")


def tool_model_weight() -> Optional[float]:
    return _knob("SYSTEMONE_TOOL_MODEL_WEIGHT", "tool_model_weight")


def tool_veto_threshold() -> Optional[float]:
    return _knob("SYSTEMONE_TOOL_VETO", "tool_veto_threshold")


def score_tools(
    engine: Any, task: str, tools: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Hybrid tool/MCP relevance: keyword recall + model precision.

    1. Keyword overlap (registry triggers/description/capabilities) surfaces
       candidates with high recall — the zero-shot head alone drowns specific
       tools (weather, browserclaw) under generic priors.
    2. One multi-label model call scores every tool; per-label priors
       (measured on neutral probes) are subtracted to remove the head's
       label bias.
    3. Model veto: keyword overlap is ignored when the model's raw score is
       below the veto threshold (kills false positives like "prices" matching
       runescape on a scraping task).
    4. relevance = kw_weight * kw_norm + model_weight * adj_norm, both
       min-max normalized; rank by relevance desc.

    Fail-open: any engine error -> [] (callers treat as skipped). If tuning
    weights are missing, falls back to model-only ranking.
    """
    if not tools:
        return []
    task_snip = task if len(task) <= 1500 else task[:1497] + "..."
    options = [_tool_option_text(t) for t in tools]
    task_kw = _task_keywords(task_snip)
    tool_kw = [_tool_keywords(t) for t in tools]
    kw_hits = [len(task_kw & kws) for kws in tool_kw]

    w_kw = tool_kw_weight()
    w_model = tool_model_weight()
    veto = tool_veto_threshold()
    try:
        if hasattr(engine, "raw_scores"):
            try:
                sdicts = engine.raw_scores(
                    [task_snip], [options], [_TOOL_CHOICE_PROMPT],
                    classification_type="multi_label")
            except TypeError:
                sdicts = engine.raw_scores(
                    [task_snip], [options], [_TOOL_CHOICE_PROMPT])
            scores = sdicts[0] if sdicts else {}
            raw = [float(scores.get(o, 0.0)) for o in options]
        else:
            ans = engine.systemone(task_snip, [{
                "name": "tools",
                "type": "choice",
                "options": options,
                "prompt": _TOOL_CHOICE_PROMPT,
            }])
            probs = (ans.get("tools") or {}).get("probabilities", {})
            raw = [float(probs.get(o, 0.0)) for o in options]
    except Exception:
        return []

    priors = _measure_priors(engine, options) if hasattr(engine, "raw_scores") else {}
    adj = []
    for o, r in zip(options, raw):
        p = priors.get(o, 0.0)
        adj.append(max(0.0, r - p))

    # model veto on keyword false positives
    if veto is not None:
        kw_eff = [h if r >= veto else 0 for h, r in zip(kw_hits, raw)]
    else:
        kw_eff = list(kw_hits)

    max_kw = max(kw_eff) if kw_eff else 0
    max_adj = max(adj) if adj else 0.0
    kw_norm = [(h / max_kw) if max_kw > 0 else 0.0 for h in kw_eff]
    adj_norm = [(a / max_adj) if max_adj > 0 else 0.0 for a in adj]

    if w_kw is None or w_model is None:
        # no tuning: model-only ranking (fail-open, no invented weights)
        combined = list(adj_norm)
    else:
        combined = [w_kw * k + w_model * a
                    for k, a in zip(kw_norm, adj_norm)]

    ranked = []
    for t, v in zip(tools, combined):
        ranked.append({
            "id": t["id"],
            "kind": t.get("kind", "tool"),
            "relevance": round(max(0.0, min(1.0, v)), 4),
        })
    ranked.sort(key=lambda r: r["relevance"], reverse=True)
    return ranked


# -- model ranking --------------------------------------------------------------

def _model_quality(entry: Dict[str, Any], tier: str) -> Optional[float]:
    """Per-tier quality or None when the registry entry lacks it.

    Missing quality is a config gap: callers skip the model rather than
    inventing a number.
    """
    q = entry.get("quality")
    if q is None:
        return None
    if isinstance(q, dict):
        v = q.get(tier, q.get("default"))
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None
    try:
        return float(q)
    except (TypeError, ValueError):
        return None


def _model_cost(entry: Dict[str, Any]) -> Optional[float]:
    """Relative cost units or None when the registry entry lacks it."""
    c = entry.get("cost")
    if c is None:
        return None
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def model_top_n() -> Optional[int]:
    raw = os.environ.get("SYSTEMONE_MODEL_TOPN")
    if raw is not None:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    val = load_tuning().get("model_top_n")
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# -- rank fusion (combine rankings from several judges) -------------------
#
# Pulled from the search/eval world: reciprocal rank fusion (Cormack et
# al., SIGIR 2009 — the hybrid-search default), Borda voting, and the
# extremized mean from forecast aggregation (which beats the plain mean
# when judges share information). EnsembleBackend consumes these.


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: float = 60.0
) -> Dict[str, float]:
    """Reciprocal rank fusion: score(c) = Σ_rankings 1/(k + rank_c).

    Each ranking lists candidate ids best-first; a candidate missing
    from a ranking contributes 0 for it. k=60 is the literature
    standard. Raises ValueError on empty input.
    """
    if not rankings:
        raise ValueError("need at least one ranking to fuse")
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, cand in enumerate(ranking):
            scores[cand] = scores.get(cand, 0.0) + 1.0 / (k + rank + 1)
    return scores


def borda_fuse(rankings: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Borda count fusion: mean (len-1-rank) points per ranking.

    First place earns len-1 points down to 0 for last; candidates
    missing from a ranking earn 0 there. Averaged (not summed) so the
    scale stays comparable as judges join. Raises ValueError on empty
    input.
    """
    if not rankings:
        raise ValueError("need at least one ranking to fuse")
    totals: Dict[str, float] = {}
    for ranking in rankings:
        n = len(ranking)
        for rank, cand in enumerate(ranking):
            totals[cand] = totals.get(cand, 0.0) + max(0, n - 1 - rank)
    return {c: v / len(rankings) for c, v in totals.items()}


def mean_distributions(
    distributions: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """Plain (linear-opinion-pool) mean of per-option distributions.

    Missing options count as 0 in that judge; the mean renormalizes to
    a simplex. Raises ValueError on empty input.
    """
    if not distributions:
        raise ValueError("need at least one distribution to average")
    keys: set = set()
    for d in distributions:
        keys.update(d)
    mean = {c: sum(float(d.get(c, 0.0)) for d in distributions)
            / len(distributions) for c in keys}
    total = sum(mean.values())
    if total <= 0:
        n = len(mean)
        return {c: 1.0 / n for c in mean} if n else {}
    return {c: v / total for c, v in mean.items()}


def extremized_average(
    distributions: Sequence[Dict[str, float]],
    strength: float = 1.0,
) -> Dict[str, float]:
    """Extremized mean: push the average away from uniform.

    e_c ∝ p̄_c + strength·(p̄_c − 1/K), clipped at 0, renormalized.
    strength=0 is the plain mean; strength=1 doubles deviations (the
    usual starting point). Extremizing beats averaging when judges'
    errors are correlated — the normal case for same-family models.
    Raises ValueError on empty input or negative strength.
    """
    if strength < 0:
        raise ValueError(f"strength must be >= 0, got {strength!r}")
    mean = mean_distributions(distributions)
    if not mean:
        return {}
    uniform = 1.0 / len(mean)
    pushed = {c: max(0.0, p + strength * (p - uniform))
              for c, p in mean.items()}
    total = sum(pushed.values())
    if total <= 0:
        return {c: uniform for c in mean}
    return {c: v / total for c, v in pushed.items()}


def rank_models(
    registry: Dict[str, Any],
    cal_probs: Dict[str, float],
    lam: Optional[float] = None,
    topn: Optional[int] = None,
    sort: str = "utility",
) -> List[Dict[str, Any]]:
    """Expected-utility rank: U(m) = Σ_t P(t)·quality(m,t) − λ·cost(m).

    Only models with available=true are ranked. Models missing quality or
    cost are skipped (can't score them; fail-open, not invented defaults).
    Advisory — never causes a model load or switch. When no cost lambda is
    configured (no tuning.json and no env), ranking degrades to pure
    expected quality (lambda = 0). topn=None returns all ranked models.

    sort (OpenRouter-style per-request control): "utility" (default),
    "quality" (expected quality desc), "cost" (cost asc), "latency"
    (latency_ms_p50 asc; entries missing it sort last). Raises
    ValueError on an unknown sort.
    """
    if sort not in ("utility", "quality", "cost", "latency"):
        raise ValueError(
            f"unknown sort {sort!r}; want utility|quality|cost|latency")
    lam = cost_lambda() if lam is None else lam
    if lam is None:
        lam = 0.0
    if topn is None:
        topn = model_top_n()
    tiers = registry.get("tiers", registry)
    if not isinstance(tiers, dict):
        return []
    scored = []
    for tier_name, entry in tiers.items():
        if not isinstance(entry, dict):
            continue
        for m in entry.get("models", []) or []:
            if not isinstance(m, dict) or not m.get("available", True):
                continue
            model_id = m.get("model_id")
            if not model_id:
                continue
            qualities = [_model_quality(m, t) for t in tiers]
            if any(q is None for q in qualities):
                continue
            cost = _model_cost(m)
            if cost is None:
                continue
            exp_quality = sum(
                float(cal_probs.get(t, 0.0)) * q
                for t, q in zip(tiers, qualities)
            )
            utility = exp_quality - lam * cost
            try:
                latency = float(m["latency_ms_p50"])
            except (KeyError, TypeError, ValueError):
                latency = None
            scored.append({
                "model_id": model_id,
                "tier": tier_name,
                "utility": round(utility, 4),
                "quality": round(exp_quality, 4),
                "cost": cost,
                "latency_ms_p50": latency,
            })
    if sort == "quality":
        scored.sort(key=lambda s: s["quality"], reverse=True)
    elif sort == "cost":
        scored.sort(key=lambda s: s["cost"])
    elif sort == "latency":
        scored.sort(key=lambda s: (s["latency_ms_p50"] is None,
                                   s["latency_ms_p50"] or 0.0))
    else:
        scored.sort(key=lambda s: s["utility"], reverse=True)
    return scored[:topn] if topn is not None else scored


# -- LM Studio inventory (availability refresh, fail-open) ------------------------

def fetch_lmstudio_models(
    base_url: Optional[str] = None, timeout: float = 1.5
) -> Optional[List[str]]:
    """GET <lmstudio>/v1/models -> [ids]. None on any failure (fail-open)."""
    from .patterns import require_http_url

    try:
        base = require_http_url(
            (base_url or os.environ.get("LMSTUDIO_BASE_URL")
             or "http://127.0.0.1:1234").rstrip("/"),
            what="LM Studio URL",
        )
    except ValueError:
        return None
    try:
        req = urllib.request.Request(base + "/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310 -- scheme enforced above via patterns.require_http_url; nosemgrep
            data = json.loads(resp.read().decode("utf-8"))
        items = data.get("data", []) if isinstance(data, dict) else []
        return [it.get("id") for it in items
                if isinstance(it, dict) and it.get("id")]
    except Exception:
        return None


def apply_inventory(
    registry: Dict[str, Any], available_ids: List[str]
) -> Dict[str, Any]:
    """Mark models[] available by live LM Studio inventory (in place)."""
    have = set(available_ids)
    tiers = registry.get("tiers", registry)
    if isinstance(tiers, dict):
        with _inventory_lock:
            for entry in tiers.values():
                if not isinstance(entry, dict):
                    continue
                for m in entry.get("models", []) or []:
                    if isinstance(m, dict) and m.get("model_id"):
                        m["available"] = m["model_id"] in have
    return registry


def start_inventory_refresher(
    registry: Dict[str, Any], interval_s: Optional[float] = None
) -> threading.Thread:
    """Background thread: refresh model availability; never raises."""
    min_interval = inventory_min_interval()
    if min_interval is None:
        # No configured floor: 30s safety minimum (liveness guard, not a
        # tuned value — prevents accidental hammering of LM Studio).
        min_interval = 30.0

    def _loop() -> None:
        ttl = interval_s or inventory_ttl() or min_interval
        while True:
            try:
                ids = fetch_lmstudio_models()
                if ids is not None:
                    apply_inventory(registry, ids)
            except Exception:
                pass
            time.sleep(max(min_interval, ttl))

    t = threading.Thread(target=_loop, name="systemone-inventory",
                         daemon=True)
    t.start()
    return t


# -- plan ranking -----------------------------------------------------------------

_STEP_RE = re.compile(r"^\s*(?:\d+[.)\-:]|[-*•])\s+\S", re.MULTILINE)


def estimate_steps(plan_text: str) -> int:
    """Rough step count: enumerated/bulleted lines, else words/30, min 1."""
    text = plan_text or ""
    steps = len(_STEP_RE.findall(text))
    if steps == 0:
        steps = max(1, len(text.split()) // 30)
    return max(1, steps)


_PLAN_STATEMENT_TEMPLATE = (
    "Task: {task}\n\n"
    "Proposed plan:\n{plan}\n\n"
    "This plan is likely to succeed at the task."
)


def rank_plans(
    engine: Any,
    task: str,
    plans: List[Dict[str, Any]],
    tier_cost: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Score candidate plans: P(plan succeeds | task) minus a cost penalty.

    One batched engine call (noul per plan). cost_penalty =
    est_steps × tier_cost / 100 (normalized); tier_cost=None means no
    penalty (fail-open when the registry lacks cost data).
    Fail-open: engine error -> plans in original order with score=None.
    """
    if not plans:
        return []
    task_snip = task if len(task) <= 1500 else task[:1497] + "..."
    questions = [
        {
            "name": f"plan_{i}",
            "type": "noul",
            "statement": _PLAN_STATEMENT_TEMPLATE.format(
                task=task_snip,
                plan=(p.get("text") or "")[:2000]),
        }
        for i, p in enumerate(plans)
    ]
    try:
        answers = engine.systemone(task_snip, questions)
    except Exception:
        return [
            {"id": p.get("id", f"plan_{i}"), "score": None,
             "p_success": None, "cost_penalty": None,
             "est_steps": estimate_steps(p.get("text") or "")}
            for i, p in enumerate(plans)
        ]
    ranked = []
    for i, p in enumerate(plans):
        pid = p.get("id", f"plan_{i}")
        try:
            ans = answers.get(f"plan_{i}") or {}
            p_success = float(ans.get("probability", 0.5))
            steps = estimate_steps(p.get("text") or "")
            penalty = (round(steps * float(tier_cost) / 100.0, 4)
                       if tier_cost is not None else 0.0)
            ranked.append({
                "id": pid,
                "score": round(p_success - penalty, 4),
                "p_success": round(p_success, 4),
                "cost_penalty": penalty,
                "est_steps": steps,
            })
        except (TypeError, ValueError, ArithmeticError):
            # One malformed answer degrades its plan, never the ranking.
            ranked.append({
                "id": pid, "score": None, "p_success": None,
                "cost_penalty": None,
                "est_steps": estimate_steps(p.get("text") or ""),
            })
    ranked.sort(key=lambda r: (r["score"] is not None, r["score"]),
                reverse=True)
    return ranked
