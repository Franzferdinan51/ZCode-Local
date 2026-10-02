"""Decision patterns: light building blocks shared by every engine.

Pure-python pieces with no torch dependency (numpy only, a base dependency),
so the SGLang path, the shim, the CLI, and the demos can use them on a slim
install:

- SystemOneError / MAX_STATE_CHARS: sanitized failure type + state bound
- validate_distribution / validate_choice: response-contract checks
- with_abstain / ABSTAIN_LABEL: explicit "none of the above" option
- StallGuard: fail-fast loop/stall detector for agent loops
- LatencyStats: p50/p95 aggregator over per-call latency_ms
- make_questions: convenience builder for question lists
- choice/score/noul_confidence: TypeSafe-compatible confidence definitions
- build_decision_prompts: state-first prompt rows with option shuffling
- MODEL_CANDIDATES / _resolve_candidates: smallest-first load order + env override

The decision patterns are ported from Ryan's jev-ultrafast / mobile-jev
agent repos (both TypeSafe-hosted apps; what transfers is their
battle-tested decision-engineering discipline, not their transport).
``systemone.api`` re-exports every name here, so ``systemone.api.X`` and
``systemone.X`` keep working unchanged.
"""

from __future__ import annotations

import math
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np

# States larger than this are capped before inference (same bound Loki uses
# for the routing task). The encoder truncates to 512 tokens anyway, so the
# cap only bounds memory/log noise — it does not change judgments.
MAX_STATE_CHARS = 6000

# Largest request body the shim/sidecar will read (DoS bound). Legitimate
# decision requests are kilobytes; even media-heavy Clef bodies with data
# URLs stay far below this. Over-cap requests get HTTP 413.
MAX_BODY_BYTES = 2 * 1024 * 1024

# Largest question batch per request (each question is inference work).
MAX_QUESTIONS_PER_REQUEST = 64

# Largest plan list per rank-plans request (one inference call per plan).
MAX_PLANS_PER_REQUEST = 32


class BodyTooLarge(Exception):
    """Request body exceeds MAX_BODY_BYTES (HTTP 413)."""


def check_body_length(headers: Any) -> int:
    """Validate Content-Length against MAX_BODY_BYTES.

    Returns the byte count to read. Garbage values raise ValueError (HTTP
    400); over-cap values raise BodyTooLarge (HTTP 413).
    """
    raw = headers.get("Content-Length", 0) if headers is not None else 0
    try:
        length = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"bad Content-Length: {raw!r}") from None
    if length < 0:
        raise ValueError(f"bad Content-Length: {raw!r}")
    if length > MAX_BODY_BYTES:
        raise BodyTooLarge(
            f"request body {length} bytes exceeds the {MAX_BODY_BYTES}-byte cap"
        )
    return length


API_TOKEN_ENV = "SYSTEMONE_API_TOKEN"


def api_token_required() -> str:
    """Shared-token gate for mutating routes ("" = open, the default).

    When $SYSTEMONE_API_TOKEN is set, POST routes require
    `Authorization: Bearer <token>` and answer 401 otherwise. Unset keeps
    the historic open-localhost behavior — fail-open by default.
    """
    return (os.environ.get(API_TOKEN_ENV) or "").strip()


def api_token_ok(authorization: str | None) -> bool:
    """True when the request satisfies the shared-token gate."""
    import hmac

    want = api_token_required()
    if not want:
        return True
    got = (authorization or "").strip()
    return hmac.compare_digest(got, f"Bearer {want}")

# Smallest-first candidates; the first that loads wins.
MODEL_CANDIDATES = [
    "knowledgator/gliclass-edge-v3.0",
    "knowledgator/gliclass-small-v1.0",
    "knowledgator/gliclass-base-v1.0",
]


def _resolve_candidates(model_name: str | None) -> List[str]:
    """Model load order: explicit arg wins, then the SYSTEMONE_MODEL env var,
    then MODEL_CANDIDATES smallest-first. Keeps the error hint below honest —
    an override that doesn't change load order is just aspirational."""
    chosen = (model_name or "").strip() or (os.environ.get("SYSTEMONE_MODEL") or "").strip()
    return [chosen] if chosen else list(MODEL_CANDIDATES)


REVISION_ENV = "SYSTEMONE_REVISION"


def resolve_revision(explicit: str | None = None) -> str | None:
    """HF revision pin: explicit arg wins, then $SYSTEMONE_REVISION.

    Returns None when unpinned (default branch). Threaded through to
    from_pretrained calls so model/tokenizer downloads are reproducible
    and immune to tag moves.
    """
    rev = (explicit or "").strip() or (os.environ.get(REVISION_ENV) or "").strip()
    return rev or None


def require_http_url(url: str, *, what: str = "URL") -> str:
    """Fail closed unless `url` is an http(s) URL.

    urllib honors file:// and other schemes, so every operator-configured
    base URL (shim, sidecar, SGLang, LM Studio) passes through here before
    any request is built. Raises ValueError on empty/unparseable input or
    any non-http(s) scheme. No host allowlist: operators may point at
    tailnets and LAN hosts freely.
    """
    from urllib.parse import urlsplit

    text = (url or "").strip()
    try:
        scheme = urlsplit(text).scheme.lower()
    except ValueError:
        scheme = ""
    if scheme not in ("http", "https"):
        raise ValueError(f"{what} must be an http(s) URL, got {text[:80]!r}")
    return text


class SystemOneError(RuntimeError):
    """Sanitized engine failure.

    Mirrors Loki's TypeSafeRequestError philosophy: the message is safe to
    surface to callers and logs. It never echoes environment-provided
    secrets, absolute paths, or transport internals — only what went wrong
    and what to do about it.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


def validate_distribution(
    prob_map: Dict[str, float], ids: Sequence[str], best: str, tol: float = 0.02
) -> Dict[str, float]:
    """Response-contract check on a probability distribution.

    Port of jev-ultrafast's ``validate_choice`` (model.py): asserts the best
    label is one of the ids, the probability keys exactly match the ids, every
    value is finite in [0, 1], the values sum to ~1, and the best label actually
    holds the max probability. Raises SystemOneError on any violation —
    catches degenerate or malformed model outputs before they become decisions.
    """
    try:
        numbers = list(prob_map.values())
        valid = (
            best in ids
            and set(prob_map) == set(ids)
            and all(
                isinstance(n, (int, float)) and math.isfinite(n) and 0 <= n <= 1
                for n in numbers
            )
            and abs(sum(numbers) - 1.0) < tol
            and prob_map[best] >= max(numbers) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise SystemOneError(
            "model returned an invalid decision distribution",
            hint="keys must match the question options, values must be "
            "finite probabilities summing to ~1, and the chosen label must "
            "hold the max probability",
        )
    return prob_map


def validate_choice(answer: Dict[str, Any], ids: Sequence[str], tol: float = 0.02) -> Dict[str, Any]:
    """Validate a Jev-shaped choice answer: {choice, probabilities, confidence}.

    Thin wrapper over validate_distribution matching jev-ultrafast's shape.
    """
    try:
        probs = answer["probabilities"]
        choice = answer["choice"]
    except (KeyError, TypeError):
        raise SystemOneError("model returned a malformed choice answer") from None
    validate_distribution(probs, ids, choice, tol=tol)
    return answer


ABSTAIN_LABEL = "none"


def with_abstain(options: Sequence[str], label: str = ABSTAIN_LABEL) -> List[str]:
    """Append an explicit abstain option to a choice question.

    Port of mobile-jev's NONE pattern (policy.mjs): never force the model to
    pick when nothing fits — "If the desired value is missing, select NONE."
    """
    opts = list(options)
    if label not in opts:
        opts.append(label)
    return opts


class StallGuard:
    """Fail-fast loop/stall detector for decision-driven agents.

    Port of jev-ultrafast's executor discipline (agent.py): consecutive
    no-progress observations trip a stall instead of letting an agent spin
    forever. Call observe() after each executed decision.
    """

    def __init__(self, max_stalls: int = 3) -> None:
        self.max_stalls = max_stalls
        self.stalls = 0

    def observe(self, progressed: bool) -> str:
        """Record whether the last decision made progress.

        Returns "ok", or "stalled" once max_stalls consecutive no-progress
        observations accumulate.
        """
        self.stalls = 0 if progressed else self.stalls + 1
        return "stalled" if self.stalls >= self.max_stalls else "ok"

    def reset(self) -> None:
        self.stalls = 0


class LatencyStats:
    """Tiny p50/p95 latency aggregator for bench and calibration runs.

    Port of mobile-jev's metrics.mjs stats(): SystemOne already reports
    per-call latency_ms in _meta; this aggregates them across runs.
    """

    def __init__(self) -> None:
        self.values: List[float] = []

    def add(self, ms: float) -> None:
        if isinstance(ms, (int, float)) and math.isfinite(ms):
            self.values.append(float(ms))

    def summary(self) -> Dict[str, Any]:
        vals = sorted(self.values)
        n = len(vals)
        if not n:
            return {"count": 0, "median_ms": None, "p95_ms": None, "mean_ms": None}
        mid = n // 2
        median = vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2
        return {
            "count": n,
            "median_ms": round(median, 1),
            "p95_ms": round(vals[math.ceil(n * 0.95) - 1], 1),
            "mean_ms": round(sum(vals) / n, 1),
        }


def make_questions(
    choices: Dict[str, List[str]] | None = None,
    scores: Dict[str, List[str]] | None = None,
    nouls: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Convenience builder for question lists."""
    qs: List[Dict[str, Any]] = []
    for name, options in (choices or {}).items():
        qs.append({"name": name, "type": "choice", "options": options})
    for name, levels in (scores or {}).items():
        qs.append({"name": name, "type": "score", "levels": levels})
    for name, statement in (nouls or {}).items():
        qs.append({"name": name, "type": "noul", "statement": statement})
    return qs


# ---------------------------------------------------------------------------
# TypeSafe-compatible confidence definitions + state-first prompt rows.
#
# Adapted from Mapika/decider (Apache-2.0): decider/systemone.py (confidence
# formulas) and decider/prompt.py (state-first prompt-row template with
# option shuffling). The score formula is the shared reference also
# shipped by Kev (kev/api.py) and SGLang (serving_decisions.py).
# ---------------------------------------------------------------------------


_MONTHS = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
_DATE_RE = re.compile(
    rf"\b(?:{_MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"
)


def date_facts(text: str) -> str:
    """Deterministic date arithmetic, ported from Kev (kev/api.py).

    Every pair of absolute dates found in `text`, as one sentence each
    ("August 3, 2026 is 12 days after July 22, 2026."). Judges cannot
    subtract dates reliably; they can use a stated day count. Returns ""
    when fewer than two dates are found; dates in order of appearance.
    """
    found: List[tuple] = []
    for m in _DATE_RE.finditer(text or ""):
        raw = m.group(0)
        try:
            d = (
                datetime.strptime(raw, "%B %d, %Y")
                if "," in raw
                else datetime.strptime(raw, "%Y-%m-%d")
            )
        except ValueError:
            continue
        if raw not in [r for r, _ in found]:
            found.append((raw, d))
    facts = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            n = (found[j][1] - found[i][1]).days
            unit = "day" if abs(n) == 1 else "days"
            if n:
                rel = "after" if n > 0 else "before"
                facts.append(
                    f"{found[j][0]} is {abs(n)} {unit} {rel} {found[i][0]}."
                )
            else:
                facts.append(
                    f"{found[j][0]} is the same day as {found[i][0]}."
                )
    return " ".join(facts)


def with_date_facts(state: Any) -> Any:
    """State plus a `date_facts` field/paragraph when 2+ absolute dates appear.

    Object states gain a `date_facts` key, list states a trailing
    {"date_facts": ...} item, string states an appended paragraph.
    Returns the state unchanged when fewer than two dates are found.
    """
    facts = date_facts(state if isinstance(state, str) else str(state))
    if not facts:
        return state
    if isinstance(state, dict):
        return {**state, "date_facts": facts}
    if isinstance(state, list):
        return state + [{"date_facts": facts}]
    if isinstance(state, str):
        return f"{state}\n\ndate_facts: {facts}"
    return state


def choice_confidence(probs: Sequence[float]) -> float:
    """TypeSafe choice confidence: (n * p_max - 1) / (n - 1).

    The shape of the distribution collapsed to 0-1: a delta on one option
    scores 1, a uniform distribution scores 0. A single-option question
    degenerates to p_max.
    """
    p = np.asarray(list(probs), dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    if n == 1:
        return float(np.clip(p[0], 0.0, 1.0))
    return float(np.clip((n * p.max() - 1.0) / (n - 1.0), 0.0, 1.0))


def score_confidence(probs: Sequence[float]) -> float:
    """TypeSafe score confidence: max(0, 1 - spread / uniform_spread).

    `probs` must be ordered by level; spread is the probability-weighted
    distance from the mode (first argmax), and uniform_spread is the mean
    absolute deviation of a uniform distribution over the levels around
    (n-1)/2. Scores 1 when all mass sits on one level, 0 at uniform (or
    anything as spread). This is the reference formula shared by
    TypeSafe's system-one-adapter 0.2.1 (via Kev's api.py), SGLang's
    /v1/decisions + /v1/systemone serving, and Kev's api.py — identical
    values whichever judge answers. Inputs are normalized (all-zeros ->
    uniform); a single level degenerates to 1.0.
    """
    p = np.asarray(list(probs), dtype=np.float64)
    n = len(p)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    total = float(p.sum())
    if total <= 0:
        p = np.full(n, 1.0 / n)
    else:
        p = p / total
    k = int(p.argmax())
    spread = float(p @ np.abs(np.arange(n) - k))
    uniform_spread = float(np.abs(np.arange(n) - (n - 1.0) / 2.0).mean())
    return float(max(0.0, 1.0 - spread / uniform_spread))


def noul_confidence(p_yes: float) -> float:
    """TypeSafe noul confidence: the probability of the chosen answer.

    Value-is-probability: max(P(yes), P(no)).
    """
    p = float(p_yes)
    return float(max(p, 1.0 - p))


def _prompt_row(state: str, k: int, text: str, labels: Sequence[str]) -> str:
    """One state-first decision row (decider's prompt template)."""
    lines = ["Context:", state, "", f"Question [{k}]: {text}", "Options:"]
    for i, lab in enumerate(labels):
        tag = chr(ord("A") + i) if i < 26 else str(i + 1)
        lines.append(f"({tag}) {lab}")
    lines += ["", f"Answer [{k}]: ("]
    return "\n".join(lines)


def build_decision_prompts(
    state: str,
    questions: Sequence[Dict[str, Any]],
    *,
    shuffle_options: bool = False,
    seed: int | None = None,
) -> tuple[List[str], List[List[str]]]:
    """Build one state-first prompt row per question.

    Row template (adapted from decider/prompt.py)::

        Context:
        <state>

        Question [k]: <text>
        Options:
        (A) <option 1>
        (B) <option 2>

        Answer [k]: (

    Question text defaults: the explicit "prompt" for choice/score, the
    "statement" for noul, else the question name. Shuffling applies to
    choice options only -- score levels keep their order and the abstain
    label (ABSTAIN_LABEL) stays last. Deterministic when `seed` is given.

    Returns (prompts, label_lists): the row per question and the label list
    the row refers to (shuffled order included), ready for raw_scores().
    """
    rng = random.Random(seed)
    prompts: List[str] = []
    label_lists: List[List[str]] = []
    for k, q in enumerate(questions, 1):
        qtype = q.get("type")
        if qtype == "choice":
            labs = list(q.get("options") or [])
            text = q.get("prompt") or f"Choose the best option ({q.get('name')})"
        elif qtype == "score":
            labs = list(q.get("levels") or [])
            text = q.get("prompt") or f"Rate the level ({q.get('name')})"
        elif qtype == "noul":
            labs = ["yes", "no"]
            text = q.get("statement") or q.get("prompt") or q.get("name")
        else:
            raise SystemOneError(
                f"unknown question type: {qtype!r}",
                hint="expected one of: choice, score, noul",
            )
        if not labs:
            raise SystemOneError(
                f"question {q.get('name')!r} has no options/levels",
                hint="choice needs 'options', score needs 'levels'",
            )
        if shuffle_options and qtype == "choice" and len(labs) > 1:
            pinned = [lab for lab in labs if lab == ABSTAIN_LABEL]
            rest = [lab for lab in labs if lab != ABSTAIN_LABEL]
            rng.shuffle(rest)
            labs = rest + pinned
        prompts.append(_prompt_row(state, k, text, labs))
        label_lists.append(labs)
    return prompts, label_lists
