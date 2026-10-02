"""SGLang backend for SystemOne decisions.

Same question/answer shapes as :class:`systemone.api.SystemOne`
(choice / score / noul, one batched call), but the judgments come from a
model served by SGLang's ``/v1/decisions`` endpoint instead of a local
GLiClass checkpoint.

Why this exists
---------------
SGLang turns any served chat model (including VLMs) into a *decision model*:
one POST carries the state plus N typed questions, and the server reads
per-option probabilities straight off next-token scores at the answer
position — no text is generated, no output is parsed. That buys three
things the local GLiClass engine cannot do:

* **Bigger judges** — route hard calls to a 27B-class model while trivial
  calls stay on the tiny local engine.
* **Multimodal decisions** — pass screenshots (or any image) alongside the
  state text when the served model is a VLM.
* **Prefix-cache speed** — ``/v1/decisions`` is prefill-only, so a stable
  prompt prefix is served from SGLang's RadixAttention KV cache; sub-100ms
  decisions are architecturally plausible on a decent GPU.

This module never imports torch or gliclass (only stdlib plus the shared
patterns helper), so it can run on any machine that can reach the SGLang
server — e.g. the Mac calling an SGLang instance on the PC.

Configuration (env):
    SGLANG_BASE_URL   e.g. http://127.0.0.1:30000  (default)
    SGLANG_MODEL      model name to send; unset -> omitted (the server's
                      served model is used)
    SGLANG_TIMEOUT    request timeout seconds (default 30)
    SGLANG_TEMPERATURE divides label logits pre-softmax (default 1.0)

Serve SGLang with e.g.:
    python -m sglang.launch_server --model-path Qwen/Qwen3.8-27B \
        --host 127.0.0.1 --port 30000

Schema notes (SGLang ``/v1/decisions``, docs.sglang.io, 2026-09-30):
    request:  {"input": str,
               "questions": [{"id": str, "type": "choice"|"score"|"yes_no",
                              "question": str,
                              "options": [{"name": str}, ...] |  (choice)
                              "levels": [{"name": str}, ...]}]} (score)
    choice supports 2-26 options (beyond 26 SGLang switches to two-letter
    labels with order-dependent priors — we refuse >26 rather than let
    calibration silently rot). score supports 2-10 levels.
    response: {"answers": {id: {"type": ..., "probabilities": {...},
               "choice": <argmax> | "score": <prob-weighted mean>,
               "label_mass": <full-vocab label probability>}}, ...}
    ``label_mass`` is the uncertainty signal: a low value means the model
    wanted an out-of-vocabulary answer. We surface it on every answer.

Calibration caveat: SGLang's probabilities can drift ~0.07 between cold
and prefix-cached requests. For the calibration battery, either pin the
cache state or quantify the drift before comparing against decider-4b.

Version status (verified 2026-09-30): /v1/decisions and /v1/systemone are
main-branch/nightly only — not in any tagged SGLang release (newest PyPI
was 0.5.20). Pin a nightly build; re-check before relying on a release.

Image input is UNVERIFIED: the decisions docs describe `input` as
string/object/array rendered as compact JSON, with no image path (the
Pokemon demo's "live game state" was structured data, not screenshots).
`images=` sends OpenAI-style content parts anyway, but confirm against a
live nightly server before depending on it.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence

from .patterns import choice_confidence, noul_confidence, score_confidence

# SGLang's documented limits for /v1/decisions. We enforce them client-side
# so a mis-shaped call fails fast and loudly instead of returning quietly
# miscalibrated answers (see module docstring on >26 options).
_MAX_CHOICE_OPTIONS = 26
_MIN_CHOICE_OPTIONS = 2
_MAX_SCORE_LEVELS = 10
_MIN_SCORE_LEVELS = 2

# Key carrying the ordered levels of a score question in the /v1/decisions
# request body. Kept as a constant so it can be adjusted in one place if a
# future SGLang version renames it.
_SCORE_LEVELS_KEY = "levels"


class SGLangError(RuntimeError):
    """Sanitized SGLang backend failure.

    Mirrors SystemOneError's philosophy: safe to surface to callers and
    logs. Never echoes URLs with credentials or response bodies beyond a
    short, scrubbed summary.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


def _check_distribution(
    prob_map: Dict[str, float], ids: Sequence[str], best: str, tol: float = 0.02
) -> Dict[str, float]:
    """Response-contract check, mirroring api.validate_distribution.

    Kept local (stdlib-only) so this backend never imports torch/gliclass.
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
        raise SGLangError(
            "SGLang returned an invalid decision distribution",
            hint="keys must match the question options, values must be "
            "finite probabilities summing to ~1, and the chosen label must "
            "hold the max probability",
        )
    return prob_map


def _option_names(options: Sequence[Any]) -> List[str]:
    """Accept SGLang option items as {"name": str} or bare strings."""
    names: List[str] = []
    for o in options:
        if isinstance(o, dict):
            names.append(str(o.get("name", o)))
        else:
            names.append(str(o))
    return names


class SGLangBackend:
    """Decision engine backed by SGLang's /v1/decisions.

    Drop-in for :class:`systemone.api.SystemOne` wherever only
    ``systemone()`` / ``speculative_decide()`` are used: identical question
    shapes in, identical Jev-shaped answers out (plus ``label_mass`` on
    every answer and ``backend: "sglang"`` in ``_meta``).
    """

    backend_name = "sglang"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        temperature: float = 1.0,
    ) -> None:
        from .patterns import require_http_url

        self.base_url = require_http_url(
            (base_url or os.environ.get("SGLANG_BASE_URL") or "http://127.0.0.1:30000")
            .strip()
            .rstrip("/"),
            what="SGLang base URL",
        )
        self.model = model or (os.environ.get("SGLANG_MODEL") or "").strip() or None
        env_timeout = (os.environ.get("SGLANG_TIMEOUT") or "").strip()
        self.timeout = (
            timeout
            if timeout is not None
            else (float(env_timeout) if env_timeout else 30.0)
        )
        env_temp = (os.environ.get("SGLANG_TEMPERATURE") or "").strip()
        self.temperature = float(env_temp) if env_temp else float(temperature)
        # Friendly name for logs/_meta; the served model is authoritative.
        self.model_name = self.model or f"sglang@{self.base_url}"

    # -- transport ------------------------------------------------------
    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                return json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise SGLangError(
                f"SGLang request failed: HTTP {e.code}",
                hint=f"detail: {detail} (is the server up at {self.base_url}?)"
                if detail
                else f"is the server up at {self.base_url}?",
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise SGLangError(
                "could not reach the SGLang server",
                hint=f"{type(e).__name__}; tried {self.base_url} "
                "(set SGLANG_BASE_URL)",
            ) from None

    def health(self) -> bool:
        """True when the SGLang server answers a health probe."""
        for path in ("/health", "/healthz"):
            try:
                req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    # -- question translation -------------------------------------------
    def _build_questions(
        self, questions: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """api.py question shapes -> SGLang /v1/decisions questions."""
        out: List[Dict[str, Any]] = []
        for q in questions:
            name = q.get("name", "q")
            qtype = q.get("type")
            if qtype == "choice":
                options = list(q.get("options") or [])
                if not (_MIN_CHOICE_OPTIONS <= len(options) <= _MAX_CHOICE_OPTIONS):
                    raise SGLangError(
                        f"choice question {name!r} has {len(options)} options",
                        hint=f"/v1/decisions supports "
                        f"{_MIN_CHOICE_OPTIONS}-{_MAX_CHOICE_OPTIONS} options",
                    )
                prompt = q.get("prompt") or (
                    "Choose the best option for the input above."
                )
                out.append({
                    "id": name,
                    "type": "choice",
                    "question": prompt,
                    "options": [{"name": o} for o in options],
                })
            elif qtype == "score":
                levels = list(q.get("levels") or [])
                if not (_MIN_SCORE_LEVELS <= len(levels) <= _MAX_SCORE_LEVELS):
                    raise SGLangError(
                        f"score question {name!r} has {len(levels)} levels",
                        hint=f"/v1/decisions supports "
                        f"{_MIN_SCORE_LEVELS}-{_MAX_SCORE_LEVELS} levels",
                    )
                prompt = q.get("prompt") or "Rate the input above."
                out.append({
                    "id": name,
                    "type": "score",
                    "question": prompt,
                    _SCORE_LEVELS_KEY: [{"name": lv} for lv in levels],
                })
            elif qtype == "noul":
                statement = q.get("statement") or q.get("prompt") or ""
                if not statement:
                    raise SGLangError(
                        f"noul question {name!r} has no statement",
                        hint="provide 'statement' (or 'prompt')",
                    )
                out.append({"id": name, "type": "yes_no", "question": statement})
            else:
                raise SGLangError(
                    f"unknown question type: {qtype!r}",
                    hint="expected one of: choice, score, noul",
                )
        # Question ids must be unique — the response is keyed by id.
        ids = [qq["id"] for qq in out]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise SGLangError(
                f"duplicate question ids: {sorted(dupes)}",
                hint="question 'name' values must be unique",
            )
        return out

    def _build_input(
        self, state: str, images: Sequence[str] | None
    ) -> Any:
        """State text (+ optional images) -> /v1/decisions input.

        Plain string when text-only. When images are provided (URLs or
        data URIs), OpenAI-style content parts — EXPERIMENTAL: the
        decisions docs describe no image path (input is string/object/array
        rendered as compact JSON), so verify against a live nightly server
        before relying on it. The parts form is only used when images are
        present.
        """
        if not images:
            return state
        parts: List[Dict[str, Any]] = [{"type": "text", "text": state}]
        for img in images:
            parts.append({"type": "image_url", "image_url": {"url": img}})
        return parts

    # -- main entry point ------------------------------------------------
    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[str] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer typed questions about `state` via SGLang /v1/decisions.

        Question shapes are identical to api.SystemOne.systemone; `images`
        optionally carries image URLs / data URIs (experimental — the
        endpoint's image support is undocumented; verify live first).
        `videos` is accepted for protocol uniformity and reported in
        ``_meta["media_dropped"]`` (/v1/decisions has no video path).
        Returns {name: answer_dict, ..., "_meta": {...}} with the same
        answer shapes as the local engine, plus "label_mass" per answer.
        """
        questions = list(questions)
        if not questions:
            raise SGLangError("questions must be non-empty")
        if len(state) > 6000:
            state = state[:6000]

        sgl_questions = self._build_questions(questions)
        body: Dict[str, Any] = {
            "input": self._build_input(state, images),
            "questions": sgl_questions,
        }
        if self.model:
            body["model"] = self.model
        if self.temperature != 1.0:
            body["temperature"] = self.temperature

        t0 = time.perf_counter()
        payload = self._post("/v1/decisions", body)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        raw_answers = payload.get("answers")
        if not isinstance(raw_answers, dict):
            raise SGLangError(
                "SGLang returned no 'answers' mapping",
                hint="expected {\"answers\": {id: {...}}}",
            )

        answers: Dict[str, Any] = {}
        label_lists: Dict[str, List[str]] = {}
        for q in questions:
            if q["type"] == "choice":
                label_lists[q["name"]] = list(q["options"])
            elif q["type"] == "score":
                label_lists[q["name"]] = list(q["levels"])
            else:
                label_lists[q["name"]] = ["yes", "no"]

        for q in questions:
            name, qtype = q["name"], q["type"]
            ans = raw_answers.get(name)
            if not isinstance(ans, dict):
                raise SGLangError(
                    f"SGLang returned no answer for question {name!r}",
                    hint="the server may have dropped or renamed the id",
                )
            label_mass = ans.get("label_mass")
            labs = label_lists[name]
            if qtype == "choice":
                probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
                best = ans.get("choice")
                if best is None:
                    best = max(probs, key=probs.get) if probs else None
                _check_distribution(probs, labs, best)
                answers[name] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": probs,
                    "confidence": choice_confidence(list(probs.values())),
                    "label_mass": label_mass,
                }
            elif qtype == "score":
                probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
                best = max(probs, key=probs.get) if probs else None
                _check_distribution(probs, labs, best)
                # SGLang's prob-weighted mean, when provided; else argmax index.
                wmean = ans.get("score")
                if not isinstance(wmean, (int, float)):
                    wmean = float(labs.index(best)) if best in labs else 0.0
                answers[name] = {
                    "type": "score",
                    "level": best,
                    "distribution": probs,
                    "score": float(wmean),
                    "confidence": score_confidence(
                        [probs[lv] for lv in labs]),
                    "label_mass": label_mass,
                    "legend": dict(q.get("legend") or {}),
                }
            else:  # noul <- yes_no
                p_yes = ans.get("probability")
                if not isinstance(p_yes, (int, float)):
                    # Fallback: some servers return a yes/no distribution.
                    probs = (ans.get("probabilities") or {})
                    p_yes = float(probs.get("yes", 0.5))
                p_yes = float(p_yes)
                answers[name] = {
                    "type": "noul",
                    "probability": p_yes,
                    "answer": bool(p_yes >= 0.5),
                    "confidence": noul_confidence(p_yes),
                    "label_mass": label_mass,
                }

        answers["_meta"] = {
            "backend": "sglang",
            "model": self.model_name,
            "base_url": self.base_url,
            "n_questions": len(questions),
            "latency_ms": round(latency_ms, 1),
            "state_chars": len(state),
        }
        if videos:
            answers["_meta"]["media_dropped"] = {
                "images": 0,
                "videos": len(list(videos)),
            }
        return answers

    def speculative_decide(
        self,
        state: str,
        operation: Dict[str, Any],
        targets: Dict[str, Dict[str, Any]],
        images: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Decide an operation AND its argument in a single batched call.

        Same contract as api.SystemOne.speculative_decide: one request
        carries the operation choice plus one target choice-head per
        operation; only the head selected by the chosen operation is used.
        """
        op_name = operation.get("name", "operation")
        op_options = list(operation["options"])
        questions: List[Dict[str, Any]] = [operation]
        for op in op_options:
            if op in targets:
                questions.append(targets[op])

        answers = self.systemone(state, questions, images=images)

        op_answer = answers[op_name]
        _check_distribution(
            op_answer["probabilities"], op_options, op_answer["choice"]
        )
        op_choice = op_answer["choice"]

        target = None
        target_conf: float | None = None
        target_probs: Dict[str, float] = {}
        if op_choice in targets:
            tq = targets[op_choice]
            t_name = tq.get("name", f"{op_choice}_target")
            t_options = list(tq["options"])
            t_answer = answers[t_name]
            _check_distribution(
                t_answer["probabilities"], t_options, t_answer["choice"]
            )
            target = t_answer["choice"]
            target_conf = t_answer["confidence"]
            target_probs = t_answer["probabilities"]

        return {
            "operation": op_choice,
            "target": target,
            "confidence": op_answer["confidence"],
            "target_confidence": target_conf,
            "probabilities": target_probs,
            "operation_probabilities": op_answer["probabilities"],
            "_meta": answers["_meta"],
        }


class HybridBackend:
    """Local-first engine with SGLang escalation on low confidence.

    Every call is first judged by `local_engine` (any object with
    ``systemone(state, questions)`` — usually api.SystemOne). When every
    answer's confidence meets `escalate_below`, the local answers are
    returned as-is. When any answer falls below it, the whole call is
    re-judged by `sglang` (a SGLangBackend) and the SGLang answers win.

    Escalation is whole-call, not per-question: mixing two judges'
    calibrations inside one answer set would silently rot the confidence
    semantics. ``_meta`` records which judge answered
    (``backend: "hybrid/local" | "hybrid/sglang"``) plus the minimum local
    confidence observed, so callers can audit the escalation rate.

    Either side may be omitted: local=None degrades to pure SGLang,
    sglang=None degrades to pure local. SGLang failures while escalating
    fail open to the local answers with ``escalation_error`` in ``_meta``
    instead of raising — the local judge already answered, so dropping its
    answers for a transport error would be strictly worse.

    Torch-free and duck-typed: this class never imports torch/gliclass,
    so it can be constructed (with sglang-only) on a slim install.
    """

    backend_name = "hybrid"

    def __init__(
        self,
        local_engine: Any | None = None,
        sglang: "SGLangBackend | None" = None,
        escalate_below: float = 0.6,
    ) -> None:
        self.local = local_engine
        self.sglang = sglang
        self.escalate_below = float(escalate_below)
        local_name = getattr(local_engine, "model_name", None) or "none"
        sglang_name = getattr(sglang, "model_name", None) or "none"
        self.model_name = f"hybrid(local={local_name},sglang={sglang_name})"

    @staticmethod
    def _min_confidence(answers: Dict[str, Any]) -> float:
        confs = [
            ans.get("confidence", 0.0)
            for key, ans in answers.items()
            if key != "_meta" and isinstance(ans, dict)
        ]
        vals = [float(c) for c in confs if isinstance(c, (int, float))]
        return min(vals) if vals else 0.0

    @staticmethod
    def _call(
        engine: Any,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None,
        videos: Sequence[Any] | None,
    ) -> Dict[str, Any]:
        """Call engine.systemone, forwarding media it accepts.

        Non-empty media is passed only for keywords the engine declares,
        so older stubs with ``systemone(state, questions)`` keep working.
        """
        kwargs: Dict[str, Any] = {}
        if images or videos:
            try:
                params = inspect.signature(engine.systemone).parameters
            except (TypeError, ValueError):
                params = {}
            if images and "images" in params:
                kwargs["images"] = images
            if videos and "videos" in params:
                kwargs["videos"] = videos
        return engine.systemone(state, questions, **kwargs)

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[str] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        if self.local is None:
            if self.sglang is None:
                raise SGLangError(
                    "HybridBackend has neither a local engine nor an "
                    "SGLang backend configured"
                )
            answers = self._call(self.sglang, state, questions, images, videos)
            meta = dict(answers.get("_meta", {}))
            meta["backend"] = "hybrid/sglang"
            meta["escalated"] = True
            meta["escalation_reason"] = "no local engine configured"
            answers["_meta"] = meta
            return answers

        local_answers = self._call(self.local, state, questions, images, videos)
        floor = self._min_confidence(local_answers)
        if floor >= self.escalate_below or self.sglang is None:
            meta = dict(local_answers.get("_meta", {}))
            meta["backend"] = "hybrid/local"
            meta["escalated"] = False
            meta["min_local_confidence"] = round(floor, 4)
            local_answers["_meta"] = meta
            return local_answers

        try:
            answers = self._call(self.sglang, state, questions, images, videos)
        except Exception as exc:
            # Any escalation failure fails open to the local answers (see
            # the class docstring): the local judge already answered, so
            # dropping its answers would be strictly worse.
            meta = dict(local_answers.get("_meta", {}))
            meta["backend"] = "hybrid/local"
            meta["escalated"] = False
            meta["min_local_confidence"] = round(floor, 4)
            meta["escalation_error"] = str(exc)[:200]
            local_answers["_meta"] = meta
            return local_answers
        meta = dict(answers.get("_meta", {}))
        meta["backend"] = "hybrid/sglang"
        meta["escalated"] = True
        meta["min_local_confidence"] = round(floor, 4)
        answers["_meta"] = meta
        return answers


def decide_fn_for(engine: Any):
    """Adapt any engine to a See-Decide-Act loop judge.

    Returns fn(state, questions, images, videos) -> answers matching the
    DecisionLoop judge protocol. Non-empty media is forwarded only for
    keywords the engine declares, so older stubs with
    ``systemone(state, questions)`` keep working. Lets agent loops (see
    examples/demo_decision_loop.py) swap engines without changing loop code.
    """
    try:
        params = inspect.signature(engine.systemone).parameters
    except (TypeError, ValueError):
        params = {}

    def fn(state: str, questions: List[Dict[str, Any]],
           images: Any = (), videos: Any = ()) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if images and "images" in params:
            kwargs["images"] = images
        if videos and "videos" in params:
            kwargs["videos"] = videos
        return engine.systemone(state, questions, **kwargs)

    return fn
