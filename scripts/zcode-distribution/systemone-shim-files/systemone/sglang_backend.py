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
  calls stay on the tiny local engine (multimodal decisions need
  SYSTEMONE_ENGINE=jev with a VLM — /v1/decisions itself is text-only).
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

Schema notes (SGLang ``/v1/decisions``, verified 2026-10-02 against
sglang main ``python/sglang/srt/entrypoints/openai/{protocol,serving_decisions}.py``):
    request:  {"input": str (non-blank),
               "questions": [{"id": str (non-blank, distinct),
                              "type": "choice"|"score"|"yes_no",
                              "question": str (non-blank),
                              "options": [{"name": str, "description"?: ...}, ...] |  (choice)
                              "levels": [str, ...]}]}                             (score)
    choice supports 2-26 options (A-Z single-token labels; /v1/decisions
    refuses more — the two-letter scheme past 26 exists only on SGLang's
    /v1/systemone route). score supports 2-10 levels, sent as BARE
    strings; the server renders dicts as compact JSON, so {"name": ...}
    items would corrupt the prompt.
    option names must be non-blank, free of control/line-break chars,
    and distinct case-insensitively (mirrored client-side).
    response: {"object": "decisions", "model": str,
               "prompt_format_version": int,
               "answers": {id: {"type": ..., "probabilities": {...},
               "choice": <argmax name> | "score": <index-weighted mean>,
               "label_mass": <full-vocab label probability>}},
               "usage": {"prompt_tokens": ..., "total_tokens": ...}}
    score probabilities are keyed by LEVEL INDEX ("0"-"9"), not by level
    name — remapped positionally onto the caller's levels here.
    yes_no answers carry probabilities {"yes": p, "no": 1-p} (no
    "probability"/"answer" keys upstream).
    ``label_mass`` is the uncertainty signal: a low value means the model
    wanted an out-of-vocabulary answer. We surface it on every answer,
    plus ``prompt_format_version``/``usage`` in ``_meta``.

Calibration caveat: SGLang's probabilities can drift ~0.07 between cold
and prefix-cached requests. For the calibration battery, either pin the
cache state or quantify the drift before comparing against decider-4b.

Version status (verified 2026-10-02): /v1/decisions and /v1/systemone are
main-branch only — not in any tagged SGLang release. Pin a nightly
build; re-check before relying on a release.

Image input is CONFIRMED UNSUPPORTED upstream: `input` is
string/object/array rendered as compact JSON, with no image path, so
`images=` is dropped (reported in ``_meta["media_dropped"]``) and only
the state text is sent. Point SYSTEMONE_ENGINE=jev at a VLM for images.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import time
import unicodedata
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


def check_option_names(names: Sequence[str], *, what: str = "option") -> None:
    """Mirror SGLang's ``check_option_names`` (protocol.py) client-side.

    Names must be non-blank, free of control/line-break characters (each
    option renders as one prompt line), and distinct case-insensitively.
    Fail fast here so a mis-shaped call never pays a server round-trip.
    """
    seen = set()
    for name in names:
        if not isinstance(name, str):
            raise SGLangError(
                f"{what} names must be strings, got {type(name).__name__}",
                hint="pass a list of strings (or {'name': ...} mappings)",
            )
        key = name.strip().casefold()
        if not key:
            raise SGLangError(
                f"{what} names must be nonempty",
                hint="drop the blank entry",
            )
        if any(unicodedata.category(c) in ("Cc", "Zl", "Zp") for c in name):
            raise SGLangError(
                f"{what} name {name!r} must not contain control or "
                "line break characters",
                hint="each option renders as one prompt line upstream",
            )
        if key in seen:
            raise SGLangError(
                f"{what} name {name!r} repeats another option",
                hint="names must be distinct (case-insensitive)",
            )
        seen.add(key)


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
            if not isinstance(name, str) or not name.strip():
                raise SGLangError(
                    f"question id must be a non-blank string, got {name!r}",
                    hint="/v1/decisions refuses blank ids ('must not be blank')",
                )
            qtype = q.get("type")
            if qtype == "choice":
                options = list(q.get("options") or [])
                if not (_MIN_CHOICE_OPTIONS <= len(options) <= _MAX_CHOICE_OPTIONS):
                    raise SGLangError(
                        f"choice question {name!r} has {len(options)} options",
                        hint=f"/v1/decisions supports "
                        f"{_MIN_CHOICE_OPTIONS}-{_MAX_CHOICE_OPTIONS} options",
                    )
                names = _option_names(options)
                check_option_names(names)
                prompt = q.get("prompt") or (
                    "Choose the best option for the input above."
                )
                wire_options: List[Dict[str, Any]] = []
                for raw, nm in zip(options, names):
                    item: Dict[str, Any] = {"name": nm}
                    if isinstance(raw, dict) and raw.get("description") is not None:
                        item["description"] = raw["description"]
                    wire_options.append(item)
                out.append({
                    "id": name,
                    "type": "choice",
                    "question": prompt,
                    "options": wire_options,
                })
            elif qtype == "score":
                levels = _option_names(list(q.get("levels") or []))
                if not (_MIN_SCORE_LEVELS <= len(levels) <= _MAX_SCORE_LEVELS):
                    raise SGLangError(
                        f"score question {name!r} has {len(levels)} levels",
                        hint=f"/v1/decisions supports "
                        f"{_MIN_SCORE_LEVELS}-{_MAX_SCORE_LEVELS} levels",
                    )
                if any(not lv.strip() for lv in levels):
                    raise SGLangError(
                        f"score question {name!r} has a blank level",
                        hint="/v1/decisions levels must not be blank",
                    )
                prompt = q.get("prompt") or "Rate the input above."
                out.append({
                    "id": name,
                    "type": "score",
                    "question": prompt,
                    # Bare strings: the server renders dicts as compact
                    # JSON, so {"name": ...} items would corrupt the prompt.
                    _SCORE_LEVELS_KEY: levels,
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
        """State text -> /v1/decisions input.

        Always the plain state string: /v1/decisions has no image path
        (input is string/object/array rendered as compact JSON), so
        `images` are dropped and reported in ``_meta["media_dropped"]``
        instead of being sent as content parts the server would render
        as JSON text.
        """
        return state

    # -- main entry point ------------------------------------------------
    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[str] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer typed questions about `state` via SGLang /v1/decisions.

        Question shapes are identical to api.SystemOne.systemone; choice
        options may be bare strings or {"name", "description"?} mappings
        (descriptions are forwarded), score levels may likewise be bare
        strings or {"name"} mappings (sent as bare strings — the server
        renders dicts as JSON). `images`/`videos` are accepted for
        protocol uniformity and reported in ``_meta["media_dropped"]``
        (/v1/decisions has no media path).
        Returns {name: answer_dict, ..., "_meta": {...}} with the same
        answer shapes as the local engine, plus "label_mass" per answer
        and the server's prompt_format_version/usage in "_meta".
        """
        questions = list(questions)
        if not questions:
            raise SGLangError("questions must be non-empty")
        if not isinstance(state, str) or not state.strip():
            raise SGLangError(
                "state must be a non-blank string",
                hint="/v1/decisions refuses blank input ('must not be blank')",
            )
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
                label_lists[q["name"]] = _option_names(list(q["options"]))
            elif q["type"] == "score":
                label_lists[q["name"]] = _option_names(list(q["levels"]))
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
                raw = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
                # Upstream keys score probabilities by level INDEX
                # ("0"-"9"); remap positionally onto the caller's levels.
                missing = [str(i) for i in range(len(labs)) if str(i) not in raw]
                if missing:
                    raise SGLangError(
                        f"SGLang answer for {name!r} lacks level keys "
                        f"{missing}",
                        hint="expected index-keyed probabilities "
                        "('0'-'9') per /v1/decisions",
                    )
                probs = {lv: raw[str(i)] for i, lv in enumerate(labs)}
                best = max(probs, key=probs.get) if probs else None
                _check_distribution(probs, labs, best)
                # SGLang's index-weighted mean, when provided; else argmax index.
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
                # Upstream shape: probabilities {"yes": p, "no": 1-p}.
                probs = (ans.get("probabilities") or {})
                p_yes = probs.get("yes", ans.get("probability", 0.5))
                if not isinstance(p_yes, (int, float)):
                    p_yes = 0.5
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
        if payload.get("prompt_format_version") is not None:
            answers["_meta"]["prompt_format_version"] = payload[
                "prompt_format_version"
            ]
        if isinstance(payload.get("usage"), dict):
            answers["_meta"]["usage"] = dict(payload["usage"])
        dropped_images = len(list(images or []))
        dropped_videos = len(list(videos or []))
        if dropped_images or dropped_videos:
            answers["_meta"]["media_dropped"] = {
                "images": dropped_images,
                "videos": dropped_videos,
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
        op_options = _option_names(list(operation["options"]))
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
            t_options = _option_names(list(tq["options"]))
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


class CascadeBackend:
    """FrugalGPT-style budgeted cascade over N engines, cheap first.

    Each call is judged by stage 0; when every answer's confidence meets
    `escalate_below` those answers win. Otherwise the whole call
    re-judges at the next stage, until a stage clears the bar, the
    `budget` (sum of per-stage `costs`) would be exceeded, or stages run
    out. Escalation is whole-call like HybridBackend (never a mix of two
    judges' calibrations), and a failing stage fails open to the last
    good answers with ``escalation_error`` in ``_meta``.

    The gate is the calibrated min-confidence (this package's fitted
    per-type temperatures make it honest); pick the threshold offline
    with calibration.route_threshold_for_target instead of guessing.
    ``_meta`` records the answering stage, stages run, cost spent, and
    whether the budget stopped an escalation, so operators can audit
    the cascade's behavior per call.

    Torch-free and duck-typed like HybridBackend.
    """

    backend_name = "cascade"

    def __init__(
        self,
        stages: Sequence[Any],
        escalate_below: float = 0.6,
        budget: float | None = None,
        costs: Sequence[float] | None = None,
    ) -> None:
        stages = list(stages)
        if not stages:
            raise ValueError("a cascade needs at least one stage")
        if not 0.0 <= float(escalate_below) <= 1.0:
            raise ValueError(
                f"escalate_below must be in [0, 1], got {escalate_below!r}")
        if costs is None:
            costs = [0.0] * len(stages)
        costs = [float(c) for c in costs]
        if len(costs) != len(stages):
            raise ValueError(
                f"{len(costs)} costs for {len(stages)} stages")
        if any(c < 0 for c in costs):
            raise ValueError("stage costs must be >= 0")
        if budget is not None and float(budget) < 0:
            raise ValueError(f"budget must be >= 0, got {budget!r}")
        self.stages = stages
        self.escalate_below = float(escalate_below)
        self.budget = None if budget is None else float(budget)
        self.costs = costs
        self.model_name = (
            "cascade(" + ",".join(
                str(getattr(s, "model_name", "?")) for s in stages) + ")"
        )

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[str] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        current = self._call(self.stages[0], state, questions, images, videos)
        spent = self.costs[0]
        floor = HybridBackend._min_confidence(current)
        stage = 0
        budget_stopped = False
        error: str | None = None
        while floor < self.escalate_below and stage + 1 < len(self.stages):
            nxt = self.costs[stage + 1]
            if self.budget is not None and spent + nxt > self.budget:
                budget_stopped = True
                break
            try:
                current = self._call(
                    self.stages[stage + 1], state, questions, images, videos)
            except Exception as exc:
                error = str(exc)[:200]
                break
            stage += 1
            spent += nxt
            floor = HybridBackend._min_confidence(current)
        meta = dict(current.get("_meta", {}))
        meta["backend"] = "cascade"
        meta["stage"] = stage
        meta["stages_run"] = stage + 1
        meta["escalated"] = stage > 0
        meta["min_confidence"] = round(floor, 4)
        meta["cost_spent"] = round(spent, 4)
        meta["budget_exhausted"] = budget_stopped
        if error is not None:
            meta["escalation_error"] = error
        current["_meta"] = meta
        return current

    @staticmethod
    def _call(
        engine: Any,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None,
        videos: Sequence[Any] | None,
    ) -> Dict[str, Any]:
        return HybridBackend._call(engine, state, questions, images, videos)


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
