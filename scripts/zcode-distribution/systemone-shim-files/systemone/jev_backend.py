"""systemone.jev_backend — JEV decision-model backend (AutoTrust JEV-27B-VL style).

Speaks the decision-model wire format from the JEV-27B-VL model card
(https://huggingface.co/autotrust/JEV-27B-VL) and the JEV-27B-DEMO client
(https://github.com/yuhai-china/JEV-27B-DEMO, ``common/jev_client.py``):

- System 1, hosted (default): ``POST {JEV_URL}/v1/decide`` with
  ``{kind, state, question, options?}`` where kind is ``noul`` | ``choice``
  | ``score`` and state is a string, a JSON object, or a list mixing text
  and images. Works against ``serve_decide.py`` (vLLM + ``/v1/decide``),
  hosted Jev APIs (``Authorization: Bearer $JEV_API_KEY``), and JevK5
  servers exposing the same route. One forward pass, calibrated
  probabilities, no text generation.
- System 1, vLLM-raw (``JEV_BACKEND=vllm``): one-token
  ``/v1/completions`` call with client-side bias + per-kind temperature
  math from a *local* ``JEV_BUNDLE`` directory (``adapter_vllm/
  decision_head.json`` + ``calibration.json``). Stdlib math only.
- System 2: :meth:`JevDecideBackend.chat` — the unmodified base model in
  the same engine, optionally thinking step by step.

Configuration — no hard-coded knobs:

- ``JEV_URL``      server base URL (default ``http://127.0.0.1:8000``)
- ``JEV_API_KEY``  bearer token for hosted APIs (default: none)
- ``JEV_BACKEND``  ``decide`` (default) | ``vllm``
- ``JEV_BUNDLE``   local bundle dir for the vllm-raw path (default: none)
- ``JEV_MODEL``    served model name (default ``autotrust/JEV-27B-VL``)
- ``JEV_TIMEOUT``  per-request seconds (default ``120``)

Fail-open everywhere: unreachable servers, missing bundles, and malformed
replies raise :class:`JevError` with a hint (never a bare traceback), and
the shim's engine selection treats an unhealthy JEV server as absent.
"""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Sequence

JEV_KINDS = ("noul", "choice", "score")
JEV_NOUL_OPTIONS = ["false", "true"]
JEV_SCORE_LEVELS = [str(i) for i in range(6)]
JEV_DEFAULT_MODEL = "autotrust/JEV-27B-VL"
JEV_MAX_CHOICE_OPTIONS = 256  # per the JEV-27B-VL model card
_LETTERS = "ABCDEFGHIJKLMNOP"


class JevError(RuntimeError):
    """JEV backend failure (unreachable server, bad reply, missing bundle)."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint

    def __str__(self) -> str:  # pragma: no cover - trivial
        base = super().__str__()
        return f"{base} ({self.hint})" if self.hint else base


def as_text(x: Any) -> str:
    """JEV state scalar -> text (mirrors jev_client.as_text)."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def image_part(img: Any) -> Dict[str, Any]:
    """Image reference -> JEV ``{"image": url}`` state part.

    Accepts http(s) URLs, data URLs, local file paths, and OpenAI-style
    ``{"type": "image_url", "image_url": {"url": ...}}`` parts (normalized
    to the model card's ``{"image": ...}`` form).
    """
    if isinstance(img, dict):
        if "image" in img:
            return {"image": str(img["image"])}
        inner = img.get("image_url") or {}
        if isinstance(inner, dict) and inner.get("url"):
            return {"image": str(inner["url"])}
        if isinstance(inner, str) and inner:
            return {"image": inner}
        raise JevError(f"unrecognized image part: {img!r:.120}")
    if isinstance(img, str) and img.startswith(("http://", "https://", "data:")):
        return {"image": img}
    try:
        path = str(img)
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            blob = base64.b64encode(f.read()).decode("ascii")
        return {"image": f"data:{mime};base64,{blob}"}
    except JevError:
        raise
    except Exception as exc:
        raise JevError(
            f"could not read image {img!r}",
            hint=f"{type(exc).__name__}: pass a URL, data URL, or path",
        ) from None


def build_state(
    state: Any, images: Sequence[Any] | None = None
) -> Any:
    """Assemble a JEV ``state`` value: text (+ optional image parts).

    Imageless states pass through untouched — strings stay strings, JSON
    values (including dicts) stay as-is, lists stay lists. When images are
    present the state becomes an ordered list mixing the original state
    with ``{"image": ...}`` parts, per the model card.
    """
    imgs = [image_part(img) for img in (images or [])]
    if not imgs:
        return state if state is not None else ""
    if isinstance(state, list):
        return list(state) + imgs
    if state is None:
        return imgs
    return [state] + imgs


class JevDecideBackend:
    """Decision engine backed by a JEV decision model (System 1 + System 2).

    Drop-in for :class:`systemone.api.SystemOne` wherever ``systemone()``
    is used: identical question shapes in, identical Jev-shaped answers
    out (plus ``backend: "jev"`` in ``_meta``). :meth:`decide` exposes the
    raw JEV ``/v1/decide`` response; :meth:`chat` exposes System 2.
    """

    backend_name = "jev"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        backend: str | None = None,
        bundle_dir: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        from .patterns import require_http_url

        self.base_url = require_http_url(
            (base_url or os.environ.get("JEV_URL") or "http://127.0.0.1:8000")
            .strip()
            .rstrip("/")
            .removesuffix("/v1"),
            what="JEV base URL",
        )
        key = api_key if api_key is not None else (os.environ.get("JEV_API_KEY") or "")
        self.api_key = key.strip()
        sel = (backend or os.environ.get("JEV_BACKEND") or "decide").strip().lower()
        if sel not in ("decide", "vllm"):
            raise JevError(
                f"unknown JEV backend {sel!r}",
                hint="JEV_BACKEND must be 'decide' or 'vllm'",
            )
        self.backend = sel
        self.bundle_dir = (
            bundle_dir if bundle_dir is not None else (os.environ.get("JEV_BUNDLE") or "")
        ).strip()
        self.model = (
            model or (os.environ.get("JEV_MODEL") or "").strip() or JEV_DEFAULT_MODEL
        )
        env_timeout = (os.environ.get("JEV_TIMEOUT") or "").strip()
        self.timeout = (
            timeout
            if timeout is not None
            else (float(env_timeout) if env_timeout else 120.0)
        )
        self.model_name = self.model
        self._bundle_cache: tuple[Dict[str, Any], Dict[str, float]] | None = None

    # -- transport ------------------------------------------------------
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=self._headers(), method="POST"
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
            raise JevError(
                f"JEV request failed: HTTP {e.code} on {path}",
                hint=f"detail: {detail} (is the server up at {self.base_url}?)"
                if detail
                else f"is the server up at {self.base_url}?",
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise JevError(
                "could not reach the JEV server",
                hint=f"{type(e).__name__}; tried {self.base_url} (set JEV_URL)",
            ) from None

    def health(self) -> bool:
        """True when the JEV server answers a probe."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/decide/info",
                headers=self._headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                if resp.status == 200:
                    return True
        except Exception:
            pass
        for path in ("/health", "/healthz"):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}{path}", headers=self._headers(), method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                    if resp.status == 200:
                        return True
            except Exception:
                continue
        return False

    # -- System 1 --------------------------------------------------------
    def decide(
        self,
        kind: str,
        state: Any,
        question: str,
        options: Sequence[str] | None = None,
        images: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """One System 1 decision: calibrated probabilities, one forward pass.

        Returns the JEV response mapping: ``options`` (list),
        ``probabilities`` (parallel list), ``choice`` / ``choice_index``,
        ``kind`` / ``effective_kind``, ``model``, ``usage``,
        ``num_model_requests``, and ``elapsed_seconds``.
        """
        if kind not in JEV_KINDS:
            raise JevError(
                f"unknown decision kind {kind!r}",
                hint=f"kind must be one of {', '.join(JEV_KINDS)}",
            )
        if kind == "noul":
            opts = list(JEV_NOUL_OPTIONS)
        elif kind == "score":
            opts = list(JEV_SCORE_LEVELS)
        else:
            opts = [str(o) for o in (options or [])]
            if not (2 <= len(opts) <= JEV_MAX_CHOICE_OPTIONS):
                raise JevError(
                    f"choice needs 2-{JEV_MAX_CHOICE_OPTIONS} options, "
                    f"got {len(opts)}",
                    hint="pass options=[...] with at least 2 entries",
                )
        if not isinstance(question, str) or not question.strip():
            raise JevError(
                "question must be a non-empty string",
                hint="ask one question about the state",
            )
        full_state = build_state(state, images)
        if self.backend == "vllm":
            return self._decide_vllm(kind, full_state, question.strip(), opts)
        return self._decide_hosted(kind, full_state, question.strip(), opts)

    def _decide_hosted(
        self, kind: str, state: Any, question: str, options: List[str]
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"kind": kind, "state": state, "question": question}
        if kind == "choice":
            body["options"] = options
        t0 = time.perf_counter()
        payload = self._post("/v1/decide", body)
        elapsed = time.perf_counter() - t0
        try:
            opts = [str(o) for o in payload["options"]]
            probs = [float(p) for p in payload["probabilities"]]
        except (KeyError, TypeError, ValueError):
            raise JevError(
                "JEV server returned no options/probabilities",
                hint='expected {"options": [...], "probabilities": [...]}',
            ) from None
        if len(opts) != len(probs) or not opts:
            raise JevError(
                "JEV server returned mismatched options/probabilities",
                hint=f"got {len(opts)} options vs {len(probs)} probabilities",
            )
        idx = int(payload.get("choice_index", max(range(len(probs)), key=probs.__getitem__)))
        if not (0 <= idx < len(opts)):
            idx = max(range(len(probs)), key=probs.__getitem__)
        # The probabilities are the source of truth: derive the choice from
        # the normalized index instead of trusting a possibly disagreeing
        # server label, so choice == options[choice_index] always holds.
        return {
            "kind": kind,
            "effective_kind": str(payload.get("effective_kind", kind)),
            "options": opts,
            "probabilities": probs,
            "choice_index": idx,
            "choice": opts[idx],
            "adaptation": payload.get("adaptation", "native"),
            "protocol": payload.get("protocol", "jev27-bare-v1"),
            "model": payload.get("model", self.model),
            "usage": payload.get("usage", {}),
            "num_model_requests": int(payload.get("num_model_requests", 1)),
            "elapsed_seconds": float(payload.get("elapsed_seconds", elapsed)),
        }

    def _load_bundle(self) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Decision-head bias + per-kind temperatures from JEV_BUNDLE."""
        if self._bundle_cache is not None:
            return self._bundle_cache
        if not self.bundle_dir:
            raise JevError(
                "the vllm-raw path needs a local decision bundle",
                hint="set JEV_BUNDLE to the model dir (adapter_vllm/ + calibration.json)",
            )
        try:
            with open(
                os.path.join(self.bundle_dir, "adapter_vllm", "decision_head.json"),
                encoding="utf-8",
            ) as f:
                head = json.load(f)
            with open(
                os.path.join(self.bundle_dir, "calibration.json"), encoding="utf-8"
            ) as f:
                temps = json.load(f)["per_kind"]
        except (OSError, ValueError, KeyError) as exc:
            raise JevError(
                f"could not load the decision bundle at {self.bundle_dir}",
                hint=f"{type(exc).__name__}: need adapter_vllm/decision_head.json "
                "and calibration.json['per_kind']",
            ) from None
        self._bundle_cache = (head, {k: float(v) for k, v in temps.items()})
        return self._bundle_cache

    def _decide_vllm(
        self, kind: str, state: Any, question: str, options: List[str]
    ) -> Dict[str, Any]:
        """Client-side decision math over plain ``vllm serve``.

        Mirrors ``jev_client._decide_vllm``: one token, option-token
        logprobs, decision-head bias, per-kind temperature. ``top_k: 0``
        and ``top_p: 1.0`` are required — the model's generation config
        sets ``top_k=20`` / ``top_p=0.95``, and vLLM would otherwise
        truncate the returned distribution (per the model card).
        """
        head, temps = self._load_bundle()
        try:
            slot = int(head["slots"]["ranges"][kind][0])
            verbalizer = [int(t) for t in head["verbalizer_ids"]]
            bias = [float(b) for b in head["bias"]]
            temperature = float(temps[kind])
        except (KeyError, TypeError, ValueError) as exc:
            raise JevError(
                "decision bundle has no usable head for kind "
                f"{kind!r}",
                hint=f"{type(exc).__name__}: need slots/ranges, verbalizer_ids, bias",
            ) from None
        ids = verbalizer[slot : slot + len(options)]
        if len(ids) != len(options):
            raise JevError(
                f"bundle verbalizer covers {len(ids)} options, need {len(options)}",
                hint="the vllm-raw path caps at the bundle's slot width",
            )
        lines = (
            options
            if kind != "choice"
            else [f"{_LETTERS[i]}) {o}" for i, o in enumerate(options)]
        )
        prompt = (
            f"[kind] {kind}\n[state] {as_text(state)}\n[question] {question}\n"
            "[options]\n" + "\n".join(lines) + "\n[decision]:"
        )
        t0 = time.perf_counter()
        payload = self._post(
            "/v1/completions",
            {
                "model": "jev-decision",
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "logprobs": len(options),
                "allowed_token_ids": ids,
                "add_special_tokens": False,
                "return_tokens_as_token_ids": True,
            },
        )
        elapsed = time.perf_counter() - t0
        try:
            top = payload["choices"][0]["logprobs"]["top_logprobs"][0]
            logp = {int(k.split(":")[1]): float(v) for k, v in top.items()}
        except (KeyError, IndexError, TypeError, ValueError):
            raise JevError(
                "vLLM server returned no token logprobs",
                hint="need choices[0].logprobs.top_logprobs with token ids",
            ) from None
        logits = [
            (logp.get(t, -1e9) + bias[slot + i]) / temperature
            for i, t in enumerate(ids)
        ]
        peak = max(logits)
        exps = [math.exp(x - peak) for x in logits]
        total = sum(exps)
        probs = [x / total for x in exps]
        idx = max(range(len(probs)), key=probs.__getitem__)
        return {
            "kind": kind,
            "effective_kind": kind,
            "options": options,
            "probabilities": probs,
            "choice_index": idx,
            "choice": options[idx],
            "adaptation": "client-math",
            "protocol": "jev27-bare-v1",
            "model": self.model,
            "usage": payload.get("usage", {}),
            "num_model_requests": 1,
            "elapsed_seconds": elapsed,
        }

    def decide_many(
        self,
        reqs: Sequence[tuple],
        workers: int = 8,
    ) -> List[Dict[str, Any]]:
        """Batch decisions concurrently; the server batches them on GPU.

        ``reqs`` are ``(kind, state, question[, options])`` tuples, mirroring
        ``jev_client.decide_many``.
        """
        import concurrent.futures as cf

        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            return list(ex.map(lambda a: self.decide(*a), reqs))

    # -- System 1 in SystemOne shapes ------------------------------------
    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer typed questions about `state` via the JEV server.

        Question shapes are identical to api.SystemOne.systemone; `images`
        optionally carries image URLs / data URIs / paths, which travel in
        the JEV state list (a VLM-backed server such as JEV-27B-VL reads
        them; text-only servers ignore or reject them per their build).
        `videos` is accepted for protocol uniformity and reported in
        ``_meta["media_dropped"]`` (JEV decision models read images, not
        video). Returns {name: answer_dict, ..., "_meta": {...}} with the
        same answer shapes as the local engine.
        """
        from .patterns import (
            choice_confidence,
            noul_confidence,
            score_confidence,
        )

        questions = list(questions)
        if not questions:
            raise JevError("questions must be non-empty")
        answers: Dict[str, Any] = {}
        t0 = time.perf_counter()
        for q in questions:
            name = q.get("name", "q")
            qtype = q.get("type")
            prompt = q.get("prompt") or q.get("question") or ""
            if qtype == "choice":
                options = [str(o) for o in (q.get("options") or [])]
                resp = self.decide(
                    "choice", state, prompt or "Which option fits best?",
                    options, images=images,
                )
                probs = dict(zip(resp["options"], resp["probabilities"]))
                answers[name] = {
                    "type": "choice",
                    "choice": resp["choice"],
                    "probabilities": probs,
                    "confidence": choice_confidence(resp["probabilities"]),
                    "label_mass": None,
                }
            elif qtype == "score":
                levels = [str(lv) for lv in (q.get("levels") or [])]
                if levels == JEV_SCORE_LEVELS:
                    resp = self.decide(
                        "score", state, prompt or "Rate the input above.",
                        images=images,
                    )
                    probs = dict(zip(resp["options"], resp["probabilities"]))
                else:
                    if len(levels) < 2:
                        raise JevError(
                            f"score question {name!r} needs >= 2 levels",
                            hint="provide 'levels' with at least 2 entries",
                        )
                    # Custom level sets travel as a choice over the levels.
                    resp = self.decide(
                        "choice", state, prompt or "Rate the input above.",
                        levels, images=images,
                    )
                    probs = dict(zip(resp["options"], resp["probabilities"]))
                best = max(probs, key=probs.get) if probs else None
                try:
                    coords = [float(lv) for lv in probs]
                except ValueError:
                    # Non-numeric levels: expected position, not value.
                    coords = [float(i) for i in range(len(probs))]
                total = sum(probs.values())
                wmean = (
                    sum(c * p for c, p in zip(coords, probs.values())) / total
                    if probs and total else 0.0
                )
                answers[name] = {
                    "type": "score",
                    "level": best,
                    "distribution": probs,
                    "score": float(wmean),
                    "confidence": score_confidence(
                        [probs[o] for o in resp["options"]])
                    if probs else 0.0,
                    "label_mass": None,
                    "legend": dict(q.get("legend") or {}),
                }
            else:  # noul <- yes/no statement
                statement = q.get("statement") or prompt or ""
                if not statement:
                    raise JevError(
                        f"noul question {name!r} needs a statement",
                        hint="provide 'statement' (or 'prompt')",
                    )
                resp = self.decide(
                    "noul", state,
                    f"Is this scenario one where: {statement}",
                    images=images,
                )
                probs = dict(zip(resp["options"], resp["probabilities"]))
                p_true = float(probs.get("true", 0.5))
                answers[name] = {
                    "type": "noul",
                    "probability": p_true,
                    "answer": bool(p_true >= 0.5),
                    "confidence": noul_confidence(p_true),
                    "label_mass": None,
                }
        answers["_meta"] = {
            "backend": "jev",
            "model": self.model_name,
            "base_url": self.base_url,
            "n_questions": len(questions),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "state_chars": len(state),
        }
        if videos:
            answers["_meta"]["media_dropped"] = {
                "images": 0,
                "videos": len(list(videos)),
            }
        return answers

    # -- System 2 ----------------------------------------------------------
    def chat(
        self,
        prompt: str,
        thinking: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        """System 2: the unmodified base model, optionally thinking.

        Returns the final answer text (reasoning stripped when the server
        returns ``<think>...</think>``-style content).
        """
        payload = self._post(
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.6 if thinking else 0.0,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )
        try:
            msg = payload["choices"][0]["message"]
            content = msg.get("content") or ""
        except (KeyError, IndexError, TypeError):
            raise JevError(
                "JEV server returned no chat message",
                hint="need choices[0].message.content",
            ) from None
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        return content.replace("<think>", "").strip()
