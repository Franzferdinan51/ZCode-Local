"""Kev-server backend: judge through a Kev /v1/systemone server.

Kev (Franzferdinan51/kev, Apache-2.0) serves TypeSafe-dialect typed
decisions from open Qwen3.5/Qwen3.8 weights in four sizes — Kev-0.8B
(laptop), Kev-4B (desktop GPU / 32 GB Mac), Kev-9B (workstation),
Kev-27B (datacentre GPU) — via ``python -m kev.serve --port 8008``.
This backend is the client side: API-style questions are rendered
into the TypeSafe dialect, judged remotely, and mapped back onto
SystemOne answers with the shared reference confidence.

Wire notes (kev/serve.py + kev/api.py, verified 2026-10-02):
- POST /v1/systemone; no /health — liveness is GET /v1/models (200
  with a "models" list). Auth is ``Authorization: Bearer <KEV_API_KEY>``
  when the server sets KEV_API_KEY (unset = open server).
- choice answers: {choice, confidence, probabilities} keyed by name.
- score answers: probabilities keyed by level INDEX ("0"-"9") with a
  {"index": level} legend — remapped positionally here.
- noul answers: {"noul": P(true)} only.
- Kev computes the same reference confidence formulas this package
  now uses (patterns.score_confidence), so recomputing locally
  reproduces the server's values.
- States are NOT truncated client-side: Kev's headline feature is
  long documents (up to 65,536 tokens on Kev-27B); over-limit
  states get a server 422 (or server-side truncation marks with
  KEV_TRUNCATE_STATES=1), which surfaces as a loud KevError.

Never imports torch: a Mac can judge through a GPU box running
kev.serve on a slim install, exactly like SGLangBackend.

Env: KEV_BASE_URL (default http://127.0.0.1:8008), KEV_MODEL
(default kev-latest), KEV_API_KEY (optional), KEV_TIMEOUT.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Sequence


class KevError(Exception):
    """Kev-server transport or contract failure (message is sanitized)."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


class KevBackend:
    """Drop-in engine over a remote Kev /v1/systemone server."""

    backend_name = "kev"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> None:
        from .patterns import require_http_url

        self.base_url = require_http_url(
            (base_url or os.environ.get("KEV_BASE_URL")
             or "http://127.0.0.1:8008").strip().rstrip("/"),
            what="Kev base URL",
        )
        self.model_name = (
            model or os.environ.get("KEV_MODEL") or "kev-latest"
        ).strip()
        env_timeout = (os.environ.get("KEV_TIMEOUT") or "").strip()
        self.timeout = (
            timeout
            if timeout is not None
            else (float(env_timeout) if env_timeout else 120.0)
        )
        self.api_key = (
            api_key or os.environ.get("KEV_API_KEY") or ""
        ).strip()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health(self) -> bool:
        """True when GET /v1/models answers 200 with a models list."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/models",
                headers=self._headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                body = json.loads(resp.read().decode("utf-8") or "{}")
            return resp.status == 200 and isinstance(
                body.get("models"), list
            )
        except Exception:
            return False

    @staticmethod
    def _to_typesafe(q: Dict[str, Any]) -> Dict[str, Any]:
        name = q.get("name", "q")
        qtype = q.get("type")
        if qtype == "choice":
            options = list(q.get("options") or [])
            if len(options) < 2:
                raise KevError(f"question {name!r} needs >= 2 options")
            return {
                "type": "choice",
                "instructions": q.get("prompt") or f"Choose the best option ({name})",
                "criteria": {str(o.get("name", o) if isinstance(o, dict) else o): (
                    o.get("description") if isinstance(o, dict) else None
                ) for o in options},
            }
        if qtype == "score":
            levels = [
                str(lv.get("name", lv) if isinstance(lv, dict) else lv)
                for lv in (q.get("levels") or [])
            ]
            if len(levels) < 2:
                raise KevError(f"question {name!r} needs >= 2 levels")
            return {
                "type": "score",
                "instructions": q.get("prompt") or f"Rate the level ({name})",
                "criteria": levels,
            }
        if qtype == "noul":
            statement = q.get("statement") or q.get("prompt") or name
            return {"type": "noul", "instructions": statement}
        raise KevError(f"question {name!r}: unknown type {qtype!r}")

    @staticmethod
    def _from_typesafe(
        name: str,
        qtype: str,
        answer: Dict[str, Any],
        labels: Sequence[str],
    ) -> Dict[str, Any]:
        from .patterns import (
            choice_confidence,
            noul_confidence,
            score_confidence,
            validate_choice,
            validate_distribution,
        )

        if qtype == "choice":
            probs = {str(k): float(v) for k, v in
                     (answer.get("probabilities") or {}).items()}
            choice = answer.get("choice") or (
                max(probs, key=probs.get) if probs else None)
            if choice is None:
                raise KevError(f"question {name!r}: empty choice answer")
            validate_choice({"choice": choice, "probabilities": probs},
                            list(probs))
            return {"type": "choice", "choice": choice,
                    "probabilities": probs,
                    "confidence": choice_confidence(list(probs.values()))}
        if qtype == "score":
            raw = {str(k): float(v) for k, v in
                   (answer.get("probabilities") or {}).items()}
            # Kev keys by level index; remap positionally via the legend
            # when present, else by the caller's level order.
            legend = answer.get("legend") or {}
            if all(str(i) in raw for i in range(len(labels))):
                dist = {labels[i]: raw[str(i)] for i in range(len(labels))}
            elif set(raw) == set(labels):
                dist = {lv: raw[lv] for lv in labels}
            elif legend and set(raw) == set(legend):
                inv = {str(v): str(k) for k, v in legend.items()}
                dist = {lv: raw[inv[lv]] for lv in labels if lv in inv}
                if set(dist) != set(labels):
                    raise KevError(
                        f"question {name!r}: legend does not cover levels")
            else:
                raise KevError(
                    f"question {name!r}: score probabilities match "
                    "neither level indices nor names")
            level = max(dist, key=dist.get) if dist else None
            if level is None:
                raise KevError(f"question {name!r}: empty score answer")
            validate_distribution(dist, list(dist), level)
            return {"type": "score", "level": level, "distribution": dist,
                    "score": sum(i * p for i, p in enumerate(dist.values())),
                    "confidence": score_confidence(list(dist.values())),
                    "legend": {lv: legend.get(str(i), lv)
                               for i, lv in enumerate(labels)} if legend
                    else {lv: lv for lv in labels}}
        # noul: Kev answers carry "noul" = P(true), and nothing else.
        try:
            p = float(answer.get("noul", answer.get("probability", 0.5)))
        except (TypeError, ValueError):
            raise KevError(f"question {name!r}: bad noul answer") from None
        p = max(0.0, min(1.0, p))
        return {"type": "noul", "probability": p, "answer": p >= 0.5,
                "confidence": noul_confidence(p)}

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer via the server's /v1/systemone.

        Kev is text-only (no media path): images/videos are reported in
        ``_meta["media_dropped"]`` and never sent.
        """
        questions = list(questions)
        if not questions:
            raise KevError("questions must be non-empty")
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "state": state,
            "questions": {q.get("name", f"q{i}"): self._to_typesafe(q)
                          for i, q in enumerate(questions)},
        }
        types = {q.get("name", f"q{i}"): q.get("type")
                 for i, q in enumerate(questions)}
        labels: Dict[str, list] = {}
        for i, q in enumerate(questions):
            nm = q.get("name", f"q{i}")
            if q.get("type") == "score":
                labels[nm] = [
                    str(lv.get("name", lv) if isinstance(lv, dict) else lv)
                    for lv in (q.get("levels") or [])]
            else:
                labels[nm] = []
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                body = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "replace")[:160]
            except Exception:
                pass
            raise KevError(
                f"server rejected the request (HTTP {e.code})",
                hint=f"{detail + '; ' if detail else ''}POST "
                f"{self.base_url}/v1/systemone; is kev.serve running? "
                "(422 = state past the serving context)",
            ) from None
        except Exception as e:
            raise KevError(
                f"could not reach the Kev server: {type(e).__name__}",
                hint=f"checked {self.base_url} (set KEV_BASE_URL)",
            ) from None
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        raw = body.get("answers") or {}
        answers: Dict[str, Any] = {}
        for name, qtype in types.items():
            ans = raw.get(name)
            if not isinstance(ans, dict):
                raise KevError(f"question {name!r}: missing answer")
            answers[name] = self._from_typesafe(name, qtype, ans, labels[name])
        meta: Dict[str, Any] = {"model": self.model_name, "backend": "kev",
                                "base_url": self.base_url,
                                "n_questions": len(questions),
                                "latency_ms": latency_ms}
        if isinstance(body.get("usage"), dict):
            meta["usage"] = dict(body["usage"])
        if body.get("truncated") is not None:
            meta["truncated"] = body["truncated"]
        dropped_images = len(list(images or []))
        dropped_videos = len(list(videos or []))
        if dropped_images or dropped_videos:
            meta["media_dropped"] = {"images": dropped_images,
                                     "videos": dropped_videos}
        answers["_meta"] = meta
        return answers
