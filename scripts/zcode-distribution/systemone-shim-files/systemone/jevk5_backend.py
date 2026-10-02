"""JevK5-server backend: judge through a JevK5 /v1/systemone server.

JevK5 (allebee/jevk5, Apache-2.0) serves TypeSafe-dialect typed decisions
from open Qwen3.5 weights — `jevk5-serve --port 8090` exposes
POST /v1/systemone plus GET /health. This backend is the client side:
API-style questions are rendered into the TypeSafe dialect, judged
remotely, and mapped back onto Jev answers with TypeSafe confidence.

Never imports torch: a Mac can judge through a GPU box running jevk5-serve
on a slim install, exactly like SGLangBackend.

Env: JEVK5_BASE_URL (default http://127.0.0.1:8090), JEVK5_MODEL.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Sequence


class JevK5Error(Exception):
    """JevK5-server transport or contract failure (message is sanitized)."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        self.hint = hint
        super().__init__(f"{message} {hint}".strip() if hint else message)


class JevK5ServerBackend:
    """Drop-in engine over a remote JevK5 /v1/systemone server."""

    backend_name = "jevk5"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        from .patterns import require_http_url

        self.base_url = require_http_url(
            (base_url or os.environ.get("JEVK5_BASE_URL")
             or "http://127.0.0.1:8090").strip().rstrip("/"),
            what="JevK5 base URL",
        )
        self.model_name = (
            model or os.environ.get("JEVK5_MODEL") or "jevk5"
        ).strip()
        self.timeout = timeout

    def health(self) -> bool:
        """True when the server answers a health probe (/health, then /healthz)."""
        for path in ("/health", "/healthz"):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}{path}", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                    body = json.loads(resp.read().decode("utf-8") or "{}")
                if resp.status == 200 and body.get("ok") is True:
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _to_typesafe(q: Dict[str, Any]) -> Dict[str, Any]:
        name = q.get("name", "q")
        qtype = q.get("type")
        if qtype == "choice":
            options = list(q.get("options") or [])
            if len(options) < 2:
                raise JevK5Error(f"question {name!r} needs >= 2 options")
            return {
                "type": "choice",
                "instructions": q.get("prompt") or f"Choose the best option ({name})",
                "criteria": {o: o for o in options},
            }
        if qtype == "score":
            levels = list(q.get("levels") or [])
            if len(levels) < 2:
                raise JevK5Error(f"question {name!r} needs >= 2 levels")
            return {
                "type": "score",
                "instructions": q.get("prompt") or f"Rate the level ({name})",
                "criteria": levels,
            }
        if qtype == "noul":
            statement = q.get("statement") or q.get("prompt") or name
            return {"type": "noul", "instructions": statement}
        raise JevK5Error(f"question {name!r}: unknown type {qtype!r}")

    @staticmethod
    def _from_typesafe(name: str, qtype: str, answer: Dict[str, Any]) -> Dict[str, Any]:
        from .patterns import (
            choice_confidence,
            noul_confidence,
            score_confidence,
            validate_choice,
            validate_distribution,
        )

        if qtype == "choice":
            probs = dict(answer.get("probabilities") or {})
            choice = answer.get("choice") or (max(probs, key=probs.get) if probs else None)
            if choice is None:
                raise JevK5Error(f"question {name!r}: empty choice answer")
            validate_choice({"choice": choice, "probabilities": probs},
                            list(probs))
            return {"type": "choice", "choice": choice, "probabilities": probs,
                    "confidence": choice_confidence(list(probs.values()))}
        if qtype == "score":
            dist = dict(answer.get("distribution") or answer.get("probabilities") or {})
            level = answer.get("level") or (max(dist, key=dist.get) if dist else None)
            if level is None or level not in dist:
                raise JevK5Error(f"question {name!r}: empty score answer")
            validate_distribution(dist, list(dist), level)
            return {"type": "score", "level": level, "distribution": dist,
                    "score": sum(i * p for i, p in enumerate(dist.values())),
                    "confidence": score_confidence(list(dist.values())),
                    "legend": dict(answer.get("legend") or {})}
        # noul: JevK5 answers carry "noul" = P(true).
        try:
            p = float(answer.get("noul", answer.get("probability", 0.5)))
        except (TypeError, ValueError):
            raise JevK5Error(f"question {name!r}: bad noul answer") from None
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
        """Answer via the server's /v1/systemone, forwarding Clef media."""
        if isinstance(state, str) and len(state) > 6000:
            state = state[:6000]
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "state": state,
            "questions": {q.get("name", f"q{i}"): self._to_typesafe(q)
                          for i, q in enumerate(questions)},
        }
        if images:
            payload["images"] = list(images)
        if videos:
            payload["videos"] = list(videos)
        types = {q.get("name", f"q{i}"): q.get("type")
                 for i, q in enumerate(questions)}
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                body = json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            raise JevK5Error(
                f"server rejected the request (HTTP {e.code})",
                hint=f"POST {self.base_url}/v1/systemone; is jevk5-serve running?",
            ) from None
        except Exception as e:
            raise JevK5Error(
                f"could not reach the JevK5 server: {type(e).__name__}",
                hint=f"checked {self.base_url} (set JEVK5_BASE_URL)",
            ) from None
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        raw = body.get("answers") or {}
        answers: Dict[str, Any] = {}
        for name, qtype in types.items():
            ans = raw.get(name)
            if not isinstance(ans, dict):
                raise JevK5Error(f"question {name!r}: missing answer")
            answers[name] = self._from_typesafe(name, qtype, ans)
        answers["_meta"] = {"model": self.model_name, "backend": "jevk5",
                            "latency_ms": latency_ms}
        return answers
