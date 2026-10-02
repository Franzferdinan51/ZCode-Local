"""systemone.client — thin HTTP client for a live SystemOne shim.

The agent layer (CLI, MCP tools, ACP adapter) talks to a *deployed* shim
over HTTP and never loads models itself. The shim URL is overridable:

    SYSTEMONE_SHIM_URL=http://macmini:8765   # env var
    SystemOneClient("http://macmini:8765")   # explicit argument
    # default when neither is given: http://127.0.0.1:8765

Stdlib only (urllib) — no extra dependencies for the agent layer.

No model IDs are hard-coded here: routing and decisions are delegated to
the shim, which owns its model registry. The decide endpoint is served by
the decision sidecar — the sole backend is Mapika/decider-4b v2.1
(Apache-2.0) — with a fail-open local GLiClass fallback (backend
"fallback") when the sidecar is unreachable. See the shim's decide
contract.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

SHIM_URL_ENV = "SYSTEMONE_SHIM_URL"
DEFAULT_SHIM_URL = "http://127.0.0.1:8765"


def default_shim_url() -> str:
    """Shim base URL: $SYSTEMONE_SHIM_URL, else the localhost default."""
    return (os.environ.get(SHIM_URL_ENV) or "").strip() or DEFAULT_SHIM_URL


class ShimError(Exception):
    """The shim refused the request, or the shim could not be reached."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


class SystemOneClient:
    """HTTP client for the SystemOne shim's agent endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0) -> None:
        from .patterns import require_http_url

        self.base_url = require_http_url(
            (base_url or default_shim_url()).rstrip("/"), what="shim URL"
        )
        self.timeout = timeout

    # -- transport ------------------------------------------------------
    def _request(
        self, method: str, path: str, payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        data: Optional[bytes] = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(detail or "{}")
                message = parsed.get("error") or detail or f"HTTP {exc.code}"
            except ValueError:
                parsed, message = None, detail or f"HTTP {exc.code}"
            raise ShimError(
                f"shim refused ({exc.code}): {message}",
                status=exc.code,
                payload=parsed,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ShimError(
                f"cannot reach shim at {self.base_url}{path}: {reason}"
            ) from exc
        try:
            parsed_body = json.loads(body.decode("utf-8") or "{}")
        except ValueError as exc:
            raise ShimError(f"shim returned non-JSON at {path}: {exc}") from exc
        if not isinstance(parsed_body, dict):
            raise ShimError(f"shim returned a non-object at {path}")
        return parsed_body

    # -- endpoints ------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """GET /healthz — shim liveness plus the loaded engine model."""
        return self._request("GET", "/healthz")

    def route(
        self,
        task: str,
        cost_bias: str = "balanced",
        tiers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """POST /v1/systemone/route — cheapest sufficient tier for *task*.

        Returns the full shim payload: {"route": {...}, "model": ..., ...}.
        The interesting bits live under payload["route"]: tier, confidence,
        margin, calibrated_probabilities, rationale, model_id, effort.
        """
        body: Dict[str, Any] = {"task": task, "cost_bias": cost_bias}
        if tiers:
            body["tiers"] = tiers
        return self._request("POST", "/v1/systemone/route", body)

    def decide(
        self,
        state: Any,
        instructions: str,
        criteria: Any = None,
        type: str = "choice",
    ) -> Dict[str, Any]:
        """POST /v1/systemone/decide — one typed decision.

        type: "choice" | "noul" | "score". criteria: choice -> {label: desc},
        noul -> {"yes": ..., "no": ...} or {"true": ..., "false": ...}
        (optional), score -> ordered list of level descriptions. Returns
        the decide payload, including "backend": "decider" (sidecar) or
        "fallback" (local GLiClass fail-open).
        """
        body: Dict[str, Any] = {
            "state": state,
            "instructions": instructions,
            "type": type,
        }
        if criteria is not None:
            body["criteria"] = criteria
        return self._request("POST", "/v1/systemone/decide", body)

    def rank_plans(
        self, task: str, plans: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """POST /v1/systemone/rank-plans — order plans by P(success | task)."""
        return self._request(
            "POST", "/v1/systemone/rank-plans", {"task": task, "plans": plans}
        )

    def permute(
        self,
        state: Any,
        question: Dict[str, Any],
        n_perm: int = 8,
        seed: int = 0,
    ) -> Dict[str, Any]:
        """POST /v1/systemone/permute — one choice under n_perm orders.

        question: TypeSafe choice question. Returns {"runs",
        "argmax_stable", "spread", ...}.
        """
        return self._request(
            "POST",
            "/v1/systemone/permute",
            {"state": state, "question": question,
             "n_perm": n_perm, "seed": seed},
        )

    def batch(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """POST /v1/systemone/batch — judge many TypeSafe bodies at once.

        items: [{state, questions, ...}] in /v1/systemone shape (1..32).
        Returns {"results": [{"status": 200, "answers", ...} |
        {"status": 4xx/5xx, "error": ...}], "model", "n_items"}.
        """
        return self._request(
            "POST", "/v1/systemone/batch", {"items": items}
        )

    def decisions(
        self,
        input: Any,
        questions: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/decisions — batched typed questions, SGLang dialect.

        questions: [{"id": str, "type": "choice"|"score"|"yes_no",
        "question": str, "options": [{"name": str}, ...] |
        "levels": [str, ...]}]. Score probabilities come back keyed by
        level index ("0"-"9"); yes_no answers carry probabilities
        {"yes", "no"}. input: state text (image parts are
        noted-and-skipped: the endpoint is text-only upstream).
        Returns {"answers": {...}}.
        """
        body: Dict[str, Any] = {"input": input, "questions": questions}
        if model:
            body["model"] = model
        return self._request("POST", "/v1/decisions", body)

    def systemone(
        self,
        state: Any,
        questions: Any,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """POST /v1/systemone — batched typed questions, TypeSafe dialect.

        questions: {id: {"type", "instructions", "criteria"}} (also accepts
        a list of question mappings carrying "id"). Returns {"answers": ...}.
        """
        body: Dict[str, Any] = {"state": state, "questions": questions}
        if model:
            body["model"] = model
        return self._request("POST", "/v1/systemone", body)

    def status(self, probe: bool = True) -> Dict[str, Any]:
        """Composite health: shim liveness plus the decision backend in use.

        The decision backend ("decider" sidecar engine vs
        "fallback" GLiClass) is only observable by asking for a decision,
        so status() runs one tiny noul probe unless probe=False.
        """
        out: Dict[str, Any] = {
            "shim_url": self.base_url,
            "shim": None,
            "decision": None,
        }
        try:
            out["shim"] = self.health()
        except ShimError as exc:
            out["shim"] = {"ok": False, "error": str(exc)}
            return out
        if probe:
            try:
                decision = self.decide(
                    "systemone status probe",
                    "Is this a status probe?",
                    type="noul",
                )
                out["decision"] = {
                    "backend": decision.get("backend"),
                    "latency_ms": decision.get("latency_ms"),
                    "confidence": decision.get("confidence"),
                }
            except ShimError as exc:
                out["decision"] = {"error": str(exc)}
        return out
