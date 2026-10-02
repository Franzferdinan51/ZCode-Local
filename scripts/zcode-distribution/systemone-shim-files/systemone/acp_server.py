"""systemone.acp_server — SystemOne as an Agent Client Protocol (ACP) agent.

Speaks ACP v1 (agentclientprotocol.com) over stdio: newline-delimited
JSON-RPC 2.0 on stdin/stdout, logs on stderr only. Hand-rolled on the
stdlib — no ACP SDK dependency.

Exactly two capabilities, both served by the live shim (never by loading
a model here):

- `/route [--cost-bias economy|balanced|quality] <task>`
      cheapest sufficient model tier for the task
- `/decide` followed by a ```json fenced block:
      {"type": "choice"|"noul"|"score", "state": ..., "instructions": ...,
       "criteria": ...}
      one typed decision with calibrated probabilities

Protocol surface: initialize, session/new, session/prompt, session/cancel,
$/cancel_request. Everything else answers -32601 (method not found).

Run:  python -m systemone.acp_server
Shim URL: $SYSTEMONE_SHIM_URL (default http://127.0.0.1:8765).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from .client import ShimError, SystemOneClient, default_shim_url

try:
    from . import __version__ as _pkg_version
except Exception:  # pragma: no cover - import fallback
    _pkg_version = "0.2.0"

AGENT_NAME = "systemone"
AGENT_TITLE = "SystemOne"

# JSON-RPC error codes (ACP reuses JSON-RPC 2.0; -32800 is ACP's cancelled).
_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_REQUEST_CANCELLED = -32800


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class AcpAgent:
    """Minimal ACP v1 agent. Single-threaded; cancellation is best-effort."""

    def __init__(self, shim_url: Optional[str] = None) -> None:
        self.client = SystemOneClient(shim_url)
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.cancelled: Dict[str, bool] = {}
        self.initialized = False

    # -- IO -------------------------------------------------------------
    def _send(self, message: Dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    def _log(self, text: str) -> None:
        sys.stderr.write(f"[systemone-acp] {text}\n")
        sys.stderr.flush()

    def _respond(self, msg_id: Any, result: Any) -> None:
        self._send({"jsonrpc": "2.0", "id": msg_id, "result": result})

    def _error(self, msg_id: Any, code: int, message: str) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": code, "message": message},
            }
        )

    def _notify_session_update(
        self, session_id: str, update: Dict[str, Any]
    ) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    def _chunk(
        self, session_id: str, message_id: str, text: str
    ) -> None:
        self._notify_session_update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "messageId": message_id,
                "content": {"type": "text", "text": text},
            },
        )

    def _tool_call(self, session_id: str, tool_call_id: str, title: str) -> None:
        self._notify_session_update(
            session_id,
            {
                "sessionUpdate": "tool_call",
                "toolCallId": tool_call_id,
                "title": title,
                "kind": "other",
                "status": "pending",
            },
        )

    def _tool_update(
        self,
        session_id: str,
        tool_call_id: str,
        status: str,
        text: Optional[str] = None,
    ) -> None:
        update: Dict[str, Any] = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": tool_call_id,
            "status": status,
        }
        if text is not None:
            update["content"] = [
                {"type": "content", "content": {"type": "text", "text": text}}
            ]
        self._notify_session_update(session_id, update)

    # -- message handling -----------------------------------------------
    def handle_line(self, line: str) -> None:
        try:
            msg = json.loads(line)
        except ValueError:
            self._error(None, _PARSE_ERROR, "Parse error: invalid JSON")
            return
        if isinstance(msg, list):  # JSON-RPC batch
            responses = [self._handle_one(m) for m in msg]
            responses = [r for r in responses if r is not None]
            if responses:
                self._send(responses[0] if len(responses) == 1 else responses)
            return
        response = self._handle_one(msg)
        if response is not None:
            self._send(response)

    def _handle_one(self, msg: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id") if isinstance(msg, dict) else None,
                "error": {"code": _INVALID_REQUEST, "message": "Invalid Request"},
            }
        method = msg.get("method")
        msg_id = msg.get("id")
        is_notification = "id" not in msg
        if not isinstance(method, str):
            if is_notification:
                return None
            return self._err_obj(msg_id, _INVALID_REQUEST, "missing method")

        handler = {
            "initialize": self._on_initialize,
            "session/new": self._on_session_new,
            "session/prompt": self._on_session_prompt,
            "session/cancel": self._on_session_cancel,
            "$/cancel_request": self._on_cancel_request,
        }.get(method)
        if handler is None:
            if is_notification:
                return None
            return self._err_obj(msg_id, _METHOD_NOT_FOUND, f"unknown method: {method}")
        try:
            result = handler(msg.get("params") or {}, msg_id, is_notification)
        except _AcpError as exc:
            if is_notification:
                return None
            return self._err_obj(msg_id, exc.code, exc.message)
        except Exception as exc:  # never leak tracebacks over the wire
            self._log(f"internal error in {method}: {type(exc).__name__}: {exc}")
            if is_notification:
                return None
            return self._err_obj(msg_id, _INTERNAL_ERROR, "Internal error")
        if is_notification or msg_id is None:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    @staticmethod
    def _err_obj(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": code, "message": message},
        }

    # -- protocol methods -------------------------------------------------
    def _on_initialize(
        self, params: Dict[str, Any], msg_id: Any, is_notification: bool
    ) -> Dict[str, Any]:
        # We speak ACP v1 only; per spec we answer with the latest version
        # we support when the client asks for something else.
        self.initialized = True
        return {
            "protocolVersion": 1,
            "agentCapabilities": {
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": False,
                },
            },
            "agentInfo": {
                "name": AGENT_NAME,
                "title": AGENT_TITLE,
                "version": _pkg_version,
            },
            "authMethods": [],
        }

    def _require_init(self) -> None:
        if not self.initialized:
            raise _AcpError(_INVALID_REQUEST, "not initialized: call initialize first")

    def _on_session_new(
        self, params: Dict[str, Any], msg_id: Any, is_notification: bool
    ) -> Dict[str, Any]:
        self._require_init()
        session_id = _new_id("sess")
        cwd = params.get("cwd") or os.getcwd()
        mcp_servers = params.get("mcpServers") or []
        self.sessions[session_id] = {
            "cwd": cwd,
            "mcp_servers": mcp_servers,
        }
        self.cancelled[session_id] = False
        message_id = _new_id("msg")
        note = ""
        if mcp_servers:
            note = (
                "\n\nNote: this agent does not connect to client-supplied MCP "
                "servers; its two tools call the SystemOne shim directly."
            )
        self._chunk(
            session_id,
            message_id,
            "SystemOne agent ready — decisions via the live shim at "
            f"{self.client.base_url}.{note}\n\n"
            "- `/route [--cost-bias economy|balanced|quality] <task>` — "
            "cheapest sufficient model tier for the task\n"
            "- `/decide` + a ```json block "
            '`{"type": "choice"|"noul"|"score", "state": ..., '
            '"instructions": ..., "criteria": ...}` — one typed decision',
        )
        return {"sessionId": session_id}

    def _on_session_prompt(
        self, params: Dict[str, Any], msg_id: Any, is_notification: bool
    ) -> Dict[str, Any]:
        self._require_init()
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise _AcpError(_INVALID_PARAMS, "unknown or missing sessionId")
        self.cancelled[session_id] = False
        text = _prompt_text(params.get("prompt"))
        message_id = _new_id("msg")
        if text.startswith("/route"):
            stop = self._flow_route(session_id, message_id, text[len("/route"):].strip())
        elif text.startswith("/decide"):
            stop = self._flow_decide(session_id, message_id, text[len("/decide"):].strip())
        else:
            self._chunk(session_id, message_id, _USAGE)
            stop = "end_turn"
        return {"stopReason": stop}

    def _on_session_cancel(
        self, params: Dict[str, Any], msg_id: Any, is_notification: bool
    ) -> None:
        session_id = params.get("sessionId")
        if isinstance(session_id, str):
            self.cancelled[session_id] = True
        return None

    def _on_cancel_request(
        self, params: Dict[str, Any], msg_id: Any, is_notification: bool
    ) -> None:
        # We make no agent->client requests, so there is nothing to cancel.
        # Notification: no response is sent either way.
        return None

    # -- capability flows -------------------------------------------------
    def _was_cancelled(self, session_id: str) -> bool:
        return self.cancelled.get(session_id, False)

    def _flow_route(self, session_id: str, message_id: str, rest: str) -> str:
        cost_bias, task = _parse_route_args(rest)
        call_id = _new_id("call")
        if not task:
            self._chunk(session_id, message_id,
                        "Usage: `/route [--cost-bias economy|balanced|quality] <task>`")
            return "end_turn"
        self._tool_call(session_id, call_id, "systemone_route: tier routing")
        self._tool_update(session_id, call_id, "in_progress")
        if self._was_cancelled(session_id):
            self._tool_update(session_id, call_id, "cancelled")
            return "cancelled"
        try:
            payload = self.client.route(task, cost_bias=cost_bias)
        except ShimError as exc:
            summary = f"Route failed: {exc}"
            self._tool_update(session_id, call_id, "failed", summary)
            self._chunk(session_id, message_id, summary)
            return "end_turn"
        route = payload.get("route", {})
        summary = _format_route(route)
        self._tool_update(session_id, call_id, "completed", summary)
        self._chunk(session_id, message_id, summary)
        return "cancelled" if self._was_cancelled(session_id) else "end_turn"

    def _flow_decide(self, session_id: str, message_id: str, rest: str) -> str:
        call_id = _new_id("call")
        try:
            request = _parse_decide_block(rest)
        except _AcpError as exc:
            self._chunk(session_id, message_id, f"{exc.message}\n\n{_DECIDE_USAGE}")
            return "end_turn"
        self._tool_call(session_id, call_id, "systemone_decide: typed decision")
        self._tool_update(session_id, call_id, "in_progress")
        if self._was_cancelled(session_id):
            self._tool_update(session_id, call_id, "cancelled")
            return "cancelled"
        try:
            decision = self.client.decide(
                request["state"],
                request["instructions"],
                criteria=request.get("criteria"),
                type=request["type"],
            )
        except ShimError as exc:
            summary = f"Decide failed: {exc}"
            self._tool_update(session_id, call_id, "failed", summary)
            self._chunk(session_id, message_id, summary)
            return "end_turn"
        summary = _format_decision(decision)
        self._tool_update(session_id, call_id, "completed", summary)
        self._chunk(session_id, message_id, summary)
        return "cancelled" if self._was_cancelled(session_id) else "end_turn"

    # -- main loop ----------------------------------------------------------
    def serve_forever(self) -> None:
        for line in sys.stdin:
            if line.strip():
                try:
                    self.handle_line(line)
                except Exception as exc:  # last-resort guard for the loop
                    self._log(f"loop error: {type(exc).__name__}: {exc}")


class _AcpError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


_USAGE = (
    "I understand two commands:\n\n"
    "- `/route [--cost-bias economy|balanced|quality] <task>` — "
    "route a task to the cheapest sufficient model tier\n"
    "- `/decide` followed by a ```json fenced block with "
    '`{"type": "choice"|"noul"|"score", "state": ..., "instructions": ..., '
    '"criteria": ...}` — one typed decision with calibrated probabilities'
)

_DECIDE_USAGE = (
    "Decide request format — a ```json fenced block:\n"
    "```json\n"
    '{"type": "choice", "state": "the situation",\n'
    ' "instructions": "the judgment to make",\n'
    ' "criteria": {"label-a": "description", "label-b": "description"}}\n'
    "```\n"
    "criteria: choice -> {label: description}; "
    "score -> ordered [level descriptions]; "
    'noul -> {"yes": "...", "no": "..."} or {"true": "...", "false": "..."} (optional).'
)


def _prompt_text(prompt: Any) -> str:
    parts: List[str] = []
    if isinstance(prompt, list):
        for block in prompt:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts).strip()


def _parse_route_args(rest: str) -> tuple:
    """Split an optional --cost-bias flag off the front of a /route task."""
    cost_bias = "balanced"
    task = rest
    if task.startswith("--cost-bias="):
        cost_bias, _, task = task[len("--cost-bias="):].partition(" ")
    elif task.startswith("--cost-bias"):
        _, _, after = task.partition(" ")
        cost_bias, _, task = after.strip().partition(" ")
    task = task.strip()
    if cost_bias not in ("economy", "balanced", "quality"):
        raise _AcpError(_INVALID_PARAMS, f"unknown cost_bias: {cost_bias!r}")
    return cost_bias, task


def _parse_decide_block(rest: str) -> Dict[str, Any]:
    """Extract and validate the ```json decide request after /decide."""
    start = rest.find("```")
    if start < 0:
        raise _AcpError(_INVALID_PARAMS, "missing ```json block with the decide request")
    fence = rest[start + 3:]
    if fence.lstrip().startswith("json"):
        fence = fence.lstrip()[4:]
    end = fence.find("```")
    if end < 0:
        raise _AcpError(_INVALID_PARAMS, "unterminated ```json block")
    try:
        request = json.loads(fence[:end])
    except ValueError as exc:
        raise _AcpError(_INVALID_PARAMS, f"decide block is not valid JSON: {exc}")
    if not isinstance(request, dict):
        raise _AcpError(_INVALID_PARAMS, "decide request must be a JSON object")
    dtype = request.get("type")
    if dtype not in ("choice", "noul", "score"):
        raise _AcpError(
            _INVALID_PARAMS, "'type' must be one of 'choice', 'noul', 'score'")
    if not isinstance(request.get("instructions"), str) or not request["instructions"].strip():
        raise _AcpError(_INVALID_PARAMS, "'instructions' must be a non-empty string")
    if "state" not in request:
        raise _AcpError(_INVALID_PARAMS, "'state' is required")
    return request


def _fmt_num(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) else "n/a"


def _format_route(route: Dict[str, Any]) -> str:
    tier = route.get("tier", "?")
    conf = _fmt_num(route.get("confidence"))
    margin = route.get("margin")
    margin_s = _fmt_num(margin) if margin is not None else "n/a"
    lines = [
        f"**Tier: {tier}** (confidence {conf}, margin {margin_s})",
        "",
        f"Model: `{route.get('model_id', 'n/a')}`",
        f"Effort: {route.get('effort', 'n/a')}",
    ]
    if route.get("uncertain") is not None:
        lines.append(f"Uncertain: {route.get('uncertain')}")
    rationale = route.get("rationale")
    if rationale:
        lines += ["", rationale]
    return "\n".join(lines)


def _format_decision(decision: Dict[str, Any]) -> str:
    dtype = decision.get("type", "?")
    conf = _fmt_num(decision.get("confidence"))
    backend = decision.get("backend", "n/a")
    lines = [f"_via {backend}_", ""]
    if dtype == "choice":
        lines.append(f"**{decision.get('label', '?')}** (confidence {conf})")
        lines.append("")
        for label, prob in (decision.get("probabilities") or {}).items():
            lines.append(f"- {label}: {_fmt_num(prob)}")
    elif dtype == "score":
        lines.append(f"**Level: {decision.get('level', '?')}** (confidence {conf})")
        lines.append("")
        for label, prob in (decision.get("distribution") or {}).items():
            lines.append(f"- level {label}: {_fmt_num(prob)}")
    else:  # noul
        probs = decision.get("probabilities") or {}
        lines.append(
            f"**{decision.get('label', '?')}** "
            f"(P(yes)={_fmt_num(probs.get('yes'))}, confidence {conf})"
        )
    latency = decision.get("latency_ms")
    if latency is not None:
        lines += ["", f"Latency: {latency} ms"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="systemone-acp",
        description="SystemOne ACP v1 agent over stdio (route + decide via the live shim).",
    )
    parser.add_argument(
        "--shim-url", default=None,
        help=f"shim base URL (default: $SYSTEMONE_SHIM_URL or {default_shim_url()})",
    )
    args = parser.parse_args(argv)
    agent = AcpAgent(shim_url=args.shim_url)
    agent._log(f"listening on stdio; shim {agent.client.base_url}")
    agent.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
