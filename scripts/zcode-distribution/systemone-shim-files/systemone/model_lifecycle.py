"""Model lifecycle management for SystemOne.

Problem: SystemOne routes model selection but never unloads models from
LM Studio. When sessions end (or hang), models stay resident in unified
memory. On a 128GB Mac Studio, 15 loaded models = ~75GB wired = system crawl.

This module tracks model usage and reaps idle models via the `lms` CLI
(the only reliable unload interface — LM Studio has no stable HTTP
unload endpoint).

Usage:
    from systemone.model_lifecycle import track_model_use, start_reaper

    # Call when SystemOne selects a model for a task:
    track_model_use("ornith-1.5-35b-a3b")

    # Start background reaper (call once at startup):
    start_reaper(idle_timeout_s=900)  # unload models idle >15min

    # Call when a session ends (e.g. grok-local exit):
    release_session_models(session_id)

Environment overrides:
    SYSTEMONE_IDLE_TIMEOUT_S    idle seconds before unload (default 900)
    SYSTEMONE_REAPER_INTERVAL_S  reaper check interval (default 60)
    SYSTEMONE_NO_REAPER         set to 1 to disable the background reaper
    SYSTEMONE_LMS_BIN           path to `lms` binary
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)

# model_id -> last-used unix timestamp
_usage: Dict[str, float] = {}
# session_id -> set of model_ids used in that session
_sessions: Dict[str, Set[str]] = {}
_lock = threading.Lock()
_reaper_thread: Optional[threading.Thread] = None


def _lms_bin() -> Optional[str]:
    explicit = os.environ.get("SYSTEMONE_LMS_BIN")
    if explicit and os.path.isfile(explicit):
        return explicit
    found = shutil.which("lms")
    if found:
        return found
    # LM Studio default install location (macOS)
    default = os.path.expanduser("~/.lmstudio/bin/lms")
    if os.path.isfile(default):
        return default
    return None


def _unload_model(model_id: str) -> bool:
    """Unload one model via `lms unload`. Returns True on success."""
    lms = _lms_bin()
    if not lms:
        logger.warning("model_lifecycle: lms binary not found, cannot unload %s", model_id)
        return False
    try:
        result = subprocess.run(
            [lms, "unload", model_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            logger.info("model_lifecycle: unloaded idle model %s", model_id)
            return True
        logger.warning(
            "model_lifecycle: lms unload %s failed: %s",
            model_id,
            (result.stderr or result.stdout or "unknown error")[:200],
        )
        return False
    except Exception as e:
        logger.warning("model_lifecycle: unload %s raised %s", model_id, e)
        return False


def track_model_use(model_id: str, session_id: Optional[str] = None) -> None:
    """Record that a model was used (call on every route decision)."""
    if not model_id:
        return
    now = time.time()
    with _lock:
        _usage[model_id] = now
        if session_id:
            _sessions.setdefault(session_id, set()).add(model_id)


def release_session_models(session_id: str, unload: bool = True) -> None:
    """Forget a session's models; optionally unload models no other session uses.

    Call this when a grok-local/agent session ends cleanly. Models still
    referenced by other live sessions are kept.
    """
    with _lock:
        models = _sessions.pop(session_id, set())
        if not models:
            return
        # Find models only this session was using
        still_needed: Set[str] = set()
        for other_models in _sessions.values():
            still_needed.update(other_models)
        to_unload = [m for m in models if m not in still_needed]
    if unload:
        for m in to_unload:
            # Only unload if idle (not used in the last 60s by anyone)
            with _lock:
                last = _usage.get(m, 0)
            if time.time() - last > 60:
                _unload_model(m)
                with _lock:
                    _usage.pop(m, None)


def reap_idle(idle_timeout_s: Optional[float] = None) -> int:
    """Unload all models idle longer than the timeout. Returns count unloaded."""
    timeout = idle_timeout_s
    if timeout is None:
        timeout = float(os.environ.get("SYSTEMONE_IDLE_TIMEOUT_S", "900"))
    now = time.time()
    with _lock:
        idle = [m for m, ts in _usage.items() if now - ts > timeout]
        # Don't unload models tied to live sessions unless they're idle too —
        # idle timeout already covers that; just avoid yanking active ones.
    unloaded = 0
    for m in idle:
        if _unload_model(m):
            unloaded += 1
            with _lock:
                _usage.pop(m, None)
    return unloaded


def start_reaper(
    idle_timeout_s: Optional[float] = None,
    interval_s: Optional[float] = None,
) -> Optional[threading.Thread]:
    """Start background thread that periodically reaps idle models. Idempotent."""
    global _reaper_thread
    if os.environ.get("SYSTEMONE_NO_REAPER") == "1":
        logger.info("model_lifecycle: reaper disabled via SYSTEMONE_NO_REAPER")
        return None
    with _lock:
        if _reaper_thread and _reaper_thread.is_alive():
            return _reaper_thread

    timeout = idle_timeout_s
    if timeout is None:
        timeout = float(os.environ.get("SYSTEMONE_IDLE_TIMEOUT_S", "900"))
    interval = interval_s
    if interval is None:
        interval = float(os.environ.get("SYSTEMONE_REAPER_INTERVAL_S", "60"))

    def _loop() -> None:
        logger.info(
            "model_lifecycle: reaper started (idle_timeout=%ss, interval=%ss)",
            timeout,
            interval,
        )
        while True:
            try:
                time.sleep(interval)
                n = reap_idle(timeout)
                if n:
                    logger.info("model_lifecycle: reaper unloaded %d idle model(s)", n)
            except Exception as e:
                logger.warning("model_lifecycle: reaper loop error: %s", e)

    t = threading.Thread(target=_loop, daemon=True, name="systemone-model-reaper")
    t.start()
    with _lock:
        _reaper_thread = t
    return t


def unload_all() -> int:
    """Emergency: unload every tracked model. Returns count unloaded."""
    with _lock:
        models = list(_usage.keys())
    n = 0
    for m in models:
        if _unload_model(m):
            n += 1
    with _lock:
        _usage.clear()
        _sessions.clear()
    return n
