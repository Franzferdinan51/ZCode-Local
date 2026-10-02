"""JevBench adapter: run SystemOne over JevBench-style typed-decision items.

JevBench (fstandhartinger/jevbench) ranks Jev-class systems on public +
sealed items shaped like::

    {"id": ..., "family": ..., "labels": [...],
     "question": {"type": ..., "instructions": ..., "criteria": {...}},
     "expected": ..., "state": ..., "provenance": {"exclude_reason": ...}}

This module scores such items with any SystemOne engine (duck-typed
``systemone(state, questions)``) and summarizes accuracy. Conventions mirror
JevBench's own adapter base (adapters/base.py): the request never carries
the expected label, probabilities are native distributions (Brier/ECE
eligible — never verbalized), and latency is wall-clock per item.

Two entry points:

- score_item(item, engine): one DecisionResult.
- run_file(path, engine, ...): a whole jsonl split -> summary (+ optional
  predictions file). ``systemone jevbench`` wraps this for the CLI.

Registering upstream (their repo, their harness) would vendor score_item's
mapping into jevbench/adapters/; this module keeps the mapping tested here
so a registration patch is mechanical.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionResult:
    """Per-item outcome, mirroring JevBench adapters/base.py."""

    adapter: str
    ok: bool
    probs: Optional[dict] = None  # native distribution over the answer keys
    probs_source: str = "native"  # "native" | "verbalized" (never logprobs)
    model: str = ""
    error: Optional[str] = None
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    label: Optional[str] = None

    def to_public(self) -> dict:
        """Public-safe view: no raw response text, no request body."""
        return {
            "adapter": self.adapter,
            "ok": self.ok,
            "probs": self.probs,
            "probs_source": self.probs_source,
            "model": self.model,
            "error": self.error,
            "latency_s": self.latency_s,
            "usage": self.usage,
            "label": self.label,
        }


def item_to_body(item: Dict[str, Any]) -> Dict[str, Any]:
    """JevBench item -> TypeSafe-dialect request body (no expected leakage).

    The question key is fixed ('decision'); the expected answer never
    appears in the request. Labels appear only as the answer vocabulary
    the judge must choose among (options/levels), never as the answer.

    Score items whose criteria list holds level *descriptions* get their
    labels as the level names, with descriptions folded into the
    instructions — otherwise bare level names ("0".."3") carry no
    meaning for the judge.
    """
    question = item.get("question") or {}
    qtype = question.get("type")
    instructions = question.get("instructions")
    criteria = question.get("criteria")
    if qtype == "score" and isinstance(criteria, list):
        labels = item.get("labels") or []
        if labels and len(labels) == len(criteria):
            descs = "\n".join(f"{lab}: {c}" for lab, c in zip(labels, criteria))
            level_block = f"Levels:\n{descs}"
            instructions = f"{instructions}\n{level_block}" if instructions else level_block
            criteria = list(labels)
    q: Dict[str, Any] = {"type": qtype, "instructions": instructions}
    if criteria is not None:
        q["criteria"] = criteria
    return {"state": item.get("state"), "questions": {"decision": q}}


def normalize_answer(answer: Dict[str, Any]) -> tuple[Optional[str], Optional[dict]]:
    """Engine answer -> (label, probs), following the djev-adapter mapping.

    - choice: probabilities as-is, label = argmax.
    - noul: {"yes": P(yes), "no": 1-P(yes)}, label = argmax. Reads
      "probability" (local) or "noul" (Kev/SGLang serve only that key).
    - score: distribution as-is, label = level. Remote servers answer
      index-keyed probabilities + a legend instead of a distribution —
      remapped through the legend onto level names.
    """
    atype = answer.get("type")
    if atype == "choice":
        probs = dict(answer.get("probabilities") or {})
        label = answer.get("choice") or (max(probs, key=probs.get) if probs else None)
        return label, probs or None
    if atype == "noul":
        raw = answer.get("probability", answer.get("noul", 0.5))
        try:
            p = float(raw)
        except (TypeError, ValueError):
            p = 0.5
        p = max(0.0, min(1.0, p))
        probs = {"yes": p, "no": 1.0 - p}
        return ("yes" if p >= 0.5 else "no"), probs
    if atype == "score":
        dist = dict(answer.get("distribution") or {})
        if not dist:
            probs = dict(answer.get("probabilities") or {})
            legend = answer.get("legend") or {}
            str_legend = {str(k): v for k, v in legend.items()}
            if probs and str_legend and set(probs) <= set(str_legend):
                dist = {str(str_legend[k]): float(probs[k]) for k in probs}
            else:
                dist = {str(k): float(v) for k, v in probs.items()}
        label = answer.get("level") or (max(dist, key=dist.get) if dist else None)
        return label, dist or None
    return None, None


def score_item(
    item: Dict[str, Any],
    engine: Any,
    *,
    adapter: str = "systemone",
    model: str = "",
) -> DecisionResult:
    """Score one JevBench item with a SystemOne engine."""
    if (item.get("provenance") or {}).get("exclude_reason"):
        return DecisionResult(adapter=adapter, ok=False, model=model,
                              error="excluded by provenance")
    from .shim import translate_body

    t0 = time.perf_counter()
    try:
        state_text, questions = translate_body(item_to_body(item))
        answers = engine.systemone(state_text, questions)
        label, probs = normalize_answer(answers.get("decision") or {})
        if label is None:
            raise ValueError("engine returned no decision")
        latency = time.perf_counter() - t0
        return DecisionResult(
            adapter=adapter, ok=True, probs=probs, model=model or getattr(
                engine, "model_name", ""),
            latency_s=latency, label=label,
        )
    except Exception as exc:  # fail-open per item: one bad item never kills a run
        return DecisionResult(
            adapter=adapter, ok=False, model=model,
            error=f"{type(exc).__name__}: {exc}"[:200],
            latency_s=time.perf_counter() - t0,
        )


def run_file(
    path: str,
    engine: Any,
    *,
    out: Optional[str] = None,
    limit: Optional[int] = None,
    adapter: str = "systemone",
    model: str = "",
) -> Dict[str, Any]:
    """Score a JevBench jsonl split; optionally write predictions jsonl.

    Returns a summary: {n, scored, correct, accuracy, mean_latency_ms,
    by_family: {family: {n, correct, accuracy}}, adapter, model}.
    """
    model = model or getattr(engine, "model_name", "")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[: max(0, limit)]

    scored = correct = 0
    lat_total = 0.0
    by_family: Dict[str, Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []
    for item in rows:
        res = score_item(item, engine, adapter=adapter, model=model)
        predictions.append({"id": item.get("id"), **res.to_public()})
        if not res.ok or res.label is None:
            continue
        scored += 1
        lat_total += res.latency_s
        fam = by_family.setdefault(item.get("family", "unknown"),
                                   {"n": 0, "correct": 0})
        fam["n"] += 1
        if res.label == item.get("expected"):
            correct += 1
            fam["correct"] += 1
    for fam in by_family.values():
        fam["accuracy"] = (fam["correct"] / fam["n"]) if fam["n"] else None
    summary = {
        "adapter": adapter,
        "model": model,
        "n": len(rows),
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored) if scored else None,
        "mean_latency_ms": round(lat_total / scored * 1000, 1) if scored else None,
        "by_family": by_family,
    }
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        summary["predictions"] = out
    return summary


# -- remote scoring (kev.benchmark --remote style) ---------------------------
#
# Score any TypeSafe-dialect HTTP endpoint (this shim, kev.serve, SGLang's
# /v1/systemone) on a JevBench split. Retry policy mirrors kev's
# RemotePredictor: 408/429/5xx + transport errors are retried with
# backoff; other 4xx fail fast (auth/config — every item would fail
# identically); a refused request (422, e.g. state past the serving
# context) is recorded as that item's error without retry.


class RemoteEndpoint:
    """POST JevBench bodies to a /v1/systemone endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str = "systemone",
        api_key: str | None = None,
        timeout: float = 120.0,
        retries: int = 3,
    ) -> None:
        from .patterns import require_http_url

        self.base_url = require_http_url(
            base_url.strip().rstrip("/"), what="bench target URL")
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self.retries = max(1, retries)
        self.served_model: str | None = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST one body; returns the decoded answers mapping.

        Raises RuntimeError on fast-fail client errors, ValueError on
        refused requests (recorded per item, not retried).
        """
        import urllib.error
        import urllib.request

        payload = json.dumps({**body, "model": self.model}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/systemone", data=payload,
            headers=self._headers(), method="POST",
        )
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # nosec B310 -- scheme enforced in __init__ via patterns.require_http_url; nosemgrep
                    data = json.loads(resp.read().decode("utf-8") or "{}")
                self._latency_ms = (time.perf_counter() - t0) * 1000.0
                self.served_model = data.get("model", self.served_model)
                return data.get("answers") or {}
            except urllib.error.HTTPError as e:
                if e.code == 422:
                    raise ValueError(
                        "endpoint refused the request (HTTP 422)") from None
                if e.code not in (408, 429) and e.code < 500:
                    raise RuntimeError(
                        f"endpoint answered HTTP {e.code}; check auth/URL")
                last = e
            except Exception as e:  # transport errors and timeouts: retried
                last = e
            if attempt + 1 < self.retries:
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"endpoint failed after {self.retries} attempts: {last}")

    def score_item(
        self,
        item: Dict[str, Any],
        *,
        adapter: str = "systemone-remote",
        model: str = "",
    ) -> DecisionResult:
        """Score one JevBench item against the remote endpoint."""
        if (item.get("provenance") or {}).get("exclude_reason"):
            return DecisionResult(adapter=adapter, ok=False, model=model,
                                  error="excluded by provenance")
        t0 = time.perf_counter()
        try:
            answers = self.post(item_to_body(item))
            label, probs = normalize_answer(answers.get("decision") or {})
            if label is None:
                raise ValueError("endpoint returned no decision")
            return DecisionResult(
                adapter=adapter, ok=True, probs=probs,
                model=model or self.served_model or "",
                latency_s=time.perf_counter() - t0, label=label)
        except RuntimeError:
            raise  # fast-fail client errors stop the run (kev semantics)
        except Exception as exc:  # fail-open per item
            return DecisionResult(
                adapter=adapter, ok=False, model=model,
                error=f"{type(exc).__name__}: {exc}"[:200],
                latency_s=time.perf_counter() - t0)


def _read_items(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit is not None:
        rows = rows[: max(0, limit)]
    return rows


def run_remote(
    path: str,
    base_url: str,
    *,
    out: str | None = None,
    limit: int | None = None,
    adapter: str = "systemone-remote",
    model: str = "",
    api_key: str | None = None,
    timeout: float = 120.0,
    retries: int = 3,
    concurrency: int = 1,
) -> Dict[str, Any]:
    """Score a JevBench jsonl split against a remote /v1/systemone endpoint.

    Same summary shape as run_file, plus "served_model" (the model id
    the endpoint reported) and "target". `concurrency` keeps that many
    requests in flight (order-preserving); 1 scores sequentially.
    """
    from concurrent.futures import ThreadPoolExecutor

    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    endpoint = RemoteEndpoint(base_url, model=model or "systemone",
                              api_key=api_key, timeout=timeout,
                              retries=retries)
    rows = _read_items(path, limit)
    if concurrency > 1 and len(rows) > 1:
        with ThreadPoolExecutor(
                max_workers=min(concurrency, len(rows))) as pool:
            results = list(pool.map(
                lambda it: endpoint.score_item(
                    it, adapter=adapter, model=model), rows))
    else:
        results = [endpoint.score_item(it, adapter=adapter, model=model)
                   for it in rows]

    scored = correct = 0
    lat_total = 0.0
    by_family: Dict[str, Dict[str, Any]] = {}
    predictions: List[Dict[str, Any]] = []
    for item, res in zip(rows, results):
        predictions.append({"id": item.get("id"), **res.to_public()})
        if not res.ok or res.label is None:
            continue
        scored += 1
        lat_total += res.latency_s
        fam = by_family.setdefault(item.get("family", "unknown"),
                                   {"n": 0, "correct": 0})
        fam["n"] += 1
        if res.label == item.get("expected"):
            correct += 1
            fam["correct"] += 1
    for fam in by_family.values():
        fam["accuracy"] = (fam["correct"] / fam["n"]) if fam["n"] else None
    summary = {
        "adapter": adapter,
        "model": model or endpoint.served_model or "",
        "served_model": endpoint.served_model,
        "target": base_url,
        "n": len(rows),
        "scored": scored,
        "correct": correct,
        "accuracy": (correct / scored) if scored else None,
        "mean_latency_ms": round(lat_total / scored * 1000, 1) if scored else None,
        "by_family": by_family,
    }
    if out:
        with open(out, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        summary["predictions"] = out
    return summary
