"""Test-time rotation averaging over choice option orders.

Ported from Kev (kev/predictors.py RotationAveraged): judges can be
position-biased (the first option wins ties it should not), so a
choice question is scored under the first ``rotations`` cyclic
rotations of its options and the per-option probabilities are
averaged geometrically (mean in log space, then softmax). Noul and
score questions keep their order — it is part of their meaning.

Costs ``rotations`` engine calls per ``systemone()`` call (fewer when
the widest choice question has fewer options than ``rotations``).
Torch-free and duck-typed: wraps any object with
``systemone(state, questions, ...)``.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, List, Sequence


class RotationAveraged:
    """Engine wrapper averaging choice judgments over option rotations."""

    backend_name = "rotation-averaged"

    def __init__(self, engine: Any, rotations: int = 3) -> None:
        if rotations < 2:
            raise ValueError("rotation averaging needs at least 2 rotations")
        self.engine = engine
        self.rotations = int(rotations)
        self.model_name = (
            f"rotation-averaged({getattr(engine, 'model_name', '?')},"
            f"rotations={self.rotations})"
        )

    @staticmethod
    def rotated(questions: Sequence[Dict[str, Any]], r: int) -> List[Dict[str, Any]]:
        """Questions with every choice's options rotated cyclically by r."""
        out = []
        for q in questions:
            if q.get("type") != "choice":
                out.append(q)
                continue
            opts = list(q.get("options") or [])
            k = r % len(opts) if opts else 0
            out.append({**q, "options": opts[k:] + opts[:k]})
        return out

    def _call(self, state: str, questions: Sequence[Dict[str, Any]],
              images: Any, videos: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if images or videos:
            try:
                params = inspect.signature(self.engine.systemone).parameters
            except (TypeError, ValueError):
                params = {}
            if images and "images" in params:
                kwargs["images"] = images
            if videos and "videos" in params:
                kwargs["videos"] = videos
        return self.engine.systemone(state, questions, **kwargs)

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer with choice questions rotation-averaged."""
        from .patterns import choice_confidence

        questions = list(questions)
        widest = max(
            (len(list(q.get("options") or [])) for q in questions
             if q.get("type") == "choice"),
            default=1,
        )
        n = min(self.rotations, widest)
        runs = [self._call(state, self.rotated(questions, r), images, videos)
                for r in range(n)]
        answers: Dict[str, Any] = {}
        for q in questions:
            name = q.get("name", "q")
            first = runs[0].get(name)
            if not isinstance(first, dict) or first.get("type") != "choice":
                answers[name] = first
                continue
            keys = list((first.get("probabilities") or {}).keys())
            log_mean = {}
            for k in keys:
                vals = []
                for run in runs:
                    probs = (run.get(name) or {}).get("probabilities") or {}
                    vals.append(math.log(max(float(probs.get(k, 0.0)), 1e-12)))
                log_mean[k] = sum(vals) / len(vals)
            top = max(log_mean.values())
            exps = {k: math.exp(v - top) for k, v in log_mean.items()}
            total = sum(exps.values())
            probs = {k: v / total for k, v in exps.items()}
            best = max(probs, key=probs.get)
            answers[name] = {
                "type": "choice",
                "choice": best,
                "probabilities": probs,
                "confidence": choice_confidence(list(probs.values())),
            }
            if first.get("label_mass") is not None:
                masses = [r.get(name, {}).get("label_mass") for r in runs]
                masses = [m for m in masses if isinstance(m, (int, float))]
                if masses:
                    answers[name]["label_mass"] = sum(masses) / len(masses)
        answers["_meta"] = {
            "backend": "rotation-averaged",
            "model": self.model_name,
            "rotations": n,
            "n_questions": len(questions),
        }
        base_meta = runs[0].get("_meta") or {}
        if isinstance(base_meta.get("latency_ms"), (int, float)):
            answers["_meta"]["latency_ms"] = round(
                sum(float(r.get("_meta", {}).get("latency_ms", 0.0))
                    for r in runs), 1)
        return answers
