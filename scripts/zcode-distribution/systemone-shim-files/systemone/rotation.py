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


class ConformalChoice:
    """Engine wrapper adding split-conformal prediction sets to choices.

    Point predictions stay the engine's; every choice answer gains
    ``prediction_set`` (options in descending-probability order) and
    ``coverage`` (1 - alpha): on exchangeable future items the set
    contains gold with marginal probability >= coverage (MAPIE/APS
    method — see calibration.fit_conformal_threshold). Noul and score
    answers pass through untouched.
    """

    backend_name = "conformal-choice"

    def __init__(self, engine: Any, tau: float, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.engine = engine
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.model_name = (
            f"conformal-choice({getattr(engine, 'model_name', '?')},"
            f"coverage={1.0 - self.alpha:.2f})"
        )

    @classmethod
    def fit(
        cls, engine: Any, records: Sequence[Dict[str, Any]],
        alpha: float = 0.1,
    ) -> "ConformalChoice":
        """Build from labeled DecisionRecord rows (calibration fit)."""
        from .calibration import fit_conformal_threshold

        return cls(engine, fit_conformal_threshold(records, alpha)["tau"],
                   alpha)

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        from .calibration import conformal_set

        kwargs: Dict[str, Any] = {}
        try:
            params = inspect.signature(self.engine.systemone).parameters
        except (TypeError, ValueError):
            params = {}
        if images and "images" in params:
            kwargs["images"] = images
        if videos and "videos" in params:
            kwargs["videos"] = videos
        answers = self.engine.systemone(state, questions, **kwargs)
        out = dict(answers)
        for name, ans in answers.items():
            if name == "_meta" or not isinstance(ans, dict):
                continue
            if ans.get("type") != "choice":
                continue
            dist = ans.get("probabilities") or {}
            if not dist:
                continue
            out[name] = {**ans,
                         "prediction_set": conformal_set(dist, self.tau),
                         "coverage": round(1.0 - self.alpha, 4)}
        meta = dict(out.get("_meta") or {})
        meta["conformal"] = {"tau": self.tau, "coverage": 1.0 - self.alpha}
        out["_meta"] = meta
        return out


class SelfConsistent:
    """Self-consistency wrapper: majority vote over sampled judgments.

    The self-consistency trick (Wang et al., 2023): sample the same
    choice question several times under different option shuffles and
    take the majority winner. Probabilities are the linear opinion pool
    (mean) across samples; ``agreement`` (fraction of samples voting the
    winner) rides along as the disagreement signal — low agreement with
    high confidence is the classic abstain trigger. Engines whose
    ``systemone()`` takes ``shuffle_options``/``prompt_seed`` get real
    shuffle diversity; others are simply called again (still useful for
    stochastic judges).
    """

    backend_name = "self-consistent"

    def __init__(self, engine: Any, samples: int = 5, seed: int = 0) -> None:
        if samples < 2:
            raise ValueError("self-consistency needs at least 2 samples")
        self.engine = engine
        self.samples = int(samples)
        self.seed = int(seed)
        self.model_name = (
            f"self-consistent({getattr(engine, 'model_name', '?')},"
            f"samples={self.samples})"
        )

    def _call(self, state: str, questions: Sequence[Dict[str, Any]],
              images: Any, videos: Any, sample: int) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        try:
            params = inspect.signature(self.engine.systemone).parameters
        except (TypeError, ValueError):
            params = {}
        if images and "images" in params:
            kwargs["images"] = images
        if videos and "videos" in params:
            kwargs["videos"] = videos
        if "shuffle_options" in params:
            kwargs["shuffle_options"] = True
        if "prompt_seed" in params:
            kwargs["prompt_seed"] = self.seed + sample
        return self.engine.systemone(state, questions, **kwargs)

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        from .patterns import choice_confidence
        from .scoring import mean_distributions

        questions = list(questions)
        runs = [self._call(state, questions, images, videos, i)
                for i in range(self.samples)]
        answers: Dict[str, Any] = {}
        agreements = []
        for q in questions:
            name = q.get("name", "q")
            first = runs[0].get(name)
            if not isinstance(first, dict) or first.get("type") != "choice":
                answers[name] = first
                continue
            dists = [(r.get(name) or {}).get("probabilities") or {}
                     for r in runs]
            dists = [d for d in dists if d]
            if not dists:
                answers[name] = first
                continue
            votes = [max(d, key=d.get) for d in dists]
            mean = mean_distributions(dists)
            winner = max(set(votes),
                         key=lambda c: (votes.count(c), mean.get(c, 0.0)))
            agreement = votes.count(winner) / len(votes)
            agreements.append(agreement)
            answers[name] = {
                "type": "choice",
                "choice": winner,
                "probabilities": mean,
                "confidence": choice_confidence(list(mean.values())),
                "agreement": round(agreement, 4),
            }
        answers["_meta"] = {
            "backend": "self-consistent",
            "model": self.model_name,
            "samples": self.samples,
            "n_questions": len(questions),
        }
        if agreements:
            answers["_meta"]["agreement_mean"] = round(
                sum(agreements) / len(agreements), 4)
        base_meta = runs[0].get("_meta") or {}
        if isinstance(base_meta.get("latency_ms"), (int, float)):
            answers["_meta"]["latency_ms"] = round(
                sum(float(r.get("_meta", {}).get("latency_ms", 0.0))
                    for r in runs), 1)
        return answers


class EnsembleBackend:
    """Fuse several engines into one judge (voting/rank/prob fusion).

    Strategies for choice questions (noul takes the mean probability,
    score the mean distribution under every strategy):

    - "average": linear opinion pool (mean distribution), argmax wins.
    - "extremized": extremized mean (strength param, default 1.0).
    - "vote": majority winner (ties break by mean probability);
      ``agreement`` rides along like SelfConsistent.
    - "rrf": reciprocal rank fusion (k param, default 60) picks the
      winner; probabilities stay the honest mean, raw fusion scores in
      ``fusion_scores``.
    - "borda": Borda-count fusion, same shape as "rrf".

    Costs one call per engine per question batch. Torch-free and
    duck-typed like RotationAveraged.
    """

    backend_name = "ensemble"

    STRATEGIES = ("average", "extremized", "vote", "rrf", "borda")

    def __init__(
        self,
        engines: Sequence[Any],
        strategy: str = "average",
        extremize: float = 1.0,
        rrf_k: float = 60.0,
    ) -> None:
        engines = list(engines)
        if len(engines) < 2:
            raise ValueError("an ensemble needs at least 2 engines")
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"unknown strategy {strategy!r}; want "
                f"{'|'.join(self.STRATEGIES)}")
        self.engines = engines
        self.strategy = strategy
        self.extremize = float(extremize)
        self.rrf_k = float(rrf_k)
        self.model_name = (
            f"ensemble({strategy},n={len(engines)}:"
            f"{','.join(str(getattr(e, 'model_name', '?')) for e in engines)})"
        )

    def _call(self, engine: Any, state: str,
              questions: Sequence[Dict[str, Any]], images: Any,
              videos: Any) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
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
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        from .patterns import (
            choice_confidence,
            noul_confidence,
            score_confidence,
            validate_distribution,
        )
        from .scoring import (
            borda_fuse,
            extremized_average,
            mean_distributions,
            rrf_fuse,
        )

        questions = list(questions)
        runs = [self._call(e, state, questions, images, videos)
                for e in self.engines]
        answers: Dict[str, Any] = {}
        for q in questions:
            name = q.get("name", "q")
            first = runs[0].get(name)
            if not isinstance(first, dict):
                answers[name] = first
                continue
            qtype = first.get("type")
            if qtype == "noul":
                ps = [float((r.get(name) or {}).get("probability", 0.5))
                      for r in runs]
                p = sum(ps) / len(ps)
                answers[name] = {"type": "noul", "probability": p,
                                 "answer": p >= 0.5,
                                 "confidence": noul_confidence(p)}
                continue
            if qtype == "score":
                dists = [(r.get(name) or {}).get("distribution") or {}
                         for r in runs]
                dists = [d for d in dists if d]
                if not dists:
                    answers[name] = first
                    continue
                dist = mean_distributions(dists)
                level = max(dist, key=dist.get)
                validate_distribution(dist, list(dist), level)
                answers[name] = {
                    "type": "score", "level": level, "distribution": dist,
                    "score": sum(i * p for i, p in enumerate(dist.values())),
                    "confidence": score_confidence(list(dist.values())),
                    "legend": (first.get("legend")
                               or {lv: lv for lv in dist}),
                }
                continue
            if qtype != "choice":
                answers[name] = first
                continue
            dists = [(r.get(name) or {}).get("probabilities") or {}
                     for r in runs]
            dists = [d for d in dists if d]
            if not dists:
                answers[name] = first
                continue
            mean = mean_distributions(dists)
            ans: Dict[str, Any] = {"type": "choice",
                                   "probabilities": mean}
            if self.strategy == "extremized":
                mean = extremized_average(dists, self.extremize)
                ans["probabilities"] = mean
                ans["choice"] = max(mean, key=mean.get)
            elif self.strategy == "vote":
                votes = [max(d, key=d.get) for d in dists]
                winner = max(set(votes),
                             key=lambda c: (votes.count(c), mean.get(c, 0.0)))
                ans["choice"] = winner
                ans["agreement"] = round(votes.count(winner) / len(votes), 4)
            elif self.strategy in ("rrf", "borda"):
                rankings = [sorted(d, key=d.get, reverse=True) for d in dists]
                fused = (rrf_fuse(rankings, self.rrf_k) if self.strategy == "rrf"
                         else borda_fuse(rankings))
                ans["choice"] = max(fused, key=fused.get)
                ans["fusion_scores"] = {k: round(v, 6)
                                        for k, v in fused.items()}
            else:
                ans["choice"] = max(mean, key=mean.get)
            ans["confidence"] = choice_confidence(
                list(ans["probabilities"].values()))
            answers[name] = ans
        answers["_meta"] = {
            "backend": "ensemble",
            "model": self.model_name,
            "strategy": self.strategy,
            "n_engines": len(self.engines),
            "n_questions": len(questions),
        }
        return answers
