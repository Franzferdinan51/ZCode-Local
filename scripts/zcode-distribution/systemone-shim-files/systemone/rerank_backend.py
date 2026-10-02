"""Cross-encoder judge backend: zero-shot decisions from reranker scores.

Any cross-encoder (query, document) -> relevance score becomes a Jev-style
judge: each option/level is scored against the state and the scores are
softmaxed into a distribution. Confidence reuses the TypeSafe definitions
in patterns.py, so a reranker judge is calibration-compatible with every
other engine.

Two pieces:

- RerankBackend: pure mapping over an injected ``score_fn`` (no ML
  dependency — unit-testable with a stub).
- OnnxCrossEncoder: ``score_fn`` backed by an ONNX cross-encoder
  (default Xenova/bge-reranker-base int8) via onnxruntime + tokenizers.
  Both are lazy imports: installing them is only required to construct
  one. Runs CPU-only — the verified no-torch local judge.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Sequence


def _softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs) if xs else 0.0
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RerankBackend:
    """Jev answers from a cross-encoder score function.

    Args:
        score_fn: (state, option_text) -> float, higher = better match.
        model_name: label reported in answers' _meta.
        noul_temperature: divisor inside sigmoid(score / T) for noul.
            Reranker scores are unbounded logits, so noul is inherently
            less calibrated than choice/score (which softmax over
            competing options); tune T per deployment.

    Option text is the label plus its criteria description when the
    question carries one ("<label>: <description>"), so bare labels like
    "yes" still carry meaning into the pair. The query is the state plus
    the question instructions (split out of the translated prompt); the
    prompt's embedded options block is dropped since it is identical for
    every candidate and only flattens score differences.
    """

    backend_name = "rerank"

    def __init__(
        self,
        score_fn: Callable[[str, str], float],
        model_name: str = "rerank",
        noul_temperature: float = 1.0,
    ) -> None:
        from .patterns import (
            choice_confidence,
            noul_confidence,
            score_confidence,
        )

        self.score_fn = score_fn
        self.model_name = model_name
        self.noul_temperature = float(noul_temperature) or 1.0
        self._choice_confidence = choice_confidence
        self._score_confidence = score_confidence
        self._noul_confidence = noul_confidence

    @staticmethod
    def _instructions(prompt: str | None) -> str:
        """Instructions half of a translated prompt (options block dropped)."""
        head, sep, _ = (prompt or "").partition("\nOptions:\n")
        return (head if sep else (prompt or "")).strip()

    @classmethod
    def _option_text(cls, label: str, descriptions: Dict[str, str] | None) -> str:
        desc = (descriptions or {}).get(label, "")
        desc = desc.strip() if isinstance(desc, str) else ""
        return f"{label}: {desc}" if desc else label

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer typed questions; media is noted and skipped (text-only).

        `images` / `videos` are accepted for protocol uniformity and
        reported in ``_meta["media_dropped"]`` — the cross-encoder reads
        text pairs only.
        """
        from .patterns import validate_distribution

        t0 = time.perf_counter()
        answers: Dict[str, Any] = {}
        for q in questions:
            name = q.get("name", "q")
            qtype = q.get("type")
            instructions = self._instructions(q.get("prompt"))
            query = f"{state}\n{instructions}" if instructions else state
            descs = q.get("descriptions") if isinstance(
                q.get("descriptions"), dict) else None
            if qtype == "choice":
                options = list(q.get("options") or [])
                if len(options) < 2:
                    raise ValueError(f"question {name!r} needs >= 2 options")
                scores = [self.score_fn(query, self._option_text(o, descs))
                          for o in options]
                probs_list = _softmax(scores)
                probs = dict(zip(options, probs_list))
                best = max(probs, key=probs.get)
                validate_distribution(probs, options, best)
                answers[name] = {
                    "type": "choice", "choice": best,
                    "probabilities": probs,
                    "confidence": self._choice_confidence(probs_list),
                }
            elif qtype == "score":
                levels = list(q.get("levels") or [])
                if len(levels) < 2:
                    raise ValueError(f"question {name!r} needs >= 2 levels")
                scores = [self.score_fn(query, self._option_text(lv, descs))
                          for lv in levels]
                dist_list = _softmax(scores)
                dist = dict(zip(levels, dist_list))
                level = max(dist, key=dist.get)
                validate_distribution(dist, levels, level)
                answers[name] = {
                    "type": "score", "level": level,
                    "distribution": dist,
                    "score": sum(i * p for i, p in enumerate(dist_list)),
                    "confidence": self._score_confidence(dist_list),
                    "legend": dict(q.get("legend") or {}),
                }
            elif qtype == "noul":
                statement = q.get("statement") or q.get("prompt") or name
                s = self.score_fn(state, statement)
                p = _sigmoid(s / self.noul_temperature)
                answers[name] = {
                    "type": "noul", "probability": p, "answer": p >= 0.5,
                    "confidence": self._noul_confidence(p),
                }
            else:
                raise ValueError(f"question {name!r}: unknown type {qtype!r}")
        answers["_meta"] = {
            "model": self.model_name,
            "backend": "rerank",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        if images or videos:
            answers["_meta"]["media_dropped"] = {
                "images": len(list(images or [])),
                "videos": len(list(videos or [])),
            }
        return answers


class OnnxCrossEncoder:
    """Cross-encoder score_fn over onnxruntime (CPU-friendly, no torch).

    Args:
        model_id: Hugging Face repo with an ONNX export + tokenizer.json.
        filename: ONNX file inside the repo (int8 default: small + fast CPU).
        max_length: pair truncation length.
        cache_dir: optional HF cache override.
        revision: HF revision pin (commit SHA / tag / branch). If None,
            $RERANK_REVISION is honored; unset -> the default branch.
            Pin a SHA in production so model downloads are reproducible
            and immune to tag moves.

    Example:
        enc = OnnxCrossEncoder()  # downloads ~280 MB once, then cached
        eng = RerankBackend(enc.score, model_name="bge-reranker-base-int8")
    """

    def __init__(
        self,
        model_id: str = "Xenova/bge-reranker-base",
        filename: str = "onnx/model_int8.onnx",
        max_length: int = 512,
        cache_dir: str | None = None,
        revision: str | None = None,
    ) -> None:
        import os

        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "OnnxCrossEncoder needs onnxruntime + tokenizers + "
                "huggingface_hub (pip install onnxruntime tokenizers "
                "huggingface_hub)"
            ) from exc

        self.model_id = model_id
        self.max_length = max_length
        self.revision = (revision or "").strip() or (
            os.environ.get("RERANK_REVISION") or "").strip() or None
        model_path = hf_hub_download(model_id, filename, cache_dir=cache_dir,
                                     revision=self.revision)
        tok_path = hf_hub_download(model_id, "tokenizer.json",
                                   cache_dir=cache_dir,
                                   revision=self.revision)
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = Tokenizer.from_file(tok_path)
        self._input_names = [i.name for i in self.session.get_inputs()]

    def score(self, query: str, doc: str) -> float:
        import numpy as np

        self.tokenizer.enable_truncation(max_length=self.max_length)
        try:
            enc = self.tokenizer.encode(query, doc)
        finally:
            self.tokenizer.no_truncation()
        feed: Dict[str, Any] = {}
        if "input_ids" in self._input_names:
            feed["input_ids"] = np.array([enc.ids], dtype=np.int64)
        if "attention_mask" in self._input_names:
            feed["attention_mask"] = np.array([enc.attention_mask],
                                              dtype=np.int64)
        if "token_type_ids" in self._input_names:
            tti = list(enc.type_ids) or [0] * len(enc.ids)
            feed["token_type_ids"] = np.array([tti], dtype=np.int64)
        logits = self.session.run(None, feed)[0]
        flat = np.asarray(logits).ravel()
        return float(flat[0])
