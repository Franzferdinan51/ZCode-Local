"""Jev-style System One API over local GLiClass models.

One call takes a `state` (text) plus multiple typed questions and answers
them all in a single batched forward pass — adding questions barely changes
latency, mirroring Jev's parallel evaluation.

Question types (mirroring Jev's three primitives):
- choice: pick from a label list -> label + per-option probabilities + confidence
- score:  rate against ordered levels  -> level + distribution + confidence
- noul:   yes/no question              -> probability the statement is true

Only ONE local model is ever loaded per SystemOne instance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer

from gliclass import GLiClassModel
from gliclass.pipeline import ZeroShotClassificationPipeline

from .calibration import (
    PerTypeTemperatureCalibrator,
    TemperatureCalibrator,
    softmax,
)
from .patterns import (
    ABSTAIN_LABEL,
    MAX_STATE_CHARS,
    MODEL_CANDIDATES,
    LatencyStats,
    StallGuard,
    SystemOneError,
    _resolve_candidates,
    build_decision_prompts,
    choice_confidence,
    make_questions,
    noul_confidence,
    resolve_revision,
    score_confidence,
    validate_choice,
    validate_distribution,
    with_abstain,
)

def default_device() -> str:
    """Best torch device for this machine: CUDA > Apple MPS > CPU.

    Stock pip torch wheels are CUDA-enabled on Linux/Windows and
    MPS-capable on macOS, so Apple Silicon gets GPU acceleration with no
    extra installs. Never raises: if the MPS backend is absent (older
    torch), it is simply skipped.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _sanitize_detail(err: Exception) -> str:
    """One-line, path/secret-free summary of an unexpected exception."""
    text = f"{type(err).__name__}: {err}".splitlines()[0]
    # strip anything that looks like a filesystem path or URL with creds
    return text[:300]


class SystemOne:
    """Local System One decision engine.

    Args:
        model_name: HF id of the GLiClass checkpoint. If None, the
            SYSTEMONE_MODEL env var is honored, else MODEL_CANDIDATES
            smallest-first.
        device: "cuda", "mps", "cpu", or None / "auto" (auto-detect:
            CUDA if available, else Apple MPS, else CPU).
        temperature: softmax temperature for output probabilities (1.0 = raw).
        calibrator: optional fitted TemperatureCalibrator or
            PerTypeTemperatureCalibrator (per-answer-type temperature map,
            adapted from Mapika/decider); overrides temperature.
        revision: HF revision pin for model/tokenizer downloads. If None,
            the SYSTEMONE_REVISION env var is honored; unset -> default
            branch (unpinned).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        temperature: float = 1.0,
        calibrator: TemperatureCalibrator | None = None,
        revision: str | None = None,
    ) -> None:
        if device is None or (
            isinstance(device, str) and device.strip().lower() == "auto"
        ):
            # "auto" (the SYSTEMONE_DEVICE default) resolves here, so every
            # entry point — shim, CLI, MCP server — gets CUDA > MPS > CPU.
            device = default_device()
        self.device = device

        candidates = _resolve_candidates(model_name)
        self.revision = resolve_revision(revision)
        rev_kwargs = {"revision": self.revision} if self.revision else {}
        last_err: Exception | None = None
        for cand in candidates:
            try:
                self.model = GLiClassModel.from_pretrained(cand, **rev_kwargs)
                # Pinned via rev_kwargs when SYSTEMONE_REVISION is set; the
                # scanner cannot see through **kwargs (B615 false positive).
                self.tokenizer = AutoTokenizer.from_pretrained(cand, **rev_kwargs)  # nosec B615
                self.model_name = cand
                last_err = None
                break
            except Exception as e:  # try next candidate
                last_err = e
        if last_err is not None:
            raise SystemOneError(
                "could not load any GLiClass model",
                hint=f"last error: {_sanitize_detail(last_err)}; "
                "check network access to huggingface.co or set a cached model via SYSTEMONE_MODEL",
            )

        self.pipeline = ZeroShotClassificationPipeline(
            self.model, self.tokenizer, device=device
        )
        self.temperature = temperature
        self.calibrator = calibrator

    # -- calibration ----------------------------------------------------
    def set_calibrator(
        self, calibrator: TemperatureCalibrator | PerTypeTemperatureCalibrator
    ) -> None:
        """Attach a fitted calibrator (overrides temperature).

        A PerTypeTemperatureCalibrator applies the question's own temperature
        (decider's >=1.4.0 semantics); a plain TemperatureCalibrator applies
        one pooled temperature to every question type.
        """
        self.calibrator = calibrator

    def _probs(self, scores: np.ndarray, qtype: str = "choice") -> np.ndarray:
        cal = self.calibrator
        if cal is not None and getattr(cal, "fitted_", False):
            if hasattr(cal, "temperature_for"):
                # per-answer-type map: the fitted temperature for THIS
                # question type is actually applied (decider >=1.4.0).
                return np.asarray(cal.predict_proba(scores, qtype))
            return np.asarray(cal.predict_proba([scores])[0])
        T = self.temperature if self.temperature > 0 else 1.0
        return softmax(np.asarray(scores, dtype=np.float64) / T)

    # -- single batched call --------------------------------------------
    def raw_scores(
        self,
        texts: List[str],
        label_lists: List[List[str]],
        prompts: List[str | None] | None = None,
        batch_size: int = 32,
        classification_type: str = "single_label",
    ) -> List[Dict[str, float]]:
        """One pipeline call; returns per-text {label: raw_score} dicts.

        classification_type="single_label" is winner-take-all (one nonzero
        entry per text) — right for tier routing. "multi_label" scores every
        label independently — right for tool/MCP relevance ranking.
        """
        results = self.pipeline(
            texts,
            label_lists,
            threshold=0.0,
            batch_size=batch_size,
            classification_type=classification_type,
            prompt=prompts,
        )
        out: List[Dict[str, float]] = []
        for res, labs in zip(results, label_lists):
            # res: list of {"label":..., "score":...}; be defensive about
            # threshold filtering by defaulting missing labels to 0.0
            got = {r["label"]: float(r["score"]) for r in res} if res else {}
            out.append({lab: got.get(lab, 0.0) for lab in labs})
        return out

    def systemone(
        self,
        state: str,
        questions: Sequence[Dict[str, Any]],
        batch_size: int = 32,
        build_prompts: bool = True,
        shuffle_options: bool = False,
        prompt_seed: int | None = None,
        images: Sequence[Any] | None = None,
        videos: Sequence[Any] | None = None,
    ) -> Dict[str, Any]:
        """Answer multiple typed questions about `state` in one batched pass.

        Each question: {"name": str, "type": "choice"|"score"|"noul", ...}
          choice: {"options": [str, ...], "prompt": optional str}
          score:  {"levels": [str, ...],  "prompt": optional str}  (ordered)
          noul:   {"statement": str}  (yes/no question about the state)

        When build_prompts is True (default), every question gets a
        state-first prompt row (see build_decision_prompts). shuffle_options
        shuffles choice options in the row (seeded by prompt_seed); score
        levels keep their order and the abstain label stays last. Pass
        build_prompts=False for the legacy behavior (explicit prompt or None
        straight to the pipeline).

        Confidence follows the TypeSafe definitions (adapted from
        Mapika/decider): choice (n*p_max-1)/(n-1), score
        max(0, 1 - sum_i p_i*|i-k|/(n-1)), noul max(P(yes), P(no)).

        Returns {name: answer_dict, ..., "_meta": {...}}.
        """
        questions = list(questions)
        if not questions:
            raise SystemOneError("questions must be non-empty")

        state_capped = False
        if len(state) > MAX_STATE_CHARS:
            state = state[:MAX_STATE_CHARS]
            state_capped = True

        if build_prompts:
            prompts, label_lists = build_decision_prompts(
                state,
                questions,
                shuffle_options=shuffle_options,
                seed=prompt_seed,
            )
        else:
            label_lists = []
            prompts = []
            for q in questions:
                qtype = q["type"]
                if qtype == "choice":
                    label_lists.append(list(q["options"]))
                    prompts.append(q.get("prompt"))
                elif qtype == "score":
                    label_lists.append(list(q["levels"]))
                    prompts.append(q.get("prompt"))
                elif qtype == "noul":
                    label_lists.append(["yes", "no"])
                    prompts.append(q.get("statement") or q.get("prompt"))
                else:
                    raise SystemOneError(
                        f"unknown question type: {qtype!r}",
                        hint="expected one of: choice, score, noul",
                    )

        t0 = time.perf_counter()
        score_dicts = self.raw_scores(
            [state] * len(questions), label_lists, prompts=prompts,
            batch_size=batch_size,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        answers: Dict[str, Any] = {}
        for q, labs, sdict in zip(questions, label_lists, score_dicts):
            scores = np.array([sdict[lab] for lab in labs], dtype=np.float64)
            qtype = q["type"]
            probs = self._probs(scores, qtype=qtype)
            prob_map = {lab: float(p) for lab, p in zip(labs, probs)}
            if qtype == "choice":
                best = labs[int(probs.argmax())]
                validate_distribution(prob_map, labs, best)
                answers[q["name"]] = {
                    "type": "choice",
                    "choice": best,
                    "probabilities": prob_map,
                    "confidence": choice_confidence(probs),
                }
            elif qtype == "score":
                best = labs[int(probs.argmax())]
                validate_distribution(prob_map, labs, best)
                answers[q["name"]] = {
                    "type": "score",
                    "level": best,
                    "distribution": prob_map,
                    "confidence": score_confidence(probs),
                    "legend": dict(q.get("legend") or {}),
                }
            else:  # noul
                p_yes = prob_map["yes"]
                best = "yes" if p_yes >= 0.5 else "no"
                validate_distribution(prob_map, ["yes", "no"], best)
                answers[q["name"]] = {
                    "type": "noul",
                    "probability": p_yes,
                    "answer": bool(p_yes >= 0.5),
                    "confidence": noul_confidence(p_yes),
                }

        answers["_meta"] = {
            "model": self.model_name,
            "device": self.device,
            "n_questions": len(questions),
            "latency_ms": round(latency_ms, 1),
            "state_chars": len(state),
            "state_capped": state_capped,
        }
        if images or videos:
            # Text-only GLiClass engine: media is noted and skipped.
            answers["_meta"]["media_dropped"] = {
                "images": len(list(images or [])),
                "videos": len(list(videos or [])),
            }
        return answers

    def speculative_decide(
        self,
        state: str,
        operation: Dict[str, Any],
        targets: Dict[str, Dict[str, Any]],
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Decide an operation AND its argument in a single batched pass.

        Speculative multi-head pattern ported from jev-ultrafast (model.py)
        and mobile-jev (policy.mjs): one request carries the operation choice
        plus one target choice-head per operation. Only the target head
        selected by the chosen operation is validated and used — *unused
        target heads cannot cause an action*.

        Args:
            state: the state text to decide about.
            operation: a choice question, e.g.
                {"name": "op", "type": "choice",
                 "options": ["click", "fill", "wait"]}
            targets: {operation_value: choice question} for operations that
                take a target, e.g. {"click": {"name": "click_target",
                "type": "choice", "options": ["btn-1", "btn-2"]}}.
                Operations without an entry need no target.

        Returns {"operation", "target" (or None), "confidence",
                 "target_confidence" (or None), "probabilities",
                 "operation_probabilities", "_meta"}.
        """
        op_name = operation.get("name", "operation")
        op_options = list(operation["options"])
        questions: List[Dict[str, Any]] = [operation]
        for op in op_options:
            if op in targets:
                questions.append(targets[op])

        answers = self.systemone(state, questions, batch_size=batch_size)

        op_answer = validate_choice(answers[op_name], op_options)
        op_choice = op_answer["choice"]

        target = None
        target_conf: float | None = None
        target_probs: Dict[str, float] = {}
        if op_choice in targets:
            tq = targets[op_choice]
            t_name = tq.get("name", f"{op_choice}_target")
            t_options = list(tq["options"])
            t_answer = validate_choice(answers[t_name], t_options)
            target = t_answer["choice"]
            target_conf = t_answer["confidence"]
            target_probs = t_answer["probabilities"]

        return {
            "operation": op_choice,
            "target": target,
            "confidence": op_answer["confidence"],
            "target_confidence": target_conf,
            "probabilities": target_probs,
            "operation_probabilities": op_answer["probabilities"],
            "_meta": answers["_meta"],
        }



