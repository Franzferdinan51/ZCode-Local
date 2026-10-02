"""Probability calibration for GLiClass raw scores.

GLiClass returns raw per-label scores that are not honest probabilities
(typically overconfident). This module implements standard post-hoc
calibration methods — the practical stand-in for Jev's RLCD-trained
calibrated confidence:

- TemperatureCalibrator: single temperature T fit by NLL minimization,
  applied as softmax(scores / T). Best for multi-class (choice/score).
- PlattCalibrator: 1-D logistic regression on the positive-class score.
  Best for binary decisions (noul, safe/unsafe).
- IsotonicCalibrator: non-parametric isotonic regression. More flexible
  than Platt when you have enough calibration data (a few hundred points).

Also provides expected_calibration_error() to measure miscalibration
before/after.
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

try:
    from scipy.optimize import minimize_scalar
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover
    _HAS_SCIPY = False

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error for binary/confidence predictions.

    Bins predictions by confidence; ECE = sum_b |acc_b - conf_b| * (n_b / n).
    A perfectly calibrated model has ECE ~ 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    assert y_true.shape == y_prob.shape, "y_true and y_prob must match"
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # last bin is inclusive on the right so p=1.0 lands somewhere
        mask = (y_prob > lo) & (y_prob <= hi) if i else (y_prob >= lo) & (y_prob <= hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = y_true[mask].mean()
        conf_b = y_prob[mask].mean()
        ece += (n_b / len(y_true)) * abs(acc_b - conf_b)
    return float(ece)


def multiclass_ece(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    n_bins: int = 10,
) -> float:
    """ECE over the predicted (max-probability) class for multi-class outputs."""
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    return expected_calibration_error(correct, conf, n_bins=n_bins)


class TemperatureCalibrator:
    """Single-parameter temperature scaling: p = softmax(scores / T).

    Fit T by minimizing negative log-likelihood on a labeled calibration
    set. T > 1 softens overconfident scores; T < 1 sharpens underconfident
    ones. Requires scipy.
    """

    def __init__(self) -> None:
        self.temperature_: float = 1.0
        self.fitted_: bool = False

    def fit(self, scores: Sequence[Sequence[float]], labels: Sequence[int]) -> "TemperatureCalibrator":
        if not _HAS_SCIPY:
            raise ImportError("scipy is required for TemperatureCalibrator.fit()")
        S = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        n = len(y)

        def nll(logT: float) -> float:
            T = float(np.exp(logT))
            P = softmax(S / T)
            # gather predicted probability of the true class
            p_true = P[np.arange(n), y]
            return float(-np.log(np.clip(p_true, 1e-12, 1.0)).mean())

        res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded",
                              options={"xatol": 1e-4})
        self.temperature_ = float(np.exp(res.x))
        self.fitted_ = True
        return self

    def predict_proba(self, scores: Sequence[Sequence[float]]) -> np.ndarray:
        return softmax(np.asarray(scores, dtype=np.float64) / self.temperature_)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (the safe alternative to pickling)."""
        return {
            "kind": "temperature",
            "temperature": self.temperature_,
            "fitted": self.fitted_,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TemperatureCalibrator":
        """Load from a to_dict()-style dict; insane values fail open to T=1."""
        obj = cls()
        try:
            T = float(d.get("temperature", 1.0))
        except (TypeError, ValueError):
            T = 1.0
        obj.temperature_ = T if (np.isfinite(T) and T > 0) else 1.0
        obj.fitted_ = bool(d.get("fitted", True))
        return obj

    def save(self, path: str) -> str:
        """Write the to_dict() payload to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "TemperatureCalibrator":
        """Load a calibrator previously written by save()."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class PlattCalibrator:
    """Binary Platt scaling: p = sigmoid(a * score + b).

    Fit with logistic regression on the positive-class score.
    Requires scikit-learn.
    """

    def __init__(self) -> None:
        self.model_: LogisticRegression | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "PlattCalibrator":
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for PlattCalibrator.fit()")
        X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(labels, dtype=int)
        self.model_ = LogisticRegression(max_iter=1000)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        assert self.model_ is not None, "call fit() first"
        X = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self.model_.predict_proba(X)[:, 1]


class IsotonicCalibrator:
    """Binary isotonic regression calibration (non-parametric).

    More flexible than Platt scaling but needs more calibration data
    (a few hundred points) to avoid overfitting. Requires scikit-learn.
    """

    def __init__(self) -> None:
        self.model_: IsotonicRegression | None = None

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        if not _HAS_SKLEARN:
            raise ImportError("scikit-learn is required for IsotonicCalibrator.fit()")
        X = np.asarray(scores, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        self.model_ = IsotonicRegression(out_of_bounds="clip")
        self.model_.fit(X, y)
        return self

    def predict_proba(self, scores: Sequence[float]) -> np.ndarray:
        assert self.model_ is not None, "call fit() first"
        X = np.asarray(scores, dtype=np.float64)
        return np.clip(self.model_.predict(X), 0.0, 1.0)


@dataclass
class CalibrationExample:
    text: str
    labels: List[str]
    true_label: str


class CalibratedScorer:
    """Wraps a raw score_fn with a fitted calibrator.

    score_fn: callable (texts: List[str], labels: List[str]) -> List[List[float]]
              returning raw per-label scores for each text.
    """

    def __init__(
        self,
        score_fn: Callable[[List[str], List[str]], List[List[float]]],
        method: str = "temperature",
    ) -> None:
        if method == "temperature":
            self.calibrator: TemperatureCalibrator | PlattCalibrator | IsotonicCalibrator = TemperatureCalibrator()
        elif method == "platt":
            self.calibrator = PlattCalibrator()
        elif method == "isotonic":
            self.calibrator = IsotonicCalibrator()
        else:
            raise ValueError(f"unknown calibration method: {method!r}")
        self.score_fn = score_fn
        self.method = method

    def fit(self, examples: Sequence[CalibrationExample]) -> "CalibratedScorer":
        texts = [e.text for e in examples]
        label_lists = [e.labels for e in examples]
        # score each example against its own label set
        raw: List[List[float]] = []
        for t, labs in zip(texts, label_lists):
            raw.append(self.score_fn([t], labs)[0])
        if self.method == "temperature":
            y_idx = [labs.index(e.true_label) for e, labs in zip(examples, label_lists)]
            self.calibrator.fit(raw, y_idx)
        else:
            # binary: probability of the true label being rank-1 vs not.
            # We calibrate P(correct top-1) using the top-1 raw score.
            assert all(len(e.labels) == 2 for e in examples), \
                "platt/isotonic need binary (2-label) examples; use temperature for multi-class"
            pos_scores = [max(r) for r in raw]
            # label = 1 if the top-scoring label is the true label
            y_bin = [int(labs[int(np.argmax(r))] == e.true_label)
                     for e, labs, r in zip(examples, label_lists, raw)]
            self.calibrator.fit(pos_scores, y_bin)
        return self

    def predict_proba(self, texts: List[str], labels: List[str]) -> np.ndarray:
        raw = self.score_fn(texts, labels)
        if self.method == "temperature":
            return self.calibrator.predict_proba(raw)
        # binary calibrators return P(top-1 is correct); distribute the
        # remainder uniformly over the other labels to keep a valid simplex.
        raw = np.asarray(raw, dtype=float)
        top1 = raw.argmax(axis=1)
        p_top1 = self.calibrator.predict_proba(raw.max(axis=1))
        k = raw.shape[1]
        out = np.full_like(raw, 0.0)
        for i in range(len(raw)):
            out[i, top1[i]] = p_top1[i]
            rest = [j for j in range(k) if j != top1[i]]
            if rest:
                out[i, rest] = (1.0 - p_top1[i]) / len(rest)
        return out

# ---------------------------------------------------------------------------
# Per-answer-type temperature calibration for the decision path.
#
# Adapted from Mapika/decider (Apache-2.0): decider/calibrate.py and
# decider/temperature.py. Decider fits one temperature per answer type
# {choice, noul, score} by NLL minimization (grid search + golden-section
# refinement) on T=1 logits; answer types with fewer than MIN_ROWS_PER_TYPE
# logged rows fall back to the pooled (all-types) temperature. The fitted map
# is APPLIED per type at decision time (>=1.4.0 semantics): SystemOne._probs
# looks up the question's own temperature instead of silently serving the
# pooled one.
# ---------------------------------------------------------------------------

DECISION_TYPES = ("choice", "noul", "score")
"""Answer types that get their own temperature entry."""

MIN_ROWS_PER_TYPE = 50
"""Decider's fallback threshold: types with fewer rows use the pooled T."""

DecisionRecord = Dict[str, Any]
"""Logged decision row: {"type": "choice"|"noul"|"score", "gold": int,
"logits": [...]} or {"type": ..., "gold": int, "probs": [...]}. For "probs"
rows the values are treated as the T=1 simplex (the relative temperature is
fit on their log-probs), which matches battery/fit.py's temper() math."""


def _fit_temperature_nll(rows: Sequence[tuple[np.ndarray, int]]) -> float:
    """Fit one temperature by NLL via golden-section search on logT.

    Pure numpy (same approach as battery/fit.py) so the decision-calibration
    path needs no scipy. Rows may have different class counts — they are
    grouped by width and each group is vectorized. Searches logT in
    [-4, 4]; returns T.
    """
    groups: Dict[int, List[tuple[np.ndarray, int]]] = {}
    for P, gold in rows:
        groups.setdefault(int(len(P)), []).append((P, gold))

    def nll(logT: float) -> float:
        T = float(np.exp(logT))
        total, n = 0.0, 0
        for members in groups.values():
            Pm = np.stack([m[0] for m in members])
            ym = np.asarray([m[1] for m in members], dtype=int)
            z = np.log(np.clip(Pm, 1e-12, 1.0)) / T
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            Q = e / e.sum(axis=1, keepdims=True)
            total += float(-np.log(np.clip(Q[np.arange(len(ym)), ym], 1e-12, 1.0)).sum())
            n += len(ym)
        return total / n

    lo, hi = -4.0, 4.0
    gr = (5 ** 0.5 - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = nll(c), nll(d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = nll(d)
    return float(np.exp((lo + hi) / 2))


def _record_to_row(record: DecisionRecord) -> tuple[str, np.ndarray, int]:
    """Validate one logged decision record -> (qtype, T=1 simplex, gold)."""
    qtype = str(record.get("type", "")).lower()
    if qtype not in DECISION_TYPES:
        raise ValueError(
            f"record has unknown type {record.get('type')!r}; "
            f"expected one of {DECISION_TYPES}"
        )
    if record.get("logits") is not None:
        P = softmax(np.asarray(record["logits"], dtype=np.float64))
    elif record.get("probs") is not None:
        P = np.asarray(record["probs"], dtype=np.float64)
        s = P.sum()
        if not np.isfinite(s) or s <= 0:
            raise ValueError("record 'probs' must be a non-empty finite simplex")
        P = P / s
    else:
        raise ValueError("record needs 'logits' or 'probs'")
    gold = int(record["gold"])
    if not 0 <= gold < len(P):
        raise ValueError(f"gold index {gold} out of range for {len(P)} classes")
    return qtype, P, gold


class PerTypeTemperatureCalibrator:
    """Per-answer-type temperature map for the decision endpoints.

    Fit from logged decision records (see DecisionRecord). Types with fewer
    than `min_rows` rows (default 50, decider's threshold) get NO entry and
    fall back to the pooled temperature at serve time. Attach with
    SystemOne.set_calibrator(); SystemOne._probs applies the question's own
    temperature — the map is actually applied, not just stored.

    Serializes to a calibration.json-compatible dict via to_dict():
    {"temperature": pooled_T, "temperature_by_type": {...}, ...} which merges
    cleanly alongside the existing route "temperature" entry.
    """

    def __init__(self, min_rows: int = MIN_ROWS_PER_TYPE) -> None:
        self.min_rows = int(min_rows)
        self.temperature_: float = 1.0  # pooled fallback
        self.temperature_by_type_: Dict[str, float] = {}
        self.rows_by_type_: Dict[str, int] = {}
        self.fitted_: bool = False

    def fit(
        self, records: Sequence[DecisionRecord]
    ) -> "PerTypeTemperatureCalibrator":
        records = list(records)
        if not records:
            raise ValueError("need at least one decision record")
        pooled: List[tuple[np.ndarray, int]] = []
        per_type: Dict[str, List[tuple[np.ndarray, int]]] = {
            t: [] for t in DECISION_TYPES
        }
        for record in records:
            qtype, P, gold = _record_to_row(record)
            per_type[qtype].append((P, gold))
            pooled.append((P, gold))
        self.temperature_ = _fit_temperature_nll(pooled)
        self.temperature_by_type_ = {}
        self.rows_by_type_ = {}
        for qtype, rows in per_type.items():
            self.rows_by_type_[qtype] = len(rows)
            if len(rows) >= self.min_rows:
                self.temperature_by_type_[qtype] = _fit_temperature_nll(rows)
            # else: no entry -> temperature_for() falls back to pooled
        self.fitted_ = True
        return self

    def temperature_for(self, qtype: str) -> float:
        """Temperature for an answer type; pooled fallback when unfitted."""
        if not self.fitted_:
            raise AssertionError("call fit() first")
        return float(
            self.temperature_by_type_.get(str(qtype).lower(), self.temperature_)
        )

    def predict_proba(
        self, scores: Sequence[float], qtype: str = "choice"
    ) -> np.ndarray:
        """softmax(scores / T_qtype) for one question's raw scores."""
        T = self.temperature_for(qtype)
        return softmax(np.asarray(scores, dtype=np.float64) / T)

    def to_dict(self) -> Dict[str, Any]:
        """calibration.json-compatible dict (merges with existing keys)."""
        if not self.fitted_:
            raise AssertionError("call fit() first")
        return {
            "temperature": self.temperature_,
            "temperature_by_type": dict(self.temperature_by_type_),
            "per_type_rows": dict(self.rows_by_type_),
            "min_rows_per_type": self.min_rows,
            "fit_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "note": (
                "per-answer-type temperature map (decider convention); types "
                f"with fewer than {self.min_rows} rows fall back to the pooled "
                "'temperature'"
            ),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerTypeTemperatureCalibrator":
        """Load from a to_dict()/calibration.json-style dict. Fail-open on the
        per-type entries: insane values are dropped so temperature_for()
        falls back to the pooled temperature instead of serving garbage."""
        obj = cls(min_rows=int(d.get("min_rows_per_type", MIN_ROWS_PER_TYPE)))
        pooled = float(d.get("temperature", 1.0))
        obj.temperature_ = pooled if (np.isfinite(pooled) and pooled > 0) else 1.0
        obj.temperature_by_type_ = {}
        for qtype, T in (d.get("temperature_by_type") or {}).items():
            try:
                Tf = float(T)
            except (TypeError, ValueError):
                continue
            if str(qtype).lower() in DECISION_TYPES and np.isfinite(Tf) and Tf > 0:
                obj.temperature_by_type_[str(qtype).lower()] = Tf
        rows = d.get("per_type_rows") or {}
        obj.rows_by_type_ = {
            str(k).lower(): int(v) for k, v in rows.items() if str(k).lower() in DECISION_TYPES
        }
        obj.fitted_ = True
        return obj

    def save(self, path: str) -> str:
        """Write the to_dict() payload to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "PerTypeTemperatureCalibrator":
        """Load a map previously written by save()."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def fit_temperature_by_type(
    records: Sequence[DecisionRecord], min_rows: int = MIN_ROWS_PER_TYPE
) -> PerTypeTemperatureCalibrator:
    """Fit a per-answer-type temperature map from logged decision records."""
    return PerTypeTemperatureCalibrator(min_rows=min_rows).fit(records)


def load_calibrator_file(path: str) -> Any:
    """Load a calibrator from a JSON file, any supported kind.

    Per-type maps (dicts carrying "temperature_by_type") load as
    PerTypeTemperatureCalibrator; anything else dict-shaped loads as
    TemperatureCalibrator (fail-open to T=1 on insane values). JSON-only:
    anything else raises instead of unpickling — pickle executes code at
    load time, and the calibrator path is reachable from an env var, so a
    fail-closed error beats a code-execution fallback. Re-save legacy
    pickles with calibrator.save(path).
    """
    with open(path, "rb") as f:
        raw = f.read()
    try:
        d = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        d = None
    if isinstance(d, dict):
        if "temperature_by_type" in d:
            return PerTypeTemperatureCalibrator.from_dict(d)
        return TemperatureCalibrator.from_dict(d)
    raise ValueError(
        f"calibrator at {path} is not JSON; re-save as JSON "
        "(calibrator.save(path)) — pickle files are refused"
    )


def load_type_calibration(path: str) -> PerTypeTemperatureCalibrator | None:
    """Load a per-type map from a calibration.json-style file.

    Returns None when the file is missing/unreadable or carries no
    "temperature_by_type" block (pooled-only files keep working — the caller
    serves the pooled temperature as before).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict) or "temperature_by_type" not in d:
            return None
        return PerTypeTemperatureCalibrator.from_dict(d)
    except (OSError, ValueError, TypeError):
        return None


# -- split-conformal prediction sets (MAPIE/APS style, numpy only) ---------
#
# A point prediction plus a confidence number still leaves the operator
# guessing how many options to take seriously. Conformal prediction
# converts the distribution into a SET of plausible options with a
# finite-sample marginal coverage guarantee — P(gold in set) >= 1 - α
# on exchangeable future items — with no distributional assumptions.
# Uses the Adaptive Prediction Sets (APS) nonconformity score: the
# cumulative ranked mass up to and including the gold label.


def fit_conformal_threshold(
    records: Sequence[DecisionRecord],
    alpha: float = 0.1,
    qtype: str = "choice",
) -> Dict[str, Any]:
    """Fit a split-conformal APS threshold from labeled decision rows.

    Args:
        records: DecisionRecord rows ({type, gold, logits|probs}).
        alpha: miscoverage rate; coverage target is 1 - alpha.
        qtype: only rows of this answer type calibrate the threshold.

    Returns {"tau", "alpha", "qtype", "n"}: sets built with
    conformal_set(distribution, tau) cover gold with marginal
    probability >= 1 - alpha (quantile with the (n+1) finite-sample
    correction). Raises ValueError when no rows of qtype exist or
    alpha is outside (0, 1).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    typed = [(P, gold) for t, P, gold in
             (_record_to_row(r) for r in records) if t == qtype]
    if not typed:
        raise ValueError(f"no {qtype!r} rows to fit a conformal threshold")
    scores = []
    for P, gold in typed:
        order = np.argsort(-P, kind="stable")
        rank = int(np.nonzero(order == gold)[0][0])
        scores.append(float(P[order[: rank + 1]].sum()))
    n = len(scores)
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return {"tau": float(np.quantile(scores, level, method="higher")),
            "alpha": float(alpha), "qtype": qtype, "n": n}


def conformal_set(
    distribution: Dict[str, float], tau: float
) -> List[str]:
    """Smallest top-probability option set with cumulative mass >= tau.

    Options come out in descending-probability order; the top option is
    always included even when tau <= 0. Raises ValueError on an empty
    distribution.
    """
    if not distribution:
        raise ValueError("need a non-empty distribution")
    ranked = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))
    out: List[str] = []
    mass = 0.0
    for name, p in ranked:
        out.append(name)
        mass += max(0.0, float(p))
        if mass >= tau:
            break
    return out


# -- RouteLLM-style cost/quality threshold selection -----------------------
#
# RouteLLM routes weak iff P(strong wins | q) < α and picks α to hold a
# quality target (e.g. 95% of the strong model). These helpers run that
# analysis offline on labeled cascade rows so the cascade ships with a
# defensible α instead of a guessed one.


def route_threshold_for_target(
    rows: Sequence[tuple],
    target: float = 0.95,
    weak_cost: float = 0.0,
    strong_cost: float = 1.0,
    grid: int = 101,
) -> Dict[str, float]:
    """Pick the cheapest α holding target × strong-model quality.

    Args:
        rows: (p_strong_wins, weak_ok, strong_ok) per item.
        target: keep routed quality >= target × strong quality.
        weak_cost / strong_cost: per-call cost units for cost_share.
        grid: α candidates swept over [0, 1].

    Returns {"alpha", "routed_quality", "strong_quality",
    "target_quality", "weak_share", "cost_share"}: the largest α (most
    weak traffic) meeting the target. α = 0 routes everything strong.
    Raises ValueError on empty rows or a target outside (0, 1].
    """
    if not 0.0 < target <= 1.0:
        raise ValueError(f"target must be in (0, 1], got {target!r}")
    data = [(float(p), bool(w), bool(s)) for p, w, s in rows]
    if not data:
        raise ValueError("need at least one cascade row")
    strong_q = sum(1.0 for _, _, s in data if s) / len(data)
    need = target * strong_q
    best = {"alpha": 0.0, "routed_quality": strong_q,
            "weak_share": 0.0}
    for i in range(max(2, grid)):
        alpha = i / (max(2, grid) - 1)
        ok = sum(1.0 for p, w, s in data if (w if p < alpha else s))
        quality = ok / len(data)
        if quality >= need:
            weak_share = sum(1.0 for p, _, _ in data if p < alpha) / len(data)
            best = {"alpha": alpha, "routed_quality": quality,
                    "weak_share": weak_share}
    denom = strong_cost if strong_cost else 1.0
    cost_share = (best["weak_share"] * weak_cost
                  + (1.0 - best["weak_share"]) * strong_cost) / denom
    return {"alpha": float(best["alpha"]),
            "routed_quality": float(best["routed_quality"]),
            "strong_quality": float(strong_q),
            "target_quality": float(need),
            "weak_share": float(best["weak_share"]),
            "cost_share": float(cost_share)}


def quality_cost_frontier(
    rows: Sequence[tuple],
    weak_cost: float = 0.0,
    strong_cost: float = 1.0,
    points: int = 21,
) -> List[Dict[str, float]]:
    """Routed (weak_share, quality, cost_share) points over the α grid.

    Same row shape as route_threshold_for_target; for plotting the
    cost/quality tradeoff before fixing α.
    """
    data = [(float(p), bool(w), bool(s)) for p, w, s in rows]
    if not data:
        raise ValueError("need at least one cascade row")
    denom = strong_cost if strong_cost else 1.0
    out = []
    for i in range(max(2, points)):
        alpha = i / (max(2, points) - 1)
        ok = sum(1.0 for p, w, s in data if (w if p < alpha else s))
        weak_share = sum(1.0 for p, _, _ in data if p < alpha) / len(data)
        out.append({
            "alpha": float(alpha),
            "weak_share": float(weak_share),
            "quality": float(ok / len(data)),
            "cost_share": float((weak_share * weak_cost
                                 + (1.0 - weak_share) * strong_cost) / denom),
        })
    return out
