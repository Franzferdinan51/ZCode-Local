"""Calibration & selective-prediction metrics for SystemOne.

Adapted from Mapika/decider (Apache-2.0) — see decider/metrics.py in the
decider repository. The metric definitions (15-bin top-label ECE, Brier,
NLL, AURC, selective accuracy, summarize tables) follow decider's; the
implementation here is original and numpy-based to match this repo's style.

All functions take:
    y_true: sequence of int gold labels (0..K-1)
    proba:  sequence of per-class probability vectors (rows sum to ~1)

and return plain floats / dicts of floats.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "top_label_ece",
    "brier_score",
    "nll_score",
    "aurc",
    "selective_accuracy",
    "coverage_at_error",
    "risk_coverage_curve",
    "select_threshold",
    "evaluate_threshold",
    "summarize",
    "format_table",
]


def _arrays(
    y_true: Sequence[int], proba: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    P = np.asarray(proba, dtype=np.float64)
    if y.ndim != 1 or P.ndim != 2 or len(y) != len(P):
        raise ValueError("y_true must be 1-D and proba 2-D with matching rows")
    if len(y) == 0:
        raise ValueError("need at least one example")
    return y, P


def top_label_ece(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error on the predicted (top-label) class.

    Bins predictions by their max-class confidence; ECE is the
    sample-weighted mean |accuracy - confidence| over bins. Decider's
    convention uses 15 bins (this repo's calibration.expected_calibration_error
    defaults to 10 and is binary-confidence oriented).
    """
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ece += (n_b / len(y)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(
    y_true: Sequence[int], proba: Sequence[Sequence[float]]
) -> float:
    """Multiclass Brier score: mean over rows of sum_i (p_i - 1[i == y])^2.

    0 is perfect; lower is better. Proper scoring rule.
    """
    y, P = _arrays(y_true, proba)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def nll_score(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    eps: float = 1e-12,
) -> float:
    """Mean negative log-likelihood of the gold class. Lower is better."""
    y, P = _arrays(y_true, proba)
    p_true = np.clip(P[np.arange(len(y)), y], eps, 1.0)
    return float(-np.log(p_true).mean())


def aurc(y_true: Sequence[int], proba: Sequence[Sequence[float]]) -> float:
    """Area under the risk-coverage curve (selective prediction quality).

    Sort by confidence (max class probability) descending; sweep k = 1..n
    with coverage = k/n and risk = error rate on the top-k. AURC is the
    trapezoid integral of risk over coverage: 0 for a perfect ranker,
    ~0.5 for a useless one. Lower is better.
    """
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    order = np.argsort(-conf, kind="stable")
    errors = (pred[order] != y[order]).astype(float)
    n = len(y)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    # trapezoid over (0, risk_at_first_bin) -> coverage points
    xs = np.concatenate([[0.0], coverage])
    ys = np.concatenate([[risk[0]], risk])
    return float(np.trapezoid(ys, xs))


def selective_accuracy(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    coverage: float = 0.8,
) -> float:
    """Accuracy on the most-confident `coverage` fraction of predictions.

    coverage=1.0 is plain accuracy. Higher is better.
    """
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    k = max(1, int(round(coverage * len(y))))
    order = np.argsort(-conf, kind="stable")[:k]
    return float((pred[order] == y[order]).mean())


def _risk_curve_arrays(
    y_true: Sequence[int], proba: Sequence[Sequence[float]]
) -> tuple:
    """Whole-confidence-group risk curve, ported from Kev (kev/metrics.py).

    Sort by confidence descending; rows sharing a confidence threshold
    move as one group (ties are never split). Returns (thresholds,
    accepted_counts, error_counts) with accepted[-1] == n.
    """
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    order = np.argsort(-conf, kind="stable")
    conf = conf[order]
    errors = np.cumsum((P[order].argmax(axis=1) != y[order]).astype(int))
    ends = np.r_[np.flatnonzero(conf[1:] != conf[:-1]), len(order) - 1]
    return conf[ends], ends + 1, errors[ends]


def coverage_at_error(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    budget: float,
) -> float:
    """Largest acceptable decision share at a fixed error budget.

    In descending confidence order, the biggest prefix whose empirical
    error stays <= budget (Kev's metric-policy: an in-sample maximum
    over confidence thresholds, not a deployed error guarantee).
    Honest probabilities get high coverage; confidently-wrong gets
    little, whatever the accuracy.
    """
    if not math.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("error budget must be in [0, 1]")
    _, accepted, errors = _risk_curve_arrays(y_true, proba)
    ok = np.flatnonzero(errors <= budget * accepted)
    return float(accepted[ok[-1]] / accepted[-1]) if len(ok) else 0.0


def risk_coverage_curve(
    y_true: Sequence[int], proba: Sequence[Sequence[float]]
) -> List[Dict[str, float]]:
    """Risk/coverage at every whole-confidence-group threshold."""
    thresholds, accepted, errors = _risk_curve_arrays(y_true, proba)
    total = int(accepted[-1])
    return [
        {"threshold": float(t), "accepted": int(n), "errors": int(e),
         "coverage": float(n / total), "risk": float(e / n)}
        for t, n, e in zip(thresholds, accepted, errors)
    ]


def select_threshold(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    budget: float,
    min_accepted: int = 1,
) -> float | None:
    """Highest-coverage confidence threshold within an error budget.

    Returns None when no threshold accepts >= min_accepted decisions
    within budget (the caller abstains on everything).
    """
    if not math.isfinite(budget) or not 0 <= budget <= 1 \
            or min_accepted < 1:
        raise ValueError("invalid error budget or minimum accepted count")
    thresholds, accepted, errors = _risk_curve_arrays(y_true, proba)
    ok = np.flatnonzero(
        (errors <= budget * accepted) & (accepted >= min_accepted))
    return float(thresholds[ok[-1]]) if len(ok) else None


def evaluate_threshold(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    threshold: float | None,
) -> Dict[str, float | None]:
    """Coverage/risk of accepting decisions at confidence >= threshold.

    threshold=None means abstain-all.
    """
    if threshold is not None and (
            not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("threshold must be in [0, 1] or None for abstain-all")
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    accepted = conf >= threshold if threshold is not None else \
        np.zeros(len(y), dtype=bool)
    n = int(accepted.sum())
    errors = int(((P.argmax(axis=1) != y) & accepted).sum())
    return {"threshold": threshold, "n": float(len(y)),
            "accepted": float(n), "errors": float(errors),
            "coverage": n / len(y), "risk": errors / n if n else None}


def summarize(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    name: str = "",
) -> Dict[str, float]:
    """One-row metric summary (decider's `summarize` convention).

    Returns {"name", "n", "accuracy", "ece_15", "brier", "nll", "aurc",
    "sel_acc@50", "sel_acc@80", "sel_acc@100", "cov@5%", "cov@1%"} —
    higher is better for accuracy / sel_acc / cov, lower is better for
    ece_15 / brier / nll / aurc.
    """
    y, P = _arrays(y_true, proba)
    pred = P.argmax(axis=1)
    return {
        "name": name,
        "n": float(len(y)),
        "accuracy": float((pred == y).mean()),
        "ece_15": top_label_ece(y, P),
        "brier": brier_score(y, P),
        "nll": nll_score(y, P),
        "aurc": aurc(y, P),
        "sel_acc@50": selective_accuracy(y, P, 0.50),
        "sel_acc@80": selective_accuracy(y, P, 0.80),
        "sel_acc@100": selective_accuracy(y, P, 1.00),
        "cov@5%": coverage_at_error(y, P, 0.05),
        "cov@1%": coverage_at_error(y, P, 0.01),
    }


_SUMMARY_COLS = [
    ("name", "name", "{}"),
    ("n", "n", "{:.0f}"),
    ("accuracy", "acc", "{:.3f}"),
    ("ece_15", "ece15", "{:.4f}"),
    ("brier", "brier", "{:.4f}"),
    ("nll", "nll", "{:.4f}"),
    ("aurc", "aurc", "{:.4f}"),
    ("sel_acc@50", "s@50", "{:.3f}"),
    ("sel_acc@80", "s@80", "{:.3f}"),
    ("sel_acc@100", "s@100", "{:.3f}"),
    ("cov@5%", "cov5%", "{:.3f}"),
    ("cov@1%", "cov1%", "{:.3f}"),
]


def format_table(rows: Sequence[Dict[str, float]]) -> str:
    """Render summarize() dicts (e.g. per tier) as an aligned ASCII table."""
    rows = list(rows)
    if not rows:
        return "(no rows)"
    headers = [label for _, label, _ in _SUMMARY_COLS]
    cells = [[fmt.format(r.get(key, float("nan"))) for key, _, fmt in _SUMMARY_COLS]
             for r in rows]
    # widen the name column to fit
    widths = [max(len(h), *(len(row[i]) for row in cells)) for i, h in enumerate(headers)]
    lines = [
        "  ".join(h.ljust(w) for h, w in zip(headers, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    return "\n".join(lines)
