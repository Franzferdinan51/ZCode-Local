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
    # Eval-honesty batch (pulled from the eval/monitoring ecosystem):
    "ndcg_at_k",
    "reciprocal_rank",
    "macro_f1",
    "failure_auroc",
    "brier_decomposition",
    "reliability_curve",
    "paired_bootstrap_ci",
    "mcnemar",
    "psi",
    "estimated_accuracy",
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


# -- rank + classification metrics (rank-plans / rerank / judge evals) ------


def ndcg_at_k(ranked_relevance: Sequence[float], k: int = 10) -> float:
    """nDCG@k for one ranked list (ToolRet/Clef-table convention).

    Args:
        ranked_relevance: relevance grades in the system's ranked order
            (higher = more relevant; binary 0/1 also fine).
        k: cutoff; grades past k are ignored.

    Returns DCG@k / IDCG@k in [0, 1]; 1.0 when nothing is relevant
    (no way to rank wrong) and 0.0 when k <= 0.
    """
    rel = [max(0.0, float(r)) for r in list(ranked_relevance)[: max(0, k)]]
    if not rel or k <= 0:
        return 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(rel))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(rel, reverse=True)))
    return dcg / ideal if ideal > 0 else 1.0


def reciprocal_rank(ranked_relevance: Sequence[float]) -> float:
    """1/rank of the first relevant item (rel > 0); 0.0 when none ranks."""
    for i, r in enumerate(ranked_relevance):
        if float(r) > 0:
            return 1.0 / (i + 1)
    return 0.0


def macro_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int] | None = None,
) -> float:
    """Macro-averaged F1 over labels (BANKING77/Clef-table convention).

    Labels with no true and no predicted members score 1.0 (nothing to
    get wrong); labels with members on one side only score 0.0.
    """
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    if yt.shape != yp.shape or yt.ndim != 1 or len(yt) == 0:
        raise ValueError("y_true and y_pred must be non-empty equal 1-D")
    if labels is None:
        labels = sorted(set(yt.tolist()) | set(yp.tolist()))
    f1s = []
    for lab in labels:
        tp = int(((yt == lab) & (yp == lab)).sum())
        fp = int(((yt != lab) & (yp == lab)).sum())
        fn = int(((yt == lab) & (yp != lab)).sum())
        if tp + fp + fn == 0:
            f1s.append(1.0)
        else:
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(sum(f1s) / len(f1s)) if f1s else 0.0


def failure_auroc(y_true: Sequence[int], proba: Sequence[Sequence[float]]) -> float:
    """AUROC of failure prediction from confidence (max class prob).

    Treats each item as correct/incorrect and scores the ranking by
    confidence: 1.0 means every correct item outranks every error (the
    uncertainty gate can separate them), 0.5 is chance. Returns 0.5
    when one side is empty (nothing to rank). Computed by the
    Mann-Whitney U statistic — no sklearn needed.
    """
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    ok = conf[(P.argmax(axis=1) == y)]
    bad = conf[(P.argmax(axis=1) != y)]
    if len(ok) == 0 or len(bad) == 0:
        return 0.5
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in ok for b in bad)
    return float(wins / (len(ok) * len(bad)))


def brier_decomposition(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    bins: int = 15,
) -> Dict[str, float]:
    """Murphy decomposition of the multiclass Brier score.

    Buckets predicted probabilities per class (one-vs-rest) and returns
    {"reliability", "resolution", "uncertainty", "brier"} with
    brier = reliability - resolution + uncertainty (up to binning).
    Reliability near 0 = calibrated; resolution near uncertainty =
    sharp (confident and right). Raises ValueError on empty input.
    """
    y, P = _arrays(y_true, proba)
    n, k = P.shape
    edges = np.linspace(0.0, 1.0, bins + 1)
    reliability = 0.0
    resolution = 0.0
    for c in range(P.shape[1]):
        o = (y == c).astype(float)
        rel_c = 0.0
        res_c = 0.0
        idx = np.clip(np.digitize(P[:, c], edges[1:-1]), 0, bins - 1)
        for b in range(bins):
            mask = idx == b
            nb = int(mask.sum())
            if nb == 0:
                continue
            fbar = float(P[mask, c].mean())
            obar = float(o[mask].mean())
            rel_c += (nb / n) * (fbar - obar) ** 2
            res_c += (nb / n) * (obar - o.mean()) ** 2
        reliability += rel_c
        resolution += res_c
    reliability /= k
    resolution /= k
    uncertainty = float(sum((y == c).mean() * (1.0 - (y == c).mean())
                           for c in range(k)) / k)
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": float(reliability - resolution + uncertainty),
    }


def reliability_curve(
    y_true: Sequence[int],
    proba: Sequence[Sequence[float]],
    bins: int = 15,
) -> List[Dict[str, float]]:
    """Top-label reliability bins for plotting: count, mean confidence,
    accuracy per bin (empty bins carry count 0 and NaN-free zeros)."""
    y, P = _arrays(y_true, proba)
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        nb = int(mask.sum())
        out.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "count": float(nb),
            "mean_confidence": float(conf[mask].mean()) if nb else 0.0,
            "accuracy": float(correct[mask].mean()) if nb else 0.0,
        })
    return out


# -- honest A/B: is engine B really better? -------------------------------


def paired_bootstrap_ci(
    a_correct: Sequence[bool | int | float],
    b_correct: Sequence[bool | int | float],
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap CI on the paired accuracy delta mean(B) - mean(A).

    Resamples items with replacement (paired: same items for both
    systems) and returns {"delta", "lo", "hi"} at `level` coverage.
    A CI excluding 0 is a significant win for its side. Raises
    ValueError on empty or mismatched inputs.
    """
    a = np.asarray([float(x) for x in a_correct], dtype=np.float64)
    b = np.asarray([float(x) for x in b_correct], dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("need non-empty equal-length correctness lists")
    if n_boot < 100:
        raise ValueError("n_boot must be >= 100")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    deltas = (b[idx] - a[idx]).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return {
        "delta": float(b.mean() - a.mean()),
        "lo": float(np.quantile(deltas, tail)),
        "hi": float(np.quantile(deltas, 1.0 - tail)),
    }


def mcnemar(
    a_correct: Sequence[bool | int | float],
    b_correct: Sequence[bool | int | float],
) -> Dict[str, float]:
    """McNemar's test on paired correct/incorrect outcomes.

    Returns {"statistic", "p_value", "b", "c"} where b = A-wrong/B-right
    and c = A-right/B-wrong. Exact binomial two-sided p-value when
    b + c < 25, else chi-square(1) with Edwards' continuity correction
    (survival via erfc — no scipy). p = 1.0 when b + c == 0.
    """
    a = [bool(x) for x in a_correct]
    b = [bool(x) for x in b_correct]
    if len(a) != len(b) or not a:
        raise ValueError("need non-empty equal-length correctness lists")
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": 0.0, "c": 0.0}
    if n < 25:
        k = max(n01, n10)
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
        return {"statistic": float((n01 - n10) ** 2 / n),
                "p_value": float(min(1.0, 2 * tail)),
                "b": float(n01), "c": float(n10)}
    stat = (abs(n01 - n10) - 1.0) ** 2 / n
    return {"statistic": float(stat),
            "p_value": float(math.erfc(math.sqrt(stat / 2.0))),
            "b": float(n01), "c": float(n10)}


# -- production monitoring without labels (NannyML/Evidently style) -------


def psi(
    expected: Sequence[float],
    actual: Sequence[float],
    bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Population Stability Index between reference and live samples.

    Quantile bins from `expected` (deduplicated); PSI = Σ (a-e)·ln(a/e).
    Evidently's rule of thumb: < 0.1 no shift, 0.1–0.2 moderate,
    > 0.2 significant. Use on confidences or predicted-label rates to
    catch drift before labels arrive. Raises ValueError on empties.
    """
    exp = np.asarray([float(x) for x in expected], dtype=np.float64)
    act = np.asarray([float(x) for x in actual], dtype=np.float64)
    if len(exp) == 0 or len(act) == 0:
        raise ValueError("need non-empty reference and live samples")
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    idx_e = np.clip(np.digitize(exp, edges[1:-1]), 0, len(edges) - 2)
    idx_a = np.clip(np.digitize(act, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        e = max(eps, float((idx_e == b).mean()))
        aval = max(eps, float((idx_a == b).mean()))
        total += (aval - e) * math.log(aval / e)
    return float(total)


def estimated_accuracy(proba: Sequence[Sequence[float]]) -> float:
    """CBPE-style accuracy estimate: mean max class probability.

    NannyML's confidence-based performance estimation for accuracy —
    valid when probabilities are calibrated, structurally blind to
    concept drift (confident-and-wrong looks good). Report alongside
    PSI drift, never alone.
    """
    P = np.asarray([list(map(float, row)) for row in proba], dtype=np.float64)
    if P.ndim != 2 or len(P) == 0:
        raise ValueError("need a non-empty 2-D probability array")
    return float(P.max(axis=1).mean())


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
