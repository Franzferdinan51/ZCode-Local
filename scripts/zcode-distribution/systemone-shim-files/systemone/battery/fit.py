#!/usr/bin/env python3
"""Fit the route-calibration temperature from the battery.

Runs every task in battery/tasks.jsonl through the shim's /route endpoint,
collects the blended per-tier `probabilities` plus the true tier label, then
fits a single temperature T by NLL minimization with 5-fold cross-validation.

The shim's blended distribution is already a simplex, so temperature scaling
is applied as softmax(log p / T) == p**(1/T) renormalized (T > 1 softens
overconfident distributions, T < 1 sharpens).

Writes systemone/calibration.json:
  {temperature, fit_date, battery_sha256, n_tasks, tiers,
   ece_before, ece_after, cv_mean_temperature, cv_folds}

Requires numpy (scipy optional — a pure-numpy golden-section search is used).

Usage:
  python3 fit.py [--base-url http://127.0.0.1:18765] [--out ../calibration.json]
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(HERE, "..", ".."))  # runnable as a script, too
from systemone.patterns import require_http_url  # noqa: E402

try:
    import numpy as np
except ImportError:
    sys.exit("fit.py requires numpy")


def post_probs(base_url: str, task: str) -> dict:
    body = json.dumps({"task": task}).encode("utf-8")
    req = urllib.request.Request(
        require_http_url(base_url.rstrip("/"), what="shim URL") + "/v1/systemone/route",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 -- scheme enforced above via patterns.require_http_url; nosemgrep
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["route"]["probabilities"]


def temper(p: np.ndarray, T: float) -> np.ndarray:
    """Softmax(log p / T): temperature-scale an already-normalized simplex."""
    logp = np.log(np.clip(p, 1e-12, 1.0))
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(P: np.ndarray, y: np.ndarray, logT: float) -> float:
    Q = temper(P, float(np.exp(logT)))
    return float(-np.log(np.clip(Q[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(P: np.ndarray, y: np.ndarray) -> float:
    """Golden-section search on logT in [-4, 4]; returns T."""
    lo, hi = -4.0, 4.0
    gr = (5 ** 0.5 - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = nll(P, y, c), nll(P, y, d)
    for _ in range(60):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = nll(P, y, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = nll(P, y, d)
    return float(np.exp((lo + hi) / 2))


def ece_10bin(y_true: np.ndarray, y_conf: np.ndarray) -> float:
    ece = 0.0
    n = len(y_true)
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        mask = (y_conf > lo) & (y_conf <= hi)
        if b == 0:
            mask = mask | (y_conf == 0.0)
        m = mask.sum()
        if m == 0:
            continue
        ece += (m / n) * abs(y_true[mask].mean() - y_conf[mask].mean())
    return float(ece)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit SystemOne route temperature")
    ap.add_argument("--base-url", default="http://127.0.0.1:18765")
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "calibration.json"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    with open(args.tasks, "rb") as f:
        raw = f.read()
    battery_sha = hashlib.sha256(raw).hexdigest()
    tasks = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    tiers = sorted({t["tier"] for t in tasks})
    t2i = {t: i for i, t in enumerate(tiers)}
    print(f"fit: {len(tasks)} tasks, tiers={tiers}, vs {args.base_url}")

    P_rows, y_rows = [], []
    for i, t in enumerate(tasks):
        probs = post_probs(args.base_url, t["task"])
        P_rows.append([probs.get(tier, 0.0) for tier in tiers])
        y_rows.append(t2i[t["tier"]])
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(tasks)}")
    P = np.asarray(P_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=int)

    # 5-fold CV: fit T on k-1 folds, score held-out NLL
    rng = np.random.default_rng(20260924)
    idx = rng.permutation(len(y))
    fold_size = len(y) // args.folds
    Ts, held_nll = [], []
    for k in range(args.folds):
        te = idx[k * fold_size:(k + 1) * fold_size]
        tr = np.setdiff1d(idx, te)
        T = fit_temperature(P[tr], y[tr])
        Ts.append(T)
        held_nll.append(nll(P[te], y[te], float(np.log(T))))
    T_mean = float(np.mean(Ts))
    print(f"CV temperatures: {[round(t, 3) for t in Ts]} -> mean T={T_mean:.4f}")
    print(f"held-out NLL: raw={nll(P, y, 0.0):.4f} "
          f"calibrated={np.mean(held_nll):.4f}")

    # ECE before/after on the full set
    pred = P.argmax(axis=1)
    correct = (pred == y).astype(float)
    ece_before = ece_10bin(correct, P.max(axis=1))
    Q = temper(P, T_mean)
    ece_after = ece_10bin(correct, Q.max(axis=1))
    print(f"ECE before={ece_before:.4f} after={ece_after:.4f}")

    cal = {
        "temperature": T_mean,
        "fit_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "battery_sha256": battery_sha,
        "n_tasks": len(tasks),
        "tiers": tiers,
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "cv_folds": args.folds,
        "cv_temperatures": [round(t, 4) for t in Ts],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
