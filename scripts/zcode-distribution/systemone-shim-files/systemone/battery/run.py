#!/usr/bin/env python3
"""SystemOne calibration battery / regression runner.

For each task in tasks.jsonl: POST /v1/systemone/route, record the predicted
tier/effort/labels, calibrated probabilities, margin and latency.

Asserts (full mode, against a scoring-capable shim):
  - tier accuracy >= 0.80 (warn < 0.85)
  - ECE (10-bin, on calibrated top-1 confidence) <= 0.10
  - p50 route latency <= 500ms, p95 <= 2000ms
  - response schema contains all required keys (schema regression)
  - no task regresses from its last recorded tier (battery/last_run.json)

Live mode (--mode live) runs a read-only subset against older shims that
lack the scoring fields: tier accuracy + latency + base schema only.

Stdlib only — runs anywhere, including over ssh on the Windows PC.

Usage:
  python3 run.py [--base-url http://127.0.0.1:8765] [--mode full|live]
                 [--tasks tasks.jsonl] [--last-run last_run.json]
                 [--out results.json] [--no-update-last-run]
Exit code 0 = all asserts pass, 1 = failure.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(HERE, "..", ".."))  # runnable as a script, too
from systemone.patterns import require_http_url  # noqa: E402

BASE_REQUIRED = [
    "tier", "model_id", "confidence", "probabilities", "effort", "task_labels",
]
SCORING_REQUIRED = BASE_REQUIRED + [
    "calibrated", "calibrated_probabilities", "margin", "uncertain",
]

TIER_ACCURACY_FLOOR = 0.80
TIER_ACCURACY_WARN = 0.85
ECE_CEILING = 0.10
MIN_ECE_N = 30
P50_CEILING_MS = 500.0
P95_CEILING_MS = 2000.0
REQUEST_TIMEOUT_S = 30


def post_route(base_url: str, task: str) -> tuple[dict, float]:
    body = json.dumps({"task": task}).encode("utf-8")
    req = urllib.request.Request(
        require_http_url(base_url.rstrip("/"), what="shim URL") + "/v1/systemone/route",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:  # nosec B310 -- scheme enforced above via patterns.require_http_url; nosemgrep
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")
    client_ms = (time.perf_counter() - t0) * 1000.0
    return payload, client_ms


def ece_10bin(y_true: list[int], y_conf: list[float]) -> float:
    """Expected calibration error, 10 equal-width bins, no numpy."""
    n = len(y_true)
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        idx = [
            i for i in range(n)
            if (y_conf[i] > lo and y_conf[i] <= hi) or (b == 0 and y_conf[i] == 0.0)
        ]
        if not idx:
            continue
        acc = sum(y_true[i] for i in idx) / len(idx)
        conf = sum(y_conf[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - conf)
    return ece


def per_tier_stats(results: list[dict]) -> dict:
    """Per-tier {n, accuracy, ece} over battery results.

    accuracy: fraction of rows with this true tier predicted correctly.
    ece: binary ECE of correctness vs the reported confidence (calibrated
    top-1 when the shim provides it), reusing ece_10bin. Stdlib only.
    """
    by_tier: dict[str, dict] = {}
    for r in results:
        tier = r.get("true_tier")
        d = by_tier.setdefault(tier, {"n": 0, "correct": 0, "y": [], "c": []})
        d["n"] += 1
        ok = r.get("pred_tier") == tier
        d["correct"] += 1 if ok else 0
        d["y"].append(1 if ok else 0)
        conf = r.get("confidence")
        d["c"].append(conf if isinstance(conf, (int, float)) else 0.0)
    out = {}
    for tier in sorted(by_tier):
        d = by_tier[tier]
        out[tier] = {
            "n": d["n"],
            "accuracy": d["correct"] / d["n"] if d["n"] else 0.0,
            "ece": ece_10bin(d["y"], d["c"]),
        }
    return out


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = math.ceil(p / 100.0 * len(s)) - 1
    return s[max(0, min(k, len(s) - 1))]


def main() -> int:
    ap = argparse.ArgumentParser(description="SystemOne battery / regression runner")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--mode", choices=("full", "live"), default="full",
                    help="live: subset asserts for older shims (no scoring fields)")
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.jsonl"))
    ap.add_argument("--last-run", default=os.path.join(HERE, "last_run.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-update-last-run", action="store_true")
    args = ap.parse_args()

    tasks = []
    with open(args.tasks, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"tasks.jsonl line {ln}: invalid JSON: {e}")
                    return 1
    print(f"battery: {len(tasks)} tasks vs {args.base_url} (mode={args.mode})")

    last = {}
    if os.path.exists(args.last_run):
        with open(args.last_run, encoding="utf-8") as f:
            last = json.load(f).get("tasks", {})

    results = []
    errors = 0
    schema_missing: dict[str, int] = {}
    expect_scoring: bool | None = None  # decided from first response in full mode
    for i, t in enumerate(tasks):
        try:
            payload, client_ms = post_route(args.base_url, t["task"])
        except Exception as e:  # noqa: BLE001 - harness must report, not crash
            errors += 1
            print(f"  [{i}] ERROR {t['task'][:50]!r}: {e}")
            continue
        route = payload.get("route") or {}
        if expect_scoring is None:
            expect_scoring = "calibrated" in route
            if args.mode == "full" and not expect_scoring:
                print("  WARNING: shim has no scoring fields; "
                      "schema/ECE asserts will be skipped")
        required = SCORING_REQUIRED if (args.mode == "full" and expect_scoring) \
            else BASE_REQUIRED
        for key in required:
            if key not in route:
                schema_missing[key] = schema_missing.get(key, 0) + 1
        cal = route.get("calibrated_probabilities") if expect_scoring else None
        top_conf = route.get("confidence")
        if expect_scoring and cal:
            top_conf = max(cal.values())
        results.append({
            "task": t["task"],
            "true_tier": t["tier"],
            "true_effort": t["effort"],
            "pred_tier": route.get("tier"),
            "pred_effort": route.get("effort"),
            "pred_labels": route.get("task_labels"),
            "confidence": top_conf,
            "calibrated": route.get("calibrated"),
            "margin": route.get("margin"),
            "uncertain": route.get("uncertain"),
            "latency_ms": payload.get("latency_ms", client_ms),
        })
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(tasks)}")

    print(f"done: {len(results)} ok, {errors} errors")
    failures: list[str] = []
    warnings: list[str] = []

    if errors:
        failures.append(f"{errors} request errors")

    n = len(results)
    tier_stats: dict = {}
    if n:
        correct = sum(1 for r in results if r["pred_tier"] == r["true_tier"])
        acc = correct / n
        print(f"tier accuracy: {acc:.3f} ({correct}/{n})")
        if acc < TIER_ACCURACY_FLOOR:
            failures.append(f"tier accuracy {acc:.3f} < {TIER_ACCURACY_FLOOR}")
        elif acc < TIER_ACCURACY_WARN:
            warnings.append(f"tier accuracy {acc:.3f} < warn {TIER_ACCURACY_WARN}")

        # ECE on top-1 confidence (calibrated when the shim provides it)
        y_true = [1 if r["pred_tier"] == r["true_tier"] else 0 for r in results]
        y_conf = [r["confidence"] if isinstance(r["confidence"], (int, float)) else 0.0
                  for r in results]
        if args.mode == "full" and expect_scoring:
            ece = ece_10bin(y_true, y_conf)
            print(f"ECE (10-bin, calibrated top-1): {ece:.4f}")
            if n < MIN_ECE_N:
                # ECE on a handful of items (e.g. the 14-task red-team
                # set) is noise, not signal: report it, don't gate on it.
                print(f"ECE: not gated (n={n} < {MIN_ECE_N})")
            elif ece > ECE_CEILING:
                failures.append(f"ECE {ece:.4f} > {ECE_CEILING}")
        else:
            print("ECE: skipped (no calibrated confidences)")

        # per-tier accuracy + ECE alongside the overall numbers
        tier_stats = per_tier_stats(results)
        if tier_stats:
            print("per-tier accuracy / ECE:")
            for tier, s in tier_stats.items():
                print(f"  {tier:10s} n={s['n']:3d} acc={s['accuracy']:.3f} "
                      f"ece={s['ece']:.4f}")

        lat = [r["latency_ms"] for r in results]
        p50, p95 = percentile(lat, 50), percentile(lat, 95)
        print(f"latency: p50={p50:.1f}ms p95={p95:.1f}ms")
        if p50 > P50_CEILING_MS:
            failures.append(f"p50 latency {p50:.1f}ms > {P50_CEILING_MS}ms")
        if p95 > P95_CEILING_MS:
            failures.append(f"p95 latency {p95:.1f}ms > {P95_CEILING_MS}ms")

        if schema_missing:
            detail = ", ".join(f"{k}×{v}" for k, v in sorted(schema_missing.items()))
            failures.append(f"schema regression: missing keys: {detail}")
        else:
            print("schema: all required keys present")

        # no-regress vs last_run.json: only flag when the last recorded tier
        # was CORRECT and the new prediction is wrong (an actual regression).
        regressed = []
        for r in results:
            prev = last.get(r["task"])
            if prev and prev.get("tier") == r["true_tier"] \
                    and r["pred_tier"] != r["true_tier"]:
                regressed.append(r["task"][:60])
        if regressed:
            failures.append(f"{len(regressed)} tier regressions vs last_run.json "
                            f"(e.g. {regressed[0]!r})")
        else:
            print(f"no-regress: clean ({len(last)} tasks compared)")

    out_path = args.out or os.path.join(
        HERE, f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base_url": args.base_url, "mode": args.mode,
                   "results": results, "failures": failures,
                   "warnings": warnings, "per_tier": tier_stats}, f, indent=1)
    print(f"results -> {out_path}")

    if not failures and not args.no_update_last_run and n:
        with open(args.last_run, "w", encoding="utf-8") as f:
            json.dump({
                "run_ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "base_url": args.base_url,
                "tasks": {r["task"]: {"tier": r["pred_tier"],
                                      "effort": r["pred_effort"]}
                          for r in results},
            }, f, indent=1)
        print(f"last_run.json updated ({n} tasks)")

    for w in warnings:
        print(f"WARNING: {w}")
    if failures:
        print("FAIL:")
        for fl in failures:
            print(f"  - {fl}")
        return 1
    print("BATTERY GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
