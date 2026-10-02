#!/usr/bin/env python3
"""Fit per-answer-type temperature map from logged decision records.

Reads JSONL decision records — one per line:
  {"type": "choice"|"noul"|"score", "gold": int, "logits": [...]}
or
  {"type": "choice"|"noul"|"score", "gold": int, "probs": [...]}

Fits a per-type temperature map (types with fewer than --min-rows rows fall
back to the pooled temperature — decider's <50 convention) and merges it into
a calibration.json-style file under "temperature_by_type", leaving any
existing pooled "temperature" (route calibration) and other keys untouched.

Requires numpy only.

Usage:
  python3 fit_types.py --records decision_records.jsonl \\
      [--calibration ../calibration.json] [--out ../calibration.json]
      [--min-rows 50]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

try:
    import numpy as np  # noqa: F401  (imported for the ImportError message)
except ImportError:
    sys.exit("fit_types.py requires numpy")

from calibration import (  # noqa: E402
    DECISION_TYPES,
    MIN_ROWS_PER_TYPE,
    PerTypeTemperatureCalibrator,
)


def load_records(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"{path} line {ln}: invalid JSON: {e}")
                sys.exit(1)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fit per-answer-type temperature map from decision records")
    ap.add_argument("--records", required=True,
                    help="JSONL decision records {type, gold, logits|probs}")
    ap.add_argument("--calibration",
                    default=os.path.join(HERE, "..", "calibration.json"),
                    help="calibration.json to merge temperature_by_type into")
    ap.add_argument("--out", default=None,
                    help="output path (default: overwrite --calibration)")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS_PER_TYPE,
                    help="per-type minimum rows before a dedicated T is fit")
    args = ap.parse_args()

    records = load_records(args.records)
    if not records:
        print("no records found")
        return 1
    print(f"fit_types: {len(records)} records from {args.records}")

    cal = PerTypeTemperatureCalibrator(min_rows=args.min_rows).fit(records)
    payload = cal.to_dict()
    print(f"  pooled T={payload['temperature']:.4f}")
    for qtype in DECISION_TYPES:
        n = payload["per_type_rows"].get(qtype, 0)
        T = payload["temperature_by_type"].get(qtype)
        if T is None:
            print(f"  {qtype:7s} n={n:5d} -> pooled fallback "
                  f"(fewer than {args.min_rows} rows)")
        else:
            print(f"  {qtype:7s} n={n:5d} T={T:.4f}")

    merged: dict = {}
    cal_path = args.calibration
    if os.path.exists(cal_path):
        with open(cal_path, encoding="utf-8") as f:
            merged = json.load(f)
        print(f"  merging into existing {cal_path} "
              f"(kept pooled temperature={merged.get('temperature')})")
    else:
        print(f"  no existing {cal_path}; writing fresh file")
    merged["temperature_by_type"] = payload["temperature_by_type"]
    merged["per_type_rows"] = payload["per_type_rows"]
    merged["min_rows_per_type"] = payload["min_rows_per_type"]
    merged["type_fit_date"] = payload["fit_date"]

    out = args.out or cal_path
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
