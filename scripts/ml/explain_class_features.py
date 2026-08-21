#!/usr/bin/env python3
"""
Rank features that distinguish each class from all other classes.

The score is one-vs-rest and intentionally simple:
  abs(class_median - rest_median) / robust_scale

This is not a causal proof. It is a compact way to inspect which measured
signals separate each scenario class in the current dataset.

Usage:
    python3 scripts/ml/explain_class_features.py \
      --csv results/ml_dataset_v3/window_features_augmented.csv \
      --exclude-classes load_ramp,multi_ue_contention \
      --top 12
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


META_COLS = 6
METADATA_PREFIXES = (
    "scenario_is_",
    "window_is_",
    "phase_is_",
    "step_is_multi_ue",
    "step_num_qhats",
    "step_parallel",
    "step_total_parallel_flows",
)


def is_metadata(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix.rstrip("_") for prefix in METADATA_PREFIXES)


def load_dataset(path: Path, exclude_classes: set[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    rows: list[list[str]] = []
    labels: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        feature_names = header[META_COLS:]
        for row in reader:
            if len(row) <= META_COLS:
                continue
            label = row[2].strip()
            if label in exclude_classes:
                continue
            labels.append(label)
            rows.append(row[META_COLS:])

    X = np.full((len(rows), len(feature_names)), np.nan)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            value = value.strip()
            if not value or value.lower() == "nan":
                continue
            try:
                X[r, c] = float(value)
            except ValueError:
                continue
    return feature_names, X, np.array(labels)


def robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    q75, q25 = np.nanpercentile(values, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-12:
        return iqr
    std = float(np.nanstd(values))
    return std if std > 1e-12 else 0.0


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/ml_dataset_v3/window_features_augmented.csv")
    parser.add_argument("--exclude-classes", default="")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--max-nan", type=float, default=50.0, help="Max NaN percent allowed within the target class.")
    parser.add_argument("--min-coverage", type=float, default=50.0, help="Min non-NaN percent required in class and rest.")
    parser.add_argument("--include-metadata", action="store_true")
    args = parser.parse_args()

    exclude_classes = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}
    csv_path = Path(args.csv)
    feature_names, X, y = load_dataset(csv_path, exclude_classes)
    if X.size == 0:
        raise SystemExit("ERROR: no rows loaded")

    feature_indices = [
        i for i, name in enumerate(feature_names)
        if args.include_metadata or not is_metadata(name)
    ]
    labels = sorted(set(y))

    print(f"CSV: {csv_path}")
    print(f"samples: {len(y)}")
    print(f"classes: {', '.join(labels)}")
    if exclude_classes:
        print(f"excluded: {', '.join(sorted(exclude_classes))}")

    for label in labels:
        class_mask = y == label
        rest_mask = ~class_mask
        class_n = int(class_mask.sum())
        rest_n = int(rest_mask.sum())
        scored = []

        for i in feature_indices:
            class_values = X[class_mask, i]
            rest_values = X[rest_mask, i]
            class_valid = np.isfinite(class_values)
            rest_valid = np.isfinite(rest_values)
            class_cov = class_valid.mean() * 100 if class_values.size else 0.0
            rest_cov = rest_valid.mean() * 100 if rest_values.size else 0.0
            class_nan = 100 - class_cov

            if class_nan > args.max_nan:
                continue
            if class_cov < args.min_coverage or rest_cov < args.min_coverage:
                continue

            class_median = float(np.nanmedian(class_values))
            rest_median = float(np.nanmedian(rest_values))
            scale = robust_scale(X[:, i])
            if scale <= 0:
                continue

            delta = class_median - rest_median
            score = abs(delta) / scale
            if score <= 0:
                continue

            direction = "higher" if delta > 0 else "lower"
            scored.append((
                score,
                feature_names[i],
                direction,
                class_median,
                rest_median,
                delta,
                class_cov,
                rest_cov,
            ))

        scored.sort(reverse=True, key=lambda item: item[0])

        print(f"\n{'=' * 90}")
        print(f"{label}  n={class_n}  vs rest n={rest_n}")
        print(f"{'=' * 90}")
        print(f"{'score':>7}  {'direction':<7}  {'class_med':>11}  {'rest_med':>11}  {'delta':>11}  {'cov%':>6}  feature")
        print("-" * 90)
        for score, name, direction, class_median, rest_median, delta, class_cov, _rest_cov in scored[:args.top]:
            print(
                f"{score:>7.2f}  {direction:<7}  {fmt(class_median):>11}  "
                f"{fmt(rest_median):>11}  {fmt(delta):>11}  {class_cov:>5.1f}%  {name}"
            )


if __name__ == "__main__":
    main()
