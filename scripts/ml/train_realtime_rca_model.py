#!/usr/bin/env python3
"""Train the compact RCA model used by the real-time inference exporter.

The model is intentionally trained only on observable physical features.  It
does not use scenario/window metadata, because those fields do not exist in a
live Grafana/Prometheus deployment.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score


META_COLS = 6

DEFAULT_FEATURES = [
    "direct_latest_ms_median",
    "direct_latest_ms_mean",
    "rf_oai_gnb_l1_prach_i0_db_mean",
    "upf_cpu_cores_p95",
    "upf_memory_mib_median",
    "server_cpu_cores_mean",
    "server_cpu_cores_p95",
    "server_cpu_cores_max",
    "server_memory_mib_median",
    "tcp_ack_no_match_hz_30s_median",
    "tcp_ack_observed_hz_30s_min",
    "tcp_ack_observed_hz_30s_mean",
    "container_cpu_cores_p95",
    "container_cpu_cores_max",
    "container_memory_mib_p95",
]

DEFAULT_EXCLUDE = {"load_ramp", "multi_ue_contention"}


def read_feature_file(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_FEATURES)
    features: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                name = name.split(",", 1)[0].strip()
                if name.lower() not in {"feature", "feature_name", "name"}:
                    features.append(name)
    return features


def load_dataset(path: Path, exclude_classes: set[str]) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if len(row) > META_COLS]
    if exclude_classes:
        rows = [row for row in rows if row[2].strip() not in exclude_classes]
    return header, rows


def build_matrix(
    header: list[str],
    rows: list[list[str]],
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_header = header[META_COLS:]
    missing = [name for name in features if name not in feature_header]
    if missing:
        raise SystemExit("Missing feature columns in CSV: " + ", ".join(missing))

    indices = [feature_header.index(name) for name in features]
    X = np.full((len(rows), len(features)), np.nan)
    y: list[str] = []
    runs: list[str] = []
    for r, row in enumerate(rows):
        runs.append(row[1].strip())
        y.append(row[2].strip())
        for c, idx in enumerate(indices):
            value = row[META_COLS + idx].strip()
            if value and value.lower() != "nan":
                try:
                    X[r, c] = float(value)
                except ValueError:
                    pass
    return X, np.array(y), np.array(runs)


def leave_one_run_out(
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> tuple[float, float]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for run in sorted(set(runs)):
        test_mask = runs == run
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train_mask])
        X_test = imputer.transform(X[test_mask])
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train, y[train_mask])
        pred = clf.predict(X_test)
        y_true.extend(y[test_mask])
        y_pred.extend(pred)
    return (
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="macro", zero_division=0),
    )


def robust_stats(X: np.ndarray, features: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    medians: dict[str, float] = {}
    iqrs: dict[str, float] = {}
    for i, name in enumerate(features):
        col = X[:, i]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            medians[name] = 0.0
            iqrs[name] = 1.0
            continue
        q25, q50, q75 = np.percentile(finite, [25, 50, 75])
        iqr = float(q75 - q25)
        medians[name] = float(q50)
        iqrs[name] = iqr if iqr > 1e-9 else 1.0
    return medians, iqrs


def class_medians(X: np.ndarray, y: np.ndarray, features: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for label in sorted(set(y)):
        mask = y == label
        values: dict[str, float] = {}
        for i, name in enumerate(features):
            finite = X[mask, i]
            finite = finite[np.isfinite(finite)]
            values[name] = float(np.median(finite)) if finite.size else float("nan")
        out[label] = values
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Stable window_features CSV")
    ap.add_argument("--model-out", default="results/ml_dataset_v3/realtime_rca_model.pkl")
    ap.add_argument("--features-file", help="Optional one-feature-per-line file")
    ap.add_argument("--exclude-classes", default="load_ramp,multi_ue_contention")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    model_out = Path(args.model_out)
    exclude = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}
    features = read_feature_file(Path(args.features_file) if args.features_file else None)

    header, rows = load_dataset(csv_path, exclude)
    X, y, runs = build_matrix(header, rows, features)

    loro_acc, loro_f1 = leave_one_run_out(X, y, runs, args.n_estimators, args.random_state)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_imp, y)
    train_pred = clf.predict(X_imp)
    train_acc = accuracy_score(y, train_pred)
    train_f1 = f1_score(y, train_pred, average="macro", zero_division=0)
    medians, iqrs = robust_stats(X, features)

    bundle = {
        "model": clf,
        "imputer": imputer,
        "features": features,
        "labels": list(clf.classes_),
        "feature_medians": medians,
        "feature_iqrs": iqrs,
        "class_feature_medians": class_medians(X, y, features),
        "metadata": {
            "csv": str(csv_path),
            "samples": int(len(y)),
            "classes": sorted(set(y)),
            "excluded_classes": sorted(exclude),
            "leave_one_run_out_accuracy": float(loro_acc),
            "leave_one_run_out_macro_f1": float(loro_f1),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_f1),
        },
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    with model_out.open("wb") as f:
        pickle.dump(bundle, f)

    print(f"Dataset: {csv_path}")
    print(f"samples: {len(y)}")
    print("classes:", ", ".join(sorted(set(y))))
    print(f"features: {len(features)}")
    print(f"leave-one-run-out accuracy: {loro_acc:.3f}")
    print(f"leave-one-run-out macro_f1: {loro_f1:.3f}")
    print(f"train accuracy: {train_acc:.3f}")
    print(f"train macro_f1: {train_f1:.3f}")
    print(f"model: {model_out}")


if __name__ == "__main__":
    main()
