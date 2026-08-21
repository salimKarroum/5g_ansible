#!/usr/bin/env python3
"""Recompute the report metrics for the final K=28 RCA model.

This script is intended for report reproducibility. It evaluates the fixed
K=28 feature set with Leave-One-Run-Out (LORO), compares Random Forest with
Logistic Regression and a majority-class baseline, and summarizes the optional
cross-stack validation dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


def load_features(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_matrix(path: Path, y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    matrix = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    matrix.to_csv(path)


def write_report(path: Path, y_true: list[str], y_pred: list[str], labels: list[str]) -> None:
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    ).transpose()
    report.to_csv(path)


def summarize_cross_stack(path: Path | None, out_dir: Path) -> dict | None:
    if path is None or not path.exists():
        return None

    cross = pd.read_csv(path)
    required = {"stack", "label", "run"}
    missing = sorted(required - set(cross.columns))
    if missing:
        raise SystemExit(f"Cross-stack CSV is missing required columns: {missing}")

    by_stack_label_run = cross.groupby(["stack", "label", "run"]).size().rename("windows").reset_index()
    by_stack_label_run.to_csv(out_dir / "cross_stack_windows_by_stack_label_run.csv", index=False)

    return {
        "rows": int(len(cross)),
        "unique_runs": int(cross["run"].nunique()),
        "runs_by_stack": {str(k): int(v) for k, v in cross.groupby("stack")["run"].nunique().items()},
        "windows_by_label": {str(k): int(v) for k, v in cross["label"].value_counts().sort_index().items()},
        "windows_by_stack_label": {
            f"{stack}|{label}": int(v) for (stack, label), v in cross.groupby(["stack", "label"]).size().items()
        },
        "runs_by_stack_label": {
            f"{stack}|{label}": int(v)
            for (stack, label), v in cross.groupby(["stack", "label"])["run"].nunique().items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-csv", required=True, help="Unbalanced original 7-class dataset CSV.")
    parser.add_argument("--balanced-csv", required=True, help="Balanced 48-window-per-class dataset CSV.")
    parser.add_argument("--features-file", required=True, help="Fixed K=28 selected feature list.")
    parser.add_argument("--cross-stack-csv", default="", help="Optional cross-stack validation CSV.")
    parser.add_argument("--out-dir", required=True, help="Directory where audit outputs are written.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Random Forest parallel jobs. Use -1 on Linux hosts if desired.",
    )
    args = parser.parse_args()

    original_csv = Path(args.original_csv)
    balanced_csv = Path(args.balanced_csv)
    features_file = Path(args.features_file)
    cross_stack_csv = Path(args.cross_stack_csv) if args.cross_stack_csv else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig = pd.read_csv(original_csv)
    bal = pd.read_csv(balanced_csv)
    features = load_features(features_file)

    for column in ("label", "run"):
        if column not in bal.columns:
            raise SystemExit(f"Balanced CSV is missing required column: {column}")
        if column not in orig.columns:
            raise SystemExit(f"Original CSV is missing required column: {column}")

    missing_features = [feature for feature in features if feature not in bal.columns]
    if missing_features:
        raise SystemExit(f"Balanced CSV is missing selected features: {missing_features}")

    labels = sorted(bal["label"].astype(str).unique())

    coverage_rows = []
    run_counts_rows = []
    for label in labels:
        before = orig[orig["label"].astype(str) == label]
        after = bal[bal["label"].astype(str) == label]
        before_runs = set(before["run"].astype(str))
        after_runs = set(after["run"].astype(str))
        removed = sorted(before_runs - after_runs)
        coverage_rows.append(
            {
                "label": label,
                "windows_before": int(len(before)),
                "windows_after": int(len(after)),
                "runs_before": int(len(before_runs)),
                "runs_after": int(len(after_runs)),
                "removed_runs": ";".join(removed),
            }
        )
        for run, count in after.groupby(after["run"].astype(str)).size().sort_index().items():
            run_counts_rows.append({"label": label, "run": run, "balanced_windows": int(count)})

    coverage = pd.DataFrame(coverage_rows)
    run_counts = pd.DataFrame(run_counts_rows)
    coverage.to_csv(out_dir / "undersampling_run_coverage.csv", index=False)
    run_counts.to_csv(out_dir / "balanced_7classes_windows_by_run.csv", index=False)

    X = bal[features].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = bal["label"].astype(str).to_numpy()
    runs = bal["run"].astype(str).to_numpy()

    rf_true: list[str] = []
    rf_pred: list[str] = []
    lr_true: list[str] = []
    lr_pred: list[str] = []
    fold_rows = []

    for run in sorted(set(runs)):
        test = runs == run
        train = ~test
        X_train_raw = X[train]
        X_test_raw = X[test]
        y_train = y[train]
        y_test = y[test]

        rf_imputer = SimpleImputer(strategy="median")
        X_train_rf = rf_imputer.fit_transform(X_train_raw)
        X_test_rf = rf_imputer.transform(X_test_raw)
        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            class_weight="balanced",
        )
        rf.fit(X_train_rf, y_train)
        pred_rf = rf.predict(X_test_rf)

        lr_imputer = SimpleImputer(strategy="median")
        X_train_lr = lr_imputer.fit_transform(X_train_raw)
        X_test_lr = lr_imputer.transform(X_test_raw)
        scaler = StandardScaler()
        X_train_lr = scaler.fit_transform(X_train_lr)
        X_test_lr = scaler.transform(X_test_lr)
        lr = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced",
            random_state=args.random_state,
        )
        lr.fit(X_train_lr, y_train)
        pred_lr = lr.predict(X_test_lr)

        rf_true.extend(y_test.tolist())
        rf_pred.extend(pred_rf.tolist())
        lr_true.extend(y_test.tolist())
        lr_pred.extend(pred_lr.tolist())

        fold_rows.append(
            {
                "run": run,
                "test_samples": int(test.sum()),
                "true_labels": ",".join(sorted(set(y_test))),
                "rf_correct": int((pred_rf == y_test).sum()),
                "rf_accuracy": float(accuracy_score(y_test, pred_rf)),
                "rf_macro_f1": float(f1_score(y_test, pred_rf, average="macro", zero_division=0)),
                "lr_correct": int((pred_lr == y_test).sum()),
                "lr_accuracy": float(accuracy_score(y_test, pred_lr)),
                "lr_macro_f1": float(f1_score(y_test, pred_lr, average="macro", zero_division=0)),
            }
        )

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out_dir / "k28_loro_fold_metrics_rf_lr.csv", index=False)
    write_matrix(out_dir / "k28_rf_confusion_matrix.csv", rf_true, rf_pred, labels)
    write_matrix(out_dir / "k28_logistic_regression_confusion_matrix.csv", lr_true, lr_pred, labels)
    write_report(out_dir / "k28_rf_classification_report.csv", rf_true, rf_pred, labels)
    write_report(out_dir / "k28_logistic_regression_classification_report.csv", lr_true, lr_pred, labels)

    majority_class = bal["label"].astype(str).mode().sort_values().iloc[0]
    majority_pred = np.array([majority_class] * len(y))

    summary = {
        "source_files": {
            "original_dataset": str(original_csv),
            "balanced_dataset": str(balanced_csv),
            "features_file": str(features_file),
            "cross_stack_dataset": str(cross_stack_csv) if cross_stack_csv else None,
        },
        "samples": int(len(bal)),
        "runs": int(len(set(runs))),
        "labels": labels,
        "features_count": int(len(features)),
        "rf_k28_loro": {
            "accuracy": float(accuracy_score(rf_true, rf_pred)),
            "macro_f1": float(f1_score(rf_true, rf_pred, average="macro", zero_division=0)),
            "fold_macro_f1_min": float(folds["rf_macro_f1"].min()),
            "fold_macro_f1_max": float(folds["rf_macro_f1"].max()),
            "fold_macro_f1_mean": float(folds["rf_macro_f1"].mean()),
            "fold_macro_f1_std": float(folds["rf_macro_f1"].std()),
            "fold_macro_f1_median": float(folds["rf_macro_f1"].median()),
        },
        "logistic_regression_loro": {
            "accuracy": float(accuracy_score(lr_true, lr_pred)),
            "macro_f1": float(f1_score(lr_true, lr_pred, average="macro", zero_division=0)),
            "fold_macro_f1_min": float(folds["lr_macro_f1"].min()),
            "fold_macro_f1_max": float(folds["lr_macro_f1"].max()),
            "fold_macro_f1_mean": float(folds["lr_macro_f1"].mean()),
            "fold_macro_f1_std": float(folds["lr_macro_f1"].std()),
            "fold_macro_f1_median": float(folds["lr_macro_f1"].median()),
            "params": lr.get_params(),
        },
        "majority_baseline": {
            "class": str(majority_class),
            "accuracy": float(accuracy_score(y, majority_pred)),
            "macro_f1": float(f1_score(y, majority_pred, labels=labels, average="macro", zero_division=0)),
        },
        "cross_stack": summarize_cross_stack(cross_stack_csv, out_dir),
    }

    (out_dir / "k28_report_results_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
