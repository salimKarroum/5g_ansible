#!/usr/bin/env python3
"""
Study how many observable features are needed for RCA classification and list
the class-specific features that separate each scenario.

The script intentionally excludes experiment metadata that would leak labels.
It ranks observable features globally with a Random Forest, evaluates several
top-K subsets with leave-one-run-out validation, then trains one-vs-rest Random
Forests to explain the selected compact feature set per class.

Usage:
    python3 scripts/ml/study_minimal_features.py \
      --csv results/ml_dataset_v3/window_features_augmented.csv \
      --exclude-classes load_ramp,multi_ue_contention \
      --out-dir results/ml_dataset_v3/minimal_feature_study
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - xgboost can be absent in lightweight envs.
    XGBClassifier = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


warnings.filterwarnings(
    "ignore",
    message=r".*Parameters: \{ \"use_label_encoder\" \} are not used.*",
    category=UserWarning,
)

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
FORCED_PATTERNS = ("tcp_evict", "tcp_no_match", "dl_mcs", "dl_bler")


def is_metadata(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix.rstrip("_") for prefix in METADATA_PREFIXES)


def load_csv(path: Path, exclude_classes: set[str]) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if len(row) > META_COLS]
    if exclude_classes:
        rows = [row for row in rows if row[2].strip() not in exclude_classes]
    return header, rows


def compute_nan_rates(rows: list[list[str]], n_features: int) -> np.ndarray:
    nan_counts = np.zeros(n_features)
    for row in rows:
        for i in range(n_features):
            value = row[META_COLS + i].strip()
            if not value or value.lower() == "nan":
                nan_counts[i] += 1
    return nan_counts / max(len(rows), 1) * 100


def build_matrix(rows: list[list[str]], feature_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.full((len(rows), len(feature_indices)), np.nan)
    y: list[str] = []
    runs: list[str] = []
    for r, row in enumerate(rows):
        y.append(row[2].strip())
        runs.append(row[1].strip())
        for c, feature_idx in enumerate(feature_indices):
            value = row[META_COLS + feature_idx].strip()
            if value and value.lower() != "nan":
                try:
                    X[r, c] = float(value)
                except ValueError:
                    pass
    return X, np.array(y), np.array(runs)


def add_temporal_ratios(X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    ratio_names: list[str] = []
    ratio_cols: list[np.ndarray] = []
    name_to_idx = {name: i for i, name in enumerate(names)}
    for i, name in enumerate(names):
        if "_second_half_mean" not in name:
            continue
        first_name = name.replace("_second_half_mean", "_first_half_mean")
        if first_name not in name_to_idx:
            continue
        j = name_to_idx[first_name]
        denom = np.where(np.abs(X[:, j]) > 1e-3, X[:, j], np.nan)
        ratio_cols.append(X[:, i] / denom)
        ratio_names.append(name.replace("_second_half_mean", "_half_ratio"))
    if not ratio_cols:
        return X, names
    return np.hstack([X, np.column_stack(ratio_cols)]), names + ratio_names


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def make_xgb(n_classes: int):
    if XGBClassifier is None:
        return None
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        num_class=n_classes,
    )


def rank_features(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[tuple[str, float]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    clf = make_rf()
    clf.fit(X_imp, y)
    order = np.argsort(clf.feature_importances_)[::-1]
    return [(names[i], float(clf.feature_importances_[i])) for i in order]


def fit_predict_fold(clf, X_train, y_train, X_test):
    if XGBClassifier is not None and isinstance(clf, XGBClassifier):
        fold_labels = sorted(set(y_train))
        label_to_id = {label: i for i, label in enumerate(fold_labels)}
        y_train_enc = np.array([label_to_id[label] for label in y_train])
        weights = compute_sample_weight("balanced", y_train)
        clf.fit(X_train, y_train_enc, sample_weight=weights)
        pred_enc = clf.predict(X_test)
        return np.array([fold_labels[int(i)] for i in pred_enc])

    if isinstance(clf, VotingClassifier):
        fold_labels = sorted(set(y_train))
        label_to_id = {label: i for i, label in enumerate(fold_labels)}
        y_train_enc = np.array([label_to_id[label] for label in y_train])
        weights = compute_sample_weight("balanced", y_train)
        clf.fit(X_train, y_train_enc, sample_weight=weights)
        pred_enc = clf.predict(X_test)
        return np.array([fold_labels[int(i)] for i in pred_enc])

    clf.fit(X_train, y_train)
    return clf.predict(X_test)


def evaluate_subset(X: np.ndarray, y: np.ndarray, runs: np.ndarray, model: str) -> tuple[float, float, dict[str, dict[str, float]]]:
    labels = sorted(set(y))
    y_true_all: list[str] = []
    y_pred_all: list[str] = []

    for run in np.unique(runs):
        test_mask = runs == run
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train_mask])
        X_test = imputer.transform(X[test_mask])
        y_train = y[train_mask]

        if model == "rf":
            clf = make_rf()
        elif model == "xgb":
            clf = make_xgb(len(labels))
            if clf is None:
                return float("nan"), float("nan"), {}
        elif model == "ens":
            xgb = make_xgb(len(labels))
            if xgb is None:
                return float("nan"), float("nan"), {}
            clf = VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
                    ("xgb", xgb),
                ],
                voting="soft",
            )
        else:
            raise ValueError(model)

        preds = fit_predict_fold(clf, X_train, y_train, X_test)
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(preds)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return acc, macro_f1, report


def robust_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return 0.0
    q75, q25 = np.nanpercentile(finite, [75, 25])
    iqr = float(q75 - q25)
    if iqr > 1e-12:
        return iqr
    std = float(np.nanstd(finite))
    return std if std > 1e-12 else 0.0


def class_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    top_per_class: int,
) -> list[dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    rows: list[dict[str, object]] = []

    for label in sorted(set(y)):
        target = (y == label).astype(int)
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_imp, target)
        order = np.argsort(clf.feature_importances_)[::-1]
        class_mask = y == label
        rest_mask = ~class_mask

        rank = 0
        for i in order:
            importance = float(clf.feature_importances_[i])
            if importance <= 0:
                continue
            class_med = float(np.nanmedian(X[class_mask, i]))
            rest_med = float(np.nanmedian(X[rest_mask, i]))
            delta = class_med - rest_med
            scale = robust_scale(X[:, i])
            separation = abs(delta) / scale if scale > 0 else 0.0
            rank += 1
            rows.append({
                "class": label,
                "rank": rank,
                "feature": names[i],
                "ovr_importance": importance,
                "direction": "higher" if delta > 0 else "lower",
                "class_median": class_med,
                "rest_median": rest_med,
                "delta": delta,
                "robust_separation": separation,
                "class_coverage_pct": float(np.isfinite(X[class_mask, i]).mean() * 100),
            })
            if rank >= top_per_class:
                break
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.3f}"


def save_sweep_plot(path: Path, rows: list[dict[str, object]]) -> None:
    if plt is None or not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model and math.isfinite(float(row["macro_f1"]))]
        if not model_rows:
            continue
        ax.plot(
            [int(row["top_k"]) for row in model_rows],
            [float(row["macro_f1"]) for row in model_rows],
            marker="o",
            label=model.upper(),
        )
    ax.set_xlabel("Nombre de features")
    ax.set_ylabel("Macro-F1 leave-one-run-out")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    csv_path: Path,
    labels: list[str],
    sweep_rows: list[dict[str, object]],
    selected_k: int,
    selected_model: str,
    selected_macro_f1: float,
    selected_features: list[tuple[str, float]],
    class_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# Minimal Feature Study")
    lines.append("")
    lines.append(f"Dataset: `{csv_path}`")
    lines.append(f"Classes: {', '.join(labels)}")
    lines.append("")
    lines.append("## Nombre minimal de features")
    lines.append("")
    lines.append(
        f"Choix compact: **top-{selected_k}** avec **{selected_model.upper()}**, "
        f"Macro-F1={selected_macro_f1:.3f}."
    )
    lines.append("")
    lines.append("| model | top_k | accuracy | macro_f1 |")
    lines.append("|---|---:|---:|---:|")
    for row in sweep_rows:
        lines.append(
            f"| {row['model']} | {row['top_k']} | "
            f"{float(row['accuracy']):.3f} | {float(row['macro_f1']):.3f} |"
        )
    lines.append("")
    lines.append("## Features globales retenues")
    lines.append("")
    for i, (name, importance) in enumerate(selected_features, 1):
        lines.append(f"{i}. `{name}` importance={importance:.4f}")
    lines.append("")
    lines.append("## Top features par classe")
    for label in labels:
        lines.append("")
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| rank | feature | direction | class median | rest median | importance | separation | coverage |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|")
        for row in [r for r in class_rows if r["class"] == label]:
            lines.append(
                f"| {row['rank']} | `{row['feature']}` | {row['direction']} | "
                f"{fmt(float(row['class_median']))} | {fmt(float(row['rest_median']))} | "
                f"{float(row['ovr_importance']):.4f} | {float(row['robust_separation']):.2f} | "
                f"{float(row['class_coverage_pct']):.1f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/ml_dataset_v3/window_features_augmented.csv")
    parser.add_argument("--exclude-classes", default="")
    parser.add_argument("--out-dir", default="results/ml_dataset_v3/minimal_feature_study")
    parser.add_argument("--nan-threshold", type=float, default=10.0)
    parser.add_argument("--top-k-list", default="5,10,15,20,25,30,40,60")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Macro-F1 drop tolerated vs best.")
    parser.add_argument("--top-per-class", type=int, default=8)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}
    header, rows = load_csv(csv_path, exclude)
    feature_names = header[META_COLS:]
    n_features = len(feature_names)
    nan_rates = compute_nan_rates(rows, n_features)

    observable_idx = [i for i, name in enumerate(feature_names) if not is_metadata(name)]
    selected_idx = sorted(set(
        [i for i in observable_idx if nan_rates[i] <= args.nan_threshold] +
        [i for i in observable_idx if any(pattern in feature_names[i] for pattern in FORCED_PATTERNS)]
    ))
    X, y, runs = build_matrix(rows, selected_idx)
    names = [feature_names[i] for i in selected_idx]
    X, names = add_temporal_ratios(X, names)
    labels = sorted(set(y))

    print(f"Dataset: {csv_path}")
    print(f"samples: {len(y)}")
    print(f"classes: {', '.join(labels)}")
    print(f"candidate_features: {len(names)}")

    ranked = rank_features(X, y, names)
    name_to_idx = {name: i for i, name in enumerate(names)}
    top_k_values = sorted({int(x.strip()) for x in args.top_k_list.split(",") if x.strip()})

    sweep_rows: list[dict[str, object]] = []
    print("\nFeature count sweep:")
    print(f"{'model':<5} {'top_k':>5} {'accuracy':>9} {'macro_f1':>9}")
    for top_k in top_k_values:
        feature_subset = [name_to_idx[name] for name, _ in ranked[:top_k]]
        X_subset = X[:, feature_subset]
        for model in ["rf", "xgb", "ens"]:
            acc, macro_f1, report = evaluate_subset(X_subset, y, runs, model)
            if not math.isfinite(macro_f1):
                continue
            sweep_rows.append({
                "model": model,
                "top_k": top_k,
                "accuracy": acc,
                "macro_f1": macro_f1,
            })
            print(f"{model:<5} {top_k:>5} {acc:>9.3f} {macro_f1:>9.3f}")

    best_macro = max(float(row["macro_f1"]) for row in sweep_rows)
    eligible = [row for row in sweep_rows if float(row["macro_f1"]) >= best_macro - args.tolerance]
    eligible.sort(key=lambda row: (int(row["top_k"]), -float(row["macro_f1"])))
    chosen = eligible[0]
    selected_k = int(chosen["top_k"])
    selected_model = str(chosen["model"])
    selected_macro_f1 = float(chosen["macro_f1"])
    selected_features = ranked[:selected_k]

    selected_indices = [name_to_idx[name] for name, _ in selected_features]
    class_rows = class_feature_importance(
        X[:, selected_indices],
        y,
        [name for name, _ in selected_features],
        args.top_per_class,
    )

    write_csv(out_dir / "feature_count_sweep.csv", sweep_rows)
    write_csv(
        out_dir / "selected_global_features.csv",
        [
            {"rank": i, "feature": name, "global_importance": importance}
            for i, (name, importance) in enumerate(selected_features, 1)
        ],
    )
    write_csv(out_dir / "per_class_top_features.csv", class_rows)
    save_sweep_plot(out_dir / "feature_count_sweep.png", sweep_rows)
    write_report(
        out_dir / "README.md",
        csv_path,
        labels,
        sweep_rows,
        selected_k,
        selected_model,
        selected_macro_f1,
        selected_features,
        class_rows,
    )

    print("\nDecision:")
    print(f"  best_macro_f1: {best_macro:.3f}")
    print(f"  selected_compact_set: top-{selected_k} {selected_model.upper()} macro_f1={selected_macro_f1:.3f}")
    print(f"  out_dir: {out_dir}")
    print("  files:")
    print(f"    - {out_dir / 'feature_count_sweep.csv'}")
    print(f"    - {out_dir / 'selected_global_features.csv'}")
    print(f"    - {out_dir / 'per_class_top_features.csv'}")
    print(f"    - {out_dir / 'feature_count_sweep.png'}")
    print(f"    - {out_dir / 'README.md'}")


if __name__ == "__main__":
    main()
