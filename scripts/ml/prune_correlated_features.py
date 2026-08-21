#!/usr/bin/env python3
"""
Prune redundant RCA features using correlation, then evaluate the reduced set.

The script starts from a selected feature list, removes one feature from each
highly correlated pair, and runs leave-one-run-out Random Forest evaluation for
several correlation thresholds.

Usage:
    python3 scripts/ml/prune_correlated_features.py \
      --csv results/ml_dataset_v3/window_features_augmented_stable_f1_0936.csv \
      --features /tmp/physical_feature_study_nanaware/selected_physical_features.csv \
      --importance /tmp/physical_feature_study_nanaware/physical_feature_importance.csv \
      --exclude-classes load_ramp,multi_ue_contention \
      --out-dir /tmp/physical_feature_study_nanaware/correlation_pruning
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


META_COLS = 6


def load_selected_features(path: Path) -> list[str]:
    df = pd.read_csv(path)
    return df["feature"].astype(str).tolist()


def load_importance(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "importance" in df.columns:
        importance_col = "importance"
    elif "global_importance" in df.columns:
        importance_col = "global_importance"
    else:
        return {}
    return dict(zip(df["feature"].astype(str), df[importance_col].astype(float)))


def load_dataset(path: Path, features: list[str], exclude: set[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if exclude:
        df = df[~df["label"].isin(exclude)].copy()
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: selected features missing from dataset: {missing}")
    return df[features].copy(), df["label"].astype(str).to_numpy(), df["run"].astype(str).to_numpy()


def evaluate_loro(X_df: pd.DataFrame, y: np.ndarray, runs: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    X = X_df.to_numpy(dtype=float)
    true_all: list[str] = []
    pred_all: list[str] = []
    for run in np.unique(runs):
        test = runs == run
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train])
        X_test = imputer.transform(X[test])
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train, y[train])
        pred = clf.predict(X_test)
        true_all.extend(y[test])
        pred_all.extend(pred)
    y_true = np.array(true_all)
    y_pred = np.array(pred_all)
    return float((y_true == y_pred).mean()), float(f1_score(y_true, y_pred, average="macro", zero_division=0)), y_true, y_pred


def prune_by_correlation(
    X_df: pd.DataFrame,
    features: list[str],
    importance: dict[str, float],
    threshold: float,
) -> tuple[list[str], list[dict[str, object]]]:
    corr = X_df[features].corr(method="spearman").abs()
    kept = set(features)
    removed_rows: list[dict[str, object]] = []

    pairs: list[tuple[float, str, str]] = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            value = corr.loc[a, b]
            if pd.notna(value) and float(value) >= threshold:
                pairs.append((float(value), a, b))
    pairs.sort(reverse=True)

    for value, a, b in pairs:
        if a not in kept or b not in kept:
            continue
        ia = importance.get(a, 0.0)
        ib = importance.get(b, 0.0)
        if ia >= ib:
            drop, keep = b, a
        else:
            drop, keep = a, b
        kept.remove(drop)
        removed_rows.append({
            "threshold": threshold,
            "removed_feature": drop,
            "kept_feature": keep,
            "abs_spearman_corr": value,
            "removed_importance": importance.get(drop, 0.0),
            "kept_importance": importance.get(keep, 0.0),
        })

    return [feature for feature in features if feature in kept], removed_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_corr_heatmap(X_df: pd.DataFrame, features: list[str], out: Path) -> None:
    if plt is None:
        return
    corr = X_df[features].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(8, len(features) * 0.42), max(7, len(features) * 0.38)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman correlation")
    ax.set_xticks(range(len(features)))
    ax.set_yticks(range(len(features)))
    ax.set_xticklabels(features, rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title("Correlation Between Selected Physical Features")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_sweep_plot(summary: list[dict[str, object]], out: Path) -> None:
    if plt is None:
        return
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    thresholds = [float(row["threshold"]) for row in summary]
    macro = [float(row["macro_f1"]) for row in summary]
    n_features = [int(row["n_features"]) for row in summary]
    ax1.plot(thresholds, macro, marker="o", color="#4C78A8", label="Macro-F1")
    ax1.set_xlabel("Correlation threshold")
    ax1.set_ylabel("Macro-F1", color="#4C78A8")
    ax1.tick_params(axis="y", labelcolor="#4C78A8")
    ax1.set_ylim(0, 1.02)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, n_features, marker="s", color="#F58518", label="# features")
    ax2.set_ylabel("Number of features", color="#F58518")
    ax2.tick_params(axis="y", labelcolor="#F58518")
    ax1.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--importance", default="")
    parser.add_argument("--exclude-classes", default="")
    parser.add_argument("--thresholds", default="0.95,0.9,0.85,0.8,0.75,0.7")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}
    features = load_selected_features(Path(args.features))
    importance = load_importance(Path(args.importance)) if args.importance else {}
    X_df, y, runs = load_dataset(Path(args.csv), features, exclude)

    save_corr_heatmap(X_df, features, out_dir / "feature_correlation_heatmap.png")

    summary_rows: list[dict[str, object]] = []
    all_removed: list[dict[str, object]] = []
    best_candidate: dict[str, object] | None = None
    best_macro = -1.0

    for threshold in [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        kept, removed = prune_by_correlation(X_df, features, importance, threshold)
        acc, macro_f1, y_true, y_pred = evaluate_loro(X_df[kept], y, runs)
        row = {
            "threshold": threshold,
            "n_features": len(kept),
            "removed": len(features) - len(kept),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "features": ";".join(kept),
        }
        summary_rows.append(row)
        all_removed.extend(removed)
        if macro_f1 > best_macro or (abs(macro_f1 - best_macro) < 1e-9 and len(kept) < int(best_candidate["n_features"])) if best_candidate else True:
            best_macro = macro_f1
            best_candidate = row
            labels = sorted(set(y_true))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            with (out_dir / "best_confusion_matrix.txt").open("w", encoding="utf-8") as f:
                f.write(f"threshold={threshold}\n")
                f.write(f"accuracy={acc:.6f}\n")
                f.write(f"macro_f1={macro_f1:.6f}\n")
                f.write("labels=" + ",".join(labels) + "\n")
                for i, label in enumerate(labels):
                    f.write(label + "," + ",".join(str(int(cm[i, j])) for j in range(len(labels))) + "\n")
            Path(out_dir / "best_features.txt").write_text("\n".join(kept) + "\n", encoding="utf-8")

    write_csv(out_dir / "correlation_pruning_summary.csv", summary_rows)
    write_csv(out_dir / "removed_correlated_features.csv", all_removed)
    save_sweep_plot(summary_rows, out_dir / "correlation_pruning_sweep.png")

    print(f"Input features: {len(features)}")
    print("\nCorrelation pruning sweep:")
    print(f"{'thr':>6} {'nfeat':>6} {'removed':>7} {'acc':>8} {'macro_f1':>9}")
    for row in summary_rows:
        print(
            f"{float(row['threshold']):>6.2f} {int(row['n_features']):>6} "
            f"{int(row['removed']):>7} {float(row['accuracy']):>8.3f} {float(row['macro_f1']):>9.3f}"
        )
    if best_candidate:
        print("\nBest reduced set:")
        print(
            f"  threshold={float(best_candidate['threshold']):.2f} "
            f"n_features={int(best_candidate['n_features'])} "
            f"accuracy={float(best_candidate['accuracy']):.3f} "
            f"macro_f1={float(best_candidate['macro_f1']):.3f}"
        )
    print(f"\nout_dir: {out_dir}")
    print(f"  - {out_dir / 'feature_correlation_heatmap.png'}")
    print(f"  - {out_dir / 'correlation_pruning_summary.csv'}")
    print(f"  - {out_dir / 'removed_correlated_features.csv'}")
    print(f"  - {out_dir / 'correlation_pruning_sweep.png'}")
    print(f"  - {out_dir / 'best_features.txt'}")


if __name__ == "__main__":
    main()
