#!/usr/bin/env python3
"""
Generate ML analysis plots for 5G RCA paper.

Produces:
  1. confusion_matrix_xgb.png   - normalized confusion matrix (best model)
  2. f1_per_class.png           - F1 per class for RF / XGB / ENS
  3. model_comparison.png       - accuracy + macro-F1 for all configs
  4. feature_importance.png     - top-20 RF feature importances
  5. class_distribution.png     - samples per class

Usage:
    python3 scripts/ml/plot_results.py
    python3 scripts/ml/plot_results.py --csv results/ml_dataset_v3/window_features_no_loadramp.csv
    python3 scripts/ml/plot_results.py --out-dir results/ml_plots
"""
import argparse
import csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

META_COLS = 6
METADATA_PREFIXES = ("scenario_is_", "window_is_", "phase_is_",
                     "step_is_multi_ue", "step_num_qhats", "step_parallel",
                     "step_total_parallel_flows")
FORCED_PATTERNS = ("tcp_evict", "tcp_no_match", "dl_mcs", "dl_bler")

CLASS_LABELS = {
    "clean_traffic":       "clean",
    "controlled_delay":    "ctrl_delay",
    "far_ue_poor_radio":   "far_ue",
    "multi_ue_contention": "multi_ue",
    "radio_interference":  "radio_interf",
    "server_stress":       "srv_stress",
    "tunnel_packet_loss":  "pkt_loss",
    "upf_stress":          "upf_stress",
    "load_ramp":           "load_ramp",
}

COLORS = {
    "RF":  "#2196F3",
    "XGB": "#FF9800",
    "ENS": "#4CAF50",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers (mirrors compare_classifiers.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(filepath):
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) > META_COLS:
                rows.append(row)
    return header, rows


def is_metadata(name):
    return any(name.startswith(p) or name == p.rstrip("_") for p in METADATA_PREFIXES)


def is_forced(name):
    return any(p in name for p in FORCED_PATTERNS)


def compute_nan_rates(rows, n_features):
    n = len(rows)
    counts = np.zeros(n_features)
    for row in rows:
        for i in range(n_features):
            v = row[META_COLS + i].strip()
            if v == "" or v.lower() == "nan":
                counts[i] += 1
    return counts / n * 100


def build_matrix(rows, feature_indices):
    X = np.full((len(rows), len(feature_indices)), np.nan)
    y, runs = [], []
    for r, row in enumerate(rows):
        y.append(row[2].strip())
        runs.append(row[1].strip())
        for c, fi in enumerate(feature_indices):
            v = row[META_COLS + fi].strip()
            if v and v.lower() != "nan":
                try:
                    X[r, c] = float(v)
                except ValueError:
                    pass
    return X, np.array(y), np.array(runs)


def add_ratio_features(X, fnames):
    name_to_idx = {n: i for i, n in enumerate(fnames)}
    ratio_names, ratio_cols = [], []
    for i, name in enumerate(fnames):
        if "_second_half_mean" in name:
            base = name.replace("_second_half_mean", "_first_half_mean")
            if base in name_to_idx:
                j = name_to_idx[base]
                denom = np.where(np.abs(X[:, j]) > 1e-3, X[:, j], np.nan)
                ratio_names.append(name.replace("_second_half_mean", "_half_ratio"))
                ratio_cols.append(X[:, i] / denom)
    if ratio_cols:
        X = np.hstack([X, np.column_stack(ratio_cols)])
        fnames = fnames + ratio_names
    return X, fnames


def rank_features(X, y, fnames):
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                class_weight="balanced")
    rf.fit(X_imp, y)
    idx = np.argsort(rf.feature_importances_)[::-1]
    return [(fnames[i], rf.feature_importances_[i]) for i in idx]


def leave_one_run_out(X, y, runs, clf_factory):
    all_true, all_pred = [], []
    for test_run in np.unique(runs):
        test_mask = runs == test_run
        train_mask = ~test_mask
        if not train_mask.any() or not test_mask.any():
            continue
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)
        clf = clf_factory()
        if isinstance(clf, (XGBClassifier, VotingClassifier)):
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", y_tr)
            fold_labels = sorted(set(y_tr))
            le = {lbl: i for i, lbl in enumerate(fold_labels)}
            clf.fit(X_tr, np.array([le[l] for l in y_tr]), sample_weight=sw)
            preds = np.array([fold_labels[i] for i in clf.predict(X_te)])
        else:
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
        all_true.extend(y_te)
        all_pred.extend(preds)
    return np.array(all_true), np.array(all_pred)


# ──────────────────────────────────────────────────────────────────────────────
# Plot functions
# ──────────────────────────────────────────────────────────────────────────────

def shorten(labels):
    return [CLASS_LABELS.get(l, l) for l in labels]


def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / np.where(cm.sum(axis=1, keepdims=True) == 0, 1,
                                           cm.sum(axis=1, keepdims=True))
    sl = shorten(labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(sl, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(sl, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm_norm[i, j]
            color = "white" if v > 0.55 else "black"
            ax.text(j, i, f"{v:.2f}\n({cm[i,j]})", ha="center", va="center",
                    fontsize=8, color=color)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def plot_f1_per_class(model_results, labels, path):
    models = list(model_results.keys())
    n = len(labels)
    x = np.arange(n)
    width = 0.25
    sl = shorten(labels)

    fig, ax = plt.subplots(figsize=(13, 6))
    palette = list(COLORS.values())
    for k, (model_name, (yt, yp)) in enumerate(model_results.items()):
        f1s = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
        short_model = model_name.split()[0]
        bars = ax.bar(x + k * width, f1s, width, label=short_model,
                      color=palette[k], alpha=0.88, edgecolor="white")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sl, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("F1-score", fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="F1=0.80")
    ax.set_title("F1-score par classe — Leave-One-Run-Out", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def plot_model_comparison(results, path):
    tags = [r[0] for r in results]
    accs = [r[1] for r in results]
    mf1s = [r[2] for r in results]
    x = np.arange(len(tags))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#2196F3", alpha=0.88)
    b2 = ax.bar(x + width / 2, mf1s, width, label="Macro-F1", color="#FF9800", alpha=0.88)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title("Comparaison des modèles (LORO cross-validation)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def plot_feature_importance(ranked, top_n, path):
    names = [r[0] for r in ranked[:top_n]][::-1]
    scores = [r[1] for r in ranked[:top_n]][::-1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(range(len(names)), scores, color=colors, alpha=0.88)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance (RF Gini)", fontsize=11)
    ax.set_title(f"Top-{top_n} features — Random Forest importance", fontsize=13, fontweight="bold")
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


def plot_class_distribution(y, path):
    counts = Counter(y)
    labels = sorted(counts.keys())
    sl = shorten(labels)
    values = [counts[l] for l in labels]
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.88, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(sl, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Nombre de fenêtres", fontsize=12)
    ax.set_title("Distribution des classes (dataset complet)", fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/ml_dataset_v3/window_features_no_loadramp.csv")
    ap.add_argument("--out-dir", default="results/ml_plots")
    ap.add_argument("--nan-threshold", type=float, default=10.0)
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.csv} ...")
    header, rows = load_csv(args.csv)
    feature_names = header[META_COLS:]
    print(f"  {len(rows)} samples, {len(feature_names)} features")

    observable_idx = [i for i, n in enumerate(feature_names) if not is_metadata(n)]
    nan_rates = compute_nan_rates(rows, len(feature_names))

    clean_idx = sorted(set(
        [i for i in observable_idx if nan_rates[i] <= args.nan_threshold] +
        [i for i in observable_idx if is_forced(feature_names[i])]
    ))
    print(f"  Clean + forced features: {len(clean_idx)}")

    X, y, runs = build_matrix(rows, clean_idx)
    fnames = [feature_names[i] for i in clean_idx]
    X, fnames = add_ratio_features(X, fnames)

    print("Ranking features ...")
    ranked = rank_features(X, y, fnames)

    top_names_ranked = [n for n, _ in ranked[:args.top_n]]
    forced_extra = [f for f in fnames if is_forced(f) and f not in top_names_ranked]
    top_names = top_names_ranked + forced_extra
    top_idx = [fnames.index(n) for n in top_names]
    X_top = X[:, top_idx]
    labels = sorted(set(y))

    def xgb_factory():
        return XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )

    def ens_factory():
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        )
        return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft")

    def rf_factory():
        return RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        )

    print("Running LORO classifiers ...")
    print("  RF ...")
    yt_rf, yp_rf = leave_one_run_out(X_top, y, runs, rf_factory)
    print(f"    Acc={(yt_rf==yp_rf).mean():.3f}  F1={f1_score(yt_rf,yp_rf,average='macro',zero_division=0):.3f}")

    print("  XGB ...")
    yt_xgb, yp_xgb = leave_one_run_out(X_top, y, runs, xgb_factory)
    print(f"    Acc={(yt_xgb==yp_xgb).mean():.3f}  F1={f1_score(yt_xgb,yp_xgb,average='macro',zero_division=0):.3f}")

    print("  ENS ...")
    yt_ens, yp_ens = leave_one_run_out(X_top, y, runs, ens_factory)
    print(f"    Acc={(yt_ens==yp_ens).mean():.3f}  F1={f1_score(yt_ens,yp_ens,average='macro',zero_division=0):.3f}")

    tag = f"top-{args.top_n}"
    results = [
        (f"RF {tag}",  (yt_rf==yp_rf).mean(),   f1_score(yt_rf,  yp_rf,  average="macro", zero_division=0)),
        (f"XGB {tag}", (yt_xgb==yp_xgb).mean(), f1_score(yt_xgb, yp_xgb, average="macro", zero_division=0)),
        (f"ENS {tag}", (yt_ens==yp_ens).mean(), f1_score(yt_ens, yp_ens, average="macro", zero_division=0)),
    ]
    model_results = {
        f"RF {tag}":  (yt_rf,  yp_rf),
        f"XGB {tag}": (yt_xgb, yp_xgb),
        f"ENS {tag}": (yt_ens, yp_ens),
    }

    print(f"\nGenerating plots → {out_dir}/")
    plot_confusion_matrix(yt_xgb, yp_xgb, labels,
                          f"Confusion matrix — XGB {tag} (LORO)",
                          out_dir / "confusion_matrix_xgb.png")
    plot_confusion_matrix(yt_rf, yp_rf, labels,
                          f"Confusion matrix — RF {tag} (LORO)",
                          out_dir / "confusion_matrix_rf.png")
    plot_f1_per_class(model_results, labels, out_dir / "f1_per_class.png")
    plot_model_comparison(results, out_dir / "model_comparison.png")
    plot_feature_importance(ranked, 20, out_dir / "feature_importance.png")
    plot_class_distribution(y, out_dir / "class_distribution.png")

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots saved in {out_dir}/")


if __name__ == "__main__":
    main()
