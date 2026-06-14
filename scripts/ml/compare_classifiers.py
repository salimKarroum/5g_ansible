#!/usr/bin/env python3
"""
Compare classifiers on the augmented dataset.

Tests:
  1. Random Forest   top-40 (baseline)
  2. Random Forest   top-60
  3. XGBoost         top-40
  4. XGBoost         top-60
  5. RF+XGB ensemble top-40
  6. RF+XGB ensemble top-60

Usage:
    python3 scripts/ml/compare_classifiers.py
    python3 scripts/ml/compare_classifiers.py --csv results/ml_dataset_v3/window_features_augmented.csv
    python3 scripts/ml/compare_classifiers.py --analyze-errors
"""
import argparse
import csv
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

META_COLS = 6
METADATA_PREFIXES = ("scenario_is_", "window_is_", "phase_is_",
                     "step_is_multi_ue", "step_num_qhats", "step_parallel",
                     "step_total_parallel_flows")


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


def compute_nan_rates(rows, n_features):
    n = len(rows)
    nan_counts = np.zeros(n_features)
    for row in rows:
        for i in range(n_features):
            v = row[META_COLS + i].strip()
            if v == "" or v.lower() == "nan":
                nan_counts[i] += 1
    return nan_counts / n * 100


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


def feature_importance_global(X, y, feature_names, clf):
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    clf.fit(X_imp, y)
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.zeros(len(feature_names))
    idx = np.argsort(importances)[::-1]
    return [(feature_names[i], importances[i]) for i in idx]


def make_rf():
    return RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    )


def make_xgb(n_classes, label_encoder):
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        # XGBoost doesn't support class_weight directly; we handle via sample_weight
    )


def leave_one_run_out(X, y, runs, clf_factory, label_encoder=None):
    unique_runs = np.unique(runs)
    all_true, all_pred = [], []
    labels_sorted = sorted(set(y))

    for test_run in unique_runs:
        test_mask = runs == test_run
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        clf = clf_factory()

        # Compute sample weights for XGBoost to mimic class_weight="balanced"
        if isinstance(clf, (XGBClassifier, VotingClassifier)):
            from sklearn.utils.class_weight import compute_sample_weight
            sw = compute_sample_weight("balanced", y_train)
            # Use fold-local label encoding so XGBoost always gets consecutive ints.
            # A class absent from training (e.g. only 1 run) is simply not predicted.
            fold_labels = sorted(set(y_train))
            le_map = {lbl: i for i, lbl in enumerate(fold_labels)}
            y_train_enc = np.array([le_map[l] for l in y_train])
            clf.fit(X_train, y_train_enc, sample_weight=sw)
            preds_enc = clf.predict(X_test)
            preds = np.array([fold_labels[i] for i in preds_enc])
        else:
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)

        all_true.extend(y_test)
        all_pred.extend(preds)

    return np.array(all_true), np.array(all_pred)


def make_ensemble(labels_sorted):
    rf = RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
        num_class=len(labels_sorted)
    )
    return VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft"
    )


def print_results(tag, y_true, y_pred, verbose=True):
    labels = sorted(set(y_true))
    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'='*65}")
    print(f"  {tag}")
    print(f"  Accuracy: {acc:.3f}  Macro-F1: {macro_f1:.3f}  ({(y_true==y_pred).sum()}/{len(y_true)})")
    print(f"{'='*65}")
    if verbose:
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("Confusion matrix:")
        print(f"  {'':>22}", "  ".join(f"{l[:6]:>6}" for l in labels))
        for i, rl in enumerate(labels):
            print(f"  {rl:>22}", "  ".join(f"{cm[i,j]:>6}" for j in range(len(labels))))
    return acc, macro_f1


def analyze_errors(X_top, y, runs, feature_names, top_names):
    """Print statistics comparing correct vs misclassified samples."""
    print(f"\n{'='*65}")
    print("  ERROR ANALYSIS: load_ramp vs far_ue_poor_radio")
    print(f"{'='*65}")

    # Focus on load_ramp and far_ue_poor_radio
    mask = np.isin(y, ["load_ramp", "far_ue_poor_radio"])
    X_sub = X_top[mask]
    y_sub = y[mask]

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_sub)

    # Find RSRP and spread features
    rsrp_feats = [(i, n) for i, n in enumerate(top_names)
                  if "rsrp" in n.lower() or "spread" in n.lower()]

    print(f"\n  RSRP/spread features comparison (mean value per class):")
    print(f"  {'Feature':<45} {'load_ramp':>12} {'far_ue':>12} {'delta':>10}")
    print(f"  {'-'*80}")
    for i, name in rsrp_feats[:12]:
        lr_vals = X_imp[y_sub == "load_ramp", i]
        fu_vals = X_imp[y_sub == "far_ue_poor_radio", i]
        lr_mean = np.nanmean(lr_vals) if len(lr_vals) > 0 else float("nan")
        fu_mean = np.nanmean(fu_vals) if len(fu_vals) > 0 else float("nan")
        delta = fu_mean - lr_mean
        print(f"  {name:<45} {lr_mean:>12.2f} {fu_mean:>12.2f} {delta:>+10.2f}")

    print(f"\n{'='*65}")
    print("  ERROR ANALYSIS: tunnel_packet_loss vs controlled_delay")
    print(f"{'='*65}")

    mask2 = np.isin(y, ["tunnel_packet_loss", "controlled_delay"])
    X_sub2 = X_top[mask2]
    y_sub2 = y[mask2]
    X_imp2 = imputer.fit_transform(X_sub2)

    tcp_feats = [(i, n) for i, n in enumerate(top_names)
                 if "tcp" in n.lower() or "evict" in n.lower() or "delay" in n.lower()
                 or "rtt" in n.lower() or "bler" in n.lower() or "mcs" in n.lower()]

    print(f"\n  TCP/BLER/MCS features comparison (mean value per class):")
    print(f"  {'Feature':<45} {'pkt_loss':>12} {'delay':>12} {'delta':>10}")
    print(f"  {'-'*80}")
    for i, name in tcp_feats[:12]:
        pl_vals = X_imp2[y_sub2 == "tunnel_packet_loss", i]
        cd_vals = X_imp2[y_sub2 == "controlled_delay", i]
        pl_mean = np.nanmean(pl_vals) if len(pl_vals) > 0 else float("nan")
        cd_mean = np.nanmean(cd_vals) if len(cd_vals) > 0 else float("nan")
        delta = pl_mean - cd_mean
        print(f"  {name:<45} {pl_mean:>12.2f} {cd_mean:>12.2f} {delta:>+10.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
                    default="results/ml_dataset_v3/window_features_augmented.csv")
    ap.add_argument("--nan-threshold", type=float, default=10.0)
    ap.add_argument("--analyze-errors", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.csv} ...")
    header, rows = load_csv(args.csv)
    feature_names = header[META_COLS:]
    n_features = len(feature_names)
    print(f"  {len(rows)} samples, {n_features} features")

    observable_idx = [i for i, n in enumerate(feature_names) if not is_metadata(n)]
    nan_rates = compute_nan_rates(rows, n_features)
    clean_idx = [i for i in observable_idx if nan_rates[i] <= args.nan_threshold]
    print(f"  Observable + low-NaN: {len(clean_idx)} features")

    X_clean, y, runs = build_matrix(rows, clean_idx)
    fnames_clean = [feature_names[i] for i in clean_idx]
    labels_sorted = sorted(set(y))

    # Rank by importance using RF
    print("\nRanking features by importance ...")
    ranked = feature_importance_global(X_clean, y, fnames_clean, make_rf())

    results = []

    for top_n in [40, 60]:
        top_names = [name for name, _ in ranked[:top_n]]
        top_idx = [feature_names.index(n) for n in top_names]
        X_top, y_top, runs_top = build_matrix(rows, top_idx)

        print(f"\n--- top-{top_n} features ---")

        # 1. Random Forest
        y_true, y_pred = leave_one_run_out(X_top, y_top, runs_top, make_rf)
        acc, mf1 = print_results(f"RF    top-{top_n}", y_true, y_pred,
                                  verbose=(top_n == 40))
        results.append((f"RF    top-{top_n}", acc, mf1))

        # 2. XGBoost
        def xgb_factory():
            return XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1,
            )
        y_true, y_pred = leave_one_run_out(X_top, y_top, runs_top, xgb_factory)
        acc, mf1 = print_results(f"XGB   top-{top_n}", y_true, y_pred,
                                  verbose=(top_n == 40))
        results.append((f"XGB   top-{top_n}", acc, mf1))

        # 3. Ensemble RF+XGB
        def ens_factory():
            rf = RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            )
            xgb = XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1,
            )
            return VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)], voting="soft"
            )
        y_true_e, y_pred_e = leave_one_run_out(X_top, y_top, runs_top, ens_factory)
        acc, mf1 = print_results(f"ENS   top-{top_n}", y_true_e, y_pred_e,
                                  verbose=(top_n == 40))
        results.append((f"ENS   top-{top_n}", acc, mf1))

        if args.analyze_errors and top_n == 40:
            analyze_errors(X_top, y_top, runs_top, fnames_clean, top_names)

    # Summary table
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Config':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"  {'-'*42}")
    for tag, acc, mf1 in sorted(results, key=lambda x: -x[1]):
        marker = " <-- best" if acc == max(r[1] for r in results) else ""
        print(f"  {tag:<20} {acc:>10.3f} {mf1:>10.3f}{marker}")


if __name__ == "__main__":
    main()
