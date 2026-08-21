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
import os
import warnings
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


warnings.filterwarnings(
    "ignore",
    message=r".*Parameters: \{ \"use_label_encoder\" \} are not used.*",
    category=UserWarning,
)

META_COLS = 6
METADATA_PREFIXES = ("scenario_is_", "window_is_", "phase_is_",
                     "step_is_multi_ue", "step_num_qhats", "step_parallel",
                     "step_total_parallel_flows")

# Features requiring the eBPF latency probe (not accessible to a standard operator)
EBPF_PREFIXES = (
    "direct_",           # gtp_teid_latency_* — GTP direct RTT
    "same_packet_",      # gtp_same_packet_* — gNB/UPF correlated RTT
    "tcp_data_observed", # gtp_tcp_data_observed_*
    "tcp_ack_observed",  # gtp_tcp_ack_observed_*
    "tcp_rtt_emitted",   # gtp_tcp_rtt_emitted_*
    "tcp_evict",         # gtp_tcp_pending_evicted_* (+ augmented ratio)
    "tcp_no_match",      # gtp_tcp_ack_no_match_* (+ augmented ratio)
)


ML_N_JOBS = int(os.environ.get("ML_N_JOBS", "1"))


def is_ebpf_feature(name):
    return any(name.startswith(p) for p in EBPF_PREFIXES)


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
        n_estimators=300, random_state=42, n_jobs=ML_N_JOBS, class_weight="balanced"
    )


def make_xgb(n_classes, label_encoder):
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=ML_N_JOBS,
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
        n_estimators=300, random_state=42, n_jobs=ML_N_JOBS
    )
    xgb = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42, n_jobs=ML_N_JOBS,
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


RUN_COMMANDS = {
    "controlled_delay": (
        "ansible-playbook -i ./inventory/default/hosts.ini playbooks/run_latency_validation.yml "
        "-e validation_scenario_names=28_controlled_delay_levels "
        "-e core=open5gs -e ebpf_probe_preflight_enabled=false"
    ),
    "upf_stress": (
        "ansible-playbook -i ./inventory/default/hosts.ini playbooks/run_tcp_paper_scenarios.yml "
        "-e paper_scenario_names=24_upf_stress_levels "
        "-e core=open5gs -e ebpf_probe_preflight_enabled=false"
    ),
    "radio_interference": (
        "ansible-playbook -i ./inventory/default/hosts.ini playbooks/run_tcp_paper_scenarios.yml "
        "-e paper_scenario_names=26_interference_gain_levels "
        "-e core=open5gs -e ebpf_probe_preflight_enabled=false"
    ),
    "server_stress": (
        "ansible-playbook -i ./inventory/default/hosts.ini playbooks/run_tcp_paper_scenarios.yml "
        "-e paper_scenario_names=25_server_stress_levels "
        "-e core=open5gs -e ebpf_probe_preflight_enabled=false"
    ),
    "tunnel_packet_loss": (
        "ansible-playbook -i ./inventory/default/hosts.ini playbooks/run_latency_validation.yml "
        "-e validation_scenario_names=30_packet_loss_levels "
        "-e core=open5gs -e ebpf_probe_preflight_enabled=false"
    ),
    "clean_traffic": (
        "ansible-playbook -i ./inventory/default/hosts.ini playbooks/run_latency_validation.yml "
        "-e validation_scenario_names=33_streaming_baseline "
        "-e core=open5gs -e ebpf_probe_preflight_enabled=false"
    ),
}


def print_reinforcement_plan(y_true, y_pred, target_support=30):
    labels = sorted(set(y_true))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rows = []
    for i, label in enumerate(labels):
        support = int(report[label]["support"])
        recall = float(report[label]["recall"])
        f1 = float(report[label]["f1-score"])

        off_diag = [(labels[j], int(cm[i, j])) for j in range(len(labels)) if j != i and cm[i, j] > 0]
        off_diag.sort(key=lambda item: item[1], reverse=True)
        main_confusion = off_diag[0] if off_diag else ("-", 0)

        missing = max(target_support - support, 0)
        priority = 0
        if missing:
            priority += missing / max(target_support, 1)
        if recall < 0.75:
            priority += (0.75 - recall) * 2
        if f1 < 0.80:
            priority += (0.80 - f1) * 2
        if main_confusion[1] >= 3:
            priority += min(main_confusion[1] / max(support, 1), 1.0)

        rows.append({
            "label": label,
            "support": support,
            "missing": missing,
            "recall": recall,
            "f1": f1,
            "confused_with": main_confusion[0],
            "confused_n": main_confusion[1],
            "priority": priority,
        })

    rows.sort(key=lambda r: (-r["priority"], r["support"], r["f1"]))

    print(f"\n{'='*65}")
    print("  CLASSES TO REINFORCE")
    print(f"{'='*65}")
    print(f"  Target support per class: {target_support}")
    print(f"  {'Class':<22} {'n':>4} {'need':>5} {'recall':>7} {'F1':>7} {'main confusion':<26}")
    print(f"  {'-'*82}")
    for row in rows:
        if row["priority"] <= 0:
            continue
        confusion = (
            f"{row['confused_with']} ({row['confused_n']})"
            if row["confused_n"]
            else "-"
        )
        print(
            f"  {row['label']:<22} {row['support']:>4} {row['missing']:>5} "
            f"{row['recall']:>7.2f} {row['f1']:>7.2f} {confusion:<26}"
        )

    print(f"\n{'='*65}")
    print("  NEXT RUNS")
    print(f"{'='*65}")
    for row in rows:
        if row["priority"] <= 0:
            continue
        cmd = RUN_COMMANDS.get(row["label"])
        if not cmd:
            continue
        reasons = []
        if row["missing"]:
            reasons.append(f"need {row['missing']} more windows")
        if row["recall"] < 0.75:
            reasons.append(f"low recall {row['recall']:.2f}")
        if row["f1"] < 0.80:
            reasons.append(f"low F1 {row['f1']:.2f}")
        if row["confused_n"]:
            reasons.append(f"confused with {row['confused_with']} ({row['confused_n']})")
        print(f"\n# {row['label']}: {', '.join(reasons)}")
        print(cmd)


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
    if X_sub2.shape[0] == 0:
        print("  (aucune donnée tunnel_packet_loss/controlled_delay — analyse ignorée)")
        return
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
    ap.add_argument("--exclude-classes", default="",
                    help="Comma-separated list of classes to exclude, e.g. load_ramp,server_stress")
    ap.add_argument("--standard-only", action="store_true",
                    help="Use only STANDARD features (no eBPF probe required). "
                         "Shows classifier performance without the ebpf-latency-probe.")
    ap.add_argument("--target-support", type=int, default=30,
                    help="Minimum desired windows per class for the reinforcement plan.")
    ap.add_argument("--no-recommend", action="store_true",
                    help="Do not print the automatic class reinforcement plan.")
    args = ap.parse_args()

    exclude = {c.strip() for c in args.exclude_classes.split(",") if c.strip()}

    print(f"Loading {args.csv} ...")
    header, rows = load_csv(args.csv)
    if exclude:
        rows = [r for r in rows if r[2].strip() not in exclude]
        print(f"  Excluded classes: {sorted(exclude)}")
    feature_names = header[META_COLS:]
    n_features = len(feature_names)
    print(f"  {len(rows)} samples, {n_features} features")

    observable_idx = [i for i, n in enumerate(feature_names) if not is_metadata(n)]
    nan_rates = compute_nan_rates(rows, n_features)

    # Always include tcp_evict / tcp_no_match / dl_mcs regardless of NaN rate or ranking
    # In --standard-only mode, drop the eBPF-derived forced patterns
    if args.standard_only:
        FORCED_PATTERNS = ("dl_mcs", "dl_bler")
        print("  [standard-only] Excluding all eBPF features (direct_*, same_packet_*, tcp_evict*, tcp_no_match*, ...)")
    else:
        FORCED_PATTERNS = ("tcp_evict", "tcp_no_match", "dl_mcs", "dl_bler")

    def is_forced_feat(name):
        return any(p in name for p in FORCED_PATTERNS)

    clean_idx = sorted(set(
        [i for i in observable_idx if nan_rates[i] <= args.nan_threshold] +
        [i for i in observable_idx if is_forced_feat(feature_names[i])]
    ))
    if args.standard_only:
        clean_idx = [i for i in clean_idx if not is_ebpf_feature(feature_names[i])]
    print(f"  Observable + low-NaN + forced: {len(clean_idx)} features")

    X_clean, y, runs = build_matrix(rows, clean_idx)
    fnames_clean = [feature_names[i] for i in clean_idx]
    labels_sorted = sorted(set(y))

    # Add second/first half ratio features (temporal trend — detects load_ramp ramp-up)
    ratio_names, ratio_cols = [], []
    name_to_idx = {n: i for i, n in enumerate(fnames_clean)}
    for i, name in enumerate(fnames_clean):
        if '_second_half_mean' in name:
            base = name.replace('_second_half_mean', '_first_half_mean')
            if base in name_to_idx:
                j = name_to_idx[base]
                denom = np.where(np.abs(X_clean[:, j]) > 1e-3, X_clean[:, j], np.nan)
                ratio_col = X_clean[:, i] / denom
                ratio_names.append(name.replace('_second_half_mean', '_half_ratio'))
                ratio_cols.append(ratio_col)

    if ratio_cols:
        X_clean = np.hstack([X_clean, np.column_stack(ratio_cols)])
        fnames_clean = fnames_clean + ratio_names
        print(f"  Added {len(ratio_names)} temporal ratio features")

    # Rank by importance using RF (on extended feature set)
    print("\nRanking features by importance ...")
    ranked = feature_importance_global(X_clean, y, fnames_clean, make_rf())

    results = []
    best_predictions = None

    for top_n in [40, 60]:
        top_names_ranked = [name for name, _ in ranked[:top_n]]
        # Always add forced tcp_evict features not already in ranking
        forced = [f for f in fnames_clean
                  if is_forced_feat(f) and f not in top_names_ranked]
        top_names = top_names_ranked + forced

        top_idx = [fnames_clean.index(n) for n in top_names]
        X_top = X_clean[:, top_idx]
        y_top, runs_top = y, runs

        label = f" (+{len(forced)} forced tcp_evict)" if forced else ""
        print(f"\n--- top-{top_n} features{label} ---")

        # 1. Random Forest
        y_true, y_pred = leave_one_run_out(X_top, y_top, runs_top, make_rf)
        acc, mf1 = print_results(f"RF    top-{top_n}", y_true, y_pred,
                                  verbose=(top_n == 40))
        results.append((f"RF    top-{top_n}", acc, mf1, y_true, y_pred))

        # 2. XGBoost
        def xgb_factory():
            return XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42, n_jobs=ML_N_JOBS,
            )
        y_true, y_pred = leave_one_run_out(X_top, y_top, runs_top, xgb_factory)
        acc, mf1 = print_results(f"XGB   top-{top_n}", y_true, y_pred,
                                  verbose=(top_n == 40))
        results.append((f"XGB   top-{top_n}", acc, mf1, y_true, y_pred))

        # 3. Ensemble RF+XGB
        def ens_factory():
            rf = RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=ML_N_JOBS
            )
            xgb = XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42, n_jobs=ML_N_JOBS,
            )
            return VotingClassifier(
                estimators=[("rf", rf), ("xgb", xgb)], voting="soft"
            )
        y_true_e, y_pred_e = leave_one_run_out(X_top, y_top, runs_top, ens_factory)
        acc, mf1 = print_results(f"ENS   top-{top_n}", y_true_e, y_pred_e,
                                  verbose=(top_n == 40))
        results.append((f"ENS   top-{top_n}", acc, mf1, y_true_e, y_pred_e))

        if args.analyze_errors and top_n == 40:
            analyze_errors(X_top, y_top, runs_top, fnames_clean, top_names_ranked)

    # Summary table
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Config':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"  {'-'*42}")
    best_acc = max(r[1] for r in results)
    for tag, acc, mf1, y_true, y_pred in sorted(results, key=lambda x: (-x[1], -x[2])):
        marker = " <-- best" if acc == best_acc else ""
        print(f"  {tag:<20} {acc:>10.3f} {mf1:>10.3f}{marker}")
        if best_predictions is None and acc == best_acc:
            best_predictions = (y_true, y_pred)

    if best_predictions and not args.no_recommend:
        print_reinforcement_plan(
            best_predictions[0],
            best_predictions[1],
            target_support=args.target_support,
        )


if __name__ == "__main__":
    main()
