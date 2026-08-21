#!/usr/bin/env python3
"""Mechanism-clean RCA feature reduction.

This script keeps the same scientific validation principle as
reduce_physical_features_loro.py, but makes two constraints stricter:

* tunnel_bandwidth_limit must be represented by iperf throughput features;
* far_ue_poor_radio must be represented by radio-quality features, not PRACH I0.

External validation runs, such as OAI core + OAI RAN, must not be used here.
They remain hold-out tests after the feature list has been selected.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


META_COLUMNS = {"window_id", "run", "label", "scenario", "phase", "source_path"}


@dataclass(frozen=True)
class MechanismRule:
    family: str
    min_features: int


MECHANISM_RULES: dict[str, list[MechanismRule]] = {
    "controlled_delay": [MechanismRule("latency", 3)],
    "controlled_jitter": [MechanismRule("latency", 3)],
    "far_ue_poor_radio": [MechanismRule("radio_quality", 5)],
    "radio_interference": [MechanismRule("radio_noise", 4)],
    "tunnel_bandwidth_limit": [MechanismRule("throughput", 4)],
    "tunnel_packet_loss": [MechanismRule("loss_retransmit", 3)],
    "upf_stress": [MechanismRule("upf_compute", 4)],
}


NAME_AUDIT_WORDS = (
    "label",
    "scenario",
    "phase",
    "run",
    "path",
    "reference",
    "tc_value",
    "tc_mode",
    "level",
    "valid_for_dataset",
    "invalid_reason",
    "clean",
    "interference",
    "stress",
    "delay_",
    "jitter_",
    "rate_",
)


def read_features(path: Path, columns: set[str]) -> list[str]:
    features: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        name = name.split(",", 1)[0].strip()
        if name in columns and name not in META_COLUMNS:
            features.append(name)
    if not features:
        raise SystemExit(f"No usable features from {path}")
    return list(dict.fromkeys(features))


def feature_family(feature: str) -> str:
    low = feature.lower()

    if low.startswith("upf_cpu_"):
        return "upf_compute"

    if "prach_i0" in low or "i0_db" in low or "i0_noise" in low:
        return "radio_noise"

    if any(x in low for x in ("rsrp", "cqi", "mcs", "bler", "snr")):
        return "radio_quality"

    if "iperf" in low and "mbps" in low:
        return "throughput"

    if any(x in low for x in ("packet_loss", "packets_lost", "retransmits")):
        return "loss_retransmit"

    if low.startswith("direct_") or "rtt_ms" in low or "latest_ms" in low or "mean_ms" in low:
        return "latency"

    return "other"


def make_rf(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate_loro(
    X: pd.DataFrame,
    y: np.ndarray,
    runs: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> tuple[list[str], list[str]]:
    y_true: list[str] = []
    y_pred: list[str] = []
    X_values = X.apply(pd.to_numeric, errors="coerce").to_numpy()

    for run in sorted(set(runs)):
        test = runs == run
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_values[train])
        X_test = imputer.transform(X_values[test])

        clf = make_rf(n_estimators, random_state)
        clf.fit(X_train, y[train])
        pred = clf.predict(X_test)

        y_true.extend(y[test].tolist())
        y_pred.extend(pred.tolist())

    return y_true, y_pred


def safe_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> tuple[float, float, dict[str, object]]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return float(acc), float(macro_f1), report


def class_recalls(labels: list[str], y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls: dict[str, float] = {}
    for label in labels:
        mask = y_true == label
        if mask.sum() == 0:
            recalls[label] = float("nan")
        else:
            recalls[label] = float((y_pred[mask] == label).mean())
    return recalls


def loro_permutation_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    runs: np.ndarray,
    features: list[str],
    n_estimators: int,
    random_state: int,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    labels = sorted(set(y))
    X_values = X.apply(pd.to_numeric, errors="coerce").to_numpy()
    raw_rows: list[dict[str, object]] = []

    for fold_idx, run in enumerate(sorted(set(runs))):
        test = runs == run
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_values[train])
        X_test = imputer.transform(X_values[test])

        clf = make_rf(n_estimators, random_state + fold_idx)
        clf.fit(X_train, y[train])

        base_pred = clf.predict(X_test)
        base_macro = f1_score(y[test], base_pred, average="macro", zero_division=0)
        base_recalls = class_recalls(labels, y[test], base_pred)

        for feature_idx, feature in enumerate(features):
            for repeat in range(repeats):
                X_perm = X_test.copy()
                X_perm[:, feature_idx] = rng.permutation(X_perm[:, feature_idx])
                perm_pred = clf.predict(X_perm)
                perm_macro = f1_score(y[test], perm_pred, average="macro", zero_division=0)
                perm_recalls = class_recalls(labels, y[test], perm_pred)

                row: dict[str, object] = {
                    "fold_run": run,
                    "repeat": repeat,
                    "feature": feature,
                    "family": feature_family(feature),
                    "macro_f1_drop": float(base_macro - perm_macro),
                }
                for label in labels:
                    if math.isfinite(base_recalls[label]) and math.isfinite(perm_recalls[label]):
                        row[f"recall_drop__{label}"] = float(base_recalls[label] - perm_recalls[label])
                raw_rows.append(row)

    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        raise SystemExit("No LORO permutation rows produced")

    agg_rows: list[dict[str, object]] = []
    for feature, sub in raw.groupby("feature", sort=False):
        row: dict[str, object] = {
            "feature": feature,
            "family": feature_family(feature),
            "macro_f1_drop_mean": sub["macro_f1_drop"].mean(),
            "macro_f1_drop_std": sub["macro_f1_drop"].std(ddof=0),
            "observations": len(sub),
        }
        for label in labels:
            col = f"recall_drop__{label}"
            if col in sub.columns:
                row[f"recall_drop_mean__{label}"] = sub[col].mean()
                row[f"recall_drop_std__{label}"] = sub[col].std(ddof=0)
        agg_rows.append(row)

    return pd.DataFrame(agg_rows), raw


def rank_for_class(importance: pd.DataFrame, label: str, family: str) -> pd.DataFrame:
    col = f"recall_drop_mean__{label}"
    if col not in importance.columns:
        return importance.iloc[0:0].copy()
    sub = importance[importance["family"] == family].copy()
    if sub.empty:
        return sub
    return sub.sort_values([col, "macro_f1_drop_mean", "feature"], ascending=[False, False, True])


def build_required_picks(
    importance: pd.DataFrame,
    labels: list[str],
) -> tuple[list[str], pd.DataFrame]:
    selected: list[str] = []
    rows: list[dict[str, object]] = []

    for label in labels:
        for rule in MECHANISM_RULES.get(label, []):
            ranked = rank_for_class(importance, label, rule.family)
            if ranked.empty:
                rows.append(
                    {
                        "class": label,
                        "family": rule.family,
                        "rank": 1,
                        "feature": "__MISSING_FAMILY__",
                        "required_min_features": rule.min_features,
                    }
                )
                continue

            for rank, (_, item) in enumerate(ranked.head(rule.min_features).iterrows(), start=1):
                feature = str(item["feature"])
                if feature not in selected:
                    selected.append(feature)
                rows.append(
                    {
                        "class": label,
                        "family": rule.family,
                        "rank": rank,
                        "feature": feature,
                        "required_min_features": rule.min_features,
                        "class_recall_drop_mean": item.get(f"recall_drop_mean__{label}", np.nan),
                        "macro_f1_drop_mean": item.get("macro_f1_drop_mean", np.nan),
                    }
                )

    return selected, pd.DataFrame(rows)


def build_order(
    importance: pd.DataFrame,
    required_features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen = set()

    for feature in required_features:
        item = importance[importance["feature"] == feature].iloc[0]
        rows.append(
            {
                "feature": feature,
                "family": feature_family(feature),
                "selection_block": "required_mechanism",
                "macro_f1_drop_mean": item["macro_f1_drop_mean"],
            }
        )
        seen.add(feature)

    remaining = importance[~importance["feature"].isin(seen)].copy()
    remaining = remaining.sort_values(["macro_f1_drop_mean", "feature"], ascending=[False, True])
    for _, item in remaining.iterrows():
        rows.append(
            {
                "feature": str(item["feature"]),
                "family": str(item["family"]),
                "selection_block": "global_backfill",
                "macro_f1_drop_mean": item["macro_f1_drop_mean"],
            }
        )

    order = pd.DataFrame(rows)
    order.insert(0, "rank", range(1, len(order) + 1))
    return order


def mechanism_detail(selected: list[str], labels: list[str]) -> tuple[bool, dict[str, str]]:
    selected_set = set(selected)
    detail: dict[str, str] = {}
    ok = True

    for label in labels:
        messages: list[str] = []
        for rule in MECHANISM_RULES.get(label, []):
            count = sum(1 for f in selected_set if feature_family(f) == rule.family)
            if count < rule.min_features:
                messages.append(f"{rule.family}:{count}/{rule.min_features}")
        if messages:
            detail[label] = "missing:" + "|".join(messages)
            ok = False
        else:
            detail[label] = "ok"

    return ok, detail


def evaluate_sweep(
    df: pd.DataFrame,
    y: np.ndarray,
    runs: np.ndarray,
    ordered_features: list[str],
    k_values: list[int],
    labels: list[str],
    baseline_macro_f1: float,
    n_estimators: int,
    random_state: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for k in k_values:
        selected = ordered_features[:k]
        y_true, y_pred = evaluate_loro(df[selected], y, runs, n_estimators, random_state)
        acc, macro_f1, report = safe_report(y_true, y_pred, labels)
        recalls = {label: float(report.get(label, {}).get("recall", 0.0)) for label in labels}
        worst_recall = min(recalls.values()) if recalls else 0.0
        mechanism_ok, details = mechanism_detail(selected, labels)

        row: dict[str, object] = {
            "k": k,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "macro_f1_drop_vs_baseline": baseline_macro_f1 - macro_f1,
            "worst_class_recall": worst_recall,
            "mechanism_clean": mechanism_ok,
            "mechanism_detail": json.dumps(details, sort_keys=True),
        }
        for label, recall in recalls.items():
            row[f"recall__{label}"] = recall
        rows.append(row)

    return pd.DataFrame(rows)


def write_mechanism_top4(
    importance: pd.DataFrame,
    labels: list[str],
    selected: list[str],
    out: Path,
) -> pd.DataFrame:
    selected_set = set(selected)
    rows: list[pd.DataFrame] = []

    for label in labels:
        pieces: list[pd.DataFrame] = []
        for rule in MECHANISM_RULES.get(label, []):
            ranked = rank_for_class(importance, label, rule.family)
            ranked = ranked[ranked["feature"].isin(selected_set)].head(4)
            pieces.append(ranked)
        if not pieces:
            continue
        sub = pd.concat(pieces, ignore_index=True)
        sub = sub.drop_duplicates("feature").head(4).copy()
        sub.insert(0, "class", label)
        sub.insert(1, "rank", range(1, len(sub) + 1))
        rows.append(sub)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(out, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exclude-classes", default="")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.02)
    parser.add_argument("--min-worst-recall", type=float, default=0.85)
    parser.add_argument("--k-values", default="20,22,25,28,30,35,44")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv)
    features_path = Path(args.features_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.exclude_classes:
        excluded = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}
        df = df[~df["label"].astype(str).isin(excluded)].copy()

    features = read_features(features_path, set(df.columns))
    y = df["label"].astype(str).to_numpy()
    runs = df["run"].astype(str).to_numpy()
    labels = sorted(set(y))

    y_true, y_pred = evaluate_loro(df[features], y, runs, args.n_estimators, args.random_state)
    baseline_acc, baseline_macro_f1, baseline_report = safe_report(y_true, y_pred, labels)
    baseline_cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=labels,
        columns=labels,
    )

    importance, raw_importance = loro_permutation_importance(
        df[features],
        y,
        runs,
        features,
        args.n_estimators,
        args.random_state,
        args.repeats,
    )
    importance = importance.sort_values(["macro_f1_drop_mean", "feature"], ascending=[False, True])

    required_features, required_table = build_required_picks(importance, labels)
    order = build_order(importance, required_features)
    ordered_features = order["feature"].tolist()

    requested = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    required_k = len(required_features)
    k_values = sorted({k for k in requested if 1 <= k <= len(ordered_features)} | {required_k, len(ordered_features)})

    sweep = evaluate_sweep(
        df,
        y,
        runs,
        ordered_features,
        k_values,
        labels,
        baseline_macro_f1,
        args.n_estimators,
        args.random_state,
    )

    eligible = sweep[
        (sweep["k"] >= required_k)
        & (sweep["macro_f1_drop_vs_baseline"] <= args.max_macro_f1_drop)
        & (sweep["worst_class_recall"] >= args.min_worst_recall)
        & (sweep["mechanism_clean"])
    ].copy()
    if eligible.empty:
        chosen = sweep[sweep["k"] >= required_k].sort_values(
            ["macro_f1", "worst_class_recall"], ascending=False
        ).iloc[0]
        selection_status = "fallback_best_score_no_threshold_match"
    else:
        chosen = eligible.sort_values("k").iloc[0]
        selection_status = "smallest_threshold_match"

    chosen_k = int(chosen["k"])
    selected = ordered_features[:chosen_k]

    pd.DataFrame(baseline_report).transpose().to_csv(out_dir / "baseline44_loro_classification_report.csv")
    baseline_cm.to_csv(out_dir / "baseline44_loro_confusion_matrix.csv")
    raw_importance.to_csv(out_dir / "loro_permutation_importance_raw.csv", index=False)
    importance.to_csv(out_dir / "loro_permutation_importance.csv", index=False)
    required_table.to_csv(out_dir / "mechanism_required_picks_by_class.csv", index=False)
    order.to_csv(out_dir / "mechanism_clean_feature_order.csv", index=False)
    sweep.to_csv(out_dir / "mechanism_clean_sweep.csv", index=False)
    (out_dir / "selected_features_mechanism_clean.txt").write_text(
        "\n".join(selected) + "\n",
        encoding="utf-8",
    )
    top4 = write_mechanism_top4(
        importance,
        labels,
        selected,
        out_dir / "top4_features_by_class_mechanism_clean.csv",
    )

    audit_hits = [f for f in features if any(w in f.lower() for w in NAME_AUDIT_WORDS)]
    readme = f"""# Mechanism-Clean Physical Feature Reduction

Input dataset: `{csv_path}`

Input features: `{features_path}`

Samples: {len(df)}

Classes: {", ".join(labels)}

Runs: {len(set(runs))}

Baseline feature count: {len(features)}

Baseline LORO accuracy: {baseline_acc:.4f}

Baseline LORO macro-F1: {baseline_macro_f1:.4f}

Required mechanism constraints:

{chr(10).join(f"- `{label}`: " + ", ".join(f"{r.family}>={r.min_features}" for r in rules) for label, rules in MECHANISM_RULES.items())}

Selection status: `{selection_status}`

Chosen K: {chosen_k}

Chosen macro-F1: {float(chosen["macro_f1"]):.4f}

Chosen worst-class recall: {float(chosen["worst_class_recall"]):.4f}

Selected feature file: `selected_features_mechanism_clean.txt`

Mechanism top-4 by class: `top4_features_by_class_mechanism_clean.csv`

Name-audit flags from input feature names:

{chr(10).join(f"- `{f}`" for f in audit_hits) if audit_hits else "- none"}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"dataset: {csv_path}")
    print(f"samples: {len(df)}")
    print(f"classes: {', '.join(labels)}")
    print(f"runs: {len(set(runs))}")
    print(f"baseline_features: {len(features)}")
    print(f"baseline_loro_accuracy: {baseline_acc:.3f}")
    print(f"baseline_loro_macro_f1: {baseline_macro_f1:.3f}")
    print(f"required_features: {required_k}")
    print()
    print("=== sweep ===")
    print(
        sweep[
            [
                "k",
                "accuracy",
                "macro_f1",
                "macro_f1_drop_vs_baseline",
                "worst_class_recall",
                "mechanism_clean",
            ]
        ].to_string(index=False)
    )
    print()
    print(f"selection_status: {selection_status}")
    print(f"chosen_k: {chosen_k}")
    print(f"selected_features: {out_dir / 'selected_features_mechanism_clean.txt'}")
    print(f"required_picks: {out_dir / 'mechanism_required_picks_by_class.csv'}")
    print(f"top4_by_class: {out_dir / 'top4_features_by_class_mechanism_clean.csv'}")
    print(f"report: {out_dir / 'README.md'}")
    print()
    print("=== top4 mechanism-clean by class ===")
    if top4.empty:
        print("no rows")
    else:
        cols = [
            "class",
            "rank",
            "feature",
            "family",
            "macro_f1_drop_mean",
        ]
        recall_cols = [c for c in top4.columns if c.startswith("recall_drop_mean__")]
        print(top4[cols + recall_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
