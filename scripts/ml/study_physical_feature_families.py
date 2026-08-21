#!/usr/bin/env python3
"""
Evaluate and explain RCA classification with a physically constrained feature set.

This script does not let the model choose arbitrary statistical shortcuts.  It
selects a small number of features from domain families that match RCA
mechanisms: latency, radio/MAC, UPF compute, server compute, TCP/loss, and
container context.

Usage:
    python3 scripts/ml/study_physical_feature_families.py \
      --csv results/ml_dataset_v3/window_features_augmented_stable_f1_0936.csv \
      --exclude-classes load_ramp,multi_ue_contention \
      --out-dir results/ml_dataset_v3/physical_feature_study
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


warnings.filterwarnings("ignore", category=RuntimeWarning)

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

FAMILIES: dict[str, list[str]] = {
    "latency_ebpf": [
        "direct_latest_ms_median",
        "direct_latest_ms_mean",
        "direct_mean_ms_5s_mean",
        "direct_p95_ms_5s_mean",
        "direct_p99_ms_5s_mean",
        "direct_event_rate_hz_5s_mean",
    ],
    "radio_mac": [
        "rf_oai_gnb_l1_prach_i0_db_mean",
        "prach_i0_db_mean",
        "rf_oai_gnb_mac_avg_rsrp_mean",
        "rf_oai_gnb_mac_ul_mcs_mean",
        "rf_oai_gnb_mac_dl_mcs_mean",
        "rf_oai_gnb_mac_ul_bler_mean",
        "rf_oai_gnb_mac_dl_bler_mean",
        "rf_oai_gnb_mac_cqi_mean",
        "rf_oai_gnb_mac_ul_snr_avg_db_mean",
        "rf_oai_gnb_mac_pucch_snr_avg_db_mean",
    ],
    "upf_compute": [
        "upf_cpu_cores_mean",
        "upf_cpu_cores_p95",
        "upf_cpu_cores_max",
        "upf_cpu_cores_std",
        "upf_cpu_cores_second_half_mean",
        "upf_memory_mib_median",
    ],
    "server_compute": [
        "server_cpu_cores_mean",
        "server_cpu_cores_p95",
        "server_cpu_cores_max",
        "server_cpu_cores_second_half_mean",
        "server_memory_mib_median",
    ],
    "tcp_correlation": [
        "tcp_evict_ratio_max",
        "tcp_ack_no_match_hz_30s_median",
        "tcp_ack_observed_hz_30s_min",
        "tcp_ack_no_match_hz_30s_mean",
        "tcp_ack_observed_hz_30s_mean",
    ],
    "container_context": [
        "container_cpu_cores_p95",
        "container_cpu_cores_max",
        "container_memory_mib_p95",
    ],
}

EXPECTED_FAMILIES: dict[str, list[str]] = {
    "clean_traffic": ["latency_ebpf", "radio_mac", "upf_compute", "server_compute"],
    "controlled_delay": ["latency_ebpf", "tcp_correlation"],
    "far_ue_poor_radio": ["radio_mac", "latency_ebpf"],
    "radio_interference": ["radio_mac", "latency_ebpf"],
    "server_stress": ["server_compute", "container_context"],
    "tunnel_packet_loss": ["tcp_correlation", "latency_ebpf"],
    "upf_stress": ["upf_compute", "container_context"],
}


def is_metadata(name: str) -> bool:
    return any(name.startswith(prefix) or name == prefix.rstrip("_") for prefix in METADATA_PREFIXES)


def load_dataset(path: Path, exclude_classes: set[str]) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if len(row) > META_COLS]
    if exclude_classes:
        rows = [row for row in rows if row[2].strip() not in exclude_classes]
    return header, rows


def build_matrix(rows: list[list[str]], feature_indices: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.full((len(rows), len(feature_indices)), np.nan)
    y: list[str] = []
    runs: list[str] = []
    for r, row in enumerate(rows):
        y.append(row[2].strip())
        runs.append(row[1].strip())
        for c, idx in enumerate(feature_indices):
            value = row[META_COLS + idx].strip()
            if value and value.lower() != "nan":
                try:
                    X[r, c] = float(value)
                except ValueError:
                    pass
    return X, np.array(y), np.array(runs)


def nan_pct(values: np.ndarray) -> float:
    return float((~np.isfinite(values)).mean() * 100)


def feature_values(rows: list[list[str]], column_idx: int) -> np.ndarray:
    values = []
    for row in rows:
        value = row[column_idx].strip()
        try:
            values.append(float(value) if value and value.lower() != "nan" else float("nan"))
        except ValueError:
            values.append(float("nan"))
    return np.array(values)


def select_features(
    header: list[str],
    rows: list[list[str]],
    max_global_nan: float,
    max_class_nan: float,
) -> tuple[list[str], list[int], dict[str, str], list[dict[str, object]]]:
    feature_names = header[META_COLS:]
    selected_names: list[str] = []
    name_to_family: dict[str, str] = {}
    labels = sorted({row[2].strip() for row in rows})
    audit_rows: list[dict[str, object]] = []

    for family, candidates in FAMILIES.items():
        for name in candidates:
            if name not in feature_names:
                continue
            if is_metadata(name):
                continue
            idx = feature_names.index(name)
            full_idx = META_COLS + idx
            values = feature_values(rows, full_idx)
            global_nan = nan_pct(values)
            class_nan_values: dict[str, float] = {}
            for label in labels:
                label_rows = [row for row in rows if row[2].strip() == label]
                class_nan_values[label] = nan_pct(feature_values(label_rows, full_idx))

            worst_class = max(class_nan_values, key=class_nan_values.get)
            worst_class_nan = class_nan_values[worst_class]
            selected = global_nan <= max_global_nan and worst_class_nan <= max_class_nan

            audit_rows.append({
                "feature": name,
                "family": family,
                "selected": "yes" if selected else "no",
                "global_nan_pct": global_nan,
                "worst_class": worst_class,
                "worst_class_nan_pct": worst_class_nan,
                **{f"nan_pct__{label}": class_nan_values[label] for label in labels},
            })

            if not selected:
                continue
            if name not in selected_names:
                selected_names.append(name)
                name_to_family[name] = family

    selected_indices = [feature_names.index(name) for name in selected_names]
    return selected_names, selected_indices, name_to_family, audit_rows


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )


def evaluate_loro(X: np.ndarray, y: np.ndarray, runs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        clf = make_rf()
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        y_true_all.extend(y[test_mask])
        y_pred_all.extend(preds)
    return np.array(y_true_all), np.array(y_pred_all)


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


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.3f}"


def feature_summary(X: np.ndarray, y: np.ndarray, names: list[str], name_to_family: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in sorted(set(y)):
        class_mask = y == label
        rest_mask = ~class_mask
        expected = set(EXPECTED_FAMILIES.get(label, []))
        for i, name in enumerate(names):
            family = name_to_family[name]
            class_med = float(np.nanmedian(X[class_mask, i]))
            rest_med = float(np.nanmedian(X[rest_mask, i]))
            delta = class_med - rest_med
            scale = robust_scale(X[:, i])
            sep = abs(delta) / scale if scale > 0 else 0.0
            rows.append({
                "class": label,
                "family": family,
                "expected_family": "yes" if family in expected else "no",
                "feature": name,
                "direction": "higher" if delta > 0 else "lower",
                "class_median": class_med,
                "rest_median": rest_med,
                "delta": delta,
                "robust_separation": sep,
                "class_coverage_pct": float(np.isfinite(X[class_mask, i]).mean() * 100),
            })
    rows.sort(key=lambda r: (str(r["class"]), str(r["expected_family"]) != "yes", -float(r["robust_separation"])))
    return rows


def train_importance(X: np.ndarray, y: np.ndarray, names: list[str], name_to_family: dict[str, str]) -> list[dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    clf = make_rf()
    clf.fit(X_imp, y)
    order = np.argsort(clf.feature_importances_)[::-1]
    rows = []
    for rank, i in enumerate(order, 1):
        rows.append({
            "rank": rank,
            "feature": names[i],
            "family": name_to_family[names[i]],
            "importance": float(clf.feature_importances_[i]),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_importance_plot(path: Path, rows: list[dict[str, object]], top_n: int = 25) -> None:
    if plt is None or not rows:
        return
    top = rows[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.28)))
    ax.barh([str(r["feature"]) for r in top], [float(r["importance"]) for r in top])
    ax.set_xlabel("Random Forest importance")
    ax.set_title("Physically constrained feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(
    path: Path,
    csv_path: Path,
    selected_names: list[str],
    name_to_family: dict[str, str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    importance_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    labels = sorted(set(y_true))
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    lines: list[str] = []
    lines.append("# Physical RCA Feature Study")
    lines.append("")
    lines.append(f"Dataset: `{csv_path}`")
    lines.append(f"Features physiques retenues: **{len(selected_names)}**")
    lines.append(f"Accuracy: **{acc:.3f}**")
    lines.append(f"Macro-F1: **{macro_f1:.3f}**")
    lines.append("")
    lines.append("## Features Par Famille")
    for family in FAMILIES:
        family_features = [name for name in selected_names if name_to_family[name] == family]
        if not family_features:
            continue
        lines.append("")
        lines.append(f"### {family}")
        for name in family_features:
            lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Note Sur Les Valeurs Manquantes")
    lines.append("")
    lines.append(
        "Les features sont retenues seulement si leur taux de NaN reste acceptable "
        "globalement et dans chaque classe. Cela évite de sélectionner une feature "
        "uniquement parce qu'elle est absente dans certaines classes."
    )
    lines.append("")
    lines.append("## Performance Par Classe")
    lines.append("")
    lines.append("| class | precision | recall | f1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for label in labels:
        lines.append(
            f"| {label} | {report[label]['precision']:.3f} | {report[label]['recall']:.3f} | "
            f"{report[label]['f1-score']:.3f} | {int(report[label]['support'])} |"
        )
    lines.append("")
    lines.append("## Importances Globales")
    lines.append("")
    lines.append("| rank | family | feature | importance |")
    lines.append("|---:|---|---|---:|")
    for row in importance_rows:
        lines.append(f"| {row['rank']} | {row['family']} | `{row['feature']}` | {float(row['importance']):.4f} |")
    lines.append("")
    lines.append("## Signatures Physiques Par Classe")
    for label in labels:
        lines.append("")
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| family | expected | feature | direction | class median | rest median | separation | coverage |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|")
        class_rows = [r for r in summary_rows if r["class"] == label]
        class_rows = sorted(class_rows, key=lambda r: (r["expected_family"] != "yes", -float(r["robust_separation"])))[:10]
        for row in class_rows:
            lines.append(
                f"| {row['family']} | {row['expected_family']} | `{row['feature']}` | {row['direction']} | "
                f"{fmt(float(row['class_median']))} | {fmt(float(row['rest_median']))} | "
                f"{float(row['robust_separation']):.2f} | {float(row['class_coverage_pct']):.1f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/ml_dataset_v3/window_features_augmented.csv")
    parser.add_argument("--exclude-classes", default="")
    parser.add_argument("--out-dir", default="results/ml_dataset_v3/physical_feature_study")
    parser.add_argument("--max-global-nan", type=float, default=35.0)
    parser.add_argument("--max-class-nan", type=float, default=35.0)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}

    header, rows = load_dataset(csv_path, exclude)
    selected_names, selected_indices, name_to_family, audit_rows = select_features(
        header,
        rows,
        args.max_global_nan,
        args.max_class_nan,
    )
    if not selected_names:
        raise SystemExit("ERROR: no physical features selected")

    X, y, runs = build_matrix(rows, selected_indices)
    y_true, y_pred = evaluate_loro(X, y, runs)
    importance_rows = train_importance(X, y, selected_names, name_to_family)
    summary_rows = feature_summary(X, y, selected_names, name_to_family)

    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    write_csv(out_dir / "selected_physical_features.csv", [
        {"feature": name, "family": name_to_family[name]} for name in selected_names
    ])
    write_csv(out_dir / "physical_feature_missingness_audit.csv", audit_rows)
    write_csv(out_dir / "physical_feature_importance.csv", importance_rows)
    write_csv(out_dir / "physical_class_signatures.csv", summary_rows)
    save_importance_plot(out_dir / "physical_feature_importance.png", importance_rows)
    write_report(
        out_dir / "README.md",
        csv_path,
        selected_names,
        name_to_family,
        y_true,
        y_pred,
        importance_rows,
        summary_rows,
    )

    print(f"Dataset: {csv_path}")
    print(f"samples: {len(y)}")
    print(f"physical_features: {len(selected_names)}")
    print(f"max_global_nan: {args.max_global_nan:.1f}%")
    print(f"max_class_nan: {args.max_class_nan:.1f}%")
    print(f"Accuracy: {acc:.3f}")
    print(f"Macro-F1: {macro_f1:.3f}")
    print("\nConfusion matrix:")
    print(f"{'':>22}", " ".join(f"{label[:6]:>6}" for label in labels))
    for i, label in enumerate(labels):
        print(f"{label:>22}", " ".join(f"{cm[i, j]:>6}" for j in range(len(labels))))
    print("\nTop physical importances:")
    for row in importance_rows[:15]:
        print(f"  {row['rank']:>2}. {row['feature']} [{row['family']}] {float(row['importance']):.4f}")
    print(f"\nout_dir: {out_dir}")
    print(f"  - {out_dir / 'README.md'}")
    print(f"  - {out_dir / 'selected_physical_features.csv'}")
    print(f"  - {out_dir / 'physical_feature_importance.csv'}")
    print(f"  - {out_dir / 'physical_class_signatures.csv'}")
    print(f"  - {out_dir / 'physical_feature_missingness_audit.csv'}")
    print(f"  - {out_dir / 'physical_feature_importance.png'}")


if __name__ == "__main__":
    main()
