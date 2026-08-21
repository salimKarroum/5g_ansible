#!/usr/bin/env python3
"""Generate final-report figures for 5G RCA validation datasets.

The script is intentionally data-driven: pass one or more validation CSV files
and, optionally, the pretrained RCA model bundle. It creates report-ready PNG
and PDF figures for dataset composition, model performance, feature coverage,
and per-class physical signatures.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CLASS_ORDER = [
    "controlled_delay",
    "controlled_jitter",
    "far_ue_poor_radio",
    "radio_interference",
    "tunnel_bandwidth_limit",
    "tunnel_packet_loss",
    "upf_stress",
]

CLASS_LABELS = {
    "controlled_delay": "Delay",
    "controlled_jitter": "Jitter",
    "far_ue_poor_radio": "Far UE",
    "radio_interference": "Radio int.",
    "tunnel_bandwidth_limit": "Tunnel BW",
    "tunnel_packet_loss": "Tunnel loss",
    "upf_stress": "UPF stress",
}

FEATURE_LABELS = {
    "direct_latest_ms_median": "eBPF latency median",
    "direct_p95_ms_5s_mean": "eBPF latency p95",
    "probe_ping_rtt_ms_median": "Ping RTT median",
    "probe_ping_rtt_ms_mean": "Ping RTT mean",
    "probe_ping_rtt_ms_p95": "Ping RTT p95",
    "probe_ping_rtt_ms_std": "Ping RTT std",
    "probe_ping_rtt_ms_range": "Ping RTT range",
    "probe_ping_packet_loss_pct": "Ping loss %",
    "probe_ping_packets_lost": "Ping packets lost",
    "probe_iperf_dl_interval_mbps_std": "DL iperf std",
    "probe_iperf_dl_interval_mbps_mean": "DL iperf mean",
    "probe_iperf_dl_receiver_mbps": "DL receiver Mbps",
    "probe_iperf_ul_interval_mbps_mean": "UL iperf mean",
    "probe_iperf_total_retransmits": "TCP retransmits",
    "probe_iperf_dl_retransmits": "DL retransmits",
    "probe_iperf_ul_retransmits": "UL retransmits",
    "rf_oai_gnb_mac_avg_rsrp_mean": "OAI avg RSRP",
    "rf_oai_gnb_mac_avg_rsrp_min": "OAI min RSRP",
    "rf_oai_gnb_mac_cqi_mean": "OAI CQI",
    "rf_oai_gnb_mac_ul_mcs_mean": "OAI UL MCS",
    "rf_oai_gnb_mac_dl_mcs_mean": "OAI DL MCS",
    "rf_oai_gnb_mac_ul_bler_mean": "OAI UL BLER",
    "rf_oai_gnb_mac_dl_bler_p95": "OAI DL BLER p95",
    "rf_oai_gnb_l1_prach_i0_db_mean": "OAI PRACH I0",
    "rf_oai_gnb_l1_prach_i0_db_p95": "OAI PRACH I0 p95",
    "prach_i0_db_mean": "PRACH I0",
    "prach_i0_db_p95": "PRACH I0 p95",
    "rf_du_low_ul_algo_efficiency_sinr_db_mean": "srsRAN UL SINR",
    "rf_du_low_ul_algo_efficiency_bler_mean": "srsRAN UL BLER",
    "rf_srsran_mac_ul_mcs_mean": "srsRAN UL MCS",
    "rf_srsran_mac_cqi_mean": "srsRAN CQI",
    "upf_cpu_cores_mean": "UPF CPU mean",
    "upf_cpu_cores_max": "UPF CPU max",
    "upf_cpu_cores_std": "UPF CPU std",
    "upf_cpu_cores_delta": "UPF CPU delta",
    "container_cpu_cores_mean": "Container CPU mean",
}

SIGNATURE_CANDIDATES = {
    "controlled_delay": [
        "probe_ping_rtt_ms_p95",
        "probe_ping_rtt_ms_mean",
        "direct_p95_ms_5s_mean",
        "direct_latest_ms_median",
    ],
    "controlled_jitter": [
        "probe_ping_rtt_ms_std",
        "probe_ping_rtt_ms_range",
        "direct_latest_ms_std",
    ],
    "far_ue_poor_radio": [
        "rf_oai_gnb_mac_avg_rsrp_mean",
        "rf_oai_gnb_mac_avg_rsrp_min",
        "rf_srsran_mac_cqi_mean",
        "probe_iperf_dl_interval_mbps_mean",
    ],
    "radio_interference": [
        "rf_oai_gnb_l1_prach_i0_db_mean",
        "prach_i0_db_mean",
        "rf_du_low_ul_algo_efficiency_sinr_db_mean",
        "rf_du_low_ul_algo_efficiency_bler_mean",
        "rf_oai_gnb_mac_cqi_mean",
    ],
    "tunnel_bandwidth_limit": [
        "probe_iperf_dl_interval_mbps_mean",
        "probe_iperf_ul_interval_mbps_mean",
        "probe_iperf_dl_receiver_mbps",
    ],
    "tunnel_packet_loss": [
        "probe_ping_packet_loss_pct",
        "probe_ping_packets_lost",
        "probe_iperf_total_retransmits",
        "probe_iperf_dl_retransmits",
    ],
    "upf_stress": [
        "upf_cpu_cores_mean",
        "upf_cpu_cores_max",
        "upf_cpu_cores_std",
        "upf_cpu_cores_delta",
        "container_cpu_cores_mean",
    ],
}

FAMILY_COLORS = {
    "Latency": "#4E79A7",
    "Radio quality": "#59A14F",
    "Radio noise": "#A65628",
    "Throughput": "#B07AA1",
    "Loss / retransmit": "#F28E2B",
    "UPF compute": "#76B7B2",
    "Other": "#999999",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec as NAME=PATH. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional pretrained RCA pickle bundle with keys: features, imputer, model.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/manual-network/dataset/final_report_figures",
    )
    parser.add_argument("--title-prefix", default="5G RCA")
    parser.add_argument(
        "--balance",
        choices=("none", "per-dataset"),
        default="none",
        help="Downsample classes before plotting/evaluation. Use per-dataset for fair class-balanced validation plots.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def parse_dataset_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"Invalid --dataset value, expected NAME=PATH: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit(f"Invalid empty dataset name in: {spec}")
    return name, Path(path)


def pretty_class(label: str) -> str:
    return CLASS_LABELS.get(label, label.replace("_", " "))


def pretty_feature(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_").lower()


def feature_family(feature: str) -> str:
    low = feature.lower()
    if low.startswith("upf_cpu_") or low.startswith("container_cpu"):
        return "UPF compute"
    if "prach_i0" in low or "i0_db" in low or "i0_noise" in low:
        return "Radio noise"
    if any(x in low for x in ("rsrp", "cqi", "mcs", "bler", "snr", "sinr")):
        return "Radio quality"
    if "iperf" in low and "mbps" in low:
        return "Throughput"
    if any(x in low for x in ("packet_loss", "packets_lost", "retransmits")):
        return "Loss / retransmit"
    if low.startswith("direct_") or "rtt_ms" in low or "latest_ms" in low:
        return "Latency"
    return "Other"


def savefig(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def class_order_for(labels: set[str]) -> list[str]:
    ordered = [label for label in CLASS_ORDER if label in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return ordered


def load_datasets(specs: list[str]) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for spec in specs:
        name, path = parse_dataset_spec(spec)
        if not path.exists():
            raise SystemExit(f"Dataset not found for {name}: {path}")
        df = pd.read_csv(path)
        for required in ("label",):
            if required not in df.columns:
                raise SystemExit(f"Dataset {path} is missing required column: {required}")
        if "run" not in df.columns:
            df["run"] = path.stem
        if "phase" not in df.columns:
            df["phase"] = ""
        datasets[name] = df
    return datasets


def balance_per_dataset(datasets: dict[str, pd.DataFrame], out_dir: Path, random_state: int) -> dict[str, pd.DataFrame]:
    balanced: dict[str, pd.DataFrame] = {}
    csv_dir = out_dir / "balanced_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    for name, df in datasets.items():
        counts = df["label"].astype(str).value_counts()
        target = int(counts.min())
        parts = []
        for label in sorted(counts.index):
            sub = df[df["label"].astype(str) == label]
            parts.append(sub.sample(n=target, random_state=random_state, replace=False))
        out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        balanced[name] = out
        out.to_csv(csv_dir / f"{slug(name)}_balanced.csv", index=False)

    summary_rows = []
    for name, df in balanced.items():
        for label, count in df["label"].astype(str).value_counts().sort_index().items():
            summary_rows.append({"dataset": name, "label": label, "windows": int(count)})
    pd.DataFrame(summary_rows).to_csv(csv_dir / "balanced_summary.csv", index=False)
    return balanced


def plot_class_distribution(datasets: dict[str, pd.DataFrame], out_base: Path) -> None:
    all_labels = class_order_for(set().union(*(set(df["label"].astype(str)) for df in datasets.values())))
    values = []
    for name, df in datasets.items():
        counts = df["label"].astype(str).value_counts()
        values.append([int(counts.get(label, 0)) for label in all_labels])

    x = np.arange(len(all_labels))
    width = min(0.78 / max(len(datasets), 1), 0.26)
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    for idx, (name, counts) in enumerate(zip(datasets.keys(), values)):
        offset = (idx - (len(datasets) - 1) / 2) * width
        bars = ax.bar(x + offset, counts, width, label=name)
        for bar, count in zip(bars, counts):
            if count:
                ax.text(bar.get_x() + bar.get_width() / 2, count + 0.4, str(count), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("30 s windows")
    ax.set_title("Validation dataset composition")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_class(label) for label in all_labels], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, out_base)


def plot_run_composition(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for name, df in datasets.items():
        labels = class_order_for(set(df["label"].astype(str)))
        runs = sorted(df["run"].astype(str).unique())
        if len(runs) <= 1:
            continue
        mat = np.zeros((len(labels), len(runs)))
        grouped = df.groupby([df["label"].astype(str), df["run"].astype(str)]).size()
        for i, label in enumerate(labels):
            for j, run in enumerate(runs):
                mat[i, j] = grouped.get((label, run), 0)

        fig, ax = plt.subplots(figsize=(max(9.0, len(runs) * 0.65), 5.2))
        bottom = np.zeros(len(runs))
        for i, label in enumerate(labels):
            ax.bar(runs, mat[i], bottom=bottom, label=pretty_class(label))
            bottom += mat[i]
        ax.set_ylabel("30 s windows")
        ax.set_title(f"Run composition - {name}")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.legend(frameon=False, ncol=2)
        ax.grid(axis="y", alpha=0.25)
        savefig(fig, out_dir / f"02_run_composition_{slug(name)}")


def load_model(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    model_path = Path(path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    with model_path.open("rb") as fp:
        bundle = pickle.load(fp)
    for key in ("features", "imputer", "model"):
        if key not in bundle:
            raise SystemExit(f"Model bundle is missing key: {key}")
    return bundle


def evaluate_model(datasets: dict[str, pd.DataFrame], bundle: dict[str, Any], out_dir: Path) -> pd.DataFrame:
    features = list(bundle["features"])
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []

    for name, original in datasets.items():
        df = original.copy()
        missing_cols = [feature for feature in features if feature not in df.columns]
        if missing_cols:
            df = pd.concat([df, pd.DataFrame({feature: np.nan for feature in missing_cols}, index=df.index)], axis=1)

        X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy()
        X = bundle["imputer"].transform(X)
        pred = bundle["model"].predict(X)
        truth = df["label"].astype(str).to_numpy()
        labels = class_order_for(set(truth) | set(pred.astype(str)))

        report = classification_report(truth, pred, labels=labels, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(out_dir / f"model_report_{slug(name)}.csv")
        cm = confusion_matrix(truth, pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_matrix_{slug(name)}.csv")

        rows.append(
            {
                "dataset": name,
                "samples": len(df),
                "accuracy": accuracy_score(truth, pred),
                "macro_f1": f1_score(truth, pred, labels=labels, average="macro", zero_division=0),
                "weighted_f1": f1_score(truth, pred, labels=labels, average="weighted", zero_division=0),
                "missing_model_features": len(missing_cols),
                "mean_feature_missing_ratio": float(pd.DataFrame(df[features]).isna().mean(axis=1).mean()),
            }
        )

        for feature in features:
            coverage_rows.append(
                {
                    "dataset": name,
                    "feature": feature,
                    "feature_label": pretty_feature(feature),
                    "family": feature_family(feature),
                    "coverage": numeric_series(df, feature).notna().mean(),
                }
            )

        plot_confusion(cm, labels, f"Pretrained model confusion - {name}", out_dir / f"03_confusion_{slug(name)}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "model_performance_summary.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(out_dir / "model_feature_coverage.csv", index=False)
    plot_model_performance(summary, out_dir / "04_model_performance_by_dataset")
    plot_feature_coverage(pd.DataFrame(coverage_rows), out_dir / "05_model_feature_coverage")
    plot_feature_families(features, out_dir / "06_model_feature_families")
    return summary


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, out_base: Path) -> None:
    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    norm = cm / row_sum
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([pretty_class(label) for label in labels], rotation=35, ha="right")
    ax.set_yticklabels([pretty_class(label) for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=9)
    savefig(fig, out_base)


def plot_model_performance(summary: pd.DataFrame, out_base: Path) -> None:
    x = np.arange(len(summary))
    width = 0.28
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - width, summary["accuracy"], width, label="Accuracy", color="#4E79A7")
    ax.bar(x, summary["macro_f1"], width, label="Macro-F1", color="#59A14F")
    ax.bar(x + width, summary["weighted_f1"], width, label="Weighted-F1", color="#F28E2B")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Pretrained RCA model validation")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["dataset"], rotation=20, ha="right")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    for idx, row in summary.iterrows():
        ax.text(idx, min(1.02, float(row["accuracy"]) + 0.025), f"n={int(row['samples'])}", ha="center", fontsize=8)
    savefig(fig, out_base)


def plot_feature_coverage(coverage: pd.DataFrame, out_base: Path) -> None:
    datasets = list(dict.fromkeys(coverage["dataset"].astype(str)))
    features = coverage.drop_duplicates("feature").copy()
    features["order"] = features["family"].map(
        {"Latency": 0, "Radio quality": 1, "Radio noise": 2, "Throughput": 3, "Loss / retransmit": 4, "UPF compute": 5, "Other": 6}
    )
    features = features.sort_values(["order", "feature_label"])
    matrix = np.zeros((len(features), len(datasets)))
    lookup = coverage.set_index(["feature", "dataset"])["coverage"]
    for i, feature in enumerate(features["feature"]):
        for j, dataset in enumerate(datasets):
            matrix[i, j] = float(lookup.get((feature, dataset), 0.0))

    height = max(6.0, len(features) * 0.31)
    fig, ax = plt.subplots(figsize=(8.8, height))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.03, label="Coverage")
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features["feature_label"], fontsize=8)
    ax.set_title("Pretrained model feature availability")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value * 100:.0f}%", ha="center", va="center", fontsize=7, color="white" if value > 0.65 else "black")
    savefig(fig, out_base)


def plot_feature_families(features: list[str], out_base: Path) -> None:
    counts = pd.Series([feature_family(feature) for feature in features]).value_counts().sort_values()
    colors = [FAMILY_COLORS.get(family, "#999999") for family in counts.index]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.set_xlabel("Number of model features")
    ax.set_title("Pretrained model feature families")
    ax.grid(axis="x", alpha=0.25)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, str(int(count)), va="center")
    savefig(fig, out_base)


def pick_signature_feature(df: pd.DataFrame, label: str) -> str | None:
    for feature in SIGNATURE_CANDIDATES.get(label, []):
        if feature in df.columns and numeric_series(df, feature).notna().any():
            return feature
    return None


def plot_signature_boxplots(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    for name, df in datasets.items():
        labels = [label for label in CLASS_ORDER if label in set(df["label"].astype(str))]
        panels: list[tuple[str, str]] = []
        for label in labels:
            feature = pick_signature_feature(df, label)
            if feature:
                panels.append((label, feature))
        if not panels:
            continue

        cols = 3
        rows = int(math.ceil(len(panels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(14.2, 4.2 * rows), squeeze=False)
        axes_flat = axes.ravel()

        for ax, (label, feature) in zip(axes_flat, panels):
            target = numeric_series(df[df["label"].astype(str) == label], feature).dropna()
            rest = numeric_series(df[df["label"].astype(str) != label], feature).dropna()
            if target.empty or rest.empty:
                ax.axis("off")
                continue
            ax.boxplot(
                [target, rest],
                tick_labels=[pretty_class(label), "Other classes"],
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "#EAF2F8", "edgecolor": "#4E79A7"},
                medianprops={"color": "#D62728", "linewidth": 2},
            )
            ax.set_title("\n".join(textwrap.wrap(pretty_feature(feature), width=28)))
            ax.grid(axis="y", alpha=0.25)
        for ax in axes_flat[len(panels) :]:
            ax.axis("off")
        fig.suptitle(f"Physical signatures by anomaly - {name}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        savefig(fig, out_dir / f"07_signature_boxplots_{slug(name)}")


def write_readme(out_dir: Path, datasets: dict[str, pd.DataFrame], model_summary: pd.DataFrame | None) -> None:
    lines = [
        "# Final report figures",
        "",
        "Generated figures:",
        "- `01_class_distribution.*`: sample balance per anomaly and dataset.",
        "- `02_run_composition_*.png`: contribution of each run to each dataset.",
        "- `03_confusion_*.png`: pretrained model confusion matrices, when `--model` is provided.",
        "- `04_model_performance_by_dataset.*`: accuracy and F1 comparison, when `--model` is provided.",
        "- `05_model_feature_coverage.*`: availability of model features per dataset, useful for missing-ratio discussion.",
        "- `06_model_feature_families.*`: model feature families.",
        "- `07_signature_boxplots_*.png`: representative anomaly signatures.",
        "- `balanced_csv/*.csv`: class-balanced CSVs, when `--balance per-dataset` is used.",
        "",
        "Dataset summary:",
    ]
    for name, df in datasets.items():
        lines.append(f"- `{name}`: {len(df)} windows, {df['label'].nunique()} classes")
    if model_summary is not None:
        lines.extend(["", "Model summary:", ""])
        lines.append(model_summary.to_markdown(index=False))
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_datasets(args.dataset)
    if args.balance == "per-dataset":
        datasets = balance_per_dataset(datasets, out_dir, args.random_state)

    plot_class_distribution(datasets, out_dir / "01_class_distribution")
    plot_run_composition(datasets, out_dir)
    plot_signature_boxplots(datasets, out_dir)

    bundle = load_model(args.model)
    model_summary = evaluate_model(datasets, bundle, out_dir) if bundle is not None else None
    write_readme(out_dir, datasets, model_summary)

    print(f"figures_dir: {out_dir}")
    for path in sorted(out_dir.glob("*.png")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
