#!/usr/bin/env python3
"""
Generate presentation-ready plots for the balanced PFE RCA classification.

The script consumes:
  - the balanced top-feature dataset used for classification,
  - the strict_no_leak output directory produced by strict_no_leak_classification.py.

It creates compact plots for slides:
  - class distribution,
  - confusion matrix,
  - per-class precision/recall/F1,
  - feature selection frequency,
  - feature-family selection frequency,
  - physical signature boxplots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CLASS_LABELS = {
    "controlled_delay": "Delay",
    "controlled_jitter": "Jitter",
    "far_ue_poor_radio": "Far UE",
    "radio_interference": "Radio int.",
    "server_stress": "Server stress",
    "tunnel_bandwidth": "Tunnel BW",
    "tunnel_packet_loss": "Tunnel loss",
    "upf_stress": "UPF stress",
}


FEATURE_LABELS = {
    "direct_latest_ms_mean": "Latency mean",
    "direct_latest_ms_std": "Latency std",
    "direct_latest_ms_p95": "Latency p95",
    "direct_p95_ms_5s_mean": "Latency p95 5s",
    "direct_p99_ms_5s_mean": "Latency p99 5s",
    "rf_oai_gnb_l1_prach_i0_db_mean": "PRACH I0 mean",
    "rf_oai_gnb_l1_prach_i0_db_std": "PRACH I0 std",
    "rf_oai_gnb_l1_prach_i0_db_p95": "PRACH I0 p95",
    "rf_oai_gnb_l1_prach_i0_db_max": "PRACH I0 max",
    "rf_slice_throughput_mean": "Slice throughput mean",
    "rf_slice_throughput_p95": "Slice throughput p95",
    "rf_slice_throughput_max": "Slice throughput max",
    "tcp_ack_no_match_hz_30s_mean": "TCP ACK no-match",
    "tcp_ack_observed_hz_30s_mean": "TCP ACK observed",
    "server_cpu_cores_mean": "Server CPU mean",
    "server_cpu_cores_p95": "Server CPU p95",
    "upf_cpu_cores_mean": "UPF CPU mean",
    "upf_cpu_cores_p95": "UPF CPU p95",
}


SIGNATURE_FEATURES = [
    ("controlled_delay", "direct_latest_ms_p95", "Latency p95 (ms)"),
    ("radio_interference", "rf_oai_gnb_l1_prach_i0_db_p95", "PRACH I0 p95"),
    ("tunnel_bandwidth", "rf_slice_throughput_max", "Slice throughput max"),
    ("tunnel_packet_loss", "tcp_ack_no_match_hz_30s_mean", "TCP ACK no-match"),
    ("server_stress", "server_cpu_cores_p95", "Server CPU p95"),
    ("upf_stress", "upf_cpu_cores_p95", "UPF CPU p95"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="results/ml_dataset_v3/window_features_transport_balanced_48_pfe_top18.csv",
    )
    parser.add_argument(
        "--eval-dir",
        default="results/ml_dataset_v3/strict_no_leak_balanced_pfe_top18",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ml_dataset_v3/pfe_balanced_top18_plots",
    )
    return parser.parse_args()


def class_name(label: str) -> str:
    return CLASS_LABELS.get(label, label.replace("_", " "))


def feature_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def feature_family(feature: str) -> str:
    if feature.startswith("direct_"):
        return "Latency"
    if feature.startswith("rf_oai_gnb_l1_prach_i0"):
        return "Radio interference"
    if feature.startswith("rf_slice_throughput"):
        return "Throughput"
    if feature.startswith("tcp_ack"):
        return "TCP correlation"
    if feature.startswith("server_cpu"):
        return "Server compute"
    if feature.startswith("upf_cpu"):
        return "UPF compute"
    return "Other"


def save(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([class_name(x) for x in counts.index], counts.values, color="#4C78A8")
    ax.set_ylabel("Windows")
    ax.set_title("Balanced RCA Dataset: Windows per Class")
    ax.tick_params(axis="x", rotation=35)
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.8, str(value), ha="center", va="bottom", fontsize=9)
    save(fig, out)


def plot_confusion(eval_dir: Path, out: Path) -> None:
    cm = pd.read_csv(eval_dir / "strict_no_leak_best_confusion_matrix.csv")
    labels = cm.iloc[:, 0].astype(str).tolist()
    mat = cm.iloc[:, 1:].to_numpy(dtype=float)
    row_sum = np.maximum(mat.sum(axis=1, keepdims=True), 1)
    norm = mat / row_sum

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([class_name(x) for x in labels], rotation=40, ha="right")
    ax.set_yticklabels([class_name(x) for x in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Strict Leave-One-Run-Out Confusion Matrix")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if norm[i, j] > 0.55 else "black"
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color=color, fontsize=9)
    save(fig, out)


def plot_class_report(eval_dir: Path, out: Path) -> None:
    report = pd.read_csv(eval_dir / "strict_no_leak_best_class_report.csv")
    report = report.sort_values("label")
    x = np.arange(len(report))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, report["precision"], width, label="Precision", color="#4C78A8")
    ax.bar(x, report["recall"], width, label="Recall", color="#F58518")
    ax.bar(x + width, report["f1"], width, label="F1", color="#54A24B")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Classification Scores")
    ax.set_xticks(x)
    ax.set_xticklabels([class_name(x) for x in report["label"]], rotation=35, ha="right")
    ax.legend(ncol=3, loc="lower right")
    save(fig, out)


def plot_feature_frequency(eval_dir: Path, out: Path) -> None:
    freq = pd.read_csv(eval_dir / "strict_no_leak_feature_frequency.csv")
    freq = freq.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh([feature_name(x) for x in freq["feature"]], freq["selected_folds"], color="#59A14F")
    ax.set_xlabel("Selected folds")
    ax.set_title("Stable Physical Features Selected Across Folds")
    save(fig, out)


def plot_family_frequency(eval_dir: Path, out: Path) -> None:
    freq = pd.read_csv(eval_dir / "strict_no_leak_feature_frequency.csv")
    freq["family"] = freq["feature"].map(feature_family)
    fam = freq.groupby("family")["selected_folds"].sum().sort_values()

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(fam.index, fam.values, color="#B279A2")
    ax.set_xlabel("Selected-fold count")
    ax.set_title("Selected Feature Coverage by Physical Family")
    save(fig, out)


def plot_signature_boxplots(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    axes = axes.ravel()

    for ax, (target, feature, title) in zip(axes, SIGNATURE_FEATURES):
        if feature not in df.columns:
            ax.axis("off")
            continue

        values_target = pd.to_numeric(df.loc[df["label"] == target, feature], errors="coerce").dropna()
        values_rest = pd.to_numeric(df.loc[df["label"] != target, feature], errors="coerce").dropna()
        ax.boxplot(
            [values_target, values_rest],
            tick_labels=[class_name(target), "Rest"],
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#E6F0FA"},
            medianprops={"color": "#E45756", "linewidth": 2},
        )
        ax.set_title(title)
        if values_target.size and values_rest.size:
            target_med = values_target.median()
            rest_med = values_rest.median()
            ax.text(
                0.5,
                0.95,
                f"median: {target_med:.3g} vs {rest_med:.3g}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
        if feature == "rf_slice_throughput_max":
            ax.set_yscale("symlog", linthresh=1)

    fig.suptitle("Physical Signature Checks: Target Class vs Rest", y=1.02, fontsize=14)
    save(fig, out)


def write_summary(eval_dir: Path, out_dir: Path) -> None:
    readme = eval_dir / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    (out_dir / "PLOTS_README.md").write_text(
        "# PFE Balanced RCA Plots\n\n"
        "Generated figures:\n\n"
        "- `class_distribution_balanced.png`: confirms 48 windows per anomaly class.\n"
        "- `confusion_matrix_loro.png`: strict leave-one-run-out confusion matrix.\n"
        "- `per_class_scores.png`: precision, recall, and F1 by class.\n"
        "- `feature_selection_frequency.png`: top physical features selected across folds.\n"
        "- `feature_family_frequency.png`: selected features grouped by physical family.\n"
        "- `physical_signature_boxplots.png`: class-vs-rest checks for key physical signatures.\n\n"
        "Evaluation summary:\n\n"
        f"{text}\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset)
    plot_class_distribution(df, out_dir / "class_distribution_balanced.png")
    plot_confusion(eval_dir, out_dir / "confusion_matrix_loro.png")
    plot_class_report(eval_dir, out_dir / "per_class_scores.png")
    plot_feature_frequency(eval_dir, out_dir / "feature_selection_frequency.png")
    plot_family_frequency(eval_dir, out_dir / "feature_family_frequency.png")
    plot_signature_boxplots(df, out_dir / "physical_signature_boxplots.png")
    write_summary(eval_dir, out_dir)

    print(f"Wrote plots to: {out_dir}")
    for path in sorted(out_dir.glob("*.png")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
