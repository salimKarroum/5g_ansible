#!/usr/bin/env python3
"""Plot the final mechanism-clean RCA results.

The script generates report-ready figures from:

* the original OAI/Open5GS windowed dataset,
* the final mechanism-clean feature list,
* the reduction outputs produced by reduce_physical_features_mechanism_clean.py.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


META_COLS = 6

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
    "controlled_delay": "Controlled\ndelay",
    "controlled_jitter": "Controlled\njitter",
    "far_ue_poor_radio": "Far UE\npoor radio",
    "radio_interference": "Radio\ninterference",
    "tunnel_bandwidth_limit": "Tunnel\nbandwidth\nlimit",
    "tunnel_packet_loss": "Tunnel\npacket loss",
    "upf_stress": "UPF\nstress",
}

FAMILY_LABELS = {
    "latency": "Latency",
    "radio_quality": "Radio quality",
    "radio_noise": "Radio noise",
    "throughput": "Throughput",
    "loss_retransmit": "Loss / retransmit",
    "upf_compute": "UPF compute",
    "other": "Other",
}

FAMILY_COLORS = {
    "latency": "#4E79A7",
    "radio_quality": "#59A14F",
    "radio_noise": "#A65628",
    "throughput": "#B07AA1",
    "loss_retransmit": "#F28E2B",
    "upf_compute": "#76B7B2",
    "other": "#999999",
}

FEATURE_LABELS = {
    "direct_latest_ms_mean": "eBPF latest latency mean",
    "direct_latest_ms_median": "eBPF latest latency median",
    "probe_ping_rtt_ms_std": "Ping RTT std",
    "probe_ping_rtt_ms_median": "Ping RTT median",
    "probe_ping_rtt_ms_mean": "Ping RTT mean",
    "probe_ping_rtt_ms_p95": "Ping RTT p95",
    "probe_ping_rtt_ms_range": "Ping RTT range",
    "rf_oai_gnb_mac_dl_bler_p95": "DL BLER p95",
    "rf_oai_gnb_mac_dl_bler_mean": "DL BLER mean",
    "rf_oai_gnb_mac_avg_rsrp_mean": "Avg RSRP mean",
    "rf_oai_gnb_mac_avg_rsrp_median": "Avg RSRP median",
    "rf_oai_gnb_mac_avg_rsrp_min": "Avg RSRP min",
    "rf_oai_gnb_mac_cqi_mean": "CQI mean",
    "rf_oai_gnb_l1_prach_i0_db_mean": "L1 PRACH I0 mean",
    "prach_i0_db_mean": "PRACH I0 mean",
    "prach_i0_db_p95": "PRACH I0 p95",
    "prach_i0_db_range": "PRACH I0 range",
    "rf_oai_gnb_l1_prach_i0_db_p95": "L1 PRACH I0 p95",
    "probe_iperf_dl_interval_mbps_std": "DL iperf Mbps std",
    "probe_iperf_dl_interval_mbps_mean": "DL iperf Mbps mean",
    "probe_iperf_dl_receiver_mbps": "DL receiver Mbps",
    "probe_iperf_ul_interval_mbps_mean": "UL iperf Mbps mean",
    "probe_iperf_total_retransmits": "Total TCP retransmits",
    "probe_ping_packets_lost": "Ping packets lost",
    "probe_iperf_dl_retransmits": "DL TCP retransmits",
    "probe_iperf_ul_retransmits": "UL TCP retransmits",
    "upf_cpu_cores_std": "UPF CPU std",
    "upf_cpu_cores_max": "UPF CPU max",
    "upf_cpu_cores_mean": "UPF CPU mean",
    "upf_cpu_cores_delta": "UPF CPU delta",
}


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


def pretty_feature(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def read_feature_file(path: Path) -> list[str]:
    features: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        feature = line.strip()
        if feature and not feature.startswith("#"):
            feature = feature.split(",", 1)[0].strip()
            if feature.lower() not in {"feature", "feature_name", "name"}:
                features.append(feature)
    return features


def load_dataset(path: Path, features: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if len(row) > META_COLS]

    feature_header = header[META_COLS:]
    missing = [name for name in features if name not in feature_header]
    if missing:
        raise SystemExit("Missing feature columns in dataset: " + ", ".join(missing))

    indices = [feature_header.index(name) for name in features]
    X = np.full((len(rows), len(features)), np.nan)
    y: list[str] = []
    runs: list[str] = []

    for r, row in enumerate(rows):
        runs.append(row[1].strip())
        y.append(row[2].strip())
        for c, idx in enumerate(indices):
            value = row[META_COLS + idx].strip()
            if value and value.lower() != "nan":
                try:
                    X[r, c] = float(value)
                except ValueError:
                    pass

    return X, np.array(y), np.array(runs)


def loro_predictions(
    X: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for run in sorted(set(runs)):
        test = runs == run
        train = ~test
        if train.sum() == 0 or test.sum() == 0:
            continue

        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X[train])
        X_test = imputer.transform(X[test])

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X_train, y[train])
        pred = clf.predict(X_test)

        y_true.extend(y[test].tolist())
        y_pred.extend(pred.tolist())

    return np.array(y_true), np.array(y_pred)


def savefig(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(rows: list[dict[str, float | str]], out_base: Path) -> None:
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width / 2, df["accuracy"], width, label="Accuracy", color="#4E79A7")
    ax.bar(x + width / 2, df["macro_f1"], width, label="Macro-F1", color="#59A14F")
    ax.set_ylim(0.88, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"], fontsize=10)
    ax.set_ylabel("LORO score")
    ax.set_title("Leave-One-Run-Out Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for i, row in df.iterrows():
        ax.text(i - width / 2, float(row["accuracy"]) + 0.004, f"{row['accuracy']:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, float(row["macro_f1"]) + 0.004, f"{row['macro_f1']:.3f}", ha="center", fontsize=9)
    savefig(fig, out_base)


def plot_sweep(sweep_csv: Path, chosen_k: int, out_base: Path) -> None:
    df = pd.read_csv(sweep_csv)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(df["k"], df["macro_f1"], marker="o", linewidth=2.0, label="Macro-F1", color="#59A14F")
    ax.plot(df["k"], df["accuracy"], marker="s", linewidth=2.0, label="Accuracy", color="#4E79A7")
    ax.axvline(chosen_k, linestyle="--", linewidth=1.4, color="#333333", label=f"Selected K={chosen_k}")
    ax.set_xlabel("Number of selected features")
    ax.set_ylabel("LORO score")
    ax.set_title("Feature Reduction Sweep")
    ax.set_ylim(max(0.88, float(df[["macro_f1", "accuracy"]].min().min()) - 0.02), 1.0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    savefig(fig, out_base)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], out_base: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([CLASS_LABELS.get(x, x).replace("\n", " ") for x in labels], rotation=35, ha="right")
    ax.set_yticklabels([CLASS_LABELS.get(x, x).replace("\n", " ") for x in labels])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Mechanism-Clean 25 Features - LORO Confusion Matrix\nAccuracy={accuracy:.3f}, Macro-F1={macro_f1:.3f}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() * 0.45 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    savefig(fig, out_base)


def plot_class_metrics(report: dict[str, dict[str, float]], labels: list[str], out_base: Path) -> None:
    rows = []
    for label in labels:
        values = report[label]
        rows.append(
            {
                "class": CLASS_LABELS.get(label, label).replace("\n", " "),
                "precision": values["precision"],
                "recall": values["recall"],
                "f1-score": values["f1-score"],
            }
        )
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(x - width, df["precision"], width, label="Precision", color="#4E79A7")
    ax.bar(x, df["recall"], width, label="Recall", color="#F28E2B")
    ax.bar(x + width, df["f1-score"], width, label="F1-score", color="#59A14F")
    ax.set_ylim(0.78, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class LORO Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    savefig(fig, out_base)


def plot_family_counts(features: list[str], out_base: Path) -> None:
    families = pd.Series([feature_family(f) for f in features]).value_counts().sort_values()
    colors = [FAMILY_COLORS.get(f, "#999999") for f in families.index]
    labels = [FAMILY_LABELS.get(f, f) for f in families.index]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.barh(labels, families.values, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_xlabel("Number of selected features")
    ax.set_title("Selected Feature Families - Mechanism-Clean 25")
    ax.grid(axis="x", alpha=0.25)
    for bar, count in zip(bars, families.values):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, str(int(count)), va="center")
    savefig(fig, out_base)


def plot_top4_table(top4_csv: Path, out_base: Path) -> None:
    df = pd.read_csv(top4_csv)
    df["feature_label"] = df["feature"].map(pretty_feature)
    labels = [c for c in CLASS_ORDER if c in set(df["class"])]

    fig, axes = plt.subplots(4, 2, figsize=(16.0, 13.2))
    axes_flat = axes.flatten()
    fig.subplots_adjust(top=0.92, bottom=0.09, left=0.04, right=0.98, hspace=0.42, wspace=0.16)
    fig.suptitle("Top 4 mechanism-clean features per anomaly class", fontsize=17, fontweight="bold")

    for idx, label in enumerate(labels):
        ax = axes_flat[idx]
        sub = df[df["class"] == label].sort_values("rank")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.text(
            0.0,
            0.96,
            CLASS_LABELS.get(label, label).replace("\n", " "),
            fontsize=12.5,
            fontweight="bold",
            va="top",
            ha="left",
        )

        for row_idx, row in enumerate(sub.itertuples(index=False), start=1):
            y = 0.80 - (row_idx - 1) * 0.18
            color = FAMILY_COLORS.get(row.family, "#999999")
            ax.add_patch(plt.Rectangle((0.0, y - 0.07), 0.98, 0.13, color=color, alpha=0.18, ec=color, lw=1.0))
            ax.text(0.025, y, f"{int(row.rank)}.", va="center", ha="left", fontsize=10.8, fontweight="bold")
            wrapped = "\n".join(textwrap.wrap(str(row.feature_label), width=30))
            ax.text(0.105, y, wrapped, va="center", ha="left", fontsize=10.0, linespacing=1.12)
            ax.text(
                0.74,
                y,
                FAMILY_LABELS.get(row.family, row.family),
                va="center",
                ha="left",
                fontsize=9.0,
                color="#333333",
            )

    for ax in axes_flat[len(labels) :]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=FAMILY_LABELS[f], markerfacecolor=FAMILY_COLORS[f], markersize=9)
        for f in FAMILY_LABELS
        if f in set(df["family"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.0, bbox_to_anchor=(0.5, 0.015))
    savefig(fig, out_base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--reduction-dir", required=True)
    parser.add_argument("--baseline-features-file", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    dataset_csv = Path(args.csv)
    features_file = Path(args.features_file)
    reduction_dir = Path(args.reduction_dir)
    out_dir = Path(args.out_dir) if args.out_dir else reduction_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_features = read_feature_file(features_file)
    X, y, runs = load_dataset(dataset_csv, final_features)
    labels = [label for label in CLASS_ORDER if label in set(y)]
    y_true, y_pred = loro_predictions(X, y, runs, args.n_estimators, args.random_state)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    final_accuracy = float((y_true == y_pred).mean())
    final_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    comparison_rows: list[dict[str, float | str]] = []
    if args.baseline_features_file:
        baseline_features = read_feature_file(Path(args.baseline_features_file))
        X_base, y_base, runs_base = load_dataset(dataset_csv, baseline_features)
        b_true, b_pred = loro_predictions(X_base, y_base, runs_base, args.n_estimators, args.random_state)
        comparison_rows.append(
            {
                "model": f"Baseline\n{len(baseline_features)} features",
                "accuracy": float((b_true == b_pred).mean()),
                "macro_f1": float(f1_score(b_true, b_pred, average="macro", zero_division=0)),
            }
        )
    comparison_rows.append(
        {
            "model": f"Mechanism-clean\n{len(final_features)} features",
            "accuracy": final_accuracy,
            "macro_f1": final_macro_f1,
        }
    )

    pd.DataFrame(report).transpose().to_csv(out_dir / "mechanism_clean25_loro_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels).to_csv(
        out_dir / "mechanism_clean25_loro_confusion_matrix.csv"
    )

    plot_model_comparison(comparison_rows, out_dir / "01_model_comparison")
    plot_sweep(reduction_dir / "mechanism_clean_sweep.csv", len(final_features), out_dir / "02_feature_reduction_sweep")
    plot_confusion(y_true, y_pred, labels, out_dir / "03_loro_confusion_matrix")
    plot_class_metrics(report, labels, out_dir / "04_per_class_metrics")
    plot_family_counts(final_features, out_dir / "05_selected_feature_families")
    plot_top4_table(reduction_dir / "top4_features_by_class_mechanism_clean.csv", out_dir / "06_top4_features_by_class")

    print(f"plots_dir: {out_dir}")
    print(f"features: {len(final_features)}")
    print(f"loro_accuracy: {final_accuracy:.3f}")
    print(f"loro_macro_f1: {final_macro_f1:.3f}")
    for path in sorted(out_dir.glob("*.png")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
