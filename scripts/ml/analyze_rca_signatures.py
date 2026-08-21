#!/usr/bin/env python3
"""
Generate an RCA-oriented data analysis from the ML window feature CSV.

The output is intentionally descriptive rather than only model-centric:
it produces plots and tables that connect each class to metric families
and physical root-cause interpretations.

Example:
    python3 scripts/ml/analyze_rca_signatures.py \
      --csv results/ml_dataset_v3/window_features_augmented.csv \
      --exclude-classes load_ramp,multi_ue_contention \
      --out-dir results/ml_dataset_v3/rca_signature_analysis
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


META_COLS = {"path", "run", "label", "scenario", "phase", "reference_path"}


CLASS_ORDER = [
    "clean_traffic",
    "controlled_delay",
    "tunnel_packet_loss",
    "far_ue_poor_radio",
    "radio_interference",
    "upf_stress",
    "server_stress",
]


FAMILY_PATTERNS = {
    "eBPF direct latency": (r"^direct_", r"gtp_teid"),
    "Radio / MAC / PHY": (r"^rf_", r"rsrp", r"snr", r"mcs", r"bler", r"cqi", r"prach"),
    "UPF compute": (r"^upf_cpu", r"^upf_memory"),
    "Server compute": (r"^server_cpu", r"^server_memory"),
    "gNB compute": (r"^gnb_cpu", r"^gnb_memory"),
    "TCP correlator": (r"tcp_", r"evict", r"no_match"),
    "Container generic": (r"^container_"),
    "Throughput": (r"throughput",),
}


SIGNATURE_CANDIDATES = [
    ("Latency mean", ["direct_mean_ms_5s_mean", "direct_mean_ms_5s_median"]),
    ("Latency p95", ["direct_p95_ms_5s_mean", "direct_p95_ms_5s_p95"]),
    ("Latest latency", ["direct_latest_ms_median", "direct_latest_ms_mean"]),
    ("Latency change", ["direct_mean_ms_5s_second_minus_first", "direct_p95_ms_5s_second_minus_first"]),
    ("UPF CPU", ["upf_cpu_cores_mean", "upf_cpu_cores_p95"]),
    ("Server CPU", ["server_cpu_cores_mean", "server_cpu_cores_p95"]),
    ("gNB CPU", ["gnb_cpu_cores_mean", "container_cpu_cores_mean"]),
    ("RSRP", ["rsrp_mean_all", "rf_oai_gnb_mac_avg_rsrp_mean"]),
    ("RSRP min UE", ["rsrp_min_ue_mean", "rf_oai_gnb_mac_avg_rsrp_min"]),
    ("PRACH I0", ["rf_oai_gnb_l1_prach_i0_db_mean", "prach_i0_db_mean"]),
    ("PRACH I0 change", ["rf_oai_gnb_l1_prach_i0_db_second_minus_first"]),
    ("UL MCS", ["rf_oai_gnb_mac_ul_mcs_mean", "rf_oai_gnb_mac_ul_mcs_median"]),
    ("DL MCS", ["rf_oai_gnb_mac_dl_mcs_mean", "rf_oai_gnb_mac_dl_mcs_median"]),
    ("UL BLER", ["rf_oai_gnb_mac_ul_bler_mean", "rf_oai_gnb_mac_ul_bler_p95"]),
    ("DL BLER", ["rf_oai_gnb_mac_dl_bler_mean", "rf_oai_gnb_mac_dl_bler_p95"]),
    ("CQI", ["rf_oai_gnb_mac_cqi_mean", "rf_oai_gnb_mac_cqi_median"]),
    ("Slice throughput", ["rf_slice_throughput_median", "rf_slice_throughput_mean"]),
    ("TCP evicted", ["tcp_pending_evicted_hz_30s_median", "tcp_pending_evicted_hz_30s_mean"]),
    ("TCP no match", ["tcp_ack_no_match_hz_30s_median", "tcp_no_match_mean"]),
]


PAIR_PLOTS = [
    ("direct_mean_ms_5s_mean", "upf_cpu_cores_mean", "Latency vs UPF CPU", "latency_vs_upf_cpu.png"),
    ("direct_mean_ms_5s_mean", "server_cpu_cores_mean", "Latency vs Server CPU", "latency_vs_server_cpu.png"),
    ("rf_oai_gnb_l1_prach_i0_db_mean", "rf_oai_gnb_mac_ul_mcs_mean", "Radio noise vs UL MCS", "prach_i0_vs_ul_mcs.png"),
    ("rsrp_mean_all", "rf_oai_gnb_l1_prach_i0_db_mean", "Coverage vs interference", "rsrp_vs_prach_i0.png"),
    ("direct_p95_ms_5s_mean", "rf_oai_gnb_mac_dl_bler_p95", "Latency tail vs DL BLER", "latency_vs_dl_bler.png"),
]


COLORS = {
    "clean_traffic": "#4CAF50",
    "controlled_delay": "#2196F3",
    "tunnel_packet_loss": "#E53935",
    "far_ue_poor_radio": "#8E24AA",
    "radio_interference": "#FB8C00",
    "upf_stress": "#00ACC1",
    "server_stress": "#6D4C41",
    "load_ramp": "#757575",
    "multi_ue_contention": "#3949AB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/ml_dataset_v3/window_features_augmented.csv")
    parser.add_argument("--out-dir", default="results/ml_dataset_v3/rca_signature_analysis")
    parser.add_argument("--exclude-classes", default="")
    parser.add_argument("--top-features", type=int, default=12)
    return parser.parse_args()


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in META_COLS:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ordered_labels(labels: list[str]) -> list[str]:
    known = [x for x in CLASS_ORDER if x in labels]
    rest = sorted(x for x in labels if x not in known)
    return known + rest


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def select_signature_columns(df: pd.DataFrame) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for display, candidates in SIGNATURE_CANDIDATES:
        col = first_existing(df, candidates)
        if col:
            selected.append((display, col))
    return selected


def robust_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    med = values.median(skipna=True)
    q25, q75 = values.quantile(0.25), values.quantile(0.75)
    iqr = q75 - q25
    if not np.isfinite(iqr) or abs(iqr) < 1e-12:
        std = values.std(skipna=True)
        iqr = std if np.isfinite(std) and std > 1e-12 else 1.0
    return (values - med) / iqr


def family_for_column(name: str) -> str:
    for family, patterns in FAMILY_PATTERNS.items():
        if any(re.search(pattern, name, flags=re.IGNORECASE) for pattern in patterns):
            return family
    return "Other"


def save_class_distribution(df: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    counts = df["label"].value_counts().reindex(labels).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(labels)), counts.values, color=[COLORS.get(x, "#607D8B") for x in labels])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Windows")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "01_class_distribution.png", dpi=180)
    plt.close(fig)


def save_missingness(df: pd.DataFrame, labels: list[str], out_dir: Path) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in META_COLS and c != "label"]
    rows = []
    for label in labels:
        sub = df[df["label"] == label]
        for family in FAMILY_PATTERNS:
            fam_cols = [c for c in numeric_cols if family_for_column(c) == family]
            if not fam_cols:
                continue
            missing = sub[fam_cols].isna().mean().mean() * 100
            rows.append({"label": label, "family": family, "missing_pct": missing, "n_features": len(fam_cols)})
    miss = pd.DataFrame(rows)
    pivot = miss.pivot(index="label", columns="family", values="missing_pct").reindex(labels)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.45 * len(labels))))
    im = ax.imshow(pivot.fillna(100).values, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_title("Missing Values By Class And Metric Family")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Missing (%)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "02_missingness_by_metric_family.png", dpi=180)
    plt.close(fig)

    miss.to_csv(out_dir / "missingness_by_class_family.csv", index=False)
    return miss


def save_signature_heatmap(df: pd.DataFrame, labels: list[str], out_dir: Path) -> pd.DataFrame:
    selected = select_signature_columns(df)
    records = []
    for display, col in selected:
        z = robust_z(df[col]).clip(-4, 4)
        tmp = pd.DataFrame({"label": df["label"], "feature": display, "robust_z": z})
        med = tmp.groupby("label")["robust_z"].median().reindex(labels)
        for label, value in med.items():
            records.append({"label": label, "signature": display, "column": col, "robust_z_median": value})
    sig = pd.DataFrame(records)
    pivot = sig.pivot(index="label", columns="signature", values="robust_z_median").reindex(labels)

    fig, ax = plt.subplots(figsize=(max(12, 0.65 * len(pivot.columns)), max(4.5, 0.5 * len(labels))))
    im = ax.imshow(pivot.fillna(0).values, vmin=-2.5, vmax=2.5, cmap="coolwarm", aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right")
    ax.set_title("Class Signatures: Robust Median Z-Score")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Robust z-score vs dataset")
    fig.tight_layout()
    fig.savefig(out_dir / "03_signature_heatmap.png", dpi=180)
    plt.close(fig)

    sig.to_csv(out_dir / "signature_heatmap_values.csv", index=False)
    return sig


def save_boxplots(df: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    selected = select_signature_columns(df)
    wanted_names = {
        "Latency mean",
        "Latency p95",
        "UPF CPU",
        "Server CPU",
        "RSRP",
        "PRACH I0",
        "UL MCS",
        "DL BLER",
        "Slice throughput",
    }
    selected = [(n, c) for n, c in selected if n in wanted_names]
    if not selected:
        return
    ncols = 3
    nrows = math.ceil(len(selected) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (display, col) in zip(axes, selected):
        data = [df.loc[df["label"] == label, col].dropna().values for label in labels]
        tick_labels = [short_label(x) for x in labels]
        try:
            ax.boxplot(data, tick_labels=tick_labels, showfliers=False)
        except TypeError:
            ax.boxplot(data, showfliers=False)
            ax.set_xticks(range(1, len(tick_labels) + 1))
            ax.set_xticklabels(tick_labels)
        ax.set_title(f"{display}\n{col}", fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(selected) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "04_key_metric_distributions.png", dpi=180)
    plt.close(fig)


def save_pair_plots(df: pd.DataFrame, labels: list[str], out_dir: Path) -> None:
    for xcol, ycol, title, filename in PAIR_PLOTS:
        if xcol not in df.columns or ycol not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for label in labels:
            sub = df[df["label"] == label]
            ax.scatter(
                sub[xcol],
                sub[ycol],
                s=28,
                alpha=0.75,
                label=short_label(label),
                color=COLORS.get(label),
                edgecolors="none",
            )
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"05_{filename}", dpi=180)
        plt.close(fig)


def short_label(label: str) -> str:
    return {
        "clean_traffic": "clean",
        "controlled_delay": "delay",
        "tunnel_packet_loss": "loss",
        "far_ue_poor_radio": "far UE",
        "radio_interference": "interf.",
        "upf_stress": "UPF",
        "server_stress": "server",
        "multi_ue_contention": "multi UE",
        "load_ramp": "ramp",
    }.get(label, label)


def compute_top_features(df: pd.DataFrame, labels: list[str], top_n: int, out_dir: Path) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in META_COLS and c != "label"]
    rows = []
    for label in labels:
        cls = df[df["label"] == label]
        rest = df[df["label"] != label]
        for col in numeric_cols:
            cls_v = cls[col].dropna()
            rest_v = rest[col].dropna()
            coverage = len(cls_v) / max(len(cls), 1)
            if len(cls_v) < 2 or len(rest_v) < 2 or coverage < 0.5:
                continue
            cls_med = cls_v.median()
            rest_med = rest_v.median()
            q25, q75 = df[col].quantile([0.25, 0.75])
            scale = q75 - q25
            if not np.isfinite(scale) or abs(scale) < 1e-12:
                scale = df[col].std()
            if not np.isfinite(scale) or abs(scale) < 1e-12:
                continue
            score = abs(cls_med - rest_med) / scale
            rows.append(
                {
                    "label": label,
                    "feature": col,
                    "family": family_for_column(col),
                    "score": score,
                    "direction": "higher" if cls_med > rest_med else "lower",
                    "class_median": cls_med,
                    "rest_median": rest_med,
                    "delta": cls_med - rest_med,
                    "coverage_pct": coverage * 100,
                }
            )
    top = pd.DataFrame(rows).sort_values(["label", "score"], ascending=[True, False])
    top.to_csv(out_dir / "top_discriminant_features_all.csv", index=False)
    top.groupby("label").head(top_n).to_csv(out_dir / "top_discriminant_features_by_class.csv", index=False)
    return top


def save_metric_summary(df: pd.DataFrame, labels: list[str], out_dir: Path) -> pd.DataFrame:
    selected = select_signature_columns(df)
    rows = []
    for label in labels:
        sub = df[df["label"] == label]
        for display, col in selected:
            values = sub[col].dropna()
            rows.append(
                {
                    "label": label,
                    "signature": display,
                    "column": col,
                    "n": len(sub),
                    "coverage_pct": len(values) / max(len(sub), 1) * 100,
                    "median": values.median() if len(values) else np.nan,
                    "mean": values.mean() if len(values) else np.nan,
                    "p95": values.quantile(0.95) if len(values) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "metric_summary_by_class.csv", index=False)
    return summary


def write_report(
    df: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
    top_features: pd.DataFrame,
    metric_summary: pd.DataFrame,
    missingness: pd.DataFrame,
) -> None:
    counts = df["label"].value_counts().reindex(labels).fillna(0).astype(int)
    lines = [
        "# RCA Signature Analysis",
        "",
        f"Input windows: **{len(df)}**",
        "",
        "## Class Counts",
        "",
        "| class | windows |",
        "|---|---:|",
    ]
    for label, count in counts.items():
        lines.append(f"| {label} | {count} |")

    lines += [
        "",
        "## Physical RCA Layers",
        "",
        "| layer | classes | expected physical signature |",
        "|---|---|---|",
        "| Baseline | clean_traffic | low latency, low CPU, stable radio, stable throughput |",
        "| Tunnel / transport | controlled_delay, tunnel_packet_loss | eBPF latency and tail behavior without radio or compute root cause |",
        "| Radio / RAN | far_ue_poor_radio, radio_interference | RSRP/CQI/MCS/BLER/PRACH-I0 changes |",
        "| Core user plane | upf_stress | UPF CPU pressure with secondary latency/throughput impact |",
        "| Application/server | server_stress | server CPU pressure with network path otherwise mostly normal |",
        "",
        "## Strongest Observed Signatures",
        "",
    ]

    for label in labels:
        lines += [f"### {label}", ""]
        top = top_features[top_features["label"] == label].head(8)
        if top.empty:
            lines.append("No strong feature found with current coverage thresholds.")
            lines.append("")
            continue
        lines += ["| rank | feature | family | direction | class median | rest median | coverage |", "|---:|---|---|---|---:|---:|---:|"]
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            lines.append(
                f"| {rank} | `{row.feature}` | {row.family} | {row.direction} | "
                f"{fmt(row.class_median)} | {fmt(row.rest_median)} | {row.coverage_pct:.1f}% |"
            )
        lines.append("")
        lines.extend(class_interpretation(label))
        lines.append("")

    lines += [
        "## Files Generated",
        "",
        "- `01_class_distribution.png`",
        "- `02_missingness_by_metric_family.png`",
        "- `03_signature_heatmap.png`",
        "- `04_key_metric_distributions.png`",
        "- `05_*.png` pair plots for physical relationships",
        "- `metric_summary_by_class.csv`",
        "- `missingness_by_class_family.csv`",
        "- `top_discriminant_features_by_class.csv`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def class_interpretation(label: str) -> list[str]:
    text = {
        "clean_traffic": [
            "Interpretation: nominal windows should have no dominant stress signature.",
            "If clean windows overlap with loss or interference, they should be audited for tails, RF instability, or partial metrics.",
        ],
        "controlled_delay": [
            "Interpretation: latency should increase while UPF/server CPU and radio metrics remain normal.",
            "Bad delay windows usually have missing `direct_*` latency metrics and should not be used as delay evidence.",
        ],
        "tunnel_packet_loss": [
            "Interpretation: packet loss creates tail latency and irregular transport behavior rather than stable added delay.",
            "This class can overlap with controlled delay if only latency magnitude is considered.",
        ],
        "far_ue_poor_radio": [
            "Interpretation: weak coverage should primarily appear through RSRP/CQI/MCS degradation.",
            "It differs from interference because the root issue is signal level rather than injected noise.",
        ],
        "radio_interference": [
            "Interpretation: interference should appear as PRACH/I0 or noise movement, MCS drops, BLER changes, and throughput instability.",
            "It can overlap with far-UE behavior because both are radio-layer causes.",
        ],
        "upf_stress": [
            "Interpretation: UPF CPU pressure is the root-cause metric; latency is secondary.",
            "It differs from controlled delay because CPU is high, not merely latency.",
        ],
        "server_stress": [
            "Interpretation: server CPU pressure is the root cause; UPF/radio can remain normal.",
            "It differs from UPF stress by the compute location of the bottleneck.",
        ],
    }
    return text.get(label, ["Interpretation: inspect top features and missingness for this class."])


def fmt(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 10000 or (abs(value) < 0.001 and value != 0):
        return f"{value:.3g}"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {x.strip() for x in args.exclude_classes.split(",") if x.strip()}
    df = pd.read_csv(args.csv)
    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column")
    if exclude:
        df = df[~df["label"].isin(exclude)].copy()
    df = ensure_numeric(df)
    labels = ordered_labels(sorted(df["label"].dropna().unique()))

    save_class_distribution(df, labels, out_dir)
    missingness = save_missingness(df, labels, out_dir)
    save_signature_heatmap(df, labels, out_dir)
    save_boxplots(df, labels, out_dir)
    save_pair_plots(df, labels, out_dir)
    top_features = compute_top_features(df, labels, args.top_features, out_dir)
    metric_summary = save_metric_summary(df, labels, out_dir)
    write_report(df, labels, out_dir, top_features, metric_summary, missingness)

    print(f"Input: {args.csv}")
    print(f"Rows analyzed: {len(df)}")
    print(f"Classes: {', '.join(labels)}")
    print(f"Wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()
