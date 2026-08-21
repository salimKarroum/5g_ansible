from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path("results/ml_dataset_v3")
SIGNATURE_DIR = ROOT / "single_model_physical_signature_scores_no_servercpu"
OUT_DIR = ROOT / "presentation_physical_signature_plots"


CLASS_ORDER = [
    "controlled_delay",
    "controlled_jitter",
    "far_ue_poor_radio",
    "radio_interference",
    "tunnel_bandwidth",
    "tunnel_packet_loss",
    "upf_stress",
]

CLASS_LABELS = {
    "controlled_delay": "Controlled\ndelay",
    "controlled_jitter": "Controlled\njitter",
    "far_ue_poor_radio": "Far UE\npoor radio",
    "radio_interference": "Radio\ninterference",
    "server_stress": "Server\nstress",
    "tunnel_bandwidth": "Tunnel\nbandwidth",
    "tunnel_packet_loss": "Tunnel\npacket loss",
    "upf_stress": "UPF\nstress",
}

FEATURE_LABELS = {
    "sig_controlled_delay_latency_level": "Latency\nlevel",
    "sig_controlled_jitter_latency_variability": "Latency\nvariability",
    "sig_far_ue_radio_degradation": "Radio signal\ndegradation",
    "sig_radio_interference_prach_noise": "PRACH\nnoise",
    "sig_tunnel_bandwidth_throughput": "Tunnel\nthroughput",
    "sig_tunnel_packet_loss_tcp_mismatch": "TCP/GTP\nmismatch",
    "sig_upf_stress_cpu": "UPF\nCPU",
}


def load_signature_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(SIGNATURE_DIR / "window_features_physical_signature_scores_no_servercpu.csv")
    feature_cols = [c for c in df.columns if c.startswith("sig_")]
    values = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    values["label"] = df["label"].astype(str)
    medians = values.groupby("label")[feature_cols].median().reindex(CLASS_ORDER)

    # Normalize each physical signature independently so one high-amplitude
    # signal, such as server CPU, does not hide the other signatures.
    col_min = medians.min(axis=0)
    col_range = (medians.max(axis=0) - col_min).replace(0, np.nan)
    matrix = ((medians - col_min) / col_range).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matrix = matrix.rename(index=CLASS_LABELS, columns=FEATURE_LABELS)

    importance = pd.read_csv(SIGNATURE_DIR / "global_importance.csv")
    importance["label"] = importance["feature"].map(FEATURE_LABELS)
    importance = importance.sort_values("importance", ascending=True)

    summary = pd.read_csv(SIGNATURE_DIR / "summary_metrics.csv").iloc[0]
    return matrix, importance, summary


def draw_overview(matrix: pd.DataFrame, importance: pd.DataFrame, summary: pd.Series) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    top = pd.read_csv(ROOT / "physical_specialist_ovr" / "physical_specialist_top4_features_labeled.csv")
    top_plot = top[top["class"].isin(CLASS_ORDER)].copy()
    report = pd.read_csv(ROOT / "physical_specialist_ovr" / "class_report.csv")
    cm = pd.read_csv(ROOT / "physical_specialist_ovr" / "confusion_matrix.csv")
    acc = np.trace(cm.drop(columns=["true_label"]).values) / cm.drop(columns=["true_label"]).values.sum()
    macro_f1 = report["f1"].mean()

    class_labels = {
        "controlled_delay": "Controlled delay",
        "controlled_jitter": "Controlled jitter",
        "far_ue_poor_radio": "Far UE poor radio",
        "radio_interference": "Radio interference",
        "server_stress": "Server stress",
        "tunnel_bandwidth": "Tunnel bandwidth",
        "tunnel_packet_loss": "Tunnel packet loss",
        "upf_stress": "UPF stress",
    }
    family_colors = {
        "Latency": "#4E79A7",
        "TCP/loss": "#F28E2B",
        "Far UE radio": "#59A14F",
        "Radio interference": "#8CD17D",
        "Server compute": "#B07AA1",
        "Throughput": "#EDC948",
        "UPF compute": "#76B7B2",
    }

    fig, axes = plt.subplots(4, 2, figsize=(14.5, 9.2))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.21, hspace=0.86, wspace=0.56)

    for idx, cls in enumerate(CLASS_ORDER):
        ax = axes[idx // 2, idx % 2]
        sub = top_plot[top_plot["class"] == cls].sort_values("mean_importance", ascending=True)
        families = sub["family"].tolist()
        if cls in {"controlled_delay", "controlled_jitter"}:
            families = ["Latency" if label.startswith("Latency") else fam for label, fam in zip(sub["feature_label"], families)]
        colors = [family_colors.get(fam, "#999999") for fam in families]
        ax.barh(sub["feature_label"], sub["mean_importance"], color=colors, edgecolor="#333333", linewidth=0.4)
        ax.set_title(class_labels[cls], fontsize=13.2, fontweight="bold", loc="left")
        ax.set_xlim(0, max(0.58, float(top_plot["mean_importance"].max()) * 1.08))
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9.6)
        ax.tick_params(axis="x", labelsize=9)
        if idx >= 6:
            ax.set_xlabel("Model feature importance", fontsize=9.5)
        else:
            ax.set_xlabel("")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes.flat[len(CLASS_ORDER) :]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=fam, markerfacecolor=color, markersize=9)
        for fam, color in family_colors.items()
        if fam in set(top_plot["family"])
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10.3, bbox_to_anchor=(0.5, 0.075))
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"physical_rca_signature_overview.{ext}", dpi=220, bbox_inches="tight")


def add_box(ax, xy, width, height, title, lines, facecolor, edgecolor="#2F4B7C"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + width / 2, y + height - 0.09, title, ha="center", va="top", fontsize=11.6, fontweight="bold")
    ax.text(
        x + 0.04,
        y + height - 0.23,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=9.2,
        linespacing=1.18,
    )


def add_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.8,
        color="#555555",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arrow)


def draw_workflow(summary: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.0))
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.96,
        "End-to-end RCA workflow implemented in this project",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.02,
        0.90,
        "The pipeline connects real 5G experiments, monitoring, supervised learning, interpretability, and online RCA output.",
        fontsize=12.5,
        color="#333333",
        ha="left",
        va="top",
    )

    y = 0.46
    w = 0.155
    h = 0.37
    xs = [0.03, 0.235, 0.44, 0.645, 0.85]
    colors = ["#E8F3FF", "#F0F7EA", "#FFF4DA", "#F7EAFE", "#EAF7F5"]

    boxes = [
        (
            "Controlled\n5G anomalies",
            [
                "8 root causes",
                "radio, transport,",
                "latency, compute",
                "repeatable scenarios",
            ],
        ),
        (
            "Monitoring\nstack",
            [
                "Prometheus metrics",
                "OAI gNB + Open5GS",
                "eBPF latency/TCP",
                "UPF and radio KPIs",
            ],
        ),
        (
            "Windowed\nRCA dataset",
            [
                "384 labeled windows",
                "48 windows/class",
                "run-level metadata",
                "balanced classes",
            ],
        ),
        (
            "Physical\nsignatures",
            [
                "latency level/jitter",
                "PRACH noise, RSRP",
                "TCP/GTP mismatch",
                "UPF CPU",
            ],
        ),
        (
            "RCA model\nand online service",
            [
                "Random Forest",
                f"Accuracy {float(summary['accuracy']):.3f}",
                f"Macro-F1 {float(summary['macro_f1']):.3f}",
                "Prometheus /metrics",
            ],
        ),
    ]

    for idx, (x, (title, lines)) in enumerate(zip(xs, boxes)):
        add_box(ax, (x, y), w, h, title, lines, colors[idx])
        if idx < len(xs) - 1:
            add_arrow(ax, (x + w, y + h / 2), (xs[idx + 1], y + h / 2))

    ax.text(
        0.05,
        0.29,
        "Offline evaluation",
        fontsize=13,
        fontweight="bold",
        color="#2F4B7C",
    )
    ax.text(
        0.05,
        0.23,
        "Strict leave-one-run-out validation prevents the classifier from seeing windows from the same run during training and testing.",
        fontsize=11,
        color="#333333",
    )

    ax.text(
        0.05,
        0.13,
        "Online demonstration",
        fontsize=13,
        fontweight="bold",
        color="#2F4B7C",
    )
    ax.text(
        0.05,
        0.07,
        "The trained model is deployed as a service that queries live Prometheus data and exports stable RCA predictions as Prometheus metrics.",
        fontsize=11,
        color="#333333",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"rca_project_workflow_overview.{ext}", dpi=220, bbox_inches="tight")


def draw_causal_signature_map() -> None:
    classes = [
        ("Controlled delay", "Latency level", "direct_latest_ms_p95 / mean", "#E8F3FF"),
        ("Controlled jitter", "Latency variability", "direct_latest_ms_std", "#E8F3FF"),
        ("Far UE poor radio", "Radio signal degradation", "RSRP, BLER, MCS", "#F0F7EA"),
        ("Radio interference", "Radio noise", "PRACH I0", "#F0F7EA"),
        ("Tunnel bandwidth", "Tunnel throughput", "slice_throughput", "#FFF4DA"),
        ("Tunnel packet loss", "TCP/GTP mismatch", "tcp_ack_no_match / observed", "#FFF4DA"),
        ("UPF stress", "UPF compute load", "upf_cpu_cores_p95", "#EAF7F5"),
        ("Server stress", "No direct feature retained", "needs server CPU metric", "#F4F4F4"),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.03,
        0.96,
        "Physically defensible RCA signatures",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.03,
        0.90,
        "Each class should be explained by a direct physical indicator. "
        "If the direct indicator is removed, the class becomes explainable only through indirect effects.",
        fontsize=12.5,
        color="#333333",
        ha="left",
        va="top",
    )

    headers = ["Anomaly class", "Direct physical signature", "Representative metrics"]
    x_positions = [0.05, 0.35, 0.64]
    widths = [0.27, 0.25, 0.31]
    y_top = 0.80
    row_h = 0.075

    for x, w, h in zip(x_positions, widths, headers):
        ax.add_patch(
            FancyBboxPatch(
                (x, y_top),
                w,
                row_h,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor="#2F4B7C",
                edgecolor="#2F4B7C",
            )
        )
        ax.text(x + 0.015, y_top + row_h / 2, h, color="white", fontsize=12.5, fontweight="bold", va="center")

    for idx, (cls, signature, metric, color) in enumerate(classes):
        y = y_top - (idx + 1) * row_h
        for col, (x, w) in enumerate(zip(x_positions, widths)):
            face = color if cls != "Server stress" else "#F8F8F8"
            edge = "#DDDDDD" if cls != "Server stress" else "#B00020"
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    w,
                    row_h * 0.88,
                    boxstyle="round,pad=0.01,rounding_size=0.012",
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=1.2,
                )
            )
        color_text = "#B00020" if cls == "Server stress" else "#111111"
        ax.text(x_positions[0] + 0.015, y + row_h * 0.44, cls, fontsize=11.3, va="center", color=color_text, fontweight="bold")
        ax.text(x_positions[1] + 0.015, y + row_h * 0.44, signature, fontsize=11.0, va="center", color=color_text)
        ax.text(x_positions[2] + 0.015, y + row_h * 0.44, metric, fontsize=10.7, va="center", color=color_text)

    ax.text(
        0.05,
        0.08,
        "Interpretation for the report: UPF CPU is physically valid for UPF stress only. "
        "For server stress, a direct server-side CPU metric should be retained, otherwise the class relies on indirect correlations.",
        fontsize=11.5,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#FFF8E5", "edgecolor": "#CC9900"},
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"causal_physical_signature_map.{ext}", dpi=220, bbox_inches="tight")


def main() -> None:
    matrix, importance, summary = load_signature_matrix()
    draw_overview(matrix, importance, summary)
    draw_workflow(summary)
    draw_causal_signature_map()
    print(f"Wrote: {OUT_DIR / 'physical_rca_signature_overview.png'}")
    print(f"Wrote: {OUT_DIR / 'physical_rca_signature_overview.pdf'}")
    print(f"Wrote: {OUT_DIR / 'rca_project_workflow_overview.png'}")
    print(f"Wrote: {OUT_DIR / 'rca_project_workflow_overview.pdf'}")
    print(f"Wrote: {OUT_DIR / 'causal_physical_signature_map.png'}")
    print(f"Wrote: {OUT_DIR / 'causal_physical_signature_map.pdf'}")


if __name__ == "__main__":
    main()
