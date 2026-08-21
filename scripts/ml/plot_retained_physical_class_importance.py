from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


ROOT = Path("results/ml_dataset_v3")
DATASET_CSV = ROOT / "window_features_pfe_uncorr12_plus_farue_no_servercpu.csv"
FEATURES_CSV = ROOT / "no_servercpu_plus_farue_plots" / "non_correlated_physical_features_no_servercpu.csv"
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
    "controlled_delay": "Controlled delay",
    "controlled_jitter": "Controlled jitter",
    "far_ue_poor_radio": "Far UE poor radio",
    "radio_interference": "Radio interference",
    "tunnel_bandwidth": "Tunnel bandwidth",
    "tunnel_packet_loss": "Tunnel packet loss",
    "upf_stress": "UPF stress",
}

CLASS_DIRECT_FAMILY = {
    "controlled_delay": "Latency eBPF",
    "controlled_jitter": "Latency eBPF",
    "far_ue_poor_radio": "Far UE radio",
    "radio_interference": "Radio interference",
    "tunnel_bandwidth": "Throughput",
    "tunnel_packet_loss": "TCP correlation",
    "upf_stress": "UPF compute",
}

FAMILY_COLORS = {
    "Latency eBPF": "#4DAF4A",
    "Throughput": "#984EA3",
    "TCP correlation": "#FF7F00",
    "Radio interference": "#A65628",
    "Far UE radio": "#F781BF",
    "UPF compute": "#377EB8",
}


def train_direct_family_importance(data: pd.DataFrame, feature_info: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    feature_info = feature_info[feature_info["feature"].isin(data.columns)].copy()
    labels = dict(zip(feature_info["feature"], feature_info["label"]))
    families = dict(zip(feature_info["feature"], feature_info["family"]))

    for cls in CLASS_ORDER:
        family = CLASS_DIRECT_FAMILY[cls]
        features = feature_info.loc[feature_info["family"] == family, "feature"].tolist()
        if not features:
            continue

        x_raw = data[features].apply(pd.to_numeric, errors="coerce").values
        x = SimpleImputer(strategy="median").fit_transform(x_raw)
        y = (data["label"].astype(str).values == cls).astype(int)

        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=1,
            class_weight="balanced",
        )
        clf.fit(x, y)

        for feature, importance in zip(features, clf.feature_importances_):
            rows.append(
                {
                    "class": cls,
                    "class_label": CLASS_LABELS[cls],
                    "feature": feature,
                    "feature_label": labels.get(feature, feature),
                    "family": families.get(feature, family),
                    "importance": float(importance),
                    "scope": "retained_features_direct_family",
                }
            )

    return pd.DataFrame(rows)


def plot_importance(importance: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(14.2, 8.6))
    fig.subplots_adjust(left=0.18, right=0.98, top=0.97, bottom=0.15, hspace=0.82, wspace=0.56)

    for idx, cls in enumerate(CLASS_ORDER):
        ax = axes[idx // 2, idx % 2]
        sub = importance[importance["class"] == cls].copy()
        sub = sub.sort_values("importance", ascending=True)

        colors = [FAMILY_COLORS.get(family, "#999999") for family in sub["family"]]
        ax.barh(
            sub["feature_label"],
            sub["importance"],
            color=colors,
            edgecolor="#333333",
            linewidth=0.4,
        )

        ax.set_title(CLASS_LABELS[cls], fontsize=13.2, fontweight="bold", loc="left")
        ax.set_xlim(0, 1.02)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9.8)
        ax.tick_params(axis="x", labelsize=9)
        if idx >= 5:
            ax.set_xlabel("Random Forest importance", fontsize=9.5)
        else:
            ax.set_xlabel("")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    for ax in axes.flat[len(CLASS_ORDER) :]:
        ax.axis("off")

    used_families = [CLASS_DIRECT_FAMILY[cls] for cls in CLASS_ORDER]
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=family, markerfacecolor=FAMILY_COLORS[family], markersize=9)
        for family in FAMILY_COLORS
        if family in used_families
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10.3,
        bbox_to_anchor=(0.5, 0.035),
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"physical_rca_signature_overview.{ext}", dpi=220, bbox_inches="tight")


def main() -> None:
    data = pd.read_csv(DATASET_CSV)
    data = data[data["label"].astype(str).isin(CLASS_ORDER)].copy()
    feature_info = pd.read_csv(FEATURES_CSV)

    importance = train_direct_family_importance(data, feature_info)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUT_DIR / "physical_rca_signature_overview_retained_importance.csv", index=False)
    plot_importance(importance)

    print(f"Wrote: {OUT_DIR / 'physical_rca_signature_overview.png'}")
    print(f"Wrote: {OUT_DIR / 'physical_rca_signature_overview.pdf'}")
    print(f"Wrote: {OUT_DIR / 'physical_rca_signature_overview_retained_importance.csv'}")


if __name__ == "__main__":
    main()
