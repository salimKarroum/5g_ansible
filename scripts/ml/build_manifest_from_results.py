#!/usr/bin/env python3
"""
Scan results/ directories and append new entries to manifest_clean_serverpod.csv.

Label mapping is based on scenario name in the path. Run this after each
new experiment to update the manifest before running run_hybrid_rca_pipeline.py.

Usage:
    python3 scripts/ml/build_manifest_from_results.py
    python3 scripts/ml/build_manifest_from_results.py --results-dir results/ --manifest results/ml_dataset_v3/manifest_clean_serverpod.csv
"""
import argparse
import csv
import gzip
from pathlib import Path

# Maps a substring in the path to a label.
# Order matters: first match wins.
LABEL_RULES = [
    # --- Interference / radio ---
    ("07_fit02_interference",           "radio_interference"),
    ("09_fit28_spatial_control",        "radio_interference"),
    ("10_fit02_bidir_interference",     "radio_interference"),
    ("11_fit28_bidir_interference",     "radio_interference"),
    ("26_interference_gain_levels",     "radio_interference"),
    ("21_decomp_far_ue_radio",          "far_ue_poor_radio"),

    # --- UPF / core stress ---
    ("22_decomp_upf_cpu_stress",        "upf_stress"),
    ("24_upf_stress_levels",            "upf_stress"),

    # --- Server stress ---
    ("23_decomp_iperf_server_cpu_stress", "server_stress"),
    ("25_server_stress_levels",         "server_stress"),

    # --- Load / contention ---
    ("03_tcp_load_ramp",                "load_ramp"),
    ("04_cross_slice_contention",       "multi_ue_contention"),
    ("27_multi_ue_contention_levels",   "multi_ue_contention"),
    ("29_load_ramp_levels",             "load_ramp"),

    # --- Transport impairments ---
    # Reference steps must come BEFORE their parent scenario patterns (first match wins).
    ("no_netem_reference",              "clean_traffic"),
    ("no_loss_reference",               "clean_traffic"),
    ("no_bandwidth_limit_reference",    "clean_traffic"),
    ("28_controlled_delay_levels",      "controlled_delay"),
    ("v03_controlled_delay",            "controlled_delay"),
    ("v07_controlled_packet_loss",      "tunnel_packet_loss"),
    ("v08_server_bandwidth_limit",      "tunnel_bandwidth"),
    ("30_packet_loss_levels",           "tunnel_packet_loss"),
    ("31_bandwidth_limit_levels",       "tunnel_bandwidth"),

    # --- Baseline (no anomaly) ---
    ("00_reference_baseline",           "clean_traffic"),
    ("01_clean_near_baseline",          "clean_traffic"),
    ("20_decomp_baseline",              "clean_traffic"),
]

# Scenarios to skip entirely (no useful label)
SKIP_PATTERNS = [
    "02_near_vs_far_radio_condition",
    "05_far_ue_stress",
    "06_mixed_ul_dl",
    "12_physical_near_far",
    "13_far_light_under",
]


def assign_label(path_str: str) -> str | None:
    for pattern, label in LABEL_RULES:
        if pattern in path_str:
            return label
    for skip in SKIP_PATTERNS:
        if skip in path_str:
            return None
    return None


def find_windows(results_dir: Path) -> list[dict]:
    """Find all prometheus_timeseries.csv.gz files under by_window/task/"""
    windows = []
    for gz in sorted(results_dir.rglob("by_window/task/*/prometheus_timeseries.csv.gz")):
        path_str = str(gz)
        label = assign_label(path_str)
        if label is None:
            continue

        # Extract scenario name from path
        parts = gz.parts
        task_idx = next((i for i, p in enumerate(parts) if p == "task"), None)
        scenario_folder = parts[task_idx + 1] if task_idx is not None else ""
        scenario = scenario_folder.split("__")[0] if "__" in scenario_folder else scenario_folder

        windows.append({
            "path": path_str,
            "label": label,
            "scenario": scenario,
            "phase": scenario_folder,
            "reference_path": "",
        })
    return windows


def load_existing_paths(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    with manifest.open(newline="") as f:
        return {row["path"] for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--manifest", default="results/ml_dataset_v3/manifest_clean_serverpod.csv")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be added without writing")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    manifest = Path(args.manifest)

    existing = load_existing_paths(manifest)
    all_windows = find_windows(results_dir)
    new_windows = [w for w in all_windows if w["path"] not in existing]

    if not new_windows:
        print("Aucune nouvelle fenêtre trouvée — manifest déjà à jour.")
        return

    # Count by label
    from collections import Counter
    counts = Counter(w["label"] for w in new_windows)
    print(f"Nouvelles fenêtres trouvées : {len(new_windows)}")
    for label, n in sorted(counts.items()):
        print(f"  {label}: {n}")

    if args.dry_run:
        print("\n--dry-run : rien n'a été écrit.")
        return

    # Append to manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "label", "scenario", "phase", "reference_path"]

    write_header = not manifest.exists()
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(new_windows)

    print(f"\nAjouté {len(new_windows)} lignes à {manifest}")
    print("Relance maintenant : python3 scripts/ml/run_hybrid_rca_pipeline.py")


if __name__ == "__main__":
    main()
