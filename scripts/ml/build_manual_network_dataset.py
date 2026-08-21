#!/usr/bin/env python3
"""Build a windowed ML dataset from validated manual network runs.

The input is the `results/manual-network` tree produced by
`scripts/validation/run_validated_network_campaign.sh` and by the earlier
manual ogstun validation runs.  Each level directory is expected to contain a
`prometheus_timeseries.csv` file and may contain ping/iperf evidence.

The output CSV is compatible with the existing ML scripts:

    path, run, label, scenario, phase, reference_path, <numeric features...>

Only numeric features are written after the first six metadata columns.
Configured intensity values are intentionally not emitted as features to avoid
label leakage.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
META_COLS = ["path", "run", "label", "scenario", "phase", "reference_path"]

SKIP_QUERY_PATTERNS = (
    "_created",
)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def ping_loss_pct(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"([0-9.]+)% packet loss", text)
    if not matches:
        return None
    return as_float(matches[-1])


def ul_ifb_packets(path: Path, direction: str) -> float | None:
    if not path.exists() or direction != "ul":
        return None
    section = f"=== qdisc ifb5g{direction} ==="
    in_section = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip() == section:
            in_section = True
            continue
        if in_section and " Sent " in f" {line} ":
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "Sent":
                return as_float(parts[3])
    return None


def as_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def sanitize(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "metric"


def should_skip_query(name: str) -> bool:
    return any(pattern in name for pattern in SKIP_QUERY_PATTERNS)


def stats_features(name: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    first = values[0]
    last = values[-1]
    minimum = ordered[0]
    maximum = ordered[-1]
    return {
        f"{name}_count": float(len(values)),
        f"{name}_mean": sum(values) / len(values),
        f"{name}_min": minimum,
        f"{name}_max": maximum,
        f"{name}_std": stddev(values),
        f"{name}_median": percentile(values, 0.50),
        f"{name}_p95": percentile(values, 0.95),
        f"{name}_last": last,
        f"{name}_range": maximum - minimum,
        f"{name}_delta": last - first,
        f"{name}_nonzero_ratio": sum(1 for x in values if abs(x) > 1e-12) / len(values),
    }


def parse_ping_features(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    rtts = [float(x) for x in re.findall(r"time=([0-9.]+)\s*ms", text)]
    features: dict[str, float] = {}
    if rtts:
        for key, value in stats_features("probe_ping_rtt_ms", rtts).items():
            features[key] = value
    match = re.search(r"([0-9.]+)% packet loss", text)
    if match:
        features["probe_ping_packet_loss_pct"] = float(match.group(1))
    match = re.search(r"(\d+) packets transmitted,\s*(\d+) received", text)
    if match:
        sent = float(match.group(1))
        received = float(match.group(2))
        features["probe_ping_packets_sent"] = sent
        features["probe_ping_packets_received"] = received
        features["probe_ping_packets_lost"] = sent - received
    return features


def parse_iperf_features(level_dir: Path) -> dict[str, float]:
    features: dict[str, float] = {}
    candidates = [
        ("dl", level_dir / "iperf_dl.json"),
        ("ul", level_dir / "iperf_ul.json"),
    ]
    for direction, path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            features[f"probe_iperf_{direction}_error"] = 1.0
            continue
        end = data.get("end", {})
        sent = end.get("sum_sent", {})
        received = end.get("sum_received", {})
        sent_bps = as_float(sent.get("bits_per_second"))
        recv_bps = as_float(received.get("bits_per_second"))
        retransmits = as_float(sent.get("retransmits"))
        if sent_bps is not None:
            features[f"probe_iperf_{direction}_sender_mbps"] = sent_bps / 1e6
        if recv_bps is not None:
            features[f"probe_iperf_{direction}_receiver_mbps"] = recv_bps / 1e6
        if retransmits is not None:
            features[f"probe_iperf_{direction}_retransmits"] = retransmits

        interval_rates: list[float] = []
        for interval in data.get("intervals", []):
            stream_sum = interval.get("sum", {})
            bps = as_float(stream_sum.get("bits_per_second"))
            if bps is not None:
                interval_rates.append(bps / 1e6)
        for key, value in stats_features(f"probe_iperf_{direction}_interval_mbps", interval_rates).items():
            features[key] = value
    return features


def read_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(float(path.read_text(encoding="utf-8").strip()))
    except ValueError:
        return None


def run_name_for(level_dir: Path) -> str:
    for part in reversed(level_dir.parts):
        if part.startswith("tcp-paper-"):
            return part
        if part.startswith("validated_network_campaign_"):
            return part
        if part.startswith("controlled_delay_ogstun_levels_"):
            return part
        if part.startswith("controlled_jitter_ogstun_"):
            return part
        if part.startswith("tunnel_packet_loss_ogstun_levels_"):
            return part
        if part.startswith("tunnel_bandwidth_ogstun_levels_"):
            return part
        if part.startswith("tunnel_bandwidth_dl_ogstun_levels_"):
            return part
    return level_dir.parent.name


def direction_for_level(level_dir: Path, env: dict[str, str] | None = None) -> str:
    if env and env.get("traffic_direction"):
        return env["traffic_direction"]
    name = level_dir.name.lower()
    if name.endswith("__ul") or "__ul" in name:
        return "ul"
    if name.endswith("__dl") or "__dl" in name or name.endswith("_dl"):
        return "dl"
    return ""


def infer_tcp_paper_metadata(level_dir: Path) -> dict[str, str] | None:
    text = str(level_dir).replace("\\", "/")
    name = level_dir.name

    if "by_window/stress_phase" in text and "24_upf_stress_levels" in text:
        return {
            "scenario": "upf_stress",
            "label": "upf_stress" if "stress_on" in name else "clean_traffic",
            "level_name": name,
        }

    if "by_window/interference_phase" in text and "26_interference_gain_levels" in text:
        strong_gain = ("gain55" in name) or ("gain70" in name)
        if "noise_on" in name and strong_gain:
            label = "radio_interference"
        elif "clean_before_noise" in name or "recovery_after_noise" in name:
            label = "clean_traffic"
        else:
            label = "__skip__"
        return {
            "scenario": "radio_interference",
            "label": label,
            "level_name": name,
        }

    if "by_window/task" in text and "34_far_ue_radio_levels" in text:
        if "far_qhat02_solo" in name:
            label = "far_ue_poor_radio"
        elif "near_reference_qhat01_qhat03" in name:
            label = "clean_traffic"
        else:
            label = "__skip__"
        return {
            "scenario": "far_ue_poor_radio",
            "label": label,
            "level_name": name,
        }

    return None


def infer_metadata(level_dir: Path) -> dict[str, str]:
    env = parse_env(level_dir / "metadata.env")
    if env.get("scenario") and env.get("label"):
        return {
            "scenario": env.get("scenario", ""),
            "label": env.get("label", ""),
            "level_name": env.get("level_name", level_dir.name),
            "tc_mode": env.get("tc_mode", ""),
            "tc_value": env.get("tc_value", ""),
            "traffic_direction": direction_for_level(level_dir, env),
            "ul_match_mode": env.get("ul_match_mode", ""),
            "ue_host": env.get("ue_host", ""),
            "qhat_ssh_target": env.get("qhat_ssh_target", ""),
            "ue_ip": env.get("ue_ip", ""),
            "iperf_parallel": env.get("iperf_parallel", ""),
            "iperf_seconds": env.get("iperf_seconds", ""),
            "iperf_timeout_extra": env.get("iperf_timeout_extra", ""),
            "valid_for_dataset": env.get("valid_for_dataset", ""),
            "invalid_reason": env.get("invalid_reason", ""),
        }

    name = level_dir.name
    path_text = str(level_dir)

    tcp_paper = infer_tcp_paper_metadata(level_dir)
    if tcp_paper is not None:
        tcp_paper.setdefault("traffic_direction", direction_for_level(level_dir))
        tcp_paper.setdefault("tc_mode", "")
        tcp_paper.setdefault("tc_value", "")
        tcp_paper.setdefault("ul_match_mode", "")
        tcp_paper.setdefault("ue_host", "")
        tcp_paper.setdefault("qhat_ssh_target", "")
        tcp_paper.setdefault("ue_ip", "")
        tcp_paper.setdefault("iperf_parallel", "")
        tcp_paper.setdefault("iperf_seconds", "")
        tcp_paper.setdefault("iperf_timeout_extra", "")
        tcp_paper.setdefault("valid_for_dataset", "")
        tcp_paper.setdefault("invalid_reason", "")
        return tcp_paper
    if "tcp-paper-" in path_text:
        return {
            "scenario": "tcp_paper_unknown",
            "label": "__skip__",
            "level_name": name,
            "tc_mode": "",
            "tc_value": "",
            "traffic_direction": direction_for_level(level_dir),
            "ul_match_mode": "",
            "ue_host": "",
            "qhat_ssh_target": "",
            "ue_ip": "",
            "iperf_parallel": "",
            "iperf_seconds": "",
            "iperf_timeout_extra": "",
            "valid_for_dataset": "",
            "invalid_reason": "",
        }

    scenario = ""
    label = ""

    if "controlled_delay_ogstun_levels_" in path_text or name.startswith("delay_"):
        scenario = "controlled_delay"
        delay = int(re.search(r"delay_(\d+)ms", name).group(1)) if re.search(r"delay_(\d+)ms", name) else 0
        label = "clean_traffic" if delay == 0 else "controlled_delay"
    elif "controlled_jitter_ogstun_" in path_text or name.startswith("jitter_") or name in {"baseline", "delay100", "20", "50", "80"}:
        scenario = "controlled_jitter"
        if name == "baseline":
            label = "clean_traffic"
        elif name == "delay100" or "jitter_000" in name:
            label = "controlled_delay"
        else:
            label = "controlled_jitter"
    elif "tunnel_packet_loss_ogstun_levels_" in path_text or name.startswith("loss_"):
        scenario = "tunnel_packet_loss"
        match = re.search(r"loss_([0-9_]+)pct", name)
        loss = float(match.group(1).replace("_", ".")) if match else 0.0
        label = "clean_traffic" if loss == 0 else "tunnel_packet_loss"
    elif "tunnel_bandwidth_dl_ogstun_levels_" in path_text or name.startswith("rate_"):
        scenario = "tunnel_bandwidth_limit"
        match = re.search(r"rate_([0-9_]+)mbit", name)
        rate = float(match.group(1).replace("_", ".")) if match else 0.0
        label = "clean_traffic" if rate == 0 else "tunnel_bandwidth_limit"
    else:
        scenario = level_dir.parent.name
        label = scenario

    return {
        "scenario": scenario,
        "label": label,
        "level_name": name,
        "tc_mode": scenario.replace("tunnel_packet_loss", "loss").replace("tunnel_bandwidth_limit", "bandwidth") if scenario.startswith("tunnel_") else "",
        "tc_value": "",
        "traffic_direction": direction_for_level(level_dir),
        "ul_match_mode": "",
        "ue_host": "",
        "qhat_ssh_target": "",
        "ue_ip": "",
        "iperf_parallel": "",
        "iperf_seconds": "",
        "iperf_timeout_extra": "",
        "valid_for_dataset": "",
        "invalid_reason": "",
    }


def dataset_validity(level_dir: Path, metadata: dict[str, str]) -> tuple[bool, str]:
    if metadata.get("valid_for_dataset") == "0":
        return False, metadata.get("invalid_reason", "metadata_invalid")

    scenario = metadata.get("scenario", "")
    if scenario not in {
        "controlled_delay",
        "controlled_jitter",
        "tunnel_packet_loss",
        "tunnel_bandwidth_limit",
    }:
        return True, ""

    loss = ping_loss_pct(level_dir / "ping.txt")
    if loss is not None and loss >= 95.0:
        return False, f"ping_packet_loss_{loss:g}pct"

    direction = metadata.get("traffic_direction", "")
    tc_mode = metadata.get("tc_mode", "")
    tc_value = metadata.get("tc_value", "")
    if direction == "ul" and (tc_mode == "jitter" or tc_value not in {"", "0", "0.0"}):
        packets = ul_ifb_packets(level_dir / "tc_before_cleanup.txt", direction)
        if packets is not None and packets == 0:
            return False, "ul_ifb_zero_packets"

    return True, ""


def find_prometheus_csv(level_dir: Path) -> Path | None:
    for name in ("prometheus_timeseries.csv", "prometheus_timeseries.csv.gz"):
        candidate = level_dir / name
        if candidate.exists():
            return candidate
    return None


def find_level_dirs(manual_root: Path | None, explicit_runs: list[Path]) -> list[Path]:
    roots = ([] if manual_root is None else [manual_root]) + explicit_runs
    level_dirs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "by_window").exists():
            for path in root.rglob("*"):
                if path.is_dir() and find_prometheus_csv(path):
                    level_dirs.append(path)
            continue
        if find_prometheus_csv(root):
            level_dirs.append(root)
            continue
        for path in root.rglob("*"):
            if path.is_dir() and find_prometheus_csv(path):
                level_dirs.append(path)
    return sorted(set(level_dirs))


def metric_names_for_row(row: dict[str, str]) -> list[str]:
    query = sanitize(row.get("query_name", ""))
    if not query or should_skip_query(query):
        return []

    names = [query]
    pod_text = " ".join(
        str(row.get(key, ""))
        for key in ("pod", "container", "instance", "namespace", "job", "metric_json")
    ).lower()

    if query in {"container_cpu_cores_30s_any", "container_cpu_cores_5s"}:
        names.append("container_cpu_cores")
        if "upf" in pod_text:
            names.append("upf_cpu_cores")
    if query in {"container_memory_mib_any", "container_memory_mib"}:
        names.append("container_memory_mib")
        if "upf" in pod_text:
            names.append("upf_memory_mib")
    return sorted(set(names))


def build_windows_from_prometheus(level_dir: Path, window_seconds: int) -> list[dict[str, float]]:
    csv_path = find_prometheus_csv(level_dir)
    if csv_path is None:
        return []

    start_epoch = read_epoch(level_dir / "start_epoch.txt")
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    min_ts: float | None = None

    with open_text(csv_path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            value = as_float(row.get("value"))
            timestamp = as_float(row.get("timestamp"))
            if value is None or timestamp is None:
                continue
            if min_ts is None or timestamp < min_ts:
                min_ts = timestamp
            base = float(start_epoch) if start_epoch is not None else float(math.floor(min_ts or timestamp))
            window_idx = int(max(0, math.floor((timestamp - base) / window_seconds)))
            for metric_name in metric_names_for_row(row):
                buckets[window_idx][metric_name].append(value)

    windows: list[dict[str, float]] = []
    for window_idx in sorted(buckets):
        features: dict[str, float] = {}
        for metric_name, values in sorted(buckets[window_idx].items()):
            features.update(stats_features(metric_name, values))
        windows.append(features)
    return windows


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def write_csv(path: Path, rows: list[dict[str, object]], feature_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = META_COLS + feature_names
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in header})


def build_dataset(
    level_dirs: Iterable[Path],
    window_seconds: int,
    include_probe_features: bool,
    drop_clean: bool,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    manifest: list[dict[str, str]] = []
    feature_names: set[str] = set()

    for level_dir in level_dirs:
        metadata = infer_metadata(level_dir)
        if metadata["label"] == "__skip__":
            continue
        if drop_clean and metadata["label"] == "clean_traffic":
            continue
        is_valid, _invalid_reason = dataset_validity(level_dir, metadata)
        if not is_valid:
            continue

        windows = build_windows_from_prometheus(level_dir, window_seconds)
        if not windows:
            continue

        probe_features: dict[str, float] = {}
        if include_probe_features:
            probe_features.update(parse_ping_features(level_dir / "ping.txt"))
            probe_features.update(parse_iperf_features(level_dir))

        run_name = run_name_for(level_dir)
        for idx, features in enumerate(windows):
            row: dict[str, object] = {
                "path": f"{rel(level_dir)}#window_{idx:04d}",
                "run": run_name,
                "label": metadata["label"],
                "scenario": metadata["scenario"],
                "phase": metadata["level_name"],
                "reference_path": "",
            }
            row.update(features)
            if include_probe_features:
                row.update(probe_features)
            rows.append(row)
            feature_names.update(k for k in row if k not in META_COLS)

        manifest.append({
            "level_dir": rel(level_dir),
            "run": run_name,
            "scenario": metadata["scenario"],
            "label": metadata["label"],
            "level_name": metadata["level_name"],
            "tc_mode": metadata.get("tc_mode", ""),
            "tc_value": metadata.get("tc_value", ""),
            "traffic_direction": metadata.get("traffic_direction", ""),
            "ul_match_mode": metadata.get("ul_match_mode", ""),
            "ue_host": metadata.get("ue_host", ""),
            "qhat_ssh_target": metadata.get("qhat_ssh_target", ""),
            "ue_ip": metadata.get("ue_ip", ""),
            "iperf_parallel": metadata.get("iperf_parallel", ""),
            "iperf_seconds": metadata.get("iperf_seconds", ""),
            "iperf_timeout_extra": metadata.get("iperf_timeout_extra", ""),
            "valid_for_dataset": metadata.get("valid_for_dataset", ""),
            "invalid_reason": metadata.get("invalid_reason", ""),
            "windows": str(len(windows)),
        })

    ordered_features = sorted(feature_names)
    return rows, ordered_features, manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "level_dir",
        "run",
        "scenario",
        "label",
        "level_name",
        "tc_mode",
        "tc_value",
        "traffic_direction",
        "ul_match_mode",
        "ue_host",
        "qhat_ssh_target",
        "ue_ip",
        "iperf_parallel",
        "iperf_seconds",
        "iperf_timeout_extra",
        "valid_for_dataset",
        "invalid_reason",
        "windows",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, object]], feature_names: list[str], manifest: list[dict[str, str]]) -> None:
    label_counts = Counter(str(row["label"]) for row in rows)
    scenario_counts = Counter(str(row["scenario"]) for row in rows)
    lines = [
        "# Manual Network Dataset",
        "",
        f"Rows: {len(rows)}",
        f"Features: {len(feature_names)}",
        f"Level directories: {len(manifest)}",
        "",
        "## Labels",
        "",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Scenarios", ""])
    for scenario, count in sorted(scenario_counts.items()):
        lines.append(f"- `{scenario}`: {count}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- The first six columns are metadata and are ignored by the existing ML scripts.",
        "- UE identity and traffic direction are kept in `manual_network_manifest.csv` for audit, not as ML features.",
        "- Configured intensity is intentionally not emitted as a feature.",
        "- Probe features are included only when `--include-probe-features` is used.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-root", default="results/manual-network")
    parser.add_argument("--no-manual-root", action="store_true", help="Use only directories passed with --run.")
    parser.add_argument("--run", action="append", default=[], help="Specific manual run/campaign directory. Repeatable.")
    parser.add_argument("--out", default="results/manual-network/dataset/window_features_manual_network.csv")
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--include-probe-features", action="store_true")
    parser.add_argument("--drop-clean", action="store_true")
    parser.add_argument("--manifest-out", default="")
    parser.add_argument("--report-out", default="")
    args = parser.parse_args()

    manual_root = None if args.no_manual_root else resolve(args.manual_root)
    explicit_runs = [resolve(x) for x in args.run]
    level_dirs = find_level_dirs(manual_root, explicit_runs)
    if not level_dirs:
        raise SystemExit("ERROR: no level directories with Prometheus CSV found")

    rows, feature_names, manifest = build_dataset(
        level_dirs,
        args.window_seconds,
        args.include_probe_features,
        args.drop_clean,
    )
    if not rows:
        raise SystemExit("ERROR: no dataset rows produced")

    out = resolve(args.out)
    manifest_out = resolve(args.manifest_out) if args.manifest_out else out.with_name("manual_network_manifest.csv")
    report_out = resolve(args.report_out) if args.report_out else out.with_name("README.md")

    write_csv(out, rows, feature_names)
    write_manifest(manifest_out, manifest)
    write_report(report_out, rows, feature_names, manifest)

    print(f"level_dirs: {len(level_dirs)}")
    print(f"rows: {len(rows)}")
    print(f"features: {len(feature_names)}")
    print("labels:")
    for label, count in sorted(Counter(str(row["label"]) for row in rows).items()):
        print(f"  {label}: {count}")
    print(f"dataset: {out}")
    print(f"manifest: {manifest_out}")
    print(f"report: {report_out}")


if __name__ == "__main__":
    main()
