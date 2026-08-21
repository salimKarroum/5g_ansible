#!/usr/bin/env python3
"""Summarize network signatures from one experiment run.

The script reads split Prometheus task windows when available. It is intended
for network interpretation before ML dataset integration.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FAMILIES: dict[str, tuple[str, ...]] = {
    "latency_direct": ("direct_latest", "direct_mean", "direct_p50", "direct_p95", "direct_p99", "gtp_teid_latency"),
    "tcp_correlation": ("tcp_ack_no_match", "tcp_pending_evicted", "tcp_data_observed", "tcp_ack_observed", "gtp_tcp"),
    "same_packet": ("same_packet",),
    "radio": ("rf_oai_gnb", "oai_gnb", "rsrp", "mcs", "bler", "cqi", "prach", "snr", "i0"),
    "throughput": ("throughput", "goodput", "mac_throughput"),
    "compute": ("container_cpu", "container_memory", "upf_cpu", "server_cpu", "node_cpu"),
}


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def open_text(path: Path):
    return gzip.open(path, "rt", newline="", encoding="utf-8") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def as_float(value: str) -> float | None:
    try:
        parsed = float(value)
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


def family_for(query_name: str, metric_name: str) -> str | None:
    text = f"{query_name} {metric_name}".lower()
    for family, patterns in FAMILIES.items():
        if any(pattern in text for pattern in patterns):
            return family
    return None


def task_name_from_path(path: Path, run_dir: Path) -> str:
    try:
        rel = path.relative_to(run_dir / "by_window" / "task")
        return str(rel.parent).replace("\\", "/")
    except ValueError:
        return path.parent.name


def find_task_csvs(run_dir: Path) -> list[Path]:
    task_dir = run_dir / "by_window" / "task"
    if not task_dir.exists():
        return []
    candidates = list(task_dir.glob("*/prometheus_timeseries.csv")) + list(task_dir.glob("*/prometheus_timeseries.csv.gz"))
    return sorted(candidates)


def summarize_csv(path: Path) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    with open_text(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            value = as_float(row.get("value", ""))
            if value is None:
                continue
            family = family_for(row.get("query_name", ""), row.get("__name__", ""))
            if family:
                values[family].append(value)
    return values


def format_value(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:.2f}"
    return f"{value:.4g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", help="Optional CSV summary output path")
    parser.add_argument("--max-tasks", type=int, default=80)
    args = parser.parse_args()

    run_dir = resolve(args.run)
    if not run_dir.exists():
        raise SystemExit(f"missing run directory: {run_dir}")

    task_csvs = find_task_csvs(run_dir)
    if not task_csvs:
        raise SystemExit(f"no by_window/task Prometheus CSVs found in {run_dir}")

    rows: list[dict[str, str]] = []
    print(f"Run: {run_dir}")
    print(f"Task windows: {len(task_csvs)}")
    print()
    header = f"{'task':64s} {'family':18s} {'n':>8s} {'mean':>10s} {'p95':>10s} {'max':>10s}"
    print(header)
    print("-" * len(header))

    for path in task_csvs[: args.max_tasks]:
        task = task_name_from_path(path, run_dir)
        summary = summarize_csv(path)
        for family in sorted(FAMILIES):
            vals = summary.get(family, [])
            if not vals:
                continue
            row = {
                "task": task,
                "family": family,
                "count": str(len(vals)),
                "mean": format_value(sum(vals) / len(vals)),
                "p95": format_value(percentile(vals, 0.95)),
                "max": format_value(max(vals)),
            }
            rows.append(row)
            print(f"{task[:64]:64s} {family:18s} {len(vals):8d} {row['mean']:>10s} {row['p95']:>10s} {row['max']:>10s}")

    if args.out:
        out = resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["task", "family", "count", "mean", "p95", "max"])
            writer.writeheader()
            writer.writerows(rows)
        print()
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
