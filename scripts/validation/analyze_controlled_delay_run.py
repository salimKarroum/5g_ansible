#!/usr/bin/env python3
"""Network-first report for a controlled-delay validation run.

This script is intentionally not an ML step. It checks whether a run labelled
as controlled delay actually exposes a delay signature in the network metrics.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DIRECT_QUERIES = (
    "direct_latest_ms",
    "direct_mean_ms_5s",
    "direct_p95_ms_5s",
    "direct_p99_ms_5s",
)

SAME_PACKET_QUERIES = (
    "same_packet_upf_rtt_latest_ms",
    "same_packet_gnb_rtt_latest_ms",
    "same_packet_gap_latest_ms",
    "same_packet_upf_rtt_mean_ms_5s",
    "same_packet_gnb_rtt_mean_ms_5s",
    "same_packet_gap_mean_ms_5s",
    "same_packet_upf_rtt_p95_ms_5s",
    "same_packet_gnb_rtt_p95_ms_5s",
    "same_packet_gap_p95_ms_5s",
)

DEFAULT_QUERIES = (
    *DIRECT_QUERIES,
    *SAME_PACKET_QUERIES,
    "tcp_pending_evicted_hz_30s",
    "tcp_ack_no_match_hz_30s",
    "container_cpu_cores_30s_any",
    "mac_throughput_total_bps",
    "prach_i0_db",
    "oai_gnb_mac_rsrp_per_rnti",
    "oai_gnb_mac_mcs_per_rnti",
)


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open(newline="", encoding="utf-8")


def as_float(value: str | None) -> float | None:
    try:
        parsed = float(value if value is not None else "")
    except ValueError:
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


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    return f"{value:.4f}"


def task_delay_ms(task: str) -> int:
    match = re.search(r"delay_(\d+)ms", task)
    return int(match.group(1)) if match else 0


def task_direction(task: str) -> str:
    if task.endswith("__dl"):
        return "dl"
    if task.endswith("__ul"):
        return "ul"
    return "both"


def task_csvs(run_dir: Path) -> list[Path]:
    task_dir = run_dir / "by_window" / "task"
    if not task_dir.exists():
        return []
    return sorted(
        list(task_dir.glob("*/prometheus_timeseries.csv"))
        + list(task_dir.glob("*/prometheus_timeseries.csv.gz"))
    )


def query_counts(path: Path, wanted: set[str] | None = None) -> Counter[str]:
    counts: Counter[str] = Counter()
    with open_text(path) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            query = row.get("query_name", "")
            if wanted is None or query in wanted:
                counts[query] += 1
    return counts


def summarize_run(run_dir: Path, queries: tuple[str, ...]) -> list[dict[str, str]]:
    wanted = set(queries)
    rows: list[dict[str, str]] = []

    for path in task_csvs(run_dir):
        task = path.parent.name
        values_by_query: dict[str, list[float]] = defaultdict(list)
        with open_text(path) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                query = row.get("query_name", "")
                if query not in wanted:
                    continue
                value = as_float(row.get("value"))
                if value is None:
                    continue
                values_by_query[query].append(value)

        for query in queries:
            values = values_by_query.get(query, [])
            if not values:
                continue
            rows.append(
                {
                    "task": task,
                    "delay_ms": str(task_delay_ms(task)),
                    "direction": task_direction(task),
                    "query": query,
                    "count": str(len(values)),
                    "mean": fmt(sum(values) / len(values)),
                    "median": fmt(percentile(values, 0.50)),
                    "p95": fmt(percentile(values, 0.95)),
                    "max": fmt(max(values)),
                }
            )
    return rows


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def first_metric(rows: list[dict[str, str]], candidates: tuple[str, ...]) -> str | None:
    present = {row["query"] for row in rows}
    for query in candidates:
        if query in present:
            return query
    return None


def mean_for(rows: list[dict[str, str]], query: str, delay_ms: int, direction: str = "both") -> float | None:
    vals = [
        numeric(row, "mean")
        for row in rows
        if row["query"] == query
        and int(row["delay_ms"]) == delay_ms
        and row["direction"] == direction
    ]
    return vals[0] if vals else None


def print_table(rows: list[dict[str, str]], queries: tuple[str, ...]) -> None:
    wanted = set(queries)
    selected = [row for row in rows if row["query"] in wanted]
    if not selected:
        return
    header = f"{'delay':>6s} {'dir':>4s} {'query':38s} {'n':>8s} {'mean':>10s} {'median':>10s} {'p95':>10s} {'max':>10s}"
    print(header)
    print("-" * len(header))
    for row in selected:
        print(
            f"{row['delay_ms']:>6s} {row['direction']:>4s} "
            f"{row['query'][:38]:38s} {row['count']:>8s} "
            f"{row['mean']:>10s} {row['median']:>10s} {row['p95']:>10s} {row['max']:>10s}"
        )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "delay_ms", "direction", "query", "count", "mean", "median", "p95", "max"]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", help="CSV output path")
    parser.add_argument("--min-visible-increase-ms", type=float, default=10.0)
    args = parser.parse_args()

    run_dir = resolve(args.run)
    if not run_dir.exists():
        raise SystemExit(f"missing run directory: {run_dir}")

    split_csvs = task_csvs(run_dir)
    if not split_csvs:
        raise SystemExit(f"no task CSVs found under {run_dir / 'by_window' / 'task'}")

    queries = DEFAULT_QUERIES
    rows = summarize_run(run_dir, queries)
    if args.out:
        write_csv(resolve(args.out), rows)

    print(f"Run: {run_dir}")
    print(f"Task windows: {len(split_csvs)}")
    print(f"Summary rows: {len(rows)}")
    if args.out:
        print(f"CSV: {resolve(args.out)}")
    print()

    global_csv = run_dir / "prometheus_timeseries.csv.gz"
    if global_csv.exists():
        wanted = set(DIRECT_QUERIES)
        global_counts = query_counts(global_csv, wanted)
        split_counts: Counter[str] = Counter()
        for path in split_csvs:
            split_counts.update(query_counts(path, wanted))
        print("Direct latency query availability:")
        for query in DIRECT_QUERIES:
            print(f"  {query}: global={global_counts[query]} task_windows={split_counts[query]}")
        print()

    print("Latency metrics:")
    print_table(rows, DIRECT_QUERIES)
    print()
    print("Same-packet RTT metrics:")
    print_table(rows, SAME_PACKET_QUERIES)
    print()
    print("Supporting network/context metrics:")
    print_table(
        rows,
        (
            "tcp_pending_evicted_hz_30s",
            "tcp_ack_no_match_hz_30s",
            "container_cpu_cores_30s_any",
            "mac_throughput_total_bps",
            "prach_i0_db",
            "oai_gnb_mac_rsrp_per_rnti",
            "oai_gnb_mac_mcs_per_rnti",
        ),
    )
    print()

    metric = first_metric(rows, ("direct_p95_ms_5s", "direct_mean_ms_5s", "direct_latest_ms"))
    delays = sorted({int(row["delay_ms"]) for row in rows if row["query"] == metric}) if metric else []
    print("Network verdict:")
    if not metric or not delays:
        print("  FAIL: no direct latency metric is available in task windows.")
        return 2

    ref = mean_for(rows, metric, 0, "both")
    high_delay = max(delays)
    high = mean_for(rows, metric, high_delay, "both")
    if ref is None or high is None:
        print(f"  FAIL: cannot compare reference and {high_delay} ms windows for {metric}.")
        return 2

    increase = high - ref
    print(f"  direct metric used: {metric}")
    print(f"  reference mean: {fmt(ref)} ms")
    print(f"  {high_delay} ms mean: {fmt(high)} ms")
    print(f"  observed increase: {fmt(increase)} ms")

    missing_derived = [q for q in ("direct_mean_ms_5s", "direct_p95_ms_5s", "direct_p99_ms_5s") if not any(r["query"] == q for r in rows)]
    if missing_derived:
        print("  WARN: derived direct latency metrics are absent from task windows:")
        for query in missing_derived:
            print(f"    - {query}")

    if increase < args.min_visible_increase_ms:
        print("  FAIL: injected delay is not visible in direct latency metrics.")
        print("  Do not integrate this run as controlled_delay before checking tc/netem placement and metric-window splitting.")
        return 2

    print("  OK: injected delay is visible in direct latency metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
