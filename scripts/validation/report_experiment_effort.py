#!/usr/bin/env python3
"""Summarize report experiment effort from datasets and timeline artifacts.

The script reports the number of unique runs represented in one or more CSV
datasets. If raw result directories are available, it also scans
`timeline_summary.json` files and sums campaign/traffic durations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(value)
    return path.stem, path


def read_dataset(path: Path) -> dict:
    runs = set()
    labels = set()
    rows = 0
    by_label_run: dict[tuple[str, str], int] = defaultdict(int)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"run", "label"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"{path} is missing required columns: {missing}")
        for row in reader:
            run = str(row["run"])
            label = str(row["label"])
            rows += 1
            runs.add(run)
            labels.add(label)
            by_label_run[(label, run)] += 1

    return {
        "path": str(path),
        "rows": rows,
        "runs": sorted(runs),
        "labels": sorted(labels),
        "by_label_run": by_label_run,
    }


def pick_duration(windows: list[dict[str, Any]], level: str) -> int | None:
    candidates = [w for w in windows if w.get("level") == level and isinstance(w.get("duration_s"), int)]
    if candidates:
        return max(int(w["duration_s"]) for w in candidates)
    return None


def fallback_span(windows: list[dict[str, Any]]) -> int | None:
    starts = [int(w["start_epoch"]) for w in windows if isinstance(w.get("start_epoch"), int)]
    ends = [int(w["end_epoch"]) for w in windows if isinstance(w.get("end_epoch"), int)]
    if starts and ends:
        return max(ends) - min(starts)
    return None


def scan_timelines(root: Path) -> dict[str, dict[str, Any]]:
    timelines = {}
    for path in root.rglob("timeline_summary.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        windows = data.get("windows", [])
        if not isinstance(windows, list):
            continue
        run_id = str(data.get("run_id") or path.parent.name)
        campaign_s = pick_duration(windows, "campaign")
        traffic_s = pick_duration(windows, "traffic_campaign")
        if campaign_s is None:
            campaign_s = fallback_span(windows)
        timelines[run_id] = {
            "path": str(path),
            "campaign_duration_s": campaign_s,
            "traffic_duration_s": traffic_s,
        }
    return timelines


def hours(seconds: int | None) -> float | None:
    if seconds is None:
        return None
    return seconds / 3600.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset CSV, optionally named as NAME=PATH. Can be passed multiple times.",
    )
    parser.add_argument(
        "--results-root",
        action="append",
        default=[],
        help="Results directory to scan for timeline_summary.json files.",
    )
    parser.add_argument("--out-json", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    datasets = {}
    all_dataset_runs = set()
    for item in args.dataset:
        name, path = parse_named_path(item)
        summary = read_dataset(path)
        datasets[name] = summary
        all_dataset_runs.update(summary["runs"])

    timelines = {}
    for root in args.results_root:
        timelines.update(scan_timelines(Path(root)))

    matched_runs = sorted(run for run in all_dataset_runs if run in timelines)
    missing_timeline_runs = sorted(all_dataset_runs - set(timelines))
    total_campaign_s = sum(
        timelines[run]["campaign_duration_s"] or 0 for run in matched_runs
    )
    total_traffic_s = sum(timelines[run]["traffic_duration_s"] or 0 for run in matched_runs)

    output = {
        "datasets": {
            name: {
                "path": data["path"],
                "rows": data["rows"],
                "unique_runs": len(data["runs"]),
                "unique_labels": len(data["labels"]),
                "labels": data["labels"],
            }
            for name, data in datasets.items()
        },
        "timeline_scan": {
            "timeline_files_found": len(timelines),
            "dataset_runs_with_timeline": len(matched_runs),
            "dataset_runs_missing_timeline": missing_timeline_runs,
            "total_campaign_hours_for_matched_runs": hours(total_campaign_s),
            "total_traffic_hours_for_matched_runs": hours(total_traffic_s),
        },
    }

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

