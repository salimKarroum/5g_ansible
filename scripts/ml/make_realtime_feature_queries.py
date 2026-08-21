#!/usr/bin/env python3
"""Generate realtime PromQL feature queries for a selected feature list."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

STAT_SUFFIXES = [
    "nonzero_ratio",
    "median",
    "mean",
    "p95",
    "std",
    "max",
    "min",
    "last",
    "range",
    "delta",
    "count",
]

DERIVED_BASE_QUERIES = {
    "upf_cpu_cores": 'sum(rate(container_cpu_usage_seconds_total{namespace="${NAMESPACE}",pod=~"${UPF_POD_REGEX}",container!="POD",container!=""}[30s]))',
    "upf_memory_mib": 'sum(container_memory_working_set_bytes{namespace="${NAMESPACE}",pod=~"${UPF_POD_REGEX}",container!="POD",container!=""}) / 1024 / 1024',
    "container_cpu_cores": 'sum(rate(container_cpu_usage_seconds_total{namespace="${NAMESPACE}",pod=~"${OPEN5GS_POD_REGEX}",container!="POD",container!=""}[30s]))',
    "container_memory_mib": 'sum(container_memory_working_set_bytes{namespace="${NAMESPACE}",pod=~"${OPEN5GS_POD_REGEX}",container!="POD",container!=""}) / 1024 / 1024',
}


def load_paper_queries(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fp:
        data = json.load(fp)
    return {str(item["name"]): str(item["query"]) for item in data}


def load_features(path: Path) -> list[str]:
    features: list[str] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if "," in text:
                text = text.split(",", 1)[0].strip()
            if text.lower() not in {"feature", "feature_name", "name"}:
                features.append(text)
    return features


def split_feature(feature: str) -> tuple[str, str]:
    for suffix in STAT_SUFFIXES:
        token = "_" + suffix
        if feature.endswith(token):
            return feature[: -len(token)], suffix
    return feature, "mean"


def base_query_for(base: str, paper_queries: dict[str, str]) -> str:
    if base in DERIVED_BASE_QUERIES:
        return DERIVED_BASE_QUERIES[base]
    if base in paper_queries:
        return paper_queries[base]
    if base.startswith("rf_"):
        return base[3:]
    if re.match(r"^(oai_gnb|gtp|fivegs|mac_throughput|node_|container_)", base):
        return base
    return ""


def wrap_query(expr: str, stat: str) -> dict[str, str]:
    subq = f"(({expr}))[${{WINDOW}}:5s]"
    if stat == "mean":
        return {"query": f"avg_over_time({subq})", "reduce": "mean"}
    if stat == "median":
        return {"query": f"quantile_over_time(0.50, {subq})", "reduce": "mean"}
    if stat == "p95":
        return {"query": f"quantile_over_time(0.95, {subq})", "reduce": "mean"}
    if stat == "std":
        return {"query": f"stddev_over_time({subq})", "reduce": "mean"}
    if stat == "max":
        return {"query": f"max_over_time({subq})", "reduce": "mean"}
    if stat == "min":
        return {"query": f"min_over_time({subq})", "reduce": "mean"}
    if stat == "last":
        return {"query": f"last_over_time({subq})", "reduce": "mean"}
    if stat == "count":
        return {"query": f"count_over_time({subq})", "reduce": "sum"}
    if stat == "nonzero_ratio":
        return {"query": f"avg_over_time(((({expr}) != 0))[${{WINDOW}}:5s])", "reduce": "mean"}
    if stat == "range":
        return {
            "query": f"max_over_time({subq}) - min_over_time({subq})",
            "reduce": "mean",
        }
    if stat == "delta":
        return {
            "query": f"last_over_time({subq}) - min_over_time({subq})",
            "reduce": "mean",
        }
    return {"query": f"avg_over_time({subq})", "reduce": "mean"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-file", required=True)
    parser.add_argument("--out", default="results/manual-network/dataset/feature_queries.generated.json")
    parser.add_argument("--paper-queries", default="scripts/validation/paper_prometheus_queries.json")
    parser.add_argument("--window", default="60s")
    args = parser.parse_args()

    features = load_features(Path(args.features_file))
    paper_queries = load_paper_queries(ROOT / args.paper_queries)

    cfg: dict[str, object] = {
        "window": args.window,
        "variables": {
            "NAMESPACE": "open5gs",
            "UPF_POD_REGEX": "open5gs-upf.*",
            "OPEN5GS_POD_REGEX": "open5gs.*|oai-gnb.*",
        },
        "features": {},
    }
    feature_cfg: dict[str, object] = cfg["features"]  # type: ignore[assignment]
    missing: list[str] = []
    for feature in features:
        base, stat = split_feature(feature)
        expr = base_query_for(base, paper_queries)
        if not expr:
            missing.append(feature)
            feature_cfg[feature] = {"query": "", "reduce": "mean"}
            continue
        feature_cfg[feature] = wrap_query(expr, stat)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    print(f"features: {len(features)}")
    print(f"queries: {out}")
    if missing:
        print("WARNING: no PromQL mapping for:")
        for feature in missing:
            print(f"  - {feature}")


if __name__ == "__main__":
    main()
