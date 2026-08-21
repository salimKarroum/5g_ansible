#!/usr/bin/env python3
"""Expose real-time RCA predictions as Prometheus metrics.

The exporter queries Prometheus for the compact physical feature set, applies a
pre-trained RCA model, and serves only plain Prometheus text metrics.  Grafana
then reads these metrics from Prometheus like any other monitoring signal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
import urllib.parse
import urllib.request
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np


CLASS_EXPLANATIONS = {
    "clean_traffic": "trafic normal: pas de signature anormale dominante",
    "controlled_delay": "delai controle: latence eBPF elevee sans hausse compute locale",
    "controlled_jitter": "gigue controlee: dispersion/tail latency elevee autour d'une latence mediane stable",
    "far_ue_poor_radio": "UE loin / radio faible: RSRP/MAC degrade",
    "radio_interference": "interference radio: degradation radio/MAC visible",
    "server_stress": "stress serveur: CPU/memoire serveur eleves",
    "tunnel_bandwidth_limit": "limitation de bande passante tunnel: debit plafonne sans signature radio dominante",
    "tunnel_packet_loss": "perte tunnel: correlation TCP/eBPF degradee",
    "upf_stress": "stress UPF: CPU/memoire UPF eleves",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("RCA_MODEL", "results/ml_dataset_v3/realtime_rca_model.pkl"))
    ap.add_argument("--queries", default=os.getenv("RCA_FEATURE_QUERIES", "scripts/realtime_rca/feature_queries.example.json"))
    ap.add_argument("--prometheus-url", default=os.getenv("PROMETHEUS_URL", "http://localhost:9090"))
    ap.add_argument("--listen", default=os.getenv("RCA_LISTEN", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("RCA_PORT", "9300")))
    ap.add_argument("--cache-seconds", type=float, default=float(os.getenv("RCA_CACHE_SECONDS", "5")))
    ap.add_argument("--prometheus-timeout", type=float, default=float(os.getenv("RCA_PROMETHEUS_TIMEOUT", "3")))
    ap.add_argument("--stability-window", type=int, default=int(os.getenv("RCA_STABILITY_WINDOW", "6")),
                    help="Number of recent raw predictions used for stable RCA output.")
    ap.add_argument("--stability-min-votes", type=int, default=int(os.getenv("RCA_STABILITY_MIN_VOTES", "4")),
                    help="Minimum majority votes in the stability window.")
    ap.add_argument("--stability-min-confidence", type=float,
                    default=float(os.getenv("RCA_STABILITY_MIN_CONFIDENCE", "0.50")),
                    help="Minimum average confidence for stable RCA output.")
    return ap.parse_args()


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def load_query_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def substitute(query: str, cfg: dict[str, Any]) -> str:
    values = {"WINDOW": str(cfg.get("window", "60s"))}
    values.update({str(k): str(v) for k, v in cfg.get("variables", {}).items()})
    for key, value in values.items():
        query = query.replace("${" + key + "}", value)
    return query


def reduce_values(values: list[float], mode: str) -> float:
    finite = [x for x in values if math.isfinite(x)]
    if not finite:
        return float("nan")
    if mode == "first":
        return finite[0]
    if mode == "max":
        return max(finite)
    if mode == "min":
        return min(finite)
    if mode == "sum":
        return sum(finite)
    return sum(finite) / len(finite)


def prom_query(prometheus_url: str, query: str, timeout: float) -> list[float]:
    params = urllib.parse.urlencode({"query": query})
    url = prometheus_url.rstrip("/") + "/api/v1/query?" + params
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("status") != "success":
        return []
    values: list[float] = []
    for series in payload.get("data", {}).get("result", []):
        value = series.get("value", [None, "nan"])[1]
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            pass
    return values


def family_for_feature(name: str) -> str:
    if name.startswith("direct_"):
        return "latency_ebpf"
    if name.startswith("rf_") or name.startswith("prach_"):
        return "radio_mac"
    if name.startswith("upf_"):
        return "upf_compute"
    if name.startswith("server_"):
        return "server_compute"
    if name.startswith("tcp_"):
        return "tcp_correlation"
    if name.startswith("container_"):
        return "container_context"
    return "other"


def escape_label(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", " ").replace('"', '\\"')


class RCAService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bundle = load_pickle(Path(args.model))
        self.query_cfg = load_query_config(Path(args.queries))
        self.features: list[str] = list(self.bundle["features"])
        self.cache_until = 0.0
        self.cache_text = ""
        self.cache_json: dict[str, Any] = {}
        self.history: deque[tuple[float, str, float]] = deque(maxlen=max(int(args.stability_window), 1))

    def stable_prediction(self, raw_root: str, raw_confidence: float) -> dict[str, Any]:
        now = time.time()
        self.history.append((now, raw_root, raw_confidence))
        counts = Counter(root for _, root, _ in self.history)
        majority_root, majority_votes = counts.most_common(1)[0]
        matching_confidences = [conf for _, root, conf in self.history if root == majority_root]
        avg_confidence = sum(matching_confidences) / max(len(matching_confidences), 1)
        stability_ratio = majority_votes / max(len(self.history), 1)
        is_stable = (
            majority_votes >= min(self.args.stability_min_votes, self.args.stability_window)
            and avg_confidence >= self.args.stability_min_confidence
        )
        if not is_stable:
            return {
                "root_cause": "uncertain",
                "confidence": avg_confidence,
                "stability_ratio": stability_ratio,
                "votes": majority_votes,
                "window": len(self.history),
                "is_stable": False,
                "majority_root_cause": majority_root,
            }
        return {
            "root_cause": majority_root,
            "confidence": avg_confidence,
            "stability_ratio": stability_ratio,
            "votes": majority_votes,
            "window": len(self.history),
            "is_stable": True,
            "majority_root_cause": majority_root,
        }

    def collect_features(self) -> tuple[dict[str, float], dict[str, bool], dict[str, str]]:
        queries = self.query_cfg.get("features", {})
        values: dict[str, float] = {}
        missing: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for feature in self.features:
            spec = queries.get(feature)
            if isinstance(spec, str):
                query = spec
                reduce_mode = "mean"
            elif isinstance(spec, dict):
                query = str(spec.get("query", ""))
                reduce_mode = str(spec.get("reduce", "mean"))
            else:
                query = ""
                reduce_mode = "mean"
            if not query:
                values[feature] = float("nan")
                missing[feature] = True
                errors[feature] = "no_query"
                continue
            try:
                raw = prom_query(
                    self.args.prometheus_url,
                    substitute(query, self.query_cfg),
                    self.args.prometheus_timeout,
                )
                value = reduce_values(raw, reduce_mode)
                values[feature] = value
                missing[feature] = not math.isfinite(value)
            except Exception as exc:  # Prometheus scrape must stay alive.
                values[feature] = float("nan")
                missing[feature] = True
                errors[feature] = exc.__class__.__name__
        return values, missing, errors

    def predict(self) -> dict[str, Any]:
        values, missing, errors = self.collect_features()
        X = np.array([[values.get(feature, float("nan")) for feature in self.features]], dtype=float)
        X_imp = self.bundle["imputer"].transform(X)
        model = self.bundle["model"]
        labels = [str(x) for x in model.classes_]
        probabilities = model.predict_proba(X_imp)[0]
        best_idx = int(np.argmax(probabilities))
        root_cause = labels[best_idx]
        confidence = float(probabilities[best_idx])
        stable = self.stable_prediction(root_cause, confidence)

        medians = self.bundle.get("feature_medians", {})
        iqrs = self.bundle.get("feature_iqrs", {})
        deviations: list[tuple[str, float, float]] = []
        for feature in self.features:
            value = values.get(feature, float("nan"))
            if not math.isfinite(value):
                continue
            median = float(medians.get(feature, 0.0))
            iqr = float(iqrs.get(feature, 1.0)) or 1.0
            score = abs(value - median) / iqr
            deviations.append((feature, float(score), float(value)))
        deviations.sort(key=lambda item: item[1], reverse=True)
        top = deviations[:5]
        top_text = ", ".join(f"{name}={value:.3g}" for name, _, value in top[:3])
        explanation = CLASS_EXPLANATIONS.get(root_cause, root_cause)
        if top_text:
            explanation = f"{explanation}; signaux dominants: {top_text}"
        stable_root = stable["root_cause"]
        if stable_root == "uncertain":
            stable_explanation = (
                "prediction incertaine: la classe majoritaire n'est pas encore assez stable "
                f"(majorite={stable['majority_root_cause']}, votes={stable['votes']}/{stable['window']}, "
                f"confiance_moyenne={stable['confidence']:.3f})"
            )
        else:
            stable_explanation = CLASS_EXPLANATIONS.get(str(stable_root), str(stable_root))
            if top_text:
                stable_explanation = f"{stable_explanation}; signaux dominants: {top_text}"

        return {
            "timestamp": time.time(),
            "root_cause": stable_root,
            "confidence": float(stable["confidence"]),
            "raw_root_cause": root_cause,
            "raw_confidence": confidence,
            "stable": stable,
            "missing_ratio": sum(1 for v in missing.values() if v) / max(len(self.features), 1),
            "probabilities": dict(zip(labels, [float(x) for x in probabilities])),
            "feature_values": values,
            "feature_missing": missing,
            "feature_errors": errors,
            "top_deviations": top,
            "explanation": stable_explanation,
            "raw_explanation": explanation,
        }

    def metrics_text(self) -> str:
        now = time.time()
        if self.cache_text and now < self.cache_until:
            return self.cache_text
        result = self.predict()
        self.cache_json = result
        self.cache_until = now + self.args.cache_seconds
        self.cache_text = render_metrics(result, self.features)
        return self.cache_text


def render_metrics(result: dict[str, Any], features: list[str]) -> str:
    lines = [
        "# HELP rca_prediction Stable RCA root cause, 1 for the selected class. Uses temporal voting and may emit uncertain.",
        "# TYPE rca_prediction gauge",
    ]
    root = result["root_cause"]
    explanation = result["explanation"]
    stable_labels = list(result["probabilities"].keys())
    if "uncertain" not in stable_labels:
        stable_labels.append("uncertain")
    for label in stable_labels:
        selected = 1.0 if label == root else 0.0
        lines.append(
            'rca_prediction{root_cause="%s",explanation="%s"} %.0f'
            % (escape_label(label), escape_label(explanation if label == root else ""), selected)
        )

    lines += [
        "# HELP rca_prediction_raw Raw per-scrape RCA root cause before temporal stabilization.",
        "# TYPE rca_prediction_raw gauge",
    ]
    raw_root = result["raw_root_cause"]
    raw_explanation = result["raw_explanation"]
    for label in result["probabilities"].keys():
        selected = 1.0 if label == raw_root else 0.0
        lines.append(
            'rca_prediction_raw{root_cause="%s",explanation="%s"} %.0f'
            % (escape_label(label), escape_label(raw_explanation if label == raw_root else ""), selected)
        )

    stable = result["stable"]
    lines += [
        "# HELP rca_confidence Confidence of the stable root cause. For uncertain, this is the majority-class average confidence.",
        "# TYPE rca_confidence gauge",
        'rca_confidence{root_cause="%s"} %.6f' % (escape_label(root), result["confidence"]),
        "# HELP rca_confidence_raw Probability of the raw selected root cause.",
        "# TYPE rca_confidence_raw gauge",
        'rca_confidence_raw{root_cause="%s"} %.6f' % (escape_label(raw_root), result["raw_confidence"]),
        "# HELP rca_stability_ratio Fraction of recent predictions voting for the majority class.",
        "# TYPE rca_stability_ratio gauge",
        "rca_stability_ratio %.6f" % stable["stability_ratio"],
        "# HELP rca_stability_votes Number of recent predictions voting for the majority class.",
        "# TYPE rca_stability_votes gauge",
        'rca_stability_votes{root_cause="%s"} %d'
        % (escape_label(stable["majority_root_cause"]), stable["votes"]),
        "# HELP rca_stability_window Number of recent predictions currently held in the stabilizer.",
        "# TYPE rca_stability_window gauge",
        "rca_stability_window %d" % stable["window"],
        "# HELP rca_is_stable 1 when the stable RCA output passed vote and confidence thresholds.",
        "# TYPE rca_is_stable gauge",
        "rca_is_stable %d" % (1 if stable["is_stable"] else 0),
        "# HELP rca_input_missing_ratio Fraction of model features missing from live Prometheus queries.",
        "# TYPE rca_input_missing_ratio gauge",
        "rca_input_missing_ratio %.6f" % result["missing_ratio"],
        "# HELP rca_class_probability Probability per RCA class.",
        "# TYPE rca_class_probability gauge",
    ]
    for label, prob in result["probabilities"].items():
        lines.append('rca_class_probability{root_cause="%s"} %.6f' % (escape_label(label), prob))

    lines += [
        "# HELP rca_feature_value Live feature value used by the RCA classifier.",
        "# TYPE rca_feature_value gauge",
    ]
    values = result["feature_values"]
    missing = result["feature_missing"]
    errors = result["feature_errors"]
    for feature in features:
        value = values.get(feature, float("nan"))
        if not math.isfinite(value):
            value_text = "NaN"
        else:
            value_text = f"{value:.12g}"
        lines.append(
            'rca_feature_value{feature="%s",family="%s"} %s'
            % (escape_label(feature), escape_label(family_for_feature(feature)), value_text)
        )

    lines += [
        "# HELP rca_feature_missing 1 when a live feature query returned no finite value.",
        "# TYPE rca_feature_missing gauge",
    ]
    for feature in features:
        lines.append(
            'rca_feature_missing{feature="%s",error="%s"} %d'
            % (escape_label(feature), escape_label(errors.get(feature, "")), 1 if missing.get(feature) else 0)
        )

    lines += [
        "# HELP rca_top_feature_deviation Robust deviation from training median for explanation.",
        "# TYPE rca_top_feature_deviation gauge",
    ]
    for rank, (feature, score, _) in enumerate(result["top_deviations"], 1):
        lines.append(
            'rca_top_feature_deviation{rank="%d",feature="%s",family="%s"} %.6f'
            % (rank, escape_label(feature), escape_label(family_for_feature(feature)), score)
        )

    lines += [
        "# HELP rca_last_inference_timestamp_seconds Unix timestamp of the last RCA inference.",
        "# TYPE rca_last_inference_timestamp_seconds gauge",
        "rca_last_inference_timestamp_seconds %.3f" % result["timestamp"],
    ]
    return "\n".join(lines) + "\n"


def make_handler(service: RCAService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/healthz"):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok\n")
                return
            if self.path.startswith("/predict"):
                service.metrics_text()
                payload = json.dumps(service.cache_json, default=str, indent=2).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path.startswith("/metrics"):
                payload = service.metrics_text().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, fmt: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    args = parse_args()
    service = RCAService(args)
    server = ThreadingHTTPServer((args.listen, args.port), make_handler(service))
    print(f"RCA inference exporter listening on {args.listen}:{args.port}")
    print(f"Prometheus URL: {args.prometheus_url}")
    print(f"Model: {args.model}")
    print(f"Queries: {args.queries}")
    server.serve_forever()


if __name__ == "__main__":
    main()
