# Real-time RCA Inference

This directory contains the minimal path to classify the current network state
and expose the probable root cause to Prometheus/Grafana.

## 1. Train the stable compact model

Run on Duckburg:

```bash
cd ~/5G_YASSIR/5g_ansible
source venv/bin/activate

python3 scripts/ml/train_realtime_rca_model.py \
  --csv results/ml_dataset_v3/window_features_augmented_stable_f1_0936.csv \
  --model-out results/ml_dataset_v3/realtime_rca_model.pkl
```

The model uses the 15 physical features kept after correlation pruning.  It
excludes `load_ramp` and `multi_ue_contention` by default.

For the validated manual-network campaign, build the dataset, select compact
features, generate live PromQL queries, and train the model with:

```bash
cd ~/5G_YASSIR/5g_ansible
source .venv-control-plane/bin/activate

bash scripts/ml/run_manual_network_ml_pipeline.sh
```

This writes:

- `results/manual-network/dataset/window_features_manual_network.csv`
- `results/manual-network/dataset/minimal_feature_study/`
- `results/manual-network/dataset/selected_features.txt`
- `results/manual-network/dataset/feature_queries.generated.json`
- `results/manual-network/dataset/realtime_network_rca_model.pkl`

## 2. Run locally against Prometheus

Adjust `scripts/realtime_rca/feature_queries.example.json` if the Prometheus
service or pod regexes differ, then run:

```bash
python3 scripts/realtime_rca/rca_inference_service.py \
  --model results/ml_dataset_v3/realtime_rca_model.pkl \
  --queries scripts/realtime_rca/feature_queries.example.json \
  --prometheus-url http://localhost:9090 \
  --port 9300
```

Check the output:

```bash
curl -s http://localhost:9300/metrics | grep '^rca_'
curl -s http://localhost:9300/predict
```

## 3. Grafana panels

Use these PromQL queries:

```promql
max by (root_cause, explanation) (rca_prediction == 1)
```

```promql
max by (root_cause) (rca_confidence)
```

```promql
topk(7, rca_class_probability)
```

```promql
topk(5, rca_top_feature_deviation)
```

```promql
rca_feature_missing
```

```promql
rca_input_missing_ratio
```

The first panel gives the probable root cause and a short explanation.  The
missing-feature panel is important: if several features are missing, the
prediction is less trustworthy even when the model returns a class.

## 4. Kubernetes deployment

`k8s/rca-inference-service.yaml` is a template.  Build an image containing:

- `scripts/realtime_rca/rca_inference_service.py`
- `scripts/realtime_rca/feature_queries.example.json` copied as
  `/app/feature_queries.json`
- `results/ml_dataset_v3/realtime_rca_model.pkl` copied as
  `/app/realtime_rca_model.pkl`
- Python dependencies already used by the ML environment: `numpy`,
  `scikit-learn`

Then apply the manifest and add a Grafana panel from the Prometheus metrics.
