# Reproducing the RCA Report Results

This folder contains the datasets, selected features, saved model artifact and
reference outputs needed to reproduce the main results reported for the final
Random Forest K=28 RCA model.

The goal is to make the report reproducible without requiring access to the
large raw Prometheus exports, packet captures or archived pod logs.

## Included Data

The following files are intentionally included in GitHub:

```text
reproducibility/data/window_features_physical_7_anomalies_derived.csv
reproducibility/data/window_features_physical_7_anomalies_derived_balanced48.csv
reproducibility/data/cross_stack_combined_balanced.csv
reproducibility/data/selected_features_mechanism_clean.txt
reproducibility/data/realtime_network_rca_model_mechanism_clean_balanced48.pkl
reproducibility/data/mechanism_clean_sweep.csv
reproducibility/data/k28_analysis_summary.json
reproducibility/data/k28_loro_fold_metrics_rf_lr.csv
reproducibility/data/undersampling_run_coverage.csv
reproducibility/data/balanced_7classes_windows_by_run.csv
reproducibility/data/cross_stack_windows_by_stack_label_run.csv
reproducibility/data/k28_rf_confusion_matrix_recomputed.csv
reproducibility/data/k28_rf_classification_report_recomputed.csv
reproducibility/data/k28_logistic_regression_confusion_matrix.csv
reproducibility/data/k28_logistic_regression_classification_report.csv
```

The two main datasets are:

- `window_features_physical_7_anomalies_derived.csv`: original labeled
  seven-class dataset before balancing.
- `window_features_physical_7_anomalies_derived_balanced48.csv`: final balanced
  dataset used for the main Random Forest K=28 evaluation.

The external validation dataset is:

- `cross_stack_combined_balanced.csv`: balanced 60-window cross-stack validation
  dataset.

## Python Environment

Create and activate a Python environment, then install the analysis
dependencies:

```bash
python3 -m venv .venv-rca
source .venv-rca/bin/activate
pip install -r requirements-ml.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv-rca
.\.venv-rca\Scripts\Activate.ps1
pip install -r requirements-ml.txt
```

## Recompute the Main K=28 Results

From the repository root, run:

```bash
python3 scripts/ml/evaluate_k28_report_results.py \
  --original-csv reproducibility/data/window_features_physical_7_anomalies_derived.csv \
  --balanced-csv reproducibility/data/window_features_physical_7_anomalies_derived_balanced48.csv \
  --features-file reproducibility/data/selected_features_mechanism_clean.txt \
  --cross-stack-csv reproducibility/data/cross_stack_combined_balanced.csv \
  --out-dir reproducibility/outputs \
  --n-jobs -1
```

On machines where multiprocessing is restricted, use:

```bash
python3 scripts/ml/evaluate_k28_report_results.py \
  --original-csv reproducibility/data/window_features_physical_7_anomalies_derived.csv \
  --balanced-csv reproducibility/data/window_features_physical_7_anomalies_derived_balanced48.csv \
  --features-file reproducibility/data/selected_features_mechanism_clean.txt \
  --cross-stack-csv reproducibility/data/cross_stack_combined_balanced.csv \
  --out-dir reproducibility/outputs \
  --n-jobs 1
```

This command regenerates:

- run coverage before and after undersampling;
- number of windows per class and per run;
- Leave-One-Run-Out fold metrics for the Random Forest K=28 model;
- Logistic Regression baseline with the same K=28 feature set;
- majority-class baseline;
- confusion matrices;
- classification reports;
- cross-stack dataset composition.

The main reference values reported from the original artifact are:

```text
Random Forest K=28 LORO accuracy  = 0.9494047619
Random Forest K=28 LORO macro-F1  = 0.9486261837
Logistic Regression accuracy      = 0.8482142857
Logistic Regression macro-F1      = 0.8474933567
Majority baseline accuracy        = 0.1428571429
Majority baseline macro-F1        = 0.0357142857
```

Small numerical differences may appear if the Random Forest is retrained with a
different scikit-learn version. The reference outputs in `reproducibility/data/`
are the values used for the report.

## Reproduce Feature Selection

The selected K=28 features are stored in:

```text
reproducibility/data/selected_features_mechanism_clean.txt
```

The feature-selection script is:

```text
scripts/ml/reduce_physical_features_mechanism_clean.py
```

The feature sweep output used during model development is:

```text
reproducibility/data/mechanism_clean_sweep.csv
```

## Dataset Balancing

The final balanced dataset was produced by random undersampling to 48 windows
per class:

```python
balanced = (
    df.groupby("label", group_keys=False)
      .sample(n=48, random_state=42)
      .sample(frac=1, random_state=42)
      .reset_index(drop=True)
)
```

The run coverage after undersampling is provided in:

```text
reproducibility/data/undersampling_run_coverage.csv
reproducibility/data/balanced_7classes_windows_by_run.csv
```

These files show that the undersampling preserves windows from all experimental
runs of each class.

## Cross-Stack Validation Dataset

The cross-stack dataset contains:

```text
60 windows
14 independent experimental runs
7 OAI/OAI runs
7 srsRAN/Open5GS runs
6 classes
10 windows per class
4 OAI/OAI windows and 6 srsRAN/Open5GS windows for each class
```

The exact run contribution is provided in:

```text
reproducibility/data/cross_stack_windows_by_stack_label_run.csv
```

## Reconstructing Datasets From Raw Runs

If the raw experiment directories are available, labeled windows can be rebuilt
with:

```bash
python3 scripts/ml/build_manual_network_dataset.py \
  --no-manual-root \
  --run <RUN_DIR> \
  --out <OUTPUT_CSV> \
  --window-seconds 30 \
  --include-probe-features \
  --drop-clean
```

Raw experiment directories are not included here because they contain large
Prometheus time series, packet captures, pod logs and archived UE logs. The
processed CSV datasets needed for reproducing the report metrics are included.

## Experiment Effort Summary

To summarize the number of runs and, when raw timelines are available, the total
campaign duration, run:

```bash
python3 scripts/validation/report_experiment_effort.py \
  --dataset main=reproducibility/data/window_features_physical_7_anomalies_derived_balanced48.csv \
  --dataset cross_stack=reproducibility/data/cross_stack_combined_balanced.csv \
  --results-root results \
  --out-json reproducibility/outputs/report_experiment_effort.json
```

If the `results/` directory contains the original `timeline_summary.json` files,
the script also reports cumulative campaign and traffic durations.

