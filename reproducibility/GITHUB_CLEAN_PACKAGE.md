# Clean GitHub Package

This file documents what should be committed for a clean reproduction-oriented
GitHub repository.

## Commit These Files And Directories

Core deployment and monitoring workflow already tracked by the repository:

```text
deploy.sh
playbooks/
roles/
tasks/
monitoring/
collections/requirements.yml
```

RCA scripts to add:

```text
scripts/ml/build_manual_network_dataset.py
scripts/ml/reduce_physical_features_mechanism_clean.py
scripts/ml/train_realtime_rca_model.py
scripts/ml/evaluate_k28_report_results.py
scripts/ml/make_realtime_feature_queries.py
scripts/ml/plot_final_report_figures.py
scripts/ml/plot_pfe_balanced_results.py
scripts/ml/analyze_rca_signatures.py
scripts/ml/explain_class_features.py
scripts/ml/study_physical_feature_families.py
scripts/ml/study_minimal_features.py
scripts/ml/prune_correlated_features.py
scripts/ml/plot_physical_signature_overview.py
scripts/ml/plot_retained_physical_class_importance.py
scripts/ml/plot_mechanism_clean_results.py
scripts/validation/run_validated_network_campaign.sh
scripts/validation/report_experiment_effort.py
scripts/validation/network_signature_summary.py
scripts/validation/analyze_controlled_delay_run.py
scripts/realtime_rca/
k8s/
requirements-ml.txt
reproducibility/
README.md
.gitignore
```

Report support files:

```text
docs/thierry_feedback_insertions.tex
docs/report_update_audit.md
docs/report_update_latex_blocks.tex
docs/github_readme_draft.md
```

## Do Not Commit

```text
results/
tmp_remote_artifacts/
tmp_report_check/
reproducibility/outputs/
*.pcap
*.pcapng
*.tgz
*.tar
*.tar.gz
prometheus_timeseries.csv
prometheus_timeseries.csv.gz
rapport-stage-extracted.txt
_rapport_tp2_extracted.txt
_rapport_tp2_2_extracted.txt
_root_cause_analysis_extracted.txt
IWCMC2024.pdf
.claude/
__pycache__/
```

## Suggested Add Commands

Use targeted `git add` commands. Do not use `git add .`.

```bash
git add README.md .gitignore requirements-ml.txt
git add reproducibility/README.md reproducibility/MANIFEST.md reproducibility/GITHUB_CLEAN_PACKAGE.md reproducibility/data/
git add scripts/ml/build_manual_network_dataset.py
git add scripts/ml/reduce_physical_features_mechanism_clean.py
git add scripts/ml/train_realtime_rca_model.py
git add scripts/ml/evaluate_k28_report_results.py
git add scripts/ml/make_realtime_feature_queries.py
git add scripts/ml/plot_final_report_figures.py
git add scripts/ml/plot_pfe_balanced_results.py
git add scripts/ml/analyze_rca_signatures.py
git add scripts/ml/explain_class_features.py
git add scripts/ml/study_physical_feature_families.py
git add scripts/ml/study_minimal_features.py
git add scripts/ml/prune_correlated_features.py
git add scripts/ml/plot_physical_signature_overview.py
git add scripts/ml/plot_retained_physical_class_importance.py
git add scripts/ml/plot_mechanism_clean_results.py
git add scripts/validation/run_validated_network_campaign.sh
git add scripts/validation/report_experiment_effort.py
git add scripts/validation/network_signature_summary.py
git add scripts/validation/analyze_controlled_delay_run.py
git add scripts/realtime_rca/
git add k8s/
git add docs/thierry_feedback_insertions.tex docs/report_update_audit.md docs/report_update_latex_blocks.tex docs/github_readme_draft.md
```

Then inspect the staged files:

```bash
git diff --cached --name-only
```

