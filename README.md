Fraud Risk Scoring (FRS)
Calibrated Risk Modeling • Cost-Sensitive Policy • Drift Monitoring • Reproducible ML

A production-minded fraud / risk scoring CLI that goes beyond model accuracy.

FRS trains a classifier, calibrates probabilities, selects thresholds via explicit decision policy, evaluates operational impact, monitors drift, and exports fully reproducible run artifacts.

This project demonstrates real-world ML engineering for risk systems — not just model training.

##  What This System Does

FRS models fraud as a risk scoring pipeline:

Train a binary classifier

(Optional) Calibrate probabilities

Select a decision threshold via policy

Evaluate operational impact

Monitor drift

Save everything for reproducibility

Generate a portfolio-grade report

## Why This Is Different

Most fraud ML demos stop at ROC-AUC.

FRS includes:

✅ Probability calibration

✅ Cost-sensitive threshold optimization

✅ Alert-rate guardrails

✅ Expected cost tracking

✅ Drift detection (features + scores)

✅ Walk-forward backtesting

✅ Reproducible run artifacts

✅ Deployment-ready scoring CLI

This mirrors how production fraud systems are designed.

## Example Run (Real Output)

From run: 2026-02-17_130730

Test Performance
Metric	Value
ROC-AUC	0.9675
PR-AUC	0.7812
Brier Score	0.0004
Threshold	0.07
Alert Rate	0.0012
Base Rate	0.0013
Expected Cost	192.0
Confusion Matrix @ Policy Threshold
	Pred 0	Pred 1
True 0	56874	12
True 1	18	57

This demonstrates:

High ranking performance (ROC-AUC)

Strong precision-recall under class imbalance

Operationally controlled alert rate

Explicit cost-aware thresholding

## System Architecture
Raw Data
   ↓
Feature Selection
   ↓
Preprocessing Pipeline
   ↓
Model (LightGBM / Logistic Regression)
   ↓
Probability Calibration (optional)
   ↓
Policy Threshold Selection
   ↓
Risk Score + Alert Decision
   ↓
Drift Monitoring + Reporting

🛠 CLI Commands
Train
frs train -c configs/base.yaml


Outputs:

runs/<timestamp>/
 ├─ bundle.joblib
 ├─ metrics.json
 ├─ drift.json
 └─ config.resolved.yaml

Walk-Forward Backtest (Time Drift Stability)
frs backtest -c configs/base.yaml --folds 5


Generates:

Per-fold metrics

Threshold stability tracking

Cost curves

PR curves

Backtest summary statistics

Score New Data
frs score \
  --run-dir runs/<RUN_ID> \
  --input data/new_data.csv \
  --output outputs/scored.csv \
  --summary outputs/summary.json


Adds:

risk_proba

alert

threshold

model_name

run_dir

Generate Report
frs report \
  --run-dir runs/<RUN_ID> \
  --out reports/<RUN_ID>


Creates:

reports/<RUN_ID>/
 ├─ REPORT.md
 └─ report_summary.json


Fully reproducible from saved artifacts.

⚙ Configuration (YAML-Driven)

All behavior is controlled via configs/base.yaml.

Includes:

Model hyperparameters

Calibration method

Threshold policy (cost or top-k)

Alert-rate guardrails

Prior-shift adjustment

Walk-forward window definitions

This ensures:

No hidden parameters

Fully reproducible experiments

Clear governance over risk decisions

## Drift Monitoring

FRS detects:

1) Feature Drift

KS tests on numeric features

Effect-size thresholds

2) Probability Drift

Distribution shift between validation and test probabilities

Saved in:

runs/<RUN_ID>/drift.json

## Operational Policy Layer

FRS separates:

Probability estimation

Decision policy

Policy modes:

cost → minimize expected cost

top_k → fixed alert rate

Optional constraints:

min_alert_rate

max_alert_rate

threshold grid size

prior shift correction

This mirrors real financial risk systems.

## Reproducibility

Each run saves:

Full pipeline

Calibrator (if used)

Decision policy

Metrics

Drift signals

Resolved configuration

Re-running report requires no retraining.

## Dataset

This project uses the public credit card fraud dataset (not stored in this repository due to GitHub file size limits).
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

Place dataset locally at:

data/creditcard.csv

## What This Demonstrates

This project demonstrates:

Applied ML under extreme class imbalance

Calibration-aware probability modeling

Policy-driven decision thresholds

Cost-sensitive evaluation

Time-aware validation

Drift governance

Reproducible ML engineering

CLI-based deployable architecture

## Roadmap (Next Upgrades)

To make this production-grade enterprise-ready:

SHAP explainability for alerts

Automated PDF report export

Model registry integration

REST API wrapper (FastAPI)

Monitoring dashboard

CI test coverage expansion

## Explainability (SHAP)

FRS can generate SHAP-based explainability artifacts for a scored dataset.

### 1) Score a CSV
```bash
frs score --run-dir runs/2026-02-17_130730 --input data/score_sample.csv --output outputs/scored_sample.csv --summary outputs/scoring_summary.json

frs explain --method tree ^
  --run-dir runs/2026-02-17_130730 ^
  --input outputs/scored_sample.csv ^
  --out runs/2026-02-17_130730/explain/explanations.csv ^
  --summary runs/2026-02-17_130730/explain/explain_summary.json ^
  --figures runs/2026-02-17_130730/explain/figures ^
  --top-k 5 ^
  --max-rows 2000

@Author

Wissebo Abdulmajid 
wissebo22@gmail.com
Data Science & Applied ML Systems