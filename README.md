Fraud Risk Scoring System (FRS)

Calibrated Risk Modeling • Cost-Sensitive Policy • Drift Monitoring • Reproducible ML Engineering

Author: Wissebo Abdulmajid
Data Science & Computational intelligence at IIUM

Overview

The Fraud Risk Scoring System (FRS) is a production-oriented machine learning framework for fraud detection and risk-based decision systems.

Rather than focusing solely on classification accuracy, FRS models fraud detection as an end-to-end operational risk system:

Probability estimation

Calibration of risk scores

Explicit decision policy

Cost-sensitive threshold optimization

Alert-rate governance

Drift monitoring

Walk-forward backtesting

Reproducible run artifacts

Deployable scoring API

The objective is to demonstrate how real-world financial risk systems are engineered beyond model training.

System Design Philosophy

FRS separates three critical layers:

Probability Modeling

Binary classifier (LightGBM / Logistic Regression)

Imbalance-aware training

Calibration (Platt / Isotonic)

Decision Policy

Cost-sensitive threshold optimization

Top-K alert-rate strategy

Alert-rate guardrails

Prior-shift adjustment

Governance & Monitoring

Feature drift detection

Score distribution drift

Walk-forward validation

Reproducible experiment artifacts

This separation reflects how risk systems are deployed in financial environments.

System Architecture
Raw Data
   ↓
Feature Selection
   ↓
Preprocessing Pipeline
   ↓
Classifier
   ↓
Probability Calibration (optional)
   ↓
Policy Threshold Selection
   ↓
Risk Score + Alert Decision
   ↓
Drift Monitoring + Reporting
Example Model Performance

From run: 2026-02-17_130730

Metric	Value
ROC-AUC	0.9675
PR-AUC	0.7812
Brier Score	0.0004
Threshold	0.07
Alert Rate	0.0012
Base Rate	0.0013
Expected Cost	192.0

Confusion Matrix (Policy Threshold)

	Pred 0	Pred 1
True 0	56874	12
True 1	18	57

This demonstrates:

Strong ranking under class imbalance

Controlled operational alert rate

Explicit cost-aware decision threshold

Calibrated probability outputs

Command Line Interface
Train Model
frs train -c configs/base.yaml

Outputs:

runs/<timestamp>/
 ├─ bundle.joblib
 ├─ metrics.json
 ├─ drift.json
 ├─ config.resolved.yaml
Walk-Forward Backtest
frs backtest -c configs/base.yaml --folds 5

Generates:

Per-fold metrics

Threshold stability tracking

Cost curves

Precision-Recall curves

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

Generate Portfolio Report
frs report \
  --run-dir runs/<RUN_ID> \
  --out reports/<RUN_ID>

Produces:

reports/<RUN_ID>/
 ├─ REPORT.md
 ├─ report_summary.json

Reports are fully reproducible from saved artifacts.

Explainability (SHAP)

FRS supports SHAP-based explanation generation for scored datasets.

Example:

frs explain \
  --method tree \
  --run-dir runs/2026-02-17_130730 \
  --input outputs/scored_sample.csv \
  --out runs/2026-02-17_130730/explain/explanations.csv \
  --summary runs/2026-02-17_130730/explain/explain_summary.json \
  --figures runs/2026-02-17_130730/explain/figures \
  --top-k 5 \
  --max-rows 2000

Artifacts include:

Row-level feature attribution

Aggregated importance summaries

Visual diagnostic plots

REST API Deployment

FRS includes a hardened FastAPI service for production scoring.

Endpoints

GET /health

POST /v1/score-json

POST /v1/score-csv

POST /v1/explain

POST /v1/report

Features

API key authentication

Path traversal protection

Upload size limits

In-memory model bundle caching

Strict feature validation

Versioned API structure

Example:

curl -X POST https://<deployment-url>/v1/score-json \
  -H "x-api-key: <API_KEY>" \
  -H "Content-Type: application/json" \
  -d @payload.json
Drift Monitoring

FRS monitors:

Feature Drift

Kolmogorov-Smirnov tests

Effect size tracking

Probability Drift

Validation vs test score distribution shift

Results are stored in:

runs/<RUN_ID>/drift.json
Configuration (YAML Driven)

All behavior is controlled through YAML configuration:

Model hyperparameters

Calibration method

Threshold policy

Cost parameters

Alert rate constraints

Backtest folds

Prior shift adjustment

This ensures:

No hidden parameters

Fully reproducible experiments

Clear governance of decision rules

Dataset

This project uses the public credit card fraud dataset:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Due to file size limits, the dataset is not stored in this repository.

Place locally at:

data/creditcard.csv
What This Project Demonstrates

Applied machine learning under extreme class imbalance

Calibration-aware probability modeling

Policy-driven threshold selection

Cost-sensitive evaluation

Drift governance

Walk-forward time validation

Reproducible ML experimentation

CLI-based deployable architecture

Production-ready API wrapping

@Author

Wissebo Abdulmajid
Data Science & Computational Intelligence
wissebo22@gmail.com