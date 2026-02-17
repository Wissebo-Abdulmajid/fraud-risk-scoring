# Fraud Risk Scoring (FRS) — Calibrated Risk + Policy Thresholding + Drift Signals

A production-minded **fraud / risk scoring CLI** that trains a binary classifier, optionally calibrates probabilities, selects a decision threshold using an explicit policy (cost-sensitive or top-k), evaluates performance, and exports reproducible run artifacts + a portfolio-grade report.

This repo is designed to demonstrate **applied ML engineering for risk systems**: calibrated probabilities, threshold governance, drift monitoring, and reproducibility.

---

## Why this project exists

Fraud detection is not just “train a model and pick 0.5”. Real systems need:

- **Risk scores you can trust** (calibration when required)
- **Decision policies** (cost-sensitive thresholding or constrained alert rates)
- **Operational metrics** (alert rate vs base rate, expected cost)
- **Drift signals** (feature drift + score drift)
- **Reproducible runs** (saved configs, metrics, model bundles)

FRS focuses on those principles end-to-end.

---

## What you get

### Core capabilities
- **Train**: model training + optional probability calibration + policy threshold selection
- **Evaluate**: ROC-AUC, PR-AUC, Brier score + confusion matrix at chosen threshold
- **Score**: score new CSVs and output `risk_proba` + `alert`
- **Drift signals**:
  - numeric drift checks (KS tests + effect size thresholds)
  - probability distribution drift
- **Report**: generate `REPORT.md` from saved run artifacts for clean portfolio evidence

### Implemented CLI commands
- `frs train`
- `frs backtest` (walk-forward validation; requires `time_col`)
- `frs score` / `frs evaluate`
- `frs report`

---

## Quickstart

### 1) Create environment + install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -U pip
pip install -e .
