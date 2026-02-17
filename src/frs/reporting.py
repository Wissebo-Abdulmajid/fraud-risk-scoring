from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np


@dataclass
class ReportPaths:
    out_dir: Path
    figures_dir: Path
    report_md: Path
    report_json: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe(path: Path) -> bool:
    return path.exists() and path.is_file()


def _ensure_dirs(out_dir: Path) -> ReportPaths:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return ReportPaths(
        out_dir=out_dir,
        figures_dir=figures_dir,
        report_md=out_dir / "REPORT.md",
        report_json=out_dir / "report_summary.json",
    )


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "N/A"
    try:
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{nd}f}"
        return str(x)
    except Exception:
        return str(x)


def _plot_score_hist(fig_path: Path, proba: np.ndarray, threshold: float) -> None:
    import matplotlib.pyplot as plt

    proba = np.asarray(proba, dtype=float)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(proba, bins=50)
    plt.axvline(float(threshold))
    plt.title("Score distribution (risk_proba)")
    plt.xlabel("risk_proba")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _plot_pr(fig_path: Path, y_true: np.ndarray, proba: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    p, r, _ = precision_recall_curve(y_true, proba)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(r, p)
    plt.title("Precision–Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _plot_roc(fig_path: Path, y_true: np.ndarray, proba: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, proba)

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _plot_calibration(fig_path: Path, y_true: np.ndarray, proba: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(mean_pred, frac_pos)
    # perfect calibration diagonal
    plt.plot([0, 1], [0, 1])
    plt.title("Calibration curve")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def build_report_from_run(
    *,
    run_dir: Path,
    out_dir: Path,
    include_backtest: bool = True,
) -> ReportPaths:
    """
    Build a portfolio-grade report from an existing run directory.

    Expected inputs from train/backtest commands:
      - run_dir/metrics.json
      - run_dir/config.resolved.yaml (optional, not parsed here)
      - run_dir/drift.json (optional)
      - run_dir/backtest/summary.json (optional)
      - run_dir/backtest/fold_metrics.jsonl (optional)
      - run_dir/backtest/*.png (optional)
    """
    rp = _ensure_dirs(out_dir)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in run_dir: {run_dir}")

    metrics = _load_json(metrics_path)

    drift_path = run_dir / "drift.json"
    drift = _load_json(drift_path) if _maybe(drift_path) else None

    backtest_summary_path = run_dir / "backtest" / "summary.json"
    backtest_summary = _load_json(backtest_summary_path) if (include_backtest and _maybe(backtest_summary_path)) else None

    # Optional: create a score distribution plot IF we have scoring output saved somewhere.
    # (Your current pipeline doesn't persist proba arrays on train; so we keep this minimal and truthful.)

    # Write a small JSON summary (useful for README badges later)
    summary = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "calibrated": metrics.get("calibrated", None),
        "threshold": metrics.get("threshold", metrics.get("policy_threshold", None)),
        "roc_auc": metrics.get("roc_auc", None),
        "pr_auc": metrics.get("pr_auc", None),
        "brier": metrics.get("brier", None),
        "alert_rate": metrics.get("alert_rate", None),
        "base_rate": metrics.get("base_rate", None),
        "expected_cost": metrics.get("expected_cost", None),
        "has_drift": drift is not None,
        "has_backtest": backtest_summary is not None,
    }
    with rp.report_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Build REPORT.md (elite but honest: only uses artifacts)
    md = []
    md.append("# Fraud Risk Scoring — Run Report\n")
    md.append(f"**Run directory:** `{run_dir}`  \n")
    md.append(f"**Report output:** `{out_dir}`  \n")

    md.append("\n## What this run produced\n")
    md.append("- A trained classifier with probability calibration (if enabled)\n")
    md.append("- A decision policy (threshold selection)\n")
    md.append("- Evaluation metrics and drift signals (and walk-forward backtest if available)\n")

    md.append("\n## Key metrics (test split)\n")
    md.append(f"- Calibrated: **{_fmt(metrics.get('calibrated'))}**\n")
    md.append(f"- Threshold: **{_fmt(metrics.get('threshold'))}**\n")
    md.append(f"- Alert rate: **{_fmt(metrics.get('alert_rate'))}**\n")
    md.append(f"- Base rate: **{_fmt(metrics.get('base_rate'))}**\n")
    md.append(f"- ROC-AUC: **{_fmt(metrics.get('roc_auc'))}**\n")
    md.append(f"- PR-AUC: **{_fmt(metrics.get('pr_auc'))}**\n")
    md.append(f"- Brier score: **{_fmt(metrics.get('brier'))}**\n")
    if "expected_cost" in metrics:
        md.append(f"- Expected cost: **{_fmt(metrics.get('expected_cost'))}**\n")

    # Confusion matrix if present
    cm_keys = ["tn", "fp", "fn", "tp"]
    if all(k in metrics for k in cm_keys):
        md.append("\n## Confusion matrix @ threshold\n")
        md.append("| | Pred 0 | Pred 1 |\n|---:|---:|---:|\n")
        md.append(f"| True 0 | {_fmt(metrics['tn'])} | {_fmt(metrics['fp'])} |\n")
        md.append(f"| True 1 | {_fmt(metrics['fn'])} | {_fmt(metrics['tp'])} |\n")

    # Drift
    md.append("\n## Drift checks\n")
    if drift is None:
        md.append("- Drift artifacts not found for this run.\n")
    else:
        md.append("- Numeric drift: saved in `drift.json` (KS tests + effect size thresholds).\n")
        md.append("- Probability drift: saved in `drift.json` (distribution shift signals).\n")

    # Backtest
    md.append("\n## Walk-forward backtest (stability under time drift)\n")
    if backtest_summary is None:
        md.append("- Backtest artifacts not found (run `frs backtest -c <config>` to generate them).\n")
    else:
        md.append(f"- Folds: **{_fmt(backtest_summary.get('folds'))}**\n")
        for k in ["roc_auc_mean", "pr_auc_mean", "brier_mean", "expected_cost_mean", "alert_rate_mean", "threshold_mean"]:
            if k in backtest_summary:
                md.append(f"- {k}: **{_fmt(backtest_summary.get(k))}**\n")
        md.append("\nBacktest figures (if present) are in `figures/`.\n")

        # Copy backtest pngs if they exist
        bt_dir = run_dir / "backtest"
        for png in bt_dir.glob("*.png"):
            target = rp.figures_dir / png.name
            try:
                target.write_bytes(png.read_bytes())
            except Exception:
                pass

        # Embed copied pngs
        copied = sorted(rp.figures_dir.glob("*.png"))
        if copied:
            md.append("\n### Backtest curves\n")
            for p in copied:
                md.append(f"- ![]({p.relative_to(rp.out_dir).as_posix()})\n")

    md.append("\n## Reproducibility\n")
    md.append("- This report is generated only from the saved run artifacts (`metrics.json`, `drift.json`, and optional backtest outputs).\n")

    with rp.report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md).strip() + "\n")

    return rp
