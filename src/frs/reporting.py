from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import shutil
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


def _copy_if_exists(src: Path, dst: Path) -> bool:
    try:
        if src.exists() and src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            return True
    except Exception:
        return False
    return False


def build_report_from_run(
    *,
    run_dir: Path,
    out_dir: Path,
    include_backtest: bool = True,
) -> ReportPaths:

    rp = _ensure_dirs(out_dir)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in run_dir: {run_dir}")

    metrics = _load_json(metrics_path)

    drift_path = run_dir / "drift.json"
    drift = _load_json(drift_path) if _maybe(drift_path) else None

    backtest_summary_path = run_dir / "backtest" / "summary.json"
    backtest_summary = (
        _load_json(backtest_summary_path)
        if (include_backtest and _maybe(backtest_summary_path))
        else None
    )

    # ============================
    # Explainability (canonical)
    # ============================

    explain_dir = run_dir / "explain"
    explain_summary_path = explain_dir / "explain_summary.json"
    explain_csv_path = explain_dir / "explanations.csv"
    explain_fig_dir = explain_dir / "figures"

    has_explain = _maybe(explain_summary_path) and _maybe(explain_csv_path)

    explain_meta: dict[str, Any] | None = None
    if has_explain:
        try:
            explain_meta = _load_json(explain_summary_path)
        except Exception:
            explain_meta = None

    copied_explain_figs: list[Path] = []
    if explain_fig_dir.exists():
        for png in explain_fig_dir.glob("*.png"):
            dst = rp.figures_dir / png.name
            if _copy_if_exists(png, dst):
                copied_explain_figs.append(dst)

    # ============================
    # Write summary JSON
    # ============================

    summary = {
        "run_dir": str(run_dir),
        "calibrated": metrics.get("calibrated"),
        "threshold": metrics.get("threshold"),
        "roc_auc": metrics.get("roc_auc"),
        "pr_auc": metrics.get("pr_auc"),
        "brier": metrics.get("brier"),
        "alert_rate": metrics.get("alert_rate"),
        "base_rate": metrics.get("base_rate"),
        "expected_cost": metrics.get("expected_cost"),
        "has_drift": drift is not None,
        "has_backtest": backtest_summary is not None,
        "has_explainability": has_explain,
    }

    with rp.report_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ============================
    # Build Markdown
    # ============================

    md: list[str] = []

    md.append("# Fraud Risk Scoring — Run Report\n")
    md.append(f"**Run directory:** `{run_dir}`  \n")
    md.append(f"**Report output:** `{out_dir}`  \n")

    md.append("\n## What this run produced\n")
    md.append("- A trained classifier with probability calibration (if enabled)")
    md.append("- A decision policy (threshold selection)")
    md.append("- Evaluation metrics and drift signals")
    md.append("- Optional walk-forward backtest")
    md.append("- Optional SHAP explainability\n")

    md.append("\n## Key metrics (test split)\n")
    md.append(f"- Calibrated: **{_fmt(metrics.get('calibrated'))}**")
    md.append(f"- Threshold: **{_fmt(metrics.get('threshold'))}**")
    md.append(f"- Alert rate: **{_fmt(metrics.get('alert_rate'))}**")
    md.append(f"- Base rate: **{_fmt(metrics.get('base_rate'))}**")
    md.append(f"- ROC-AUC: **{_fmt(metrics.get('roc_auc'))}**")
    md.append(f"- PR-AUC: **{_fmt(metrics.get('pr_auc'))}**")
    md.append(f"- Brier score: **{_fmt(metrics.get('brier'))}**")

    if "expected_cost" in metrics:
        md.append(f"- Expected cost: **{_fmt(metrics.get('expected_cost'))}**")

    # Confusion matrix
    if all(k in metrics for k in ["tn", "fp", "fn", "tp"]):
        md.append("\n## Confusion matrix @ threshold\n")
        md.append("| | Pred 0 | Pred 1 |")
        md.append("|---:|---:|---:|")
        md.append(f"| True 0 | {_fmt(metrics['tn'])} | {_fmt(metrics['fp'])} |")
        md.append(f"| True 1 | {_fmt(metrics['fn'])} | {_fmt(metrics['tp'])} |")

    # ============================
    # Explainability Section
    # ============================

    md.append("\n## Explainability (SHAP)\n")

    if not has_explain:
        md.append(
            "- Explainability artifacts not found (run `frs explain --run-dir <RUN_DIR> --input <CSV>`)."
        )
    else:
        if explain_meta:
            md.append(f"- Method: **{_fmt(explain_meta.get('method'))}**")
            md.append(f"- Rows explained: **{_fmt(explain_meta.get('rows_explained'))}**")
            md.append(f"- Top-k per row: **{_fmt(explain_meta.get('top_k'))}**")
        md.append(f"- Explanations CSV: `{explain_csv_path}`")

        if copied_explain_figs:
            md.append("\n### Explainability figures\n")
            for p in sorted(copied_explain_figs):
                md.append(f"- ![]({p.relative_to(rp.out_dir).as_posix()})")

    # ============================
    # Drift
    # ============================

    md.append("\n## Drift checks\n")
    if drift is None:
        md.append("- Drift artifacts not found.")
    else:
        md.append("- Numeric drift: see `drift.json`.")
        md.append("- Probability drift: see `drift.json`.")

    # ============================
    # Backtest
    # ============================

    md.append("\n## Walk-forward backtest (stability under time drift)\n")

    if backtest_summary is None:
        md.append("- Backtest artifacts not found (run `frs backtest -c <config>`).")
    else:
        md.append(f"- Folds: **{_fmt(backtest_summary.get('folds'))}**")
        for k in [
            "roc_auc_mean",
            "pr_auc_mean",
            "brier_mean",
            "expected_cost_mean",
            "alert_rate_mean",
            "threshold_mean",
        ]:
            if k in backtest_summary:
                md.append(f"- {k}: **{_fmt(backtest_summary.get(k))}**")

        bt_dir = run_dir / "backtest"
        copied_bt_figs: list[Path] = []
        for png in bt_dir.glob("*.png"):
            dst = rp.figures_dir / png.name
            if _copy_if_exists(png, dst):
                copied_bt_figs.append(dst)

        if copied_bt_figs:
            md.append("\n### Backtest curves\n")
            for p in sorted(copied_bt_figs):
                md.append(f"- ![]({p.relative_to(rp.out_dir).as_posix()})")

    md.append("\n## Reproducibility\n")
    md.append("- This report is generated strictly from saved run artifacts (`metrics.json`, `drift.json`, backtest outputs, and explain outputs).")

    with rp.report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md).strip() + "\n")

    return rp
