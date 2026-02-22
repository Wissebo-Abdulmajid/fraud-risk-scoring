from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd


@dataclass
class ExplainResult:
    explanations_csv: Path
    summary_json: Path
    figures_dir: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def explain_scored_csv(
    *,
    run_dir: Path,
    scored_csv: Path,
    output_csv: Path,
    summary_path: Path,
    figures_dir: Path,
    method: str = "tree",
    top_k: int = 10,
    max_rows: int = 2000,
) -> ExplainResult:
    """
    Production-safe explainability entry point.
    Reads a SCORED CSV and writes:
      - output_csv (per-row top-k contributions)
      - summary_path (metadata + global importance)
      - figures_dir (global plots)
    """
    import joblib
    import shap

    from frs.calibration import predict_proba  # uses pipeline predict_proba safely
    from frs.logging_setup import setup_logging

    log = setup_logging()

    bundle_path = run_dir / "bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle.joblib not found in run_dir: {run_dir}")

    bundle = joblib.load(bundle_path)
    pipeline = bundle.get("pipeline", None)
    feature_spec = bundle.get("feature_spec", None)
    model_name = bundle.get("model_name", "unknown")

    if pipeline is None:
        raise ValueError("bundle.joblib missing 'pipeline'.")
    if not isinstance(feature_spec, dict) or "numeric" not in feature_spec or "categorical" not in feature_spec:
        raise ValueError("bundle.joblib missing 'feature_spec' (numeric/categorical).")

    required_cols = list(feature_spec["numeric"]) + list(feature_spec["categorical"])

    df = pd.read_csv(scored_csv)
    if len(df) == 0:
        raise ValueError("Input scored CSV is empty.")

    # Cap rows for runtime
    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    # Validate required features exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Scored CSV missing required feature cols ({len(missing)}): {missing}")

    X = df[required_cols].copy()

    # Prefer calibrated score already in file, else recompute
    if "risk_proba" in df.columns:
        proba = df["risk_proba"].to_numpy(dtype=float)
    else:
        proba = predict_proba(pipeline, X)

    # --- Choose SHAP explainer
    method = (method or "").lower().strip()
    if method not in {"tree", "permutation"}:
        raise ValueError("method must be one of: tree, permutation")

    # Pipeline is sklearn Pipeline, final estimator likely LightGBM inside.
    # TreeExplainer works best when we can access the underlying tree model.
    estimator = getattr(pipeline, "named_steps", {}).get("model", None)

    if method == "tree" and estimator is not None:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X)
        # LightGBM binary sometimes returns list[ndarray]
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_values = np.asarray(shap_values, dtype=float)
    else:
        # model-agnostic, slower but safe for any estimator
        def f(m: np.ndarray) -> np.ndarray:
            Xtmp = pd.DataFrame(m, columns=required_cols)
            return predict_proba(pipeline, Xtmp)

        explainer = shap.PermutationExplainer(f, X, max_evals=500)
        shap_values = np.asarray(explainer(X).values, dtype=float)

    # --- Global importance
    abs_mean = np.mean(np.abs(shap_values), axis=0)
    global_rank = np.argsort(-abs_mean)

    global_table = []
    for idx in global_rank[: min(50, len(required_cols))]:
        global_table.append(
            {
                "feature": required_cols[int(idx)],
                "mean_abs_shap": float(abs_mean[int(idx)]),
            }
        )

    # --- Per-row top-k explanations
    rows = []
    for i in range(len(X)):
        sv = shap_values[i]
        top_idx = np.argsort(-np.abs(sv))[:top_k]
        row = {
            "row_index": int(i),
            "risk_proba": float(proba[i]) if i < len(proba) else None,
        }
        for j, fi in enumerate(top_idx, start=1):
            feat = required_cols[int(fi)]
            row[f"top{j}_feature"] = feat
            row[f"top{j}_shap"] = float(sv[int(fi)])
            row[f"top{j}_value"] = float(X.iloc[i][feat])
        rows.append(row)

    # Write outputs
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_dir": str(run_dir),
        "model_name": str(model_name),
        "method": method,
        "rows_explained": int(len(X)),
        "top_k": int(top_k),
        "global_importance": global_table,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figures
    _ensure_dir(figures_dir)
    try:
        shap.summary_plot(shap_values, X, show=False)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(figures_dir / "global_beeswarm.png", dpi=180)
        plt.close()

        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(figures_dir / "global_importance_bar.png", dpi=180)
        plt.close()
    except Exception as e:
        log.warning("Failed to generate SHAP figures: %s", e)

    return ExplainResult(
        explanations_csv=output_csv,
        summary_json=summary_path,
        figures_dir=figures_dir,
    )
