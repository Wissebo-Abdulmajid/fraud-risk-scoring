from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import json
import warnings

import numpy as np
import pandas as pd


@dataclass
class ExplainResult:
    out_csv: Path
    out_json: Path
    figures_dir: Path


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_bundle(run_dir: Path) -> dict[str, Any]:
    import joblib

    bundle_path = run_dir / "bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle.joblib not found in run_dir: {run_dir}")

    bundle = joblib.load(bundle_path)
    if "pipeline" not in bundle:
        raise ValueError("bundle.joblib missing 'pipeline'.")
    if "policy" not in bundle or not isinstance(bundle["policy"], dict) or "threshold" not in bundle["policy"]:
        raise ValueError("bundle.joblib missing 'policy.threshold'.")
    if "feature_spec" not in bundle:
        raise ValueError("bundle.joblib missing 'feature_spec'.")
    return bundle


def _get_required_cols(feature_spec: dict[str, Any]) -> list[str]:
    numeric = list(feature_spec.get("numeric", []))
    categorical = list(feature_spec.get("categorical", []))
    req = numeric + categorical
    if not req:
        raise ValueError("feature_spec has no features (numeric/categorical empty).")
    return req


def _topk_str(feature_names: list[str], shap_row: np.ndarray, k: int) -> str:
    idx = np.argsort(np.abs(shap_row))[::-1][:k]
    parts = [f"{feature_names[i]}:{shap_row[i]:+.4f}" for i in idx]
    return "; ".join(parts)


def _plot_global_bar(fig_path: Path, feature_names: list[str], shap_values: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = order[:20]

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.barh([feature_names[i] for i in top][::-1], mean_abs[top][::-1])
    plt.title("Global feature importance (mean |SHAP|) — Top 20")
    plt.xlabel("mean(|SHAP|)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _plot_beeswarm(fig_path: Path, X: pd.DataFrame, shap_values: np.ndarray, feature_names: list[str]) -> None:
    import shap
    import matplotlib.pyplot as plt

    # silence the warning you saw (NumPy RNG global seeding)
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message="The NumPy global RNG was seeded by calling `np.random.seed`",
    )

    fig_path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()


def _try_tree_explainer(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]] | None:
    """
    Fast path for tree models (LightGBM sklearn API).
    Returns (shap_values, feature_names) if possible, else None.
    """
    try:
        import shap
        import lightgbm as lgb
    except Exception:
        return None

    estimator = None
    preprocessor = None

    # If sklearn Pipeline: last step is estimator, earlier are preprocessors
    if hasattr(pipeline, "steps") and pipeline.steps:
        estimator = pipeline.steps[-1][1]
        if len(pipeline.steps) > 1:
            preprocessor = pipeline.steps[-2][1]
    else:
        estimator = pipeline

    if estimator is None or not isinstance(estimator, lgb.LGBMClassifier):
        return None

    # Transform X through preprocessor if present
    if preprocessor is not None and hasattr(preprocessor, "transform"):
        X_t = preprocessor.transform(X)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = [f"f{i}" for i in range(X_t.shape[1])]
    else:
        X_t = X.to_numpy()
        feature_names = list(X.columns)

    explainer = shap.TreeExplainer(estimator)

    # SHAP + LightGBM warning (binary output list)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
        )
        shap_values = explainer.shap_values(X_t)

    # For binary classification, shap_values is often [class0, class1]
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values, dtype=float)
    return shap_values, feature_names


def explain_scored_csv(
    *,
    run_dir: Path,
    scored_csv: Path,
    out_csv: Path,
    out_json: Path,
    figures_dir: Path,
    top_k: int = 10,
    max_rows: int = 2000,
    background_size: int = 200,
    method: Literal["auto", "tree", "permutation"] = "auto",
    make_beeswarm: bool = True,
) -> ExplainResult:
    """
    Explains a SCORED CSV (must contain original feature columns; may also contain risk_proba/alert).
    Writes:
      - out_csv: original rows (+ row_index) + shap_top_features
      - out_json: summary incl. global importance
      - figures_dir: global bar + beeswarm
    """
    import shap

    bundle = _load_bundle(run_dir)
    pipeline = bundle["pipeline"]
    policy = bundle["policy"]
    feature_spec = bundle["feature_spec"]
    model_name = str(bundle.get("model_name", "unknown"))

    required_cols = _get_required_cols(feature_spec)

    df = pd.read_csv(scored_csv)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Scored CSV missing required feature columns.\n"
            f"Missing ({len(missing)}): {missing}"
        )

    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})
    n_total = len(df)
    df_use = df.head(int(max_rows)).copy()

    X = df_use[required_cols].copy()

    # Ensure risk_proba + alert exist (if not, compute)
    if "risk_proba" not in df_use.columns:
        df_use["risk_proba"] = pipeline.predict_proba(X)[:, 1]
    if "alert" not in df_use.columns:
        thr = float(policy["threshold"])
        df_use["alert"] = (df_use["risk_proba"].to_numpy() >= thr).astype(int)

    shap_values: np.ndarray | None = None
    feature_names: list[str] | None = None
    used_method: str = method

    if method in ("auto", "tree"):
        tree_res = _try_tree_explainer(pipeline, X)
        if tree_res is not None:
            shap_values, feature_names = tree_res
            used_method = "tree"
        elif method == "tree":
            raise RuntimeError("Requested method=tree but TreeExplainer path is not available for this model.")

    if shap_values is None or feature_names is None:
        # fallback: permutation-like model-agnostic explainer
        used_method = "permutation"

        bg = X.sample(n=min(int(background_size), len(X)), random_state=42)

        def f_predict_proba(data: pd.DataFrame) -> np.ndarray:
            return pipeline.predict_proba(data)[:, 1]

        explainer = shap.Explainer(f_predict_proba, bg)
        exp = explainer(X)
        shap_values = np.asarray(exp.values, dtype=float)
        feature_names = list(X.columns)

    # Add per-row top-k contributions
    out = df_use.copy()
    k = int(top_k)
    out["shap_top_features"] = [
        _topk_str(feature_names, shap_values[i, :], k=k) for i in range(len(out))
    ]

    _safe_mkdir(out_csv.parent)
    _safe_mkdir(out_json.parent)
    _safe_mkdir(figures_dir)

    out.to_csv(out_csv, index=False)

    # Global importance table (mean |SHAP|)
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    global_importance = [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in np.argsort(mean_abs)[::-1]
    ]

    summary = {
        "run_dir": str(run_dir),
        "model_name": model_name,
        "method_requested": method,
        "method_used": used_method,
        "rows_total_in_csv": int(n_total),
        "rows_explained": int(len(out)),
        "top_k": int(k),
        "max_rows": int(max_rows),
        "background_size": int(background_size),
        "threshold": float(policy["threshold"]),
        "global_importance": global_importance,
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_global_bar(figures_dir / "global_importance_bar.png", feature_names, shap_values)
    if make_beeswarm:
        _plot_beeswarm(figures_dir / "global_beeswarm.png", X, shap_values, feature_names)

    return ExplainResult(out_csv=out_csv, out_json=out_json, figures_dir=figures_dir)


# ---------------------------
# Stable engine + CLI wrapper
# ---------------------------

def explain_cmd(
    *,
    run_dir: Path,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    figures_dir: Path,
    top_k: int = 10,
    max_rows: int = 2000,
    method: str = "tree",
) -> ExplainResult:
    """
    STABLE engine function for BOTH CLI and API.

    - input_path: scored CSV (or feature CSV; must include required feature columns)
    - output_path: explanations CSV output
    - summary_path: explain_summary.json output
    - figures_dir: folder for global plots
    """
    m = method.lower().strip()
    if m not in {"auto", "tree", "permutation"}:
        raise ValueError("method must be one of: auto, tree, permutation")

    return explain_scored_csv(
        run_dir=run_dir,
        scored_csv=input_path,
        out_csv=output_path,
        out_json=summary_path,
        figures_dir=figures_dir,
        top_k=top_k,
        max_rows=max_rows,
        method=m,  # type: ignore[arg-type]
        make_beeswarm=True,
    )


def explain_cli(
    run_dir: Path,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    figures_dir: Path,
    top_k: int,
    max_rows: int,
    method: str,
) -> None:
    """
    Thin wrapper for Typer CLI.
    """
    explain_cmd(
        run_dir=run_dir,
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        figures_dir=figures_dir,
        top_k=top_k,
        max_rows=max_rows,
        method=method,
    )
