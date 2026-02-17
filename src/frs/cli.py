from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from rich import print as rprint

from frs.artifacts import make_run_dir, save_bundle, save_json, save_yaml
from frs.calibration import (
    adjust_proba_for_prior_shift,
    apply_calibrator,
    fit_calibrator,
    predict_proba,
)
from frs.config import load_config
from frs.data_io import load_tabular, validate_min_schema
from frs.drift import drift_report_numeric, drift_report_proba
from frs.evaluation import compute_metrics
from frs.features import infer_features  # keep only infer here; we'll refactor features.py next
from frs.logging_setup import setup_logging
from frs.reporting import build_report_from_run
from frs.models import build_preprocessor, train_lightgbm, train_logreg
from frs.policy import (
    DecisionPolicy,
    apply_policy,
    choose_threshold_cost_sensitive,
    choose_threshold_top_k,
)
from frs.splits import stratified_train_val_test_split, time_train_val_test_split

app = typer.Typer(add_completion=False, help="Fraud / Risk Scoring System (FRS) CLI")


# ----------------------------
# Helpers (self-contained)
# ----------------------------
def _expected_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(fp * cost_fp + fn * cost_fn)


def _alert_rate(y_pred: np.ndarray) -> float:
    y_pred = np.asarray(y_pred).astype(int)
    return float(y_pred.mean()) if len(y_pred) else 0.0


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(pd.Series(rec).to_json())
            f.write("\n")


def _safe_metrics_to_dict(metrics_obj: Any) -> dict[str, Any]:
    try:
        return asdict(metrics_obj)
    except Exception:
        try:
            return dict(metrics_obj.__dict__)
        except Exception:
            return {"metrics": str(metrics_obj)}


def _time_sorted_df(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        raise ValueError(f"time_col={time_col!r} not found in columns.")
    return df.sort_values(by=time_col, kind="mergesort").reset_index(drop=True)


def _make_walkforward_folds(
    n_rows: int,
    n_folds: int,
    train_start_frac: float,
    step_frac: float,
    val_frac: float,
    test_frac: float,
) -> list[tuple[slice, slice, slice]]:
    """
    True walk-forward:
      Fold k:
        train: [0 : train_end_k]
        val  : [train_end_k : train_end_k + val_len]
        test : [train_end_k + val_len : train_end_k + val_len + test_len]
    Then train_end moves forward by step each fold.
    """
    if n_rows < 1000:
        raise ValueError("Dataset too small for walk-forward backtest (need >= 1000 rows).")

    for name, v in [
        ("train_start_frac", train_start_frac),
        ("step_frac", step_frac),
        ("val_frac", val_frac),
        ("test_frac", test_frac),
    ]:
        if not (0.0 < v < 1.0):
            raise ValueError(f"{name} must be in (0, 1). Got {v}")

    if train_start_frac + val_frac + test_frac >= 1.0:
        raise ValueError(
            "train_start_frac + val_frac + test_frac must be < 1.0 "
            f"(got {train_start_frac + val_frac + test_frac:.3f})."
        )

    val_len = int(round(n_rows * val_frac))
    test_len = int(round(n_rows * test_frac))
    step = int(round(n_rows * step_frac))
    train_end0 = int(round(n_rows * train_start_frac))

    if val_len < 1 or test_len < 1 or step < 1 or train_end0 < 1:
        raise ValueError("Fractions produce empty windows. Increase dataset size or adjust fractions.")

    folds: list[tuple[slice, slice, slice]] = []
    train_end = train_end0

    for _ in range(n_folds):
        val_start = train_end
        val_end = val_start + val_len
        test_start = val_end
        test_end = test_start + test_len

        if test_end > n_rows:
            break

        folds.append((slice(0, train_end), slice(val_start, val_end), slice(test_start, test_end)))
        train_end += step

    return folds


def _plot_cost_curve(
    out_path: Path,
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    thresholds = np.linspace(0.0, 1.0, 201)
    costs = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        costs.append(_expected_cost(y_true, y_pred, cost_fp=cost_fp, cost_fn=cost_fn))

    costs = np.asarray(costs, dtype=float)
    best_idx = int(np.argmin(costs))
    best_t = float(thresholds[best_idx])
    best_cost = float(costs[best_idx])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(thresholds, costs)
    plt.xlabel("Threshold")
    plt.ylabel("Expected Cost")
    plt.title("Cost vs Threshold")
    plt.axvline(best_t)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    return {"best_threshold_grid": best_t, "best_cost_grid": best_cost}


def _plot_pr_curve(out_path: Path, y_true: np.ndarray, proba: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    precision, recall, _ = precision_recall_curve(y_true, proba)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _choose_split(cfg, df: pd.DataFrame, log) -> Any:
    """
    Select split strategy based on config.
    - If time_col is provided => time-based split (recommended for fraud)
    - Else => stratified split
    """
    if cfg.data.time_col and str(cfg.data.time_col).strip():
        log.info("Using time-based split on column=%s", cfg.data.time_col)
        return time_train_val_test_split(
            df=df,
            target=cfg.data.target,
            time_col=cfg.data.time_col,
            test_size=cfg.split.test_size,
            val_size=cfg.split.val_size,
            seed=cfg.project.seed,
            id_col=cfg.data.id_col,
        )

    if cfg.split.strategy != "stratified":
        raise NotImplementedError("Only stratified split is implemented when time_col is not provided.")

    log.info("Using stratified split (no time_col provided)")
    return stratified_train_val_test_split(
        df=df,
        target=cfg.data.target,
        test_size=cfg.split.test_size,
        val_size=cfg.split.val_size,
        seed=cfg.project.seed,
    )


def _choose_threshold_cost_wrapper(
    *,
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    grid_size: int | None,
    min_alert_rate: float | None,
    max_alert_rate: float | None,
    prev_threshold: float | None,
) -> float:
    """
    Compatibility wrapper:
    - If frs.policy.choose_threshold_cost_sensitive supports advanced signature
      it will be used.
    - Otherwise fall back to simple signature.
    """
    try:
        return float(
            choose_threshold_cost_sensitive(
                y_true=y_true,
                proba=proba,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                grid_size=grid_size,
                min_alert_rate=min_alert_rate,
                max_alert_rate=max_alert_rate,
                prev_threshold=prev_threshold,
            )
        )
    except TypeError:
        return float(
            choose_threshold_cost_sensitive(
                y_true=y_true,
                proba=proba,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
            )
        )


def _get_features(cfg, df: pd.DataFrame, log) -> tuple[list[str], list[str]]:
    """
    Decide which columns are features (used by BOTH train + backtest).

    Rules:
    1) If YAML provides features.numeric/features.categorical => use them.
    2) Otherwise infer types from the dataframe.
    3) Always exclude columns that must never be model inputs:
       - target
       - time_col (if provided)
       - id_col (if provided)
       - anything in cfg.features.exclude (if present)
    """
    # 1) infer vs configured
    if not cfg.features.numeric and not cfg.features.categorical:
        fs = infer_features(df, cfg.data.target)
        numeric = list(fs.numeric)
        categorical = list(fs.categorical)
        log.info("Inferred features: numeric=%d categorical=%d", len(numeric), len(categorical))
    else:
        numeric = list(cfg.features.numeric)
        categorical = list(cfg.features.categorical)
        log.info("Configured features: numeric=%d categorical=%d", len(numeric), len(categorical))

    # 2) build exclusion set
    exclude = set(getattr(cfg.features, "exclude", []) or [])
    exclude.add(str(cfg.data.target))

    if cfg.data.time_col and str(cfg.data.time_col).strip():
        exclude.add(str(cfg.data.time_col))

    if cfg.data.id_col and str(cfg.data.id_col).strip():
        exclude.add(str(cfg.data.id_col))

    # 3) filter out
    numeric = [c for c in numeric if c not in exclude]
    categorical = [c for c in categorical if c not in exclude]

    log.info(
        "After exclude: numeric=%d categorical=%d excluded=%s",
        len(numeric),
        len(categorical),
        sorted(exclude),
    )

    return numeric, categorical


# ----------------------------
# Commands
# ----------------------------
@app.command("train")
def train_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to YAML config"),
):
    """
    Train a model, calibrate probabilities, select a decision threshold, and save artifacts.
    """
    log = setup_logging()
    cfg = load_config(config)

    df, meta = load_tabular(cfg.data.path)
    validate_min_schema(df, cfg.data.target)
    log.info("Loaded data: rows=%s cols=%s", meta.n_rows, meta.n_cols)

    numeric, categorical = _get_features(cfg, df, log)

    split = _choose_split(cfg, df, log)

    X_train = split.train.drop(columns=[cfg.data.target])
    y_train = split.train[cfg.data.target].to_numpy()

    X_val = split.val.drop(columns=[cfg.data.target])
    y_val = split.val[cfg.data.target].to_numpy()

    X_test = split.test.drop(columns=[cfg.data.target])
    y_test = split.test[cfg.data.target].to_numpy()

    unique = set(np.unique(y_train))
    if not unique.issubset({0, 1, True, False}):
        raise ValueError(f"Target must be binary (0/1). Found values: {sorted(unique)}")

    pre = build_preprocessor(numeric=numeric, categorical=categorical)

    # Optional baseline training (useful later for comparison; not used in scoring)
    _ = train_logreg(X_train, y_train, preprocessor=pre, seed=cfg.project.seed)

    if cfg.model.name == "lightgbm":
        model = train_lightgbm(
            X_train,
            y_train,
            preprocessor=pre,
            params=cfg.model.params,
            seed=cfg.project.seed,
        )
    else:
        model = train_logreg(X_train, y_train, preprocessor=pre, seed=cfg.project.seed)

    # Base probabilities
    val_base = predict_proba(model.pipeline, X_val)
    test_base = predict_proba(model.pipeline, X_test)

    # Calibration
    if cfg.calibration.enabled:
        cal = fit_calibrator(
            y_true=y_val,
            proba=val_base,
            method=cfg.calibration.method,
        )
        val_proba = apply_calibrator(cal, val_base)
        test_proba = apply_calibrator(cal, test_base)
        calibrator = cal
    else:
        val_proba = val_base
        test_proba = test_base
        calibrator = None
        metrics_dict["calibrated"] = bool(calibrator is not None)
        metrics_dict["calibration_method"] = cfg.calibration.method if calibrator is not None else None

    # Prior-shift correction (optional)
    if getattr(cfg.policy, "prior_shift_enabled", False):
        pi_train = float(np.mean(y_train.astype(int)))
        src = getattr(cfg.policy, "prior_shift_source", "val")
        if src == "val":
            pi_target = float(np.mean(y_val.astype(int)))
        elif src == "train_val":
            pi_target = float(np.mean(np.concatenate([y_train.astype(int), y_val.astype(int)])))
        else:
            pi_target = pi_train

        val_proba = adjust_proba_for_prior_shift(val_proba, pi_train=pi_train, pi_target=pi_target)
        test_proba = adjust_proba_for_prior_shift(test_proba, pi_train=pi_train, pi_target=pi_target)

    # Threshold / Policy
    if cfg.policy.mode == "cost":
        thr = _choose_threshold_cost_wrapper(
            y_true=y_val.astype(int),
            proba=val_proba,
            cost_fp=float(cfg.policy.cost_fp),
            cost_fn=float(cfg.policy.cost_fn),
            grid_size=int(getattr(cfg.policy, "threshold_grid_size", 401)),
            min_alert_rate=getattr(cfg.policy, "min_alert_rate", None),
            max_alert_rate=getattr(cfg.policy, "max_alert_rate", None),
            prev_threshold=None,
        )
        policy = DecisionPolicy(
            mode="cost",
            threshold=float(thr),
            cost_fp=float(cfg.policy.cost_fp),
            cost_fn=float(cfg.policy.cost_fn),
        )
    else:
        thr = choose_threshold_top_k(val_proba, top_k=float(cfg.policy.top_k))
        policy = DecisionPolicy(mode="top_k", threshold=float(thr), top_k=float(cfg.policy.top_k))

    # Evaluate on test
    y_pred = apply_policy(test_proba, threshold=float(policy.threshold))
    metrics = compute_metrics(
        y_true=y_test.astype(int),
        proba=test_proba,
        y_pred=y_pred,
        threshold=float(policy.threshold),
    )

    metrics_dict = _safe_metrics_to_dict(metrics)
    metrics_dict["expected_cost"] = _expected_cost(
        y_true=y_test.astype(int),
        y_pred=y_pred,
        cost_fp=float(cfg.policy.cost_fp),
        cost_fn=float(cfg.policy.cost_fn),
    )
    metrics_dict["alert_rate"] = _alert_rate(y_pred)
    metrics_dict["base_rate"] = float(np.mean(y_test.astype(int)))
    metrics_dict["split_mode"] = "time" if (cfg.data.time_col and str(cfg.data.time_col).strip()) else "stratified"

    # Drift signals (train vs test)
    drift = {
        "numeric": drift_report_numeric(split.train, split.test, numeric_cols=numeric, ks_alpha=0.05, ks_min=0.10),
        "proba": drift_report_proba(val_proba, test_proba),
    }

    run_dir = make_run_dir()
    save_yaml(run_dir / "config.resolved.yaml", cfg.model_dump())
    save_json(run_dir / "metrics.json", metrics_dict)
    save_json(run_dir / "drift.json", drift)

    bundle = {
        "config": cfg.model_dump(),
        "feature_spec": {"numeric": numeric, "categorical": categorical},
        "model_name": model.model_name,
        "pipeline": model.pipeline,
        "calibrator": calibrator,
        "policy": policy.__dict__,
    }
    save_bundle(run_dir / "bundle.joblib", bundle)
    log.info("Saved artifacts to: %s", run_dir)

    rprint(f"[green]Run saved:[/green] {run_dir}")
    rprint(metrics_dict)


@app.command("backtest")
def backtest_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to YAML config"),
    folds: int = typer.Option(3, "--folds", min=2, max=10, help="Number of walk-forward folds"),
    train_start_frac: float | None = typer.Option(
        None, "--train-start-frac", help="Initial train window fraction (overrides config)"
    ),
    step_frac: float | None = typer.Option(None, "--step-frac", help="How much train end moves forward each fold (overrides config)"),
    val_frac: float | None = typer.Option(None, "--val-frac", help="Validation window fraction (overrides config)"),
    test_frac: float | None = typer.Option(None, "--test-frac", help="Test window fraction (overrides config)"),
):
    """
    Walk-forward (rolling) backtest over time_col to validate stability under drift.
    Saves per-fold metrics + summary + cost/pr curves.
    """
    log = setup_logging()
    cfg = load_config(config)

    if not cfg.data.time_col or not str(cfg.data.time_col).strip():
        raise ValueError("backtest requires data.time_col (e.g., Time). Set it in your YAML config.")

    # Use config defaults unless CLI overrides are provided
    bt = getattr(cfg, "backtest", None)
    if bt is None:
        bt_train_start, bt_step, bt_val, bt_test = 0.50, 0.05, 0.10, 0.10
        ema_alpha = 0.70
    else:
        bt_train_start = float(bt.train_start_frac)
        bt_step = float(bt.step_frac)
        bt_val = float(bt.val_frac)
        bt_test = float(bt.test_frac)
        ema_alpha = float(getattr(bt, "threshold_ema_alpha", 0.70))

    train_start_frac = float(train_start_frac) if train_start_frac is not None else bt_train_start
    step_frac = float(step_frac) if step_frac is not None else bt_step
    val_frac = float(val_frac) if val_frac is not None else bt_val
    test_frac = float(test_frac) if test_frac is not None else bt_test

    df, meta = load_tabular(cfg.data.path)
    validate_min_schema(df, cfg.data.target)
    log.info("Loaded data: rows=%s cols=%s", meta.n_rows, meta.n_cols)
    log.info("Backtest: walk-forward requested_folds=%d time_col=%s", folds, cfg.data.time_col)
    log.info(
        "Backtest windows: train_start=%.2f step=%.2f val=%.2f test=%.2f",
        train_start_frac,
        step_frac,
        val_frac,
        test_frac,
    )

    numeric, categorical = _get_features(cfg, df, log)

    df_sorted = _time_sorted_df(df, cfg.data.time_col)
    n = len(df_sorted)

    fold_slices = _make_walkforward_folds(
        n_rows=n,
        n_folds=folds,
        train_start_frac=float(train_start_frac),
        step_frac=float(step_frac),
        val_frac=float(val_frac),
        test_frac=float(test_frac),
    )

    if len(fold_slices) != folds:
        raise ValueError(
            f"Requested folds={folds}, but only possible folds={len(fold_slices)} with these window fractions. "
            "Reduce val/test/step or reduce train_start_frac."
        )

    run_dir = make_run_dir()
    out_dir = run_dir / "backtest"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_records: list[dict[str, Any]] = []
    prev_thr: float | None = None
    alpha = float(ema_alpha)

    for i, (tr_sl, va_sl, te_sl) in enumerate(fold_slices, start=1):
        train_df = df_sorted.iloc[tr_sl].copy()
        val_df = df_sorted.iloc[va_sl].copy()
        test_df = df_sorted.iloc[te_sl].copy()

        X_train = train_df.drop(columns=[cfg.data.target])
        y_train = train_df[cfg.data.target].to_numpy()

        X_val = val_df.drop(columns=[cfg.data.target])
        y_val = val_df[cfg.data.target].to_numpy()

        X_test = test_df.drop(columns=[cfg.data.target])
        y_test = test_df[cfg.data.target].to_numpy()

        unique = set(np.unique(y_train))
        if not unique.issubset({0, 1, True, False}):
            raise ValueError(f"Target must be binary (0/1). Found values: {sorted(unique)}")

        pre = build_preprocessor(numeric=numeric, categorical=categorical)

        if cfg.model.name == "lightgbm":
            model = train_lightgbm(
                X_train,
                y_train,
                preprocessor=pre,
                params=cfg.model.params,
                seed=cfg.project.seed,
            )
        else:
            model = train_logreg(X_train, y_train, preprocessor=pre, seed=cfg.project.seed)

        val_base = predict_proba(model.pipeline, X_val)
        test_base = predict_proba(model.pipeline, X_test)

        # Calibration
        if cfg.calibration.enabled:
            cal = fit_calibrator(y_true=y_val, proba=val_base, method=cfg.calibration.method)
            val_proba = apply_calibrator(cal, val_base)
            test_proba = apply_calibrator(cal, test_base)
        else:
            val_proba = val_base
            test_proba = test_base

        # Prior-shift correction (fold-local, optional)
        if getattr(cfg.policy, "prior_shift_enabled", False):
            pi_train = float(np.mean(y_train.astype(int)))
            src = getattr(cfg.policy, "prior_shift_source", "val")
            if src == "val":
                pi_target = float(np.mean(y_val.astype(int)))
            elif src == "train_val":
                pi_target = float(np.mean(np.concatenate([y_train.astype(int), y_val.astype(int)])))
            else:
                pi_target = pi_train

            val_proba = adjust_proba_for_prior_shift(val_proba, pi_train=pi_train, pi_target=pi_target)
            test_proba = adjust_proba_for_prior_shift(test_proba, pi_train=pi_train, pi_target=pi_target)

        # Threshold selection (guardrails + optional EMA smoothing)
        if cfg.policy.mode == "cost":
            thr_raw = _choose_threshold_cost_wrapper(
                y_true=y_val.astype(int),
                proba=val_proba,
                cost_fp=float(cfg.policy.cost_fp),
                cost_fn=float(cfg.policy.cost_fn),
                grid_size=int(getattr(cfg.policy, "threshold_grid_size", 401)),
                min_alert_rate=getattr(cfg.policy, "min_alert_rate", None),
                max_alert_rate=getattr(cfg.policy, "max_alert_rate", None),
                prev_threshold=prev_thr,
            )

            if prev_thr is None:
                thr = float(thr_raw)
            else:
                thr = float(alpha * float(thr_raw) + (1.0 - alpha) * float(prev_thr))
        else:
            thr_raw = float(choose_threshold_top_k(val_proba, top_k=float(cfg.policy.top_k)))
            thr = float(thr_raw)

        prev_thr = float(thr)

        y_pred = apply_policy(test_proba, threshold=float(thr))
        metrics = compute_metrics(
            y_true=y_test.astype(int),
            proba=test_proba,
            y_pred=y_pred,
            threshold=float(thr),
        )

        drift_fold = {
            "numeric": drift_report_numeric(train_df, test_df, numeric_cols=numeric, ks_alpha=0.05, ks_min=0.10),
            "proba": drift_report_proba(val_proba, test_proba),
        }

        m = _safe_metrics_to_dict(metrics)
        m["fold"] = int(i)
        m["threshold_raw"] = float(thr_raw)
        m["threshold_ema"] = float(thr)
        m["threshold"] = float(thr)  # legacy key
        m["expected_cost"] = _expected_cost(
            y_true=y_test.astype(int),
            y_pred=y_pred,
            cost_fp=float(cfg.policy.cost_fp),
            cost_fn=float(cfg.policy.cost_fn),
        )
        m["alert_rate"] = _alert_rate(y_pred)
        m["base_rate"] = float(np.mean(y_test.astype(int)))
        m["drift"] = drift_fold

        best_grid = _plot_cost_curve(
            out_path=out_dir / f"cost_curve_fold_{i}.png",
            y_true=y_test.astype(int),
            proba=test_proba,
            cost_fp=float(cfg.policy.cost_fp),
            cost_fn=float(cfg.policy.cost_fn),
        )
        m.update(best_grid)
        _plot_pr_curve(out_path=out_dir / f"pr_curve_fold_{i}.png", y_true=y_test.astype(int), proba=test_proba)

        fold_records.append(m)
        rprint(f"[cyan]Fold {i}[/cyan] saved curves + metrics.")

    _write_jsonl(out_dir / "fold_metrics.jsonl", fold_records)

    dfm = pd.DataFrame(fold_records)
    summary: dict[str, Any] = {"folds": len(fold_records)}
    for col in [
        "pr_auc",
        "roc_auc",
        "brier",
        "expected_cost",
        "alert_rate",
        "base_rate",
        "tn",
        "fp",
        "fn",
        "tp",
        "threshold",
    ]:
        if col in dfm.columns:
            summary[f"{col}_mean"] = float(dfm[col].mean())
            summary[f"{col}_std"] = float(dfm[col].std(ddof=1)) if len(dfm) > 1 else 0.0

    if "threshold_raw" in dfm.columns:
        summary["threshold_raw_mean"] = float(dfm["threshold_raw"].mean())
        summary["threshold_raw_std"] = float(dfm["threshold_raw"].std(ddof=1)) if len(dfm) > 1 else 0.0
    if "threshold_ema" in dfm.columns:
        summary["threshold_ema_mean"] = float(dfm["threshold_ema"].mean())
        summary["threshold_ema_std"] = float(dfm["threshold_ema"].std(ddof=1)) if len(dfm) > 1 else 0.0
        summary["threshold_ema_alpha"] = float(alpha)

    save_yaml(run_dir / "config.resolved.yaml", cfg.model_dump())
    save_json(out_dir / "summary.json", summary)

    rprint(f"[green]Backtest saved:[/green] {out_dir}")
    rprint(summary)

@app.command("report")
def report_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, help="Path to a run directory (contains metrics.json)"),
    out: Path = typer.Option(..., "--out", help="Output report folder (e.g., reports/<run_id>)"),
    include_backtest: bool = typer.Option(True, "--include-backtest/--no-backtest", help="Include backtest artifacts if present"),
):
    """
    Generate a portfolio-grade report (REPORT.md + figures) from a saved run directory.
    """
    log = setup_logging()
    rp = build_report_from_run(run_dir=run_dir, out_dir=out, include_backtest=include_backtest)
    log.info("Report written: %s", rp.report_md)
    rprint(f"[green]Report created:[/green] {rp.report_md}")


@app.command("evaluate")
@app.command("score")
def score_cmd(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, help="Path to a training run directory (contains bundle.joblib)"),
    input_path: Path = typer.Option(..., "--input", exists=True, help="CSV file to score"),
    output_path: Path = typer.Option(..., "--output", help="Where to write scored CSV"),
    threshold: float | None = typer.Option(None, "--threshold", help="Override threshold (default: use bundle policy threshold)"),
    label_col: str | None = typer.Option(None, "--label-col", help="Optional label column (e.g., Class) to compute metrics"),
    summary_path: Path | None = typer.Option(None, "--summary", help="Optional path to write scoring summary JSON"),
):
    """
    Score new data using a saved run bundle (pipeline + optional calibrator + policy threshold).

    Output columns added:
      - risk_proba : calibrated probability (if calibrator exists), else raw model proba
      - alert      : 1 if risk_proba >= threshold else 0
      - threshold  : threshold used
      - model_name : model identifier
      - run_dir    : run directory used

    If --label-col is provided and exists in input, saves metrics as well.
    """
    log = setup_logging()

    bundle_path = run_dir / "bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle.joblib not found in run_dir: {run_dir}")

    import joblib

    bundle = joblib.load(bundle_path)

    pipeline = bundle.get("pipeline", None)
    calibrator = bundle.get("calibrator", None)
    policy = bundle.get("policy", None)
    feature_spec = bundle.get("feature_spec", None)
    model_name = bundle.get("model_name", "unknown")

    if pipeline is None:
        raise ValueError("bundle.joblib missing 'pipeline'.")

    if not isinstance(policy, dict) or "threshold" not in policy:
        raise ValueError("bundle.joblib missing 'policy.threshold'.")

    if not isinstance(feature_spec, dict) or "numeric" not in feature_spec or "categorical" not in feature_spec:
        raise ValueError("bundle.joblib missing 'feature_spec' (numeric/categorical).")

    numeric = list(feature_spec["numeric"])
    categorical = list(feature_spec["categorical"])
    required_cols = numeric + categorical

    # threshold used
    thr_used = float(threshold) if threshold is not None else float(policy["threshold"])

    # --- Load input
    df_in = pd.read_csv(input_path)
    log.info("Loaded input: rows=%d cols=%d", df_in.shape[0], df_in.shape[1])

    # --- Label handling (optional)
    y_true = None
    if label_col:
        if label_col in df_in.columns:
            y_true = df_in[label_col].to_numpy()
        else:
            raise ValueError(f"--label-col was provided as {label_col!r} but that column is not in input CSV.")

    # --- Validate required feature columns
    missing = [c for c in required_cols if c not in df_in.columns]
    if missing:
        raise ValueError(
            "Input is missing required feature columns.\n"
            f"Missing ({len(missing)}): {missing}\n"
            f"Expected features ({len(required_cols)}): {required_cols}"
        )

    # --- Predict probabilities
    X = df_in.copy()
    proba_base = predict_proba(pipeline, X)

    if calibrator is not None:
        proba = apply_calibrator(calibrator, proba_base)
        log.info("Applied calibrator from bundle.")
    else:
        proba = proba_base
        log.info("No calibrator in bundle; using raw probabilities.")

    proba = np.asarray(proba, dtype=float)
    alert = (proba >= thr_used).astype(int)

    # --- Write output CSV
    df_out = df_in.copy()
    df_out["risk_proba"] = proba
    df_out["alert"] = alert
    df_out["threshold"] = float(thr_used)
    df_out["model_name"] = str(model_name)
    df_out["run_dir"] = str(run_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)

    # --- Summary JSON (optional but recommended)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "model_name": str(model_name),
        "rows": int(len(df_out)),
        "threshold": float(thr_used),
        "alerts": int(df_out["alert"].sum()),
        "alert_rate": float(df_out["alert"].mean()) if len(df_out) else 0.0,
        "proba_min": float(np.min(proba)) if len(proba) else None,
        "proba_p50": float(np.percentile(proba, 50)) if len(proba) else None,
        "proba_p95": float(np.percentile(proba, 95)) if len(proba) else None,
        "proba_max": float(np.max(proba)) if len(proba) else None,
        "calibrated": bool(calibrator is not None),
    }

    # --- If labels provided, compute metrics too
    if y_true is not None:
        unique = set(np.unique(np.asarray(y_true)))
        if not unique.issubset({0, 1, True, False}):
            raise ValueError(f"Label column must be binary (0/1). Found values: {sorted(unique)}")

        metrics = compute_metrics(
            y_true=np.asarray(y_true).astype(int),
            proba=proba,
            y_pred=alert,
            threshold=float(thr_used),
        )
        metrics_dict = _safe_metrics_to_dict(metrics)
        metrics_dict["expected_cost"] = _expected_cost(
            y_true=np.asarray(y_true).astype(int),
            y_pred=alert,
            cost_fp=float(policy.get("cost_fp", 1.0)),
            cost_fn=float(policy.get("cost_fn", 10.0)),
        )
        summary["metrics"] = metrics_dict

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(summary_path, summary)

    rprint(f"[green]Scored file saved:[/green] {output_path}")
    if summary_path is not None:
        rprint(f"[green]Summary saved:[/green] {summary_path}")

    rprint(
        {
            "rows": summary["rows"],
            "alerts": summary["alerts"],
            "alert_rate": summary["alert_rate"],
            "threshold": summary["threshold"],
        }
    )

def main() -> None:
    app()


if __name__ == "__main__":
    main()
