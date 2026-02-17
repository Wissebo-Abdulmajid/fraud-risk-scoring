from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-8) -> float:
    """
    Population Stability Index (PSI) on 1D arrays.
    Practical guide:
      < 0.10  : small / negligible drift
      0.10-0.25: moderate drift
      > 0.25  : significant drift
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)

    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]
    if len(e) < 50 or len(a) < 50:
        return 0.0

    # Quantile bins based on expected (train) to stabilize the reference
    qs = np.linspace(0.0, 1.0, bins + 1)
    cuts = np.quantile(e, qs)

    # Ensure strictly increasing cut edges
    cuts = np.unique(cuts)
    if len(cuts) < 3:
        return 0.0

    e_hist, _ = np.histogram(e, bins=cuts)
    a_hist, _ = np.histogram(a, bins=cuts)

    e_pct = e_hist / max(e_hist.sum(), 1)
    a_pct = a_hist / max(a_hist.sum(), 1)

    e_pct = np.clip(e_pct, eps, 1.0)
    a_pct = np.clip(a_pct, eps, 1.0)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _psi_label(psi: float) -> str:
    if psi >= 0.25:
        return "high"
    if psi >= 0.10:
        return "moderate"
    return "low"


def drift_report_numeric(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
    *,
    ks_alpha: float = 0.05,
    ks_min: float = 0.10,
    psi_bins: int = 10,
    psi_moderate: float = 0.10,
    psi_high: float = 0.25,
    min_n: int = 200,
    max_cols: int = 30,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Numeric drift report (train vs test) with:

    1) KS test: flags drift only if BOTH:
        - p_value < ks_alpha
        - ks_stat >= ks_min
       (This prevents "everything drifts" on huge datasets.)

    2) PSI per column (quantile-binned on train):
        - easier to interpret magnitude

    Returns:
      - summary counts
      - per_col details
      - top_drifts ranked by PSI then KS
    """
    cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns]
    cols = cols[: int(max_cols)]

    per_col: dict[str, Any] = {}

    n_ks_drift = 0
    n_psi_moderate = 0
    n_psi_high = 0

    for c in cols:
        x = train_df[c].to_numpy()
        y = test_df[c].to_numpy()

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if len(x) < min_n or len(y) < min_n:
            continue

        # KS test
        stat, pval = ks_2samp(x, y)
        stat = float(stat)
        pval = float(pval)

        # Important: require both statistical significance + effect size
        ks_drift = bool((pval < float(ks_alpha)) and (stat >= float(ks_min)))

        # PSI magnitude
        psi = float(_psi(x, y, bins=int(psi_bins)))
        psi_level = _psi_label(psi)

        if ks_drift:
            n_ks_drift += 1
        if psi >= float(psi_moderate):
            n_psi_moderate += 1
        if psi >= float(psi_high):
            n_psi_high += 1

        per_col[c] = {
            "n_train": int(len(x)),
            "n_test": int(len(y)),
            "ks_stat": stat,
            "p_value": pval,
            "ks_drift": ks_drift,
            "ks_alpha": float(ks_alpha),
            "ks_min": float(ks_min),
            "psi": psi,
            "psi_level": psi_level,
        }

    # Rank columns by PSI then KS statistic (most practically drifting first)
    ranked = sorted(
        per_col.items(),
        key=lambda kv: (kv[1]["psi"], kv[1]["ks_stat"]),
        reverse=True,
    )
    top = [
        {"col": k, **v}
        for k, v in ranked[: int(max(1, top_k))]
    ]

    return {
        "ks_alpha": float(ks_alpha),
        "ks_min": float(ks_min),
        "psi_bins": int(psi_bins),
        "psi_moderate": float(psi_moderate),
        "psi_high": float(psi_high),
        "min_n": int(min_n),
        "n_cols_checked": int(len(per_col)),
        "n_ks_drifting": int(n_ks_drift),
        "n_psi_moderate_or_more": int(n_psi_moderate),
        "n_psi_high": int(n_psi_high),
        "top_drifts": top,
        "per_col": per_col,
    }


def drift_report_proba(
    train_proba: np.ndarray,
    test_proba: np.ndarray,
    *,
    bins: int = 10,
    min_n: int = 200,
) -> dict[str, Any]:
    """
    PSI on predicted probabilities as a simple model-drift signal.
    """
    e = np.asarray(train_proba, dtype=float)
    a = np.asarray(test_proba, dtype=float)
    e = e[np.isfinite(e)]
    a = a[np.isfinite(a)]

    if len(e) < min_n or len(a) < min_n:
        return {"psi": 0.0, "psi_level": "low"}

    psi = float(_psi(e, a, bins=int(bins)))
    return {"psi": psi, "psi_level": _psi_label(psi)}
