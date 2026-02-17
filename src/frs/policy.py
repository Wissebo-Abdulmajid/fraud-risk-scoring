from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DecisionPolicy:
    mode: str
    threshold: float
    cost_fp: float | None = None
    cost_fn: float | None = None
    top_k: float | None = None


def _expected_cost(y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(fp * cost_fp + fn * cost_fn)


def _expected_cost_rate(y_true: np.ndarray, y_pred: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    # Normalize by N so window size differences don't destabilize threshold selection
    n = int(len(y_true))
    if n <= 0:
        return float("inf")
    return _expected_cost(y_true, y_pred, cost_fp, cost_fn) / float(n)


def choose_threshold_cost_sensitive(
    y_true: np.ndarray,
    proba: np.ndarray,
    cost_fp: float,
    cost_fn: float,
    *,
    grid_size: int = 401,
    min_alert_rate: float | None = None,
    max_alert_rate: float | None = None,
    prev_threshold: float | None = None,
    tie_breaker_eps: float = 1e-12,
) -> float:
    """
    Stable cost-sensitive threshold selection.
    - Uses a fixed threshold grid (not unique(proba)) to reduce noise on small/rare labels.
    - Minimizes *cost rate* (cost / N) for stability across folds.
    - Optional alert-rate constraints for operational control.
    - Optional prev_threshold tie-breaker to reduce jitter.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(proba).astype(float)
    if len(p) == 0:
        return 0.5

    thresholds = np.linspace(0.0, 1.0, int(grid_size))
    best_t = 0.5
    best_obj = float("inf")

    # Pre-check constraints
    def _ok_alert(ar: float) -> bool:
        if min_alert_rate is not None and ar < float(min_alert_rate):
            return False
        if max_alert_rate is not None and ar > float(max_alert_rate):
            return False
        return True

    for t in thresholds:
        y_pred = (p >= t).astype(int)
        ar = float(y_pred.mean()) if len(y_pred) else 0.0
        if not _ok_alert(ar):
            continue

        obj = _expected_cost_rate(y_true, y_pred, cost_fp=cost_fp, cost_fn=cost_fn)

        if obj + tie_breaker_eps < best_obj:
            best_obj = obj
            best_t = float(t)
        elif abs(obj - best_obj) <= tie_breaker_eps and prev_threshold is not None:
            # Tie-break toward previous threshold to stabilize across folds
            if abs(float(t) - float(prev_threshold)) < abs(best_t - float(prev_threshold)):
                best_t = float(t)

    # If constraints made everything invalid, fall back to unconstrained optimum
    if not np.isfinite(best_obj):
        best_t = choose_threshold_cost_sensitive(
            y_true=y_true,
            proba=p,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            grid_size=grid_size,
            min_alert_rate=None,
            max_alert_rate=None,
            prev_threshold=prev_threshold,
        )

    return float(best_t)


def choose_threshold_top_k(proba: np.ndarray, top_k: float) -> float:
    if not (0.0 < top_k < 1.0):
        raise ValueError("top_k must be in (0, 1).")
    return float(np.quantile(np.asarray(proba, dtype=float), 1.0 - float(top_k)))


def apply_policy(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(proba, dtype=float) >= float(threshold)).astype(int)
