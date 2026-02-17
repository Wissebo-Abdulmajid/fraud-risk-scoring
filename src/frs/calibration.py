from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CalMethod = Literal["sigmoid", "isotonic"]


@dataclass(frozen=True)
class CalibratedBundle:
    method: CalMethod
    calibrator: object  # IsotonicRegression or LogisticRegression


def _to_1d_float(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=float).reshape(-1)
    # Replace NaN/inf safely before clipping
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    return p


def _clip_proba(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def fit_calibrator(y_true: np.ndarray, proba: np.ndarray, method: CalMethod) -> CalibratedBundle:
    """
    Fit a probability calibrator on (y_true, proba) where `proba` are base model probabilities.
    """
    y = np.asarray(y_true).astype(int).reshape(-1)
    p = _clip_proba(_to_1d_float(proba))

    if y.shape[0] != p.shape[0]:
        raise ValueError(f"y_true and proba must have same length. Got {len(y)} vs {len(p)}.")

    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return CalibratedBundle(method=method, calibrator=iso)

    if method == "sigmoid":
        # Platt scaling: logistic regression on logit(proba)
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000)
        lr.fit(logit, y)
        return CalibratedBundle(method=method, calibrator=lr)

    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(bundle: CalibratedBundle, proba: np.ndarray) -> np.ndarray:
    """
    Apply a fitted calibrator to base probabilities.
    """
    p = _clip_proba(_to_1d_float(proba))

    if bundle.method == "isotonic":
        out = np.asarray(bundle.calibrator.predict(p), dtype=float)
        return np.clip(out, 0.0, 1.0)

    if bundle.method == "sigmoid":
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        out = bundle.calibrator.predict_proba(logit)[:, 1].astype(float)
        return np.clip(out, 0.0, 1.0)

    raise ValueError(f"Unknown calibration method: {bundle.method}")
def adjust_proba_for_prior_shift(
    proba: np.ndarray,
    pi_train: float,
    pi_target: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Prior probability shift correction (label shift) via odds adjustment.

    odds' = odds * [(pi_target/(1-pi_target)) / (pi_train/(1-pi_train))]

    This adapts probabilities when base rates drift over time.
    """
    p = _clip_proba(np.asarray(proba, dtype=float), eps=eps)

    pi_train = float(np.clip(pi_train, eps, 1.0 - eps))
    pi_target = float(np.clip(pi_target, eps, 1.0 - eps))

    odds = p / (1.0 - p)
    mult = (pi_target / (1.0 - pi_target)) / (pi_train / (1.0 - pi_train))
    odds_adj = odds * mult
    p_adj = odds_adj / (1.0 + odds_adj)

    return _clip_proba(p_adj, eps=eps)


def predict_proba(model, X) -> np.ndarray:
    """
    Standardized probability extraction for sklearn-like estimators.
    """
    p = model.predict_proba(X)[:, 1]
    return _to_1d_float(p)
