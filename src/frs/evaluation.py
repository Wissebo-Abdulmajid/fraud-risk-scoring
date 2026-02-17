from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass(frozen=True)
class Metrics:
    pr_auc: float
    roc_auc: float
    brier: float
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, y_pred: np.ndarray, threshold: float) -> Metrics:
    pr_auc = float(average_precision_score(y_true, proba))
    roc_auc = float(roc_auc_score(y_true, proba))
    brier = float(brier_score_loss(y_true, proba))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return Metrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        brier=brier,
        threshold=float(threshold),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
