from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from frs.calibration import apply_calibrator, predict_proba


class RiskModel:
    """
    Production-safe model wrapper.
    Loads a frozen bundle.joblib and exposes a predict() method.
    """

    def __init__(self, bundle_path: Path):
        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        bundle = joblib.load(bundle_path)

        self.pipeline = bundle["pipeline"]
        self.calibrator = bundle.get("calibrator")
        self.policy = bundle["policy"]
        self.feature_spec = bundle["feature_spec"]
        self.model_name = bundle.get("model_name", "unknown")

        self.numeric = self.feature_spec["numeric"]
        self.categorical = self.feature_spec["categorical"]
        self.required_cols = self.numeric + self.categorical
        self.threshold = float(self.policy["threshold"])

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        missing = [c for c in self.required_cols if c not in data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        df = pd.DataFrame([data])

        proba = predict_proba(self.pipeline, df)

        if self.calibrator is not None:
            proba = apply_calibrator(self.calibrator, proba)

        proba = float(np.asarray(proba)[0])
        alert = int(proba >= self.threshold)

        return {
            "risk_proba": proba,
            "alert": alert,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }
