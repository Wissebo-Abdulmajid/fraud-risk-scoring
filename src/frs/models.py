from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    model_name: str


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []

    if numeric:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric))

    if categorical:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    # Keep transformed output as pandas DataFrame when supported
    try:
        pre.set_output(transform="pandas")
    except Exception:
        pass

    return pre


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    """
    Computes scale_pos_weight from TRAIN ONLY, but caps it to avoid over-amplifying the rare class.
    """
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 1:
        raise ValueError("Training split has 0 positive samples (fraud=1). Adjust split/windows.")

    spw = float(n_neg / n_pos)

    # Cap to keep learning + calibration stable on extreme imbalance datasets
    return float(min(spw, 50.0))


def train_logreg(
    X,
    y,
    preprocessor: ColumnTransformer,
    seed: int,
) -> ModelBundle:
    """
    Logistic Regression baseline.

    IMPORTANT:
    - We do NOT use class_weight='balanced' by default because it can distort probability calibration.
    - If you want weighting later, do it deliberately and re-check calibration.
    """
    clf = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=seed,
        solver="lbfgs",
    )

    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])

    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

    pipe.fit(X, y)
    return ModelBundle(pipeline=pipe, model_name="logreg")


def train_lightgbm(
    X,
    y,
    preprocessor: ColumnTransformer,
    params: dict,
    seed: int,
) -> ModelBundle:
    """
    LightGBM main model.

    We compute scale_pos_weight from the current training labels (per split/fold),
    and override any config-provided value to prevent “pumped” imbalance settings.
    """
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is not installed or failed to import.")

    spw = _compute_scale_pos_weight(y)

    # Copy + override any potentially harmful imbalance knobs from config
    safe_params = dict(params) if params else {}
    safe_params.pop("class_weight", None)
    safe_params.pop("scale_pos_weight", None)

    clf = LGBMClassifier(
        random_state=seed,
        scale_pos_weight=spw,
        **safe_params,
    )

    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])

    try:
        pipe.set_output(transform="pandas")
    except Exception:
        pass

    pipe.fit(X, y)
    return ModelBundle(pipeline=pipe, model_name="lightgbm")
