from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def infer_features(df: pd.DataFrame, target: str) -> FeatureSpec:
    """
    Infer numeric vs categorical features from dataframe dtypes,
    excluding the target column.
    """
    cols = [c for c in df.columns if c != target]

    numeric: list[str] = []
    categorical: list[str] = []

    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)

    return FeatureSpec(numeric=numeric, categorical=categorical)


def exclude_non_features(
    numeric: list[str],
    categorical: list[str],
    *,
    target: str,
    time_col: str | None = None,
    id_col: str | None = None,
    extra_exclude: Iterable[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Centralized exclusion logic.

    Removes columns that must NOT be treated as model inputs:
      - target (label)
      - time_col (used for sorting/splitting, not learning)
      - id_col (identifier leakage)
      - extra_exclude (any user-provided excludes from config)

    Returns:
      (filtered_numeric, filtered_categorical, excluded_sorted_list)
    """
    excluded: set[str] = {str(target)}

    if time_col and str(time_col).strip():
        excluded.add(str(time_col))

    if id_col and str(id_col).strip():
        excluded.add(str(id_col))

    if extra_exclude:
        excluded.update(str(x) for x in extra_exclude if str(x).strip())

    numeric_out = [c for c in numeric if c not in excluded]
    categorical_out = [c for c in categorical if c not in excluded]

    return numeric_out, categorical_out, sorted(excluded)
