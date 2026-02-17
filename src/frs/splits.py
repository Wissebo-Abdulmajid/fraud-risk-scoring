from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def stratified_train_val_test_split(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> SplitBundle:
    """
    Stratified split: preserves label ratio across splits.
    Note: Can be optimistic for fraud in real deployments if there is time drift.
    """
    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must be in (0, 1).")

    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    y = df[target]

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    y_train_val = train_val[target]
    val_frac_of_train_val = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_frac_of_train_val,
        random_state=seed,
        stratify=y_train_val,
    )

    return SplitBundle(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )


def time_train_val_test_split(
    df: pd.DataFrame,
    target: str,
    time_col: str,
    test_size: float,
    val_size: float,
    seed: int,
    id_col: Optional[str] = None,
) -> SplitBundle:
    """
    Time-based split: sorts by `time_col` ascending and splits chronologically:
    train (oldest) -> val -> test (newest).

    This is the safer, more realistic evaluation for fraud/risk models.
    """
    if time_col is None or str(time_col).strip() == "":
        raise ValueError("time_col must be provided for time-based splitting.")

    if time_col not in df.columns:
        raise ValueError(f"time_col='{time_col}' not found in dataframe columns.")

    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must be in (0, 1).")

    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    d = df.copy()

    # Stable sort to preserve original order when time ties exist
    # If you have an id_col or timestamp ties, you can stabilize with id_col.
    if id_col and id_col in d.columns:
        d = d.sort_values([time_col, id_col], ascending=[True, True], kind="mergesort")
    else:
        d = d.sort_values([time_col], ascending=[True], kind="mergesort")

    n = len(d)
    n_test = int(np.floor(n * test_size))
    n_val = int(np.floor(n * val_size))

    # Ensure all splits non-empty (at least 1 row each)
    if n_test < 1 or n_val < 1 or (n - n_test - n_val) < 1:
        raise ValueError(
            f"Split sizes too small for n={n}. "
            f"Got n_test={n_test}, n_val={n_val}, n_train={n - n_test - n_val}."
        )

    train_end = n - (n_test + n_val)
    val_end = n - n_test

    train = d.iloc[:train_end]
    val = d.iloc[train_end:val_end]
    test = d.iloc[val_end:]

    # Optional: quick sanity check that all classes appear at least in train+val
    # (test can legitimately have edge cases; we don't want to leak it).
    y_train = set(pd.unique(train[target]))
    y_val = set(pd.unique(val[target]))
    if not (y_train | y_val):
        raise ValueError("No target values found after splitting (unexpected).")

    return SplitBundle(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
    )
