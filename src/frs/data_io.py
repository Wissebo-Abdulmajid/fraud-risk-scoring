from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DataMeta:
    n_rows: int
    n_cols: int
    columns: list[str]


def load_tabular(path: str | Path) -> tuple[pd.DataFrame, DataMeta]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if p.suffix.lower() in {".csv"}:
        df = pd.read_csv(p)
    elif p.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}. Use CSV or Parquet.")

    meta = DataMeta(n_rows=int(df.shape[0]), n_cols=int(df.shape[1]), columns=list(df.columns))
    return df, meta


def validate_min_schema(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    if df[target].isna().all():
        raise ValueError(f"Target column '{target}' is all NaN.")

    # common: labels stored as strings; we tolerate for now, but warn later in train.
