from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "fraud-risk-calibrated"
    seed: int = 42


class DataConfig(BaseModel):
    path: str
    target: str
    time_col: str | None = None
    id_col: str | None = None


class SplitConfig(BaseModel):
    strategy: Literal["stratified", "time"] = "stratified"
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    val_size: float = Field(0.2, ge=0.05, le=0.5)


class BacktestConfig(BaseModel):
    train_start_frac: float = Field(0.50, ge=0.0, le=1.0)
    step_frac: float = Field(0.05, ge=0.0, le=1.0)
    val_frac: float = Field(0.10, ge=0.0, le=1.0)
    test_frac: float = Field(0.10, ge=0.0, le=1.0)
    threshold_ema_alpha: float = Field(0.70, ge=0.0, le=1.0)

class FeaturesConfig(BaseModel):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)  # columns never used as features


class ModelConfig(BaseModel):
    name: Literal["lightgbm", "logreg"] = "lightgbm"
    params: dict[str, Any] = Field(default_factory=dict)


class CalibrationConfig(BaseModel):
    enabled: bool = True
    method: Literal["sigmoid", "isotonic"] = "sigmoid"


class PolicyConfig(BaseModel):
    mode: Literal["cost", "top_k"] = "cost"
    cost_fp: float = 1.0
    cost_fn: float = 10.0
    top_k: float = Field(0.01, gt=0.0, lt=1.0)
    min_alert_rate: float | None = None
    max_alert_rate: float | None = None
    threshold_grid_size: int = Field(401, ge=101, le=5001)
    prior_shift_enabled: bool = True
    prior_shift_source: Literal["val", "train", "train_val"] = "val"

class Config(BaseModel):
    project: ProjectConfig
    data: DataConfig
    split: SplitConfig
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    features: FeaturesConfig
    model: ModelConfig
    calibration: CalibrationConfig
    policy: PolicyConfig


def load_config(path: str | Path) -> Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {p} did not parse into a dict.")
    return Config(**raw)
