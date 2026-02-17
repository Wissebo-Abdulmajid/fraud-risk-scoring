from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import yaml


def make_run_dir(base: str = "runs") -> Path:
    from datetime import datetime

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(base) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_yaml(path: Path, obj: Any) -> None:
    path.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def save_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_bundle(path: Path, bundle: Any) -> None:
    joblib.dump(bundle, path)
