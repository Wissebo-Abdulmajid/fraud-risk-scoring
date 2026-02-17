from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    # src/frs/paths.py -> src/frs -> src -> repo root
    return Path(__file__).resolve().parents[2]
