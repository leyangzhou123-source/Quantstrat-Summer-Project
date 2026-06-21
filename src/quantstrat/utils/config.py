from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib


def load_config(path: str | Path = "configs/default.toml") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        return tomllib.load(handle)
