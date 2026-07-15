from __future__ import annotations

from pathlib import Path
from typing import Any
import tomllib

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix in {".yaml", ".yml"}:
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    with config_path.open("rb") as handle:
        return tomllib.load(handle)
