from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    for section in ("model", "data", "generation", "dependency", "experiment"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing mapping section {section!r}")
    generation = config["generation"]
    region_size = int(generation["region_size"])
    if region_size not in {16, 32, 64}:
        raise ValueError("Initial pilot region_size must be 16, 32, or 64")
    if int(generation["steps"]) <= 0 or int(generation["max_new_tokens"]) <= 0:
        raise ValueError("steps and max_new_tokens must be positive")
    return config
