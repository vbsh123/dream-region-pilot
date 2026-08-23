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
    probe = config.get("probe", {})
    if probe:
        if int(probe.get("max_active_regions", 1)) <= 0:
            raise ValueError("probe.max_active_regions must be positive")
        if int(probe.get("max_progress_gap", 0)) < 0:
            raise ValueError("probe.max_progress_gap must be non-negative")
        if int(probe.get("edge_persistence", 1)) <= 0:
            raise ValueError("probe.edge_persistence must be positive")
        for key in ("spawn_readiness", "readiness_confidence_threshold"):
            value = float(probe.get(key, 0.0))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"probe.{key} must be in [0, 1]")
        mean_field = probe.get("mean_field", {})
        if mean_field and int(mean_field.get("topk", 1)) <= 0:
            raise ValueError("probe.mean_field.topk must be positive")
    return config
