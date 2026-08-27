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
    task = str(config["data"].get("task", "gsm8k"))
    if task not in {"gsm8k", "asdiv", "math500", "humaneval"}:
        raise ValueError("data.task must be gsm8k, asdiv, math500, or humaneval")
    region_size = int(generation["region_size"])
    if region_size not in {16, 20, 25, 32, 64}:
        raise ValueError("Pilot region_size must be 16, 20, 25, 32, or 64")
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
        deferral_threshold = float(
            probe.get("deferral_confidence_threshold", 0.5)
        )
        if not 0.0 <= deferral_threshold <= 1.0:
            raise ValueError(
                "probe.deferral_confidence_threshold must be in [0, 1]"
            )
        if int(probe.get("max_region_deferrals", 4)) < 0:
            raise ValueError("probe.max_region_deferrals must be non-negative")
        if int(probe.get("max_global_deferral_iterations", 4)) <= 0:
            raise ValueError(
                "probe.max_global_deferral_iterations must be positive"
            )
        deferral_cutoff = probe.get("deferral_until_revealed_tokens")
        if deferral_cutoff is not None and int(deferral_cutoff) < 0:
            raise ValueError(
                "probe.deferral_until_revealed_tokens must be non-negative"
            )
        mean_field = probe.get("mean_field", {})
        if mean_field and int(mean_field.get("topk", 1)) <= 0:
            raise ValueError("probe.mean_field.topk must be positive")
        flowblock_proxy = probe.get("flowblock_proxy", {})
        if flowblock_proxy:
            if int(flowblock_proxy.get("max_active_regions", 2)) <= 0:
                raise ValueError(
                    "probe.flowblock_proxy.max_active_regions must be positive"
                )
            for key in (
                "spawn_readiness",
                "readiness_confidence_threshold",
            ):
                value = float(flowblock_proxy.get(key, 0.0))
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"probe.flowblock_proxy.{key} must be in [0, 1]"
                    )
    mean_field_baseline = config.get("mean_field_baseline", {})
    if mean_field_baseline:
        if int(mean_field_baseline.get("block_size", 32)) <= 0:
            raise ValueError("mean_field_baseline.block_size must be positive")
        threshold = float(mean_field_baseline.get("threshold", 0.9))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("mean_field_baseline.threshold must be in [0, 1]")
        if int(mean_field_baseline.get("iterations", 2)) <= 0:
            raise ValueError("mean_field_baseline.iterations must be positive")
        if int(mean_field_baseline.get("pair_chunk_size", 16)) <= 0:
            raise ValueError(
                "mean_field_baseline.pair_chunk_size must be positive"
            )
    return config
