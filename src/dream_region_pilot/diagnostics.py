from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .regions import Region


class ExampleDiagnostics:
    """Per-iteration regional scheduling and commitment diagnostics."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.iterations: list[dict[str, Any]] = []

    def record_iteration(
        self,
        *,
        iteration: int,
        regions: list[Region],
        scheduled: list[int],
        advanced: list[int],
        committed: dict[int, list[int]],
        commitment_details: list[dict[str, Any]],
        admitted_region_count: int,
        newly_admitted: list[int],
        readiness_by_region: dict[int, float],
        control_state: dict[str, Any],
    ) -> None:
        self.iterations.append(
            {
                "iteration": iteration,
                "scheduled_regions": scheduled,
                "advanced_regions": advanced,
                "committed_response_positions": committed,
                "commitment_details": commitment_details,
                "committed_count": sum(len(values) for values in committed.values()),
                "admitted_region_count": admitted_region_count,
                "newly_admitted_regions": newly_admitted,
                "readiness_by_region": readiness_by_region,
                "control": control_state,
                "regions": [
                    {
                        "region": region.index,
                        "clock": region.clock,
                        "schedule_step": region.schedule_step,
                        "remaining_mask_indices": list(
                            region.remaining_mask_indices
                        ),
                        "mask_ratio": len(region.remaining_mask_indices)
                        / len(region.token_indices),
                    }
                    for region in regions
                ],
            }
        )

    def finalize(self) -> None:
        with (self.output_dir / "iterations.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in self.iterations:
                handle.write(json.dumps(row) + "\n")
        with (self.output_dir / "commitments.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for step in self.iterations:
                for detail in step["commitment_details"]:
                    handle.write(
                        json.dumps({"iteration": step["iteration"], **detail})
                        + "\n"
                    )
        self._write_clock_csv()
        self._plot_clocks_and_masks()

    def _write_clock_csv(self) -> None:
        path = self.output_dir / "region_state.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "iteration",
                    "region",
                    "clock",
                    "schedule_step",
                    "mask_ratio",
                    "scheduled",
                    "advanced",
                    "admitted",
                    "readiness",
                    "progress_tokens",
                    "blocked",
                    "urgent",
                ),
            )
            writer.writeheader()
            for step in self.iterations:
                advanced = set(step["advanced_regions"])
                scheduled = set(step["scheduled_regions"])
                blocked = set(step["control"]["blocked_regions"])
                urgent = set(step["control"]["urgent_regions"])
                for region in step["regions"]:
                    index = region["region"]
                    writer.writerow(
                        {
                            "iteration": step["iteration"],
                            "region": index,
                            "clock": region["clock"],
                            "schedule_step": region["schedule_step"],
                            "mask_ratio": region["mask_ratio"],
                            "scheduled": int(index in scheduled),
                            "advanced": int(index in advanced),
                            "admitted": int(
                                index < step["admitted_region_count"]
                            ),
                            "readiness": step["readiness_by_region"].get(
                                index, ""
                            ),
                            "progress_tokens": step["control"][
                                "progress_tokens_after"
                            ].get(index, 0),
                            "blocked": int(index in blocked),
                            "urgent": int(index in urgent),
                        }
                    )

    def _plot_clocks_and_masks(self) -> None:
        if not self.iterations:
            return
        region_count = len(self.iterations[0]["regions"])
        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for region_index in range(region_count):
            x = [step["iteration"] for step in self.iterations]
            clocks = [
                step["regions"][region_index]["clock"]
                for step in self.iterations
            ]
            masks = [
                step["regions"][region_index]["mask_ratio"]
                for step in self.iterations
            ]
            axes[0].plot(x, clocks, label=f"R{region_index}")
            axes[1].plot(x, masks, label=f"R{region_index}")
        axes[0].set_ylabel("local clock")
        axes[1].set_ylabel("mask ratio")
        axes[1].set_xlabel("global iteration / NFE")
        axes[0].legend(ncol=min(4, region_count), fontsize=8)
        figure.tight_layout()
        figure.savefig(self.output_dir / "clocks_and_masks.png", dpi=140)
        plt.close(figure)
