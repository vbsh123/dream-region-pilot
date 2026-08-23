from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dependencies import DependencySnapshot, graph_summary
from .regions import Region


class ExampleDiagnostics:
    def __init__(self, output_dir: Path, snapshot_interval: int):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.iterations: list[dict[str, Any]] = []
        self.graph_snapshots: list[dict[str, Any]] = []

    def should_save_dependency(self, iteration: int) -> bool:
        return iteration == 0 or iteration % self.snapshot_interval == 0

    def record_dependency(
        self, iteration: int, snapshot: DependencySnapshot
    ) -> None:
        if not self.should_save_dependency(iteration):
            return
        matrices = {
            "raw_token_dependency": snapshot.raw_token.detach().float().cpu().numpy(),
            "normalized_token_dependency": snapshot.normalized_token.detach()
            .float()
            .cpu()
            .numpy(),
        }
        matrices.update(
            {
                f"region_{name}": matrix.detach().float().cpu().numpy()
                for name, matrix in snapshot.region_matrices.items()
            }
        )
        np.savez_compressed(
            self.output_dir / f"dependencies_step_{iteration:04d}.npz", **matrices
        )

        graph = {
            "iteration": iteration,
            "edges": snapshot.edges,
            **graph_summary(snapshot),
        }
        self.graph_snapshots.append(graph)
        (self.output_dir / f"graph_step_{iteration:04d}.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8"
        )

        for name, matrix in snapshot.region_matrices.items():
            path = self.output_dir / f"region_{name}_step_{iteration:04d}.csv"
            np.savetxt(path, matrix.detach().float().cpu().numpy(), delimiter=",")
        self._plot_graph(iteration, snapshot)
        self._plot_heatmap(iteration, snapshot)

    def record_iteration(
        self,
        *,
        iteration: int,
        regions: list[Region],
        scheduled: list[int],
        advanced: list[int],
        committed: dict[int, list[int]],
        edges: list[dict[str, float | int]],
    ) -> None:
        self.iterations.append(
            {
                "iteration": iteration,
                "scheduled_regions": scheduled,
                "advanced_regions": advanced,
                "committed_response_positions": committed,
                "committed_count": sum(len(values) for values in committed.values()),
                "regions": [
                    {
                        "region": region.index,
                        "clock": region.clock,
                        "schedule_step": region.schedule_step,
                        "parents": sorted(region.parents),
                        "remaining_mask_indices": list(region.remaining_mask_indices),
                        "mask_ratio": len(region.remaining_mask_indices)
                        / len(region.token_indices),
                    }
                    for region in regions
                ],
                "edges": edges,
            }
        )

    def finalize(self) -> None:
        with (self.output_dir / "iterations.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in self.iterations:
                handle.write(json.dumps(row) + "\n")
        (self.output_dir / "graph_timeline.json").write_text(
            json.dumps(self.graph_snapshots, indent=2) + "\n", encoding="utf-8"
        )
        self._write_clock_csv()
        self._write_graph_metrics_csv()
        self._plot_clocks_and_masks()
        self._plot_graph_metrics()

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
                ),
            )
            writer.writeheader()
            for step in self.iterations:
                advanced = set(step["advanced_regions"])
                scheduled = set(step["scheduled_regions"])
                for region in step["regions"]:
                    writer.writerow(
                        {
                            "iteration": step["iteration"],
                            "region": region["region"],
                            "clock": region["clock"],
                            "schedule_step": region["schedule_step"],
                            "mask_ratio": region["mask_ratio"],
                            "scheduled": int(region["region"] in scheduled),
                            "advanced": int(region["region"] in advanced),
                        }
                    )

    def _write_graph_metrics_csv(self) -> None:
        if not self.graph_snapshots:
            return
        region_ids = sorted(
            {
                int(region)
                for snapshot in self.graph_snapshots
                for region in snapshot["parent_counts"]
            }
        )
        fieldnames = [
            "iteration",
            "region_count",
            "active_region_count",
            "edge_count",
            "possible_edge_count",
            "active_possible_edge_count",
            "edge_density",
            "edge_percentage",
            "active_edge_density",
            "active_edge_percentage",
            "average_region_degree",
            "average_active_region_degree",
            "component_count",
            "active_component_count",
            "average_parent_count",
            "max_parent_count",
            "root_count",
            "is_complete_graph",
        ] + [f"R{region}_parent_count" for region in region_ids]
        with (self.output_dir / "graph_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in self.graph_snapshots:
                row = {key: snapshot[key] for key in fieldnames if key in snapshot}
                parent_counts = snapshot["parent_counts"]
                for region in region_ids:
                    row[f"R{region}_parent_count"] = parent_counts.get(
                        region, parent_counts.get(str(region), 0)
                    )
                writer.writerow(row)

    def _plot_graph(self, iteration: int, snapshot: DependencySnapshot) -> None:
        region_count = snapshot.selected_region.shape[0]
        figure, axis = plt.subplots(figsize=(max(6, region_count), 2.8))
        x_positions = np.arange(region_count)
        for edge in snapshot.edges:
            left, right = int(edge["left"]), int(edge["right"])
            score = float(edge["score"])
            height = 0.25 + 0.12 * (right - left)
            axis.plot(
                [left, (left + right) / 2, right],
                [0, height, 0],
                color="tab:blue",
                alpha=0.6,
            )
            axis.text((left + right) / 2, height, f"{score:.3g}", fontsize=7)
        axis.scatter(x_positions, np.zeros(region_count), s=280, zorder=3)
        for index in range(region_count):
            axis.text(index, 0, f"R{index}", ha="center", va="center", color="white")
        axis.set_title(f"Region dependency graph, iteration {iteration}")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(self.output_dir / f"graph_step_{iteration:04d}.png", dpi=140)
        plt.close(figure)

    def _plot_heatmap(self, iteration: int, snapshot: DependencySnapshot) -> None:
        values = snapshot.selected_region.detach().float().cpu().numpy()
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(values, cmap="magma")
        axis.set_title(f"Selected region dependency, step {iteration}")
        axis.set_xlabel("region")
        axis.set_ylabel("region")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(
            self.output_dir / f"region_dependency_step_{iteration:04d}.png", dpi=140
        )
        plt.close(figure)

    def _plot_clocks_and_masks(self) -> None:
        if not self.iterations:
            return
        region_count = len(self.iterations[0]["regions"])
        figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for region_index in range(region_count):
            x = [step["iteration"] for step in self.iterations]
            clocks = [step["regions"][region_index]["clock"] for step in self.iterations]
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

    def _plot_graph_metrics(self) -> None:
        if not self.graph_snapshots:
            return
        iterations = [item["iteration"] for item in self.graph_snapshots]
        figure, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(
            iterations,
            [item["edge_density"] for item in self.graph_snapshots],
            marker="o",
            label="all fixed regions",
        )
        axes[0].plot(
            iterations,
            [item["active_edge_density"] for item in self.graph_snapshots],
            marker="x",
            label="active regions",
        )
        axes[0].set_ylabel("edge density")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].legend()
        axes[1].plot(
            iterations,
            [item["average_region_degree"] for item in self.graph_snapshots],
            marker="o",
            label="all fixed regions",
        )
        axes[1].plot(
            iterations,
            [item["average_active_region_degree"] for item in self.graph_snapshots],
            marker="x",
            label="active regions",
        )
        axes[1].set_ylabel("average degree")
        axes[1].legend()
        axes[2].plot(
            iterations,
            [item["component_count"] for item in self.graph_snapshots],
            marker="o",
            label="components",
        )
        axes[2].plot(
            iterations,
            [item["max_parent_count"] for item in self.graph_snapshots],
            marker="x",
            label="max parents",
        )
        axes[2].set_ylabel("count")
        axes[2].set_xlabel("global iteration / NFE")
        axes[2].legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "graph_metrics_over_time.png", dpi=140)
        plt.close(figure)
