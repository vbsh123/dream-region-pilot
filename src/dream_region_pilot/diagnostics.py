from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dependencies import DependencySnapshot, SignalGraph, graph_summary
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
        matrices.update(
            {
                f"{name}_token_dependency": matrix.detach().float().cpu().numpy()
                for name, matrix in snapshot.signal_token_matrices.items()
            }
        )
        matrices.update(
            {
                f"region_{name}": matrix.detach().float().cpu().numpy()
                for name, matrix in snapshot.signal_region_matrices.items()
            }
        )
        np.savez_compressed(
            self.output_dir / f"dependencies_step_{iteration:04d}.npz", **matrices
        )

        signals: dict[str, dict[str, Any]] = {}
        previous_signals = (
            self.graph_snapshots[-1].get("signals", {})
            if self.graph_snapshots
            else {}
        )
        for name, signal_graph in snapshot.signal_graphs.items():
            summary = graph_summary(snapshot, signal_graph)
            current_pairs = {
                (int(edge["left"]), int(edge["right"]))
                for edge in signal_graph.edges
            }
            previous_pairs = {
                (int(edge["left"]), int(edge["right"]))
                for edge in previous_signals.get(name, {}).get("edges", [])
            }
            union = current_pairs | previous_pairs
            signals[name] = {
                "threshold": signal_graph.threshold,
                "edges": signal_graph.edges,
                **summary,
                "edges_added_since_previous": [
                    {"left": left, "right": right}
                    for left, right in sorted(current_pairs - previous_pairs)
                ],
                "edges_removed_since_previous": [
                    {"left": left, "right": right}
                    for left, right in sorted(previous_pairs - current_pairs)
                ],
                "edge_jaccard_with_previous": (
                    len(current_pairs & previous_pairs) / len(union)
                    if union
                    else 1.0
                ),
            }
        graph = {
            "iteration": iteration,
            "edges": snapshot.edges,
            **graph_summary(snapshot),
            "signals": signals,
            "metadata": snapshot.metadata,
        }
        self.graph_snapshots.append(graph)
        (self.output_dir / f"graph_step_{iteration:04d}.json").write_text(
            json.dumps(graph, indent=2) + "\n", encoding="utf-8"
        )

        for name, matrix in snapshot.region_matrices.items():
            path = self.output_dir / f"region_{name}_step_{iteration:04d}.csv"
            np.savetxt(path, matrix.detach().float().cpu().numpy(), delimiter=",")
        for name, matrix in snapshot.signal_region_matrices.items():
            path = self.output_dir / f"region_{name}_step_{iteration:04d}.csv"
            np.savetxt(path, matrix.detach().float().cpu().numpy(), delimiter=",")

        plotted_graphs = {
            name: value
            for name, value in snapshot.signal_graphs.items()
            if name == "dapd"
            or name.endswith("_t0.90")
            or name.startswith("combined_")
        }
        for name, signal_graph in plotted_graphs.items():
            self._plot_graph(iteration, name, signal_graph)
        self._plot_heatmap(iteration, "dapd", snapshot.selected_region)
        mean_field_keys = [
            name
            for name in snapshot.signal_region_matrices
            if name.startswith("mean_field_")
            and not name.startswith("mean_field_raw_")
        ]
        preferred_key = next(
            (name for name in mean_field_keys if name == "mean_field_mean"),
            mean_field_keys[0] if mean_field_keys else None,
        )
        if preferred_key is not None:
            self._plot_heatmap(
                iteration,
                "mean_field",
                snapshot.signal_region_matrices[preferred_key],
            )

    def record_iteration(
        self,
        *,
        iteration: int,
        regions: list[Region],
        scheduled: list[int],
        advanced: list[int],
        committed: dict[int, list[int]],
        commitment_details: list[dict[str, Any]],
        edges: list[dict[str, float | int]],
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
        with (self.output_dir / "commitments.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for step in self.iterations:
                for detail in step["commitment_details"]:
                    handle.write(
                        json.dumps(
                            {"iteration": step["iteration"], **detail}
                        )
                        + "\n"
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
                    writer.writerow(
                        {
                            "iteration": step["iteration"],
                            "region": region["region"],
                            "clock": region["clock"],
                            "schedule_step": region["schedule_step"],
                            "mask_ratio": region["mask_ratio"],
                            "scheduled": int(region["region"] in scheduled),
                            "advanced": int(region["region"] in advanced),
                            "admitted": int(
                                region["region"] < step["admitted_region_count"]
                            ),
                            "readiness": step["readiness_by_region"].get(
                                region["region"], ""
                            ),
                            "progress_tokens": step["control"][
                                "progress_tokens"
                            ].get(region["region"], 0),
                            "blocked": int(region["region"] in blocked),
                            "urgent": int(region["region"] in urgent),
                        }
                    )

    def _write_graph_metrics_csv(self) -> None:
        if not self.graph_snapshots:
            return
        fieldnames = [
            "iteration",
            "signal",
            "threshold",
            "region_count",
            "active_region_count",
            "edge_count",
            "active_edge_count",
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
            "active_root_count",
            "adjacent_edge_count",
            "nonadjacent_edge_count",
            "is_complete_graph",
            "edges_added_count",
            "edges_removed_count",
            "edge_jaccard_with_previous",
            "parent_counts_json",
        ]
        with (self.output_dir / "graph_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for snapshot in self.graph_snapshots:
                for signal, values in snapshot["signals"].items():
                    row = {
                        key: values[key]
                        for key in fieldnames
                        if key in values
                    }
                    row.update(
                        {
                            "iteration": snapshot["iteration"],
                            "signal": signal,
                            "edges_added_count": len(
                                values["edges_added_since_previous"]
                            ),
                            "edges_removed_count": len(
                                values["edges_removed_since_previous"]
                            ),
                            "parent_counts_json": json.dumps(
                                values["parent_counts"], sort_keys=True
                            ),
                        }
                    )
                    writer.writerow(row)

    def _plot_graph(
        self, iteration: int, name: str, graph: SignalGraph
    ) -> None:
        region_count = graph.matrix.shape[0]
        figure, axis = plt.subplots(figsize=(max(6, region_count), 2.8))
        x_positions = np.arange(region_count)
        for edge in graph.edges:
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
        axis.set_title(f"{name} region graph, iteration {iteration}")
        axis.axis("off")
        figure.tight_layout()
        figure.savefig(
            self.output_dir / f"graph_{name}_step_{iteration:04d}.png", dpi=140
        )
        plt.close(figure)

    def _plot_heatmap(
        self, iteration: int, name: str, matrix
    ) -> None:
        values = matrix.detach().float().cpu().numpy()
        figure, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(values, cmap="magma")
        axis.set_title(f"{name} region dependency, step {iteration}")
        axis.set_xlabel("region")
        axis.set_ylabel("region")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(
            self.output_dir / f"region_dependency_{name}_step_{iteration:04d}.png",
            dpi=140,
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
        figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        signal_names = sorted(
            {
                name
                for item in self.graph_snapshots
                for name in item["signals"]
                if name == "dapd"
                or name.endswith("_t0.90")
                or name.startswith("combined_")
            }
        )
        for name in signal_names:
            selected = [
                (item["iteration"], item["signals"][name])
                for item in self.graph_snapshots
                if name in item["signals"]
            ]
            axes[0].plot(
                [item[0] for item in selected],
                [item[1]["active_edge_density"] for item in selected],
                marker="o",
                label=name,
            )
            axes[1].plot(
                [item[0] for item in selected],
                [item[1]["average_active_region_degree"] for item in selected],
                marker="o",
                label=name,
            )
            axes[2].plot(
                [item[0] for item in selected],
                [item[1]["edge_jaccard_with_previous"] for item in selected],
                marker="o",
                label=name,
            )
        axes[0].set_ylabel("edge density")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].legend()
        axes[1].set_ylabel("average degree")
        axes[1].legend()
        axes[2].set_ylabel("edge Jaccard")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_xlabel("global iteration / NFE")
        axes[2].legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "graph_metrics_over_time.png", dpi=140)
        plt.close(figure)
