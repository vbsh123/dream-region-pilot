from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


NUMBER = r"-?\d[\d,]*(?:\.\d+)?"


def extract_answer(text: str) -> str | None:
    hash_answers = re.findall(rf"####\s*({NUMBER})", text)
    if hash_answers:
        return hash_answers[-1].replace(",", "")
    boxed_answers = re.findall(rf"\\boxed\{{\s*({NUMBER})\s*\}}", text)
    if boxed_answers:
        return boxed_answers[-1].replace(",", "")
    numbers = re.findall(NUMBER, text)
    return numbers[-1].replace(",", "") if numbers else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strategies = sorted({str(row["strategy"]) for row in rows})
    result = []
    for strategy in strategies:
        selected = [row for row in rows if row["strategy"] == strategy]
        count = len(selected)
        densities = [
            float(row["mean_graph_edge_density"])
            for row in selected
            if row.get("mean_graph_edge_density") is not None
        ]
        graph_snapshots = [
            snapshot
            for row in selected
            for snapshot in row.get("graph_summaries", [])
        ]
        result.append(
            {
                "strategy": strategy,
                "examples": count,
                "task_accuracy": sum(bool(row["correct"]) for row in selected) / count,
                "mean_nfe": sum(int(row["nfe"]) for row in selected) / count,
                "total_nfe": sum(int(row["nfe"]) for row in selected),
                "mean_tokens_committed_per_forward": sum(
                    float(row["average_tokens_committed_per_forward"])
                    for row in selected
                )
                / count,
                "total_canvas_tokens": sum(int(row["canvas_tokens"]) for row in selected),
                "total_effective_generated_tokens": sum(
                    int(row["effective_generated_tokens"]) for row in selected
                ),
                "total_generated_tokens": sum(
                    int(row["effective_generated_tokens"]) for row in selected
                ),
                "total_wall_clock_seconds": sum(
                    float(row["wall_clock_seconds"]) for row in selected
                ),
                "mean_wall_clock_seconds": sum(
                    float(row["wall_clock_seconds"]) for row in selected
                )
                / count,
                "mean_dependency_seconds": sum(
                    float(row["dependency_seconds"]) for row in selected
                )
                / count,
                "mean_mean_field_seconds": sum(
                    float(row.get("mean_field_seconds", 0.0)) for row in selected
                )
                / count,
                "mean_graph_edge_density": (
                    sum(densities) / len(densities) if densities else None
                ),
                "mean_graph_average_degree": (
                    sum(float(item["average_region_degree"]) for item in graph_snapshots)
                    / len(graph_snapshots)
                    if graph_snapshots
                    else None
                ),
                "mean_graph_component_count": (
                    sum(float(item["component_count"]) for item in graph_snapshots)
                    / len(graph_snapshots)
                    if graph_snapshots
                    else None
                ),
                "mean_graph_max_parent_count": (
                    sum(float(item["max_parent_count"]) for item in graph_snapshots)
                    / len(graph_snapshots)
                    if graph_snapshots
                    else None
                ),
                "complete_graph_snapshot_fraction": (
                    sum(bool(item["is_complete_graph"]) for item in graph_snapshots)
                    / len(graph_snapshots)
                    if graph_snapshots
                    else None
                ),
            }
        )
    return result


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import csv

    summary = summarize(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if summary:
        with (output_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
