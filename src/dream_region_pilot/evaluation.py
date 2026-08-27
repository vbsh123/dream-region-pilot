from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


NUMBER = r"-?\d[\d,]*(?:(?:\.\d+)|(?:/\d+))?"


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
        total_nfe = sum(int(row["nfe"]) for row in selected)
        total_canvas_tokens = sum(int(row["canvas_tokens"]) for row in selected)
        total_effective_tokens = sum(
            int(row["effective_generated_tokens"]) for row in selected
        )
        total_wall_clock = sum(
            float(row["wall_clock_seconds"]) for row in selected
        )
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
                "strict_match_accuracy": (
                    sum(bool(row["strict_match_correct"]) for row in selected)
                    / count
                    if all("strict_match_correct" in row for row in selected)
                    else None
                ),
                "flexible_extract_accuracy": (
                    sum(bool(row["flexible_extract_correct"]) for row in selected)
                    / count
                    if all("flexible_extract_correct" in row for row in selected)
                    else None
                ),
                "mean_nfe": sum(int(row["nfe"]) for row in selected) / count,
                "total_nfe": total_nfe,
                "mean_tokens_committed_per_forward": (
                    total_canvas_tokens / total_nfe if total_nfe > 0 else None
                ),
                "total_canvas_tokens": total_canvas_tokens,
                "total_effective_generated_tokens": total_effective_tokens,
                "mean_effective_generated_tokens": (
                    total_effective_tokens / count
                ),
                "total_generated_tokens": total_effective_tokens,
                "total_wall_clock_seconds": total_wall_clock,
                "mean_wall_clock_seconds": total_wall_clock / count,
                "canvas_tokens_per_second": (
                    total_canvas_tokens / total_wall_clock
                    if total_wall_clock > 0
                    else None
                ),
                "effective_tokens_per_second": (
                    total_effective_tokens / total_wall_clock
                    if total_wall_clock > 0
                    else None
                ),
                "forward_passes_per_second": (
                    total_nfe / total_wall_clock
                    if total_wall_clock > 0
                    else None
                ),
                "mean_dependency_seconds": sum(
                    float(row["dependency_seconds"]) for row in selected
                )
                / count,
                "mean_mean_field_seconds": sum(
                    float(row.get("mean_field_seconds", 0.0)) for row in selected
                )
                / count,
                "mean_mean_field_forced_progress_events": sum(
                    int(row.get("mean_field_forced_progress_events", 0))
                    for row in selected
                )
                / count,
                "mean_iterations_with_blocking": sum(
                    int(row.get("iterations_with_blocking", 0)) for row in selected
                )
                / count,
                "mean_iterations_with_tail_guard": sum(
                    int(row.get("iterations_with_tail_guard", 0))
                    for row in selected
                )
                / count,
                "mean_iterations_with_deferral": sum(
                    int(row.get("iterations_with_deferral", 0))
                    for row in selected
                )
                / count,
                "mean_deferred_region_events": sum(
                    int(row.get("deferred_region_events", 0))
                    for row in selected
                )
                / count,
                "mean_forced_region_events": sum(
                    int(row.get("forced_region_events", 0))
                    for row in selected
                )
                / count,
                "mean_blocked_region_events": sum(
                    int(row.get("blocked_region_events", 0)) for row in selected
                )
                / count,
                "mean_active_dependency_edge_count": sum(
                    float(row.get("mean_active_dependency_edge_count", 0.0))
                    for row in selected
                )
                / count,
                "mean_iterations_with_pooled_commit_groups": sum(
                    int(row.get("iterations_with_pooled_commit_groups", 0))
                    for row in selected
                )
                / count,
                "mean_max_commit_group_size": sum(
                    float(row.get("mean_max_commit_group_size", 0.0))
                    for row in selected
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
    vanilla = next(
        (item for item in result if item["strategy"] == "vanilla"), None
    )
    for item in result:
        item["nfe_speedup_vs_vanilla"] = (
            vanilla["mean_nfe"] / item["mean_nfe"]
            if vanilla is not None and item["mean_nfe"] > 0
            else None
        )
        item["wall_clock_speedup_vs_vanilla"] = (
            vanilla["mean_wall_clock_seconds"] / item["mean_wall_clock_seconds"]
            if vanilla is not None and item["mean_wall_clock_seconds"] > 0
            else None
        )
        item["canvas_tps_speedup_vs_vanilla"] = (
            item["canvas_tokens_per_second"]
            / vanilla["canvas_tokens_per_second"]
            if vanilla is not None
            and item["canvas_tokens_per_second"] is not None
            and vanilla["canvas_tokens_per_second"]
            else None
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
