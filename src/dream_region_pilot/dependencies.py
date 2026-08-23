from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .regions import Region


@dataclass(frozen=True)
class DependencySnapshot:
    raw_token: torch.Tensor
    normalized_token: torch.Tensor
    region_matrices: dict[str, torch.Tensor]
    selected_region: torch.Tensor
    parents: dict[int, set[int]]
    edges: list[dict[str, float | int]]
    active_region_indices: tuple[int, ...]
    signal_token_matrices: dict[str, torch.Tensor] = field(default_factory=dict)
    signal_region_matrices: dict[str, torch.Tensor] = field(default_factory=dict)
    signal_graphs: dict[str, "SignalGraph"] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalGraph:
    matrix: torch.Tensor
    threshold: float
    parents: dict[int, set[int]]
    edges: list[dict[str, Any]]


def verify_dapd_checkout(repo: Path, expected_revision: str) -> str:
    repo = repo.resolve()
    if not (repo / ".git").is_dir():
        raise FileNotFoundError(
            f"DAPD checkout not found at {repo}; run scripts/setup_vast.sh"
        )
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise RuntimeError(
            f"DAPD revision mismatch: expected {expected_revision}, found {revision}"
        )
    return revision


class DAPDDreamAdapter:
    """Thin adapter over the pinned public DAPD Dream attention implementation."""

    def __init__(self, repo: Path, revision: str, layer_ratio: float = 0.3):
        self.repo = repo.resolve()
        self.revision = verify_dapd_checkout(self.repo, revision)
        self.layer_ratio = float(layer_ratio)
        if not 0.0 < self.layer_ratio <= 1.0:
            raise ValueError("layer_ratio must be in (0, 1]")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        from dapd.core import build_dependency_graph
        from dapd.dream_core import DreamAttentionCaptureHook, shift_logits_dream

        self._build_dependency_graph = build_dependency_graph
        self._hook_class = DreamAttentionCaptureHook
        self._shift_logits = shift_logits_dream

    def forward_logits(self, model, tokens: torch.Tensor) -> torch.Tensor:
        outputs = model.model(tokens, output_attentions=False, return_dict=True)
        return self._shift_logits(model.lm_head(outputs.last_hidden_state))

    def forward_with_dependencies(
        self,
        model,
        tokens: torch.Tensor,
        generated_mask: torch.Tensor,
        prompt_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hook = self._hook_class()
        hook.register(model, layer_ratio=self.layer_ratio)
        try:
            logits = self.forward_logits(model, tokens)
            attention = hook.compute_attention_weights()
        finally:
            hook.remove()

        full_mask = torch.zeros_like(tokens, dtype=torch.bool)
        full_mask[:, prompt_length:] = generated_mask
        raw, normalized = self._build_dependency_graph(attention, full_mask)
        response_slice = slice(prompt_length, tokens.shape[1])
        return (
            logits,
            raw[0, response_slice, response_slice],
            normalized[0, response_slice, response_slice],
        )


def aggregate_region_dependencies(
    token_dependency: torch.Tensor,
    regions: list[Region],
    *,
    method: str,
    topk_percent: float = 0.10,
    gamma: float = 2.0,
) -> torch.Tensor:
    valid = {"mean", "topk_percent", "power_mean"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}")
    if not 0.0 < topk_percent <= 1.0:
        raise ValueError("topk_percent must be in (0, 1]")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    count = len(regions)
    result = torch.zeros(
        (count, count), dtype=torch.float32, device=token_dependency.device
    )
    for left in range(count):
        if not regions[left].remaining_mask_indices:
            continue
        left_index = torch.tensor(
            regions[left].remaining_mask_indices,
            dtype=torch.long,
            device=token_dependency.device,
        )
        for right in range(left + 1, count):
            if not regions[right].remaining_mask_indices:
                continue
            right_index = torch.tensor(
                regions[right].remaining_mask_indices,
                dtype=torch.long,
                device=token_dependency.device,
            )
            cross = token_dependency.index_select(0, left_index).index_select(
                1, right_index
            )
            # Include both directions. This is redundant for symmetric scores
            # and intentional for DAPD's row-normalized matrix.
            reverse = token_dependency.index_select(0, right_index).index_select(
                1, left_index
            )
            values = torch.cat((cross.reshape(-1), reverse.reshape(-1)))
            if method == "mean":
                score = values.mean()
            elif method == "topk_percent":
                k = max(1, math.ceil(values.numel() * topk_percent))
                score = torch.topk(values, k=k).values.mean()
            else:
                score = values.pow(gamma).mean()
            result[left, right] = score
            result[right, left] = score
    return result


def aggregate_all_region_dependencies(
    token_dependency: torch.Tensor,
    regions: list[Region],
    *,
    topk_percent: float,
    gamma: float,
) -> dict[str, torch.Tensor]:
    return {
        method: aggregate_region_dependencies(
            token_dependency,
            regions,
            method=method,
            topk_percent=topk_percent,
            gamma=gamma,
        )
        for method in ("mean", "topk_percent", "power_mean")
    }


def threshold_region_graph(
    region_dependency: torch.Tensor, threshold: float
) -> tuple[dict[int, set[int]], list[dict[str, Any]]]:
    count = region_dependency.shape[0]
    if region_dependency.shape != (count, count):
        raise ValueError("region_dependency must be square")
    parents = {index: set() for index in range(count)}
    edges: list[dict[str, float | int]] = []
    for left in range(count):
        for right in range(left + 1, count):
            score = float(region_dependency[left, right].item())
            if score >= threshold:
                # Positional orientation requested by the pilot: lower is parent.
                parents[right].add(left)
                edges.append({"left": left, "right": right, "score": score})
    return parents, edges


def _binary_combination_graph(
    left: SignalGraph,
    right: SignalGraph,
    *,
    operation: str,
) -> SignalGraph:
    if operation not in {"union", "intersection"}:
        raise ValueError("operation must be union or intersection")
    left_edges = {
        (int(edge["left"]), int(edge["right"])): float(edge["score"])
        for edge in left.edges
    }
    right_edges = {
        (int(edge["left"]), int(edge["right"])): float(edge["score"])
        for edge in right.edges
    }
    if operation == "union":
        pairs = set(left_edges) | set(right_edges)
    else:
        pairs = set(left_edges) & set(right_edges)
    count = left.matrix.shape[0]
    matrix = torch.zeros((count, count), dtype=torch.float32, device=left.matrix.device)
    parents = {index: set() for index in range(count)}
    edges: list[dict[str, Any]] = []
    for low, high in sorted(pairs):
        matrix[low, high] = 1.0
        matrix[high, low] = 1.0
        parents[high].add(low)
        edges.append(
            {
                "left": low,
                "right": high,
                "score": 1.0,
                "dapd_score": left_edges.get((low, high)),
                "mean_field_score": right_edges.get((low, high)),
            }
        )
    return SignalGraph(matrix=matrix, threshold=0.5, parents=parents, edges=edges)


def make_dependency_snapshot(
    *,
    raw_token: torch.Tensor,
    normalized_token: torch.Tensor,
    regions: list[Region],
    matrix: str,
    aggregator: str,
    topk_percent: float,
    gamma: float,
    threshold: float,
    additional_token_dependencies: dict[str, torch.Tensor] | None = None,
    additional_thresholds: dict[str, list[float]] | None = None,
    additional_aggregators: dict[str, str] | None = None,
    combination_threshold: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> DependencySnapshot:
    if matrix not in {"raw", "normalized"}:
        raise ValueError("dependency matrix must be 'raw' or 'normalized'")
    selected_token = raw_token if matrix == "raw" else normalized_token
    region_matrices = aggregate_all_region_dependencies(
        selected_token,
        regions,
        topk_percent=topk_percent,
        gamma=gamma,
    )
    selected_region = region_matrices[aggregator]
    parents, edges = threshold_region_graph(selected_region, threshold)
    signal_token_matrices = dict(additional_token_dependencies or {})
    signal_region_matrices: dict[str, torch.Tensor] = {}
    signal_graphs: dict[str, SignalGraph] = {
        "dapd": SignalGraph(
            matrix=selected_region,
            threshold=threshold,
            parents=parents,
            edges=edges,
        )
    }
    additional_thresholds = additional_thresholds or {}
    additional_aggregators = additional_aggregators or {}
    for signal_name, token_matrix in signal_token_matrices.items():
        matrices = aggregate_all_region_dependencies(
            token_matrix,
            regions,
            topk_percent=topk_percent,
            gamma=gamma,
        )
        for method_name, region_matrix in matrices.items():
            signal_region_matrices[f"{signal_name}_{method_name}"] = region_matrix
        signal_aggregator = additional_aggregators.get(signal_name, aggregator)
        if signal_aggregator not in matrices:
            raise ValueError(
                f"Unknown aggregator {signal_aggregator!r} for {signal_name!r}"
            )
        selected = matrices[signal_aggregator]
        thresholds = additional_thresholds.get(signal_name, [])
        for signal_threshold in thresholds:
            extra_parents, extra_edges = threshold_region_graph(
                selected, float(signal_threshold)
            )
            key = f"{signal_name}_t{float(signal_threshold):.2f}"
            signal_graphs[key] = SignalGraph(
                matrix=selected,
                threshold=float(signal_threshold),
                parents=extra_parents,
                edges=extra_edges,
            )

        if combination_threshold is not None and thresholds:
            matching_key = f"{signal_name}_t{float(combination_threshold):.2f}"
            if matching_key not in signal_graphs:
                extra_parents, extra_edges = threshold_region_graph(
                    selected, float(combination_threshold)
                )
                signal_graphs[matching_key] = SignalGraph(
                    matrix=selected,
                    threshold=float(combination_threshold),
                    parents=extra_parents,
                    edges=extra_edges,
                )
            signal_graphs[f"combined_union_t{float(combination_threshold):.2f}"] = (
                _binary_combination_graph(
                    signal_graphs["dapd"],
                    signal_graphs[matching_key],
                    operation="union",
                )
            )
            signal_graphs[
                f"combined_intersection_t{float(combination_threshold):.2f}"
            ] = _binary_combination_graph(
                signal_graphs["dapd"],
                signal_graphs[matching_key],
                operation="intersection",
            )
    return DependencySnapshot(
        raw_token=raw_token,
        normalized_token=normalized_token,
        region_matrices=region_matrices,
        selected_region=selected_region,
        parents=parents,
        edges=edges,
        active_region_indices=tuple(
            region.index for region in regions if not region.done
        ),
        signal_token_matrices=signal_token_matrices,
        signal_region_matrices=signal_region_matrices,
        signal_graphs=signal_graphs,
        metadata=dict(metadata or {}),
    )


def graph_summary(
    snapshot: DependencySnapshot,
    graph: SignalGraph | None = None,
) -> dict[str, Any]:
    graph = graph or snapshot.signal_graphs.get("dapd")
    region_matrix = graph.matrix if graph is not None else snapshot.selected_region
    edges = graph.edges if graph is not None else snapshot.edges
    parents = graph.parents if graph is not None else snapshot.parents
    region_count = region_matrix.shape[0]
    possible = region_count * (region_count - 1) // 2
    active = set(snapshot.active_region_indices)
    active_count = len(active)
    active_possible = active_count * (active_count - 1) // 2
    adjacency = {index: set() for index in range(region_count)}
    for edge in edges:
        left, right = int(edge["left"]), int(edge["right"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    degrees = {index: len(neighbors) for index, neighbors in adjacency.items()}
    active_degrees = {
        index: len(adjacency[index] & active) for index in active
    }
    parent_counts = {
        region_index: len(parents.get(region_index, set()))
        for region_index in range(region_count)
    }
    components = 0
    unseen = set(adjacency)
    while unseen:
        components += 1
        stack = [unseen.pop()]
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unseen
            unseen -= neighbors
            stack.extend(neighbors)
    active_components = 0
    unseen_active = set(active)
    while unseen_active:
        active_components += 1
        stack = [unseen_active.pop()]
        while stack:
            node = stack.pop()
            neighbors = adjacency[node] & unseen_active
            unseen_active -= neighbors
            stack.extend(neighbors)
    density = len(edges) / possible if possible else 0.0
    active_edge_count = sum(
        int(edge["left"]) in active and int(edge["right"]) in active
        for edge in edges
    )
    active_density = active_edge_count / active_possible if active_possible else 0.0
    adjacent_edges = sum(
        int(edge["right"]) - int(edge["left"]) == 1 for edge in edges
    )
    return {
        "region_count": region_count,
        "active_region_count": active_count,
        "active_region_indices": sorted(active),
        "edge_count": len(edges),
        "active_edge_count": active_edge_count,
        "possible_edge_count": possible,
        "active_possible_edge_count": active_possible,
        "edge_density": density,
        "edge_percentage": 100.0 * density,
        "active_edge_density": active_density,
        "active_edge_percentage": 100.0 * active_density,
        "average_region_degree": (
            sum(degrees.values()) / region_count if region_count else 0.0
        ),
        "average_active_region_degree": (
            sum(active_degrees.values()) / active_count
            if active_count
            else 0.0
        ),
        "region_degrees": degrees,
        "active_region_degrees": active_degrees,
        "component_count": components,
        "active_component_count": active_components,
        "parent_counts": parent_counts,
        "average_parent_count": (
            sum(parent_counts.values()) / region_count if region_count else 0.0
        ),
        "max_parent_count": max(parent_counts.values(), default=0),
        "root_count": sum(count == 0 for count in parent_counts.values()),
        "active_root_count": sum(
            not (parents.get(index, set()) & active) for index in active
        ),
        "adjacent_edge_count": adjacent_edges,
        "nonadjacent_edge_count": len(edges) - adjacent_edges,
        "is_complete_graph": bool(possible and len(edges) == possible),
    }
