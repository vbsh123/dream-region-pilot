from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
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
            # Include both directions. This is redundant for DAPD's raw symmetric
            # score and intentional for its row-normalized matrix.
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
) -> tuple[dict[int, set[int]], list[dict[str, float | int]]]:
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
    )


def graph_summary(snapshot: DependencySnapshot) -> dict[str, Any]:
    region_count = snapshot.selected_region.shape[0]
    possible = region_count * (region_count - 1) // 2
    active = set(snapshot.active_region_indices)
    active_count = len(active)
    active_possible = active_count * (active_count - 1) // 2
    adjacency = {index: set() for index in range(region_count)}
    for edge in snapshot.edges:
        left, right = int(edge["left"]), int(edge["right"])
        adjacency[left].add(right)
        adjacency[right].add(left)
    degrees = {index: len(neighbors) for index, neighbors in adjacency.items()}
    parent_counts = {
        region_index: len(snapshot.parents.get(region_index, set()))
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
    density = len(snapshot.edges) / possible if possible else 0.0
    active_density = len(snapshot.edges) / active_possible if active_possible else 0.0
    return {
        "region_count": region_count,
        "active_region_count": active_count,
        "active_region_indices": sorted(active),
        "edge_count": len(snapshot.edges),
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
            sum(degrees[index] for index in active) / active_count
            if active_count
            else 0.0
        ),
        "region_degrees": degrees,
        "component_count": components,
        "active_component_count": active_components,
        "parent_counts": parent_counts,
        "average_parent_count": (
            sum(parent_counts.values()) / region_count if region_count else 0.0
        ),
        "max_parent_count": max(parent_counts.values(), default=0),
        "root_count": sum(count == 0 for count in parent_counts.values()),
        "is_complete_graph": bool(possible and len(snapshot.edges) == possible),
    }
