from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def exact_jsd_interaction(
    logits: torch.Tensor,
    *,
    pair_chunk_size: int = 16,
) -> torch.Tensor:
    """Construct the exact normalized interaction matrix from Algorithm 1.

    This is intentionally limited to one active block. Applying the paper's
    O(m^2 |V|) operation to the full 256-token canvas is both unnecessary for
    its block sampler and needlessly expensive on a 24 GB card.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [masked_positions, vocabulary]")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")
    count = logits.shape[0]
    interaction = torch.zeros(
        (count, count), dtype=torch.float32, device=logits.device
    )
    if count < 2:
        return interaction

    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    probabilities = log_probabilities.exp()
    entropy = _entropy_terms(probabilities)
    pairs = torch.triu_indices(count, count, offset=1, device=logits.device)
    values: list[torch.Tensor] = []
    for start in range(0, pairs.shape[1], pair_chunk_size):
        left = pairs[0, start : start + pair_chunk_size]
        right = pairs[1, start : start + pair_chunk_size]
        mixture = 0.5 * (
            probabilities.index_select(0, left)
            + probabilities.index_select(0, right)
        )
        jsd = _entropy_terms(mixture) - 0.5 * (
            entropy.index_select(0, left) + entropy.index_select(0, right)
        )
        values.append((1.0 - jsd / math.log(2.0)).clamp(0.0, 1.0))

    pair_values = torch.cat(values)
    maximum = pair_values.max()
    if float(maximum.item()) > 0:
        pair_values = pair_values / maximum
    interaction[pairs[0], pairs[1]] = pair_values
    interaction[pairs[1], pairs[0]] = pair_values
    return interaction


@torch.no_grad()
def mean_field_commit_indices(
    logits: torch.Tensor,
    *,
    threshold: float = 0.9,
    iterations: int = 2,
    pair_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Reproduce Algorithm 1 from arXiv:2606.15805.

    Returns selected row indices, final commit intensities, and whether the
    progress fallback was needed. The paper pseudocode does not specify what
    to do when thresholding selects no token; choosing the maximum-q position
    is our explicit deadlock-prevention choice and is logged by the decoder.
    """
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must contain at least one masked position")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    top_two = torch.topk(logits.float(), k=2, dim=-1).values
    # log softmax cancels in log(p_top1) - log(p_top2).
    confidence = top_two[:, 0] - top_two[:, 1]
    interaction = exact_jsd_interaction(
        logits, pair_chunk_size=pair_chunk_size
    )
    intensity = torch.sigmoid(confidence)
    for _ in range(iterations):
        intensity = torch.sigmoid(confidence - interaction @ intensity)
    selected = (intensity >= threshold).nonzero(as_tuple=False).flatten()
    fallback = selected.numel() == 0
    if fallback:
        selected = intensity.argmax().reshape(1)
    return selected, intensity, fallback


def _entropy_terms(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.where(
        probabilities > 0,
        -probabilities * probabilities.clamp_min(1e-30).log(),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
