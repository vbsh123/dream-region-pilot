from __future__ import annotations

import math
from typing import Any

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


@torch.no_grad()
def topk_tail_jsd_dependency(
    response_logits: torch.Tensor,
    generated_mask: torch.Tensor,
    *,
    topk: int = 256,
    pair_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Approximate the Mean-Field paper's token interaction matrix.

    The paper uses ``1 - JSD(p_i, p_j) / ln(2)`` over full vocabulary
    distributions, followed by max normalization over the active masked set.
    Its reported implementation bounds that set to a 20-token active block.

    Our probe can expose all 256 masked canvas positions. To keep that
    diagnostic practical on a 24 GB GPU, each pair is compared on the union of
    both positions' top-k token supports plus one shared residual-tail bucket.
    This is a coarsened JSD: it can overestimate similarity when the omitted
    tails differ. Retained mass is returned so that approximation quality is
    visible in every snapshot.
    """
    if response_logits.ndim == 3:
        if response_logits.shape[0] != 1:
            raise ValueError("The minimal probe supports batch size one")
        response_logits = response_logits[0]
    if response_logits.ndim != 2:
        raise ValueError("response_logits must have shape [N, vocabulary]")
    if generated_mask.ndim == 2:
        if generated_mask.shape[0] != 1:
            raise ValueError("The minimal probe supports batch size one")
        generated_mask = generated_mask[0]
    if generated_mask.ndim != 1 or generated_mask.shape[0] != response_logits.shape[0]:
        raise ValueError("generated_mask must match the response length")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if pair_chunk_size <= 0:
        raise ValueError("pair_chunk_size must be positive")

    active_indices = generated_mask.nonzero(as_tuple=False).flatten()
    canvas_length = response_logits.shape[0]
    result = torch.zeros(
        (canvas_length, canvas_length),
        dtype=torch.float32,
        device=response_logits.device,
    )
    active_count = int(active_indices.numel())
    if active_count < 2:
        return result, result.clone(), {
            "method": "topk_tail_jsd",
            "paper_exact": False,
            "active_masked_tokens": active_count,
            "topk": min(topk, response_logits.shape[-1]),
            "mean_topk_probability_mass": 1.0 if active_count == 1 else None,
            "minimum_topk_probability_mass": 1.0 if active_count == 1 else None,
            "maximum_topk_probability_mass": 1.0 if active_count == 1 else None,
        }

    active_logits = response_logits.index_select(0, active_indices).float()
    log_probabilities = F.log_softmax(active_logits, dim=-1)
    vocabulary_size = log_probabilities.shape[-1]
    support_size = min(int(topk), vocabulary_size)
    top_log_probabilities, top_indices = torch.topk(
        log_probabilities, k=support_size, dim=-1
    )
    retained_mass = top_log_probabilities.exp().sum(dim=-1)

    pair_indices = torch.triu_indices(
        active_count,
        active_count,
        offset=1,
        device=response_logits.device,
    )
    pair_similarities: list[torch.Tensor] = []
    for start in range(0, pair_indices.shape[1], pair_chunk_size):
        left = pair_indices[0, start : start + pair_chunk_size]
        right = pair_indices[1, start : start + pair_chunk_size]

        # Sorting the concatenated supports makes duplicate vocabulary ids
        # adjacent. Duplicate entries are masked so probability is counted once.
        support = torch.cat(
            (top_indices.index_select(0, left), top_indices.index_select(0, right)),
            dim=1,
        ).sort(dim=1).values
        unique = torch.ones_like(support, dtype=torch.bool)
        unique[:, 1:] = support[:, 1:] != support[:, :-1]

        left_log = log_probabilities[left[:, None], support]
        right_log = log_probabilities[right[:, None], support]
        left_probability = left_log.exp() * unique
        right_probability = right_log.exp() * unique
        left_tail = (1.0 - left_probability.sum(dim=1)).clamp_min(0.0)
        right_tail = (1.0 - right_probability.sum(dim=1)).clamp_min(0.0)

        left_augmented = torch.cat((left_probability, left_tail[:, None]), dim=1)
        right_augmented = torch.cat((right_probability, right_tail[:, None]), dim=1)
        mixture = 0.5 * (left_augmented + right_augmented)
        jsd = _entropy_terms(mixture) - 0.5 * (
            _entropy_terms(left_augmented) + _entropy_terms(right_augmented)
        )
        similarity = (1.0 - jsd / math.log(2.0)).clamp(0.0, 1.0)
        pair_similarities.append(similarity)

    raw_similarities = torch.cat(pair_similarities)
    similarities = raw_similarities.clone()
    maximum = raw_similarities.max()
    if float(maximum.item()) > 0:
        similarities = similarities / maximum
    raw_active_matrix = torch.zeros(
        (active_count, active_count),
        dtype=torch.float32,
        device=response_logits.device,
    )
    raw_active_matrix[pair_indices[0], pair_indices[1]] = raw_similarities
    raw_active_matrix[pair_indices[1], pair_indices[0]] = raw_similarities
    active_matrix = torch.zeros_like(raw_active_matrix)
    active_matrix[pair_indices[0], pair_indices[1]] = similarities
    active_matrix[pair_indices[1], pair_indices[0]] = similarities
    raw_result = result.clone()
    raw_result[active_indices[:, None], active_indices[None, :]] = raw_active_matrix
    result[active_indices[:, None], active_indices[None, :]] = active_matrix

    return raw_result, result, {
        "method": "topk_tail_jsd",
        "paper_exact": support_size == vocabulary_size,
        "active_masked_tokens": active_count,
        "vocabulary_size": vocabulary_size,
        "topk": support_size,
        "pair_chunk_size": pair_chunk_size,
        "mean_topk_probability_mass": float(retained_mass.mean().item()),
        "minimum_topk_probability_mass": float(retained_mass.min().item()),
        "maximum_topk_probability_mass": float(retained_mass.max().item()),
        "maximum_raw_similarity": float(maximum.item()),
        "tail_bucket_warning": (
            "A shared residual bucket can overestimate similarity when omitted "
            "vocabulary tails differ."
        ),
    }
