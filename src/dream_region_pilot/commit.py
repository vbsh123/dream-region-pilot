from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .regions import Region


def _filtered_logits(
    logits: torch.Tensor, top_p: float | None, top_k: int | None
) -> torch.Tensor:
    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            -1, sorted_indices, remove
        )
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    if top_k is not None:
        top_k = min(int(top_k), logits.shape[-1])
        cutoff = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    policy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dream's ordinary sample_tokens rule, adapted from pinned DAWN."""
    if temperature > 0:
        logits = logits / temperature
    logits = _filtered_logits(logits, top_p, top_k)
    probabilities = F.softmax(logits, dim=-1)
    if temperature > 0:
        # Match Dream's released sampler.  ``Categorical(probs=...)`` first
        # normalizes and validates its input as a simplex; for Dream's 152k
        # vocabulary that validation can reject otherwise sampleable bfloat16
        # rows because their rounded sum misses the strict simplex tolerance.
        # ``torch.multinomial`` is the path used by Dream itself and avoids the
        # extra normalization/validation pass.  Dream falls back to greedy
        # selection if multinomial sampling is numerically invalid.
        try:
            predictions = torch.multinomial(
                probabilities, num_samples=1
            ).squeeze(-1)
            confidence = probabilities.gather(
                -1, predictions.unsqueeze(-1)
            ).squeeze(-1)
        except RuntimeError:
            confidence, predictions = probabilities.max(dim=-1)
    else:
        confidence, predictions = probabilities.max(dim=-1)

    if policy == "topk_margin":
        top_two = torch.topk(probabilities, k=2, dim=-1).values
        confidence = top_two[..., 0] - top_two[..., 1]
    elif policy == "entropy":
        confidence = torch.sum(
            probabilities * torch.log(probabilities + 1e-10), dim=-1
        )
    elif policy != "maskgit_plus":
        raise ValueError("commit_policy must be entropy, topk_margin, or maskgit_plus")
    return confidence, predictions


def local_transfer_count(
    remaining_masks: int,
    *,
    clock: int,
    local_steps: int,
    eps: float,
) -> int:
    if remaining_masks <= 0:
        return 0
    if not 0 <= clock < local_steps:
        raise ValueError("clock must index an unfinished local schedule")
    if clock == local_steps - 1:
        return remaining_masks
    delta = (1.0 - eps) / local_steps
    current_time = 1.0 - clock * delta
    next_time = 1.0 - (clock + 1) * delta
    return int(remaining_masks * (1.0 - next_time / current_time))


def commit_active_regions(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    *,
    prompt_length: int,
    mask_token_id: int,
    regions: list[Region],
    local_steps: int,
    eps: float,
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    policy: str,
    alg_temp: float | None,
    region_groups: list[list[Region]] | None = None,
) -> dict[int, list[int]]:
    response_mask = tokens[0, prompt_length:] == mask_token_id
    absolute_mask = tokens == mask_token_id
    absolute_mask[:, :prompt_length] = False
    confidence, predictions = sample_tokens(
        logits[absolute_mask],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        policy=policy,
    )
    full_confidence = torch.full(
        tokens.shape,
        -torch.inf,
        dtype=logits.dtype,
        device=tokens.device,
    )
    full_predictions = tokens.clone()
    full_confidence[absolute_mask] = confidence.to(logits.dtype)
    full_predictions[absolute_mask] = predictions

    committed: dict[int, list[int]] = {region.index: [] for region in regions}
    groups = region_groups or [[region] for region in regions]
    grouped_indices = [region.index for group in groups for region in group]
    if sorted(grouped_indices) != sorted(region.index for region in regions):
        raise ValueError("region_groups must contain every active region exactly once")

    for group in groups:
        masked_by_region: dict[int, torch.Tensor] = {}
        count = 0
        terminal_relative: list[torch.Tensor] = []
        for region in group:
            relative = torch.tensor(
                region.token_indices, dtype=torch.long, device=tokens.device
            )
            masked_relative = relative[response_mask[relative]]
            masked_by_region[region.index] = masked_relative
            if region.schedule_step == local_steps - 1:
                # A pooled component may otherwise spend this region's final
                # transfer budget on higher-confidence tokens in another
                # member.  Its schedule cursor would then be exhausted while
                # masks remain, leaving it permanently ineligible.  Preserve
                # Dream's terminal-step completion invariant per region; the
                # rest of the component's budget is still selected jointly.
                terminal_relative.append(masked_relative)
            count += local_transfer_count(
                int(masked_relative.numel()),
                clock=region.schedule_step,
                local_steps=local_steps,
                eps=eps,
            )
        if count == 0:
            continue

        group_relative = torch.cat(
            [masked_by_region[region.index] for region in group]
        )
        count = min(count, int(group_relative.numel()))
        reserved_relative = (
            torch.cat(terminal_relative)
            if terminal_relative
            else group_relative[:0]
        )
        remaining_budget = count - int(reserved_relative.numel())
        if remaining_budget < 0:
            raise RuntimeError("terminal commitment exceeds the group budget")

        if remaining_budget:
            candidate_mask = ~torch.isin(group_relative, reserved_relative)
            candidate_relative = group_relative[candidate_mask]
            candidate_absolute = candidate_relative + prompt_length
            scores = full_confidence[0, candidate_absolute]
            if alg_temp is None or alg_temp == 0:
                chosen_local = torch.topk(scores, k=remaining_budget).indices
            else:
                weights = F.softmax(scores / alg_temp, dim=-1)
                chosen_local = torch.multinomial(
                    weights, num_samples=remaining_budget
                )
            selected_relative = torch.cat(
                (reserved_relative, candidate_relative[chosen_local])
            )
        else:
            selected_relative = reserved_relative
        selected_absolute = selected_relative + prompt_length
        tokens[0, selected_absolute] = full_predictions[0, selected_absolute]
        for region in group:
            region_start = region.token_indices[0]
            region_end = region.token_indices[-1] + 1
            selected_in_region = selected_relative[
                (selected_relative >= region_start)
                & (selected_relative < region_end)
            ]
            committed[region.index] = [
                int(position.item()) for position in selected_in_region
            ]
    return committed


@torch.no_grad()
def describe_commitments(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    prompt_length: int,
    regions: list[Region],
    committed: dict[int, list[int]],
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    policy: str,
) -> list[dict[str, Any]]:
    """Describe committed tokens without changing decoding behavior.

    Raw probabilities are computed from the model logits before temperature or
    top-p/top-k filtering.  Sampling probabilities reproduce the distribution
    used by ``sample_tokens``.  The latter is also the source of the entropy or
    margin score used to rank commitments under Dream's ordinary policy.

    This function is called only for explicitly requested diagnostic examples.
    Keeping it outside ``commit_active_regions`` makes its extra work easy to
    time and exclude from reported decoding latency.
    """
    selected: list[tuple[int, int]] = [
        (int(region_index), int(position))
        for region_index, positions in committed.items()
        for position in positions
    ]
    if not selected:
        return []

    device = logits.device
    response_logits = logits[0, prompt_length:].float()
    selected_relative = torch.tensor(
        [position for _, position in selected], dtype=torch.long, device=device
    )
    selected_logits = response_logits.index_select(0, selected_relative)

    raw_log_normalizer = torch.logsumexp(selected_logits, dim=-1)
    raw_top_logits, raw_top_ids = torch.topk(selected_logits, k=2, dim=-1)
    raw_top_probabilities = (
        raw_top_logits - raw_log_normalizer.unsqueeze(-1)
    ).exp()
    chosen_ids = tokens[0, selected_relative + prompt_length].long()
    raw_chosen_probability = (
        selected_logits.gather(1, chosen_ids.unsqueeze(-1)).squeeze(-1)
        - raw_log_normalizer
    ).exp()
    raw_log_probabilities = selected_logits - raw_log_normalizer.unsqueeze(-1)
    raw_probabilities = raw_log_probabilities.exp()
    raw_entropy = -(
        raw_probabilities * raw_log_probabilities
    ).sum(dim=-1)

    sampling_logits = selected_logits
    if temperature > 0:
        sampling_logits = sampling_logits / temperature
    sampling_logits = _filtered_logits(sampling_logits, top_p, top_k)
    sampling_log_probabilities = F.log_softmax(sampling_logits, dim=-1)
    sampling_probabilities = sampling_log_probabilities.exp()
    sampling_top_probabilities, sampling_top_ids = torch.topk(
        sampling_probabilities, k=2, dim=-1
    )
    sampling_chosen_probability = sampling_probabilities.gather(
        1, chosen_ids.unsqueeze(-1)
    ).squeeze(-1)
    sampling_entropy = -(
        sampling_probabilities * sampling_log_probabilities
    ).sum(dim=-1)

    # Rank by the standard, untempered top-1 confidence.  This is the quantity
    # used by the readiness gate and is the most useful value for testing a
    # future Fast-dLLM-style confidence veto.
    all_raw_confidence = (
        response_logits.max(dim=-1).values
        - torch.logsumexp(response_logits, dim=-1)
    ).exp()
    region_by_index = {region.index: region for region in regions}
    global_candidates = torch.tensor(
        [
            position
            for region in regions
            for position in region.remaining_mask_indices
        ],
        dtype=torch.long,
        device=device,
    )
    global_scores = all_raw_confidence.index_select(0, global_candidates)

    details: list[dict[str, Any]] = []
    for row, (region_index, response_position) in enumerate(selected):
        region = region_by_index[region_index]
        region_candidates = torch.tensor(
            region.remaining_mask_indices, dtype=torch.long, device=device
        )
        region_scores = all_raw_confidence.index_select(0, region_candidates)
        confidence = raw_top_probabilities[row, 0]
        if policy == "entropy":
            selection_score = -sampling_entropy[row]
        elif policy == "topk_margin":
            selection_score = (
                sampling_top_probabilities[row, 0]
                - sampling_top_probabilities[row, 1]
            )
        else:
            selection_score = sampling_chosen_probability[row]
        details.append(
            {
                "region": region_index,
                "response_position": response_position,
                "absolute_position": response_position + prompt_length,
                "token_id": int(chosen_ids[row].item()),
                "raw_chosen_probability": float(
                    raw_chosen_probability[row].item()
                ),
                "raw_top1_token_id": int(raw_top_ids[row, 0].item()),
                "raw_top1_probability": float(
                    raw_top_probabilities[row, 0].item()
                ),
                "raw_top2_token_id": int(raw_top_ids[row, 1].item()),
                "raw_top2_probability": float(
                    raw_top_probabilities[row, 1].item()
                ),
                "raw_chosen_is_top1": bool(
                    chosen_ids[row].item() == raw_top_ids[row, 0].item()
                ),
                "raw_top1_top2_margin": float(
                    (
                        raw_top_probabilities[row, 0]
                        - raw_top_probabilities[row, 1]
                    ).item()
                ),
                "raw_entropy": float(raw_entropy[row].item()),
                "sampling_chosen_probability": float(
                    sampling_chosen_probability[row].item()
                ),
                "sampling_top1_token_id": int(
                    sampling_top_ids[row, 0].item()
                ),
                "sampling_top1_probability": float(
                    sampling_top_probabilities[row, 0].item()
                ),
                "sampling_top2_token_id": int(
                    sampling_top_ids[row, 1].item()
                ),
                "sampling_top2_probability": float(
                    sampling_top_probabilities[row, 1].item()
                ),
                "sampling_chosen_is_top1": bool(
                    chosen_ids[row].item() == sampling_top_ids[row, 0].item()
                ),
                "sampling_entropy": float(sampling_entropy[row].item()),
                "policy": policy,
                "policy_selection_score": float(selection_score.item()),
                "raw_global_confidence_rank": int(
                    (global_scores > confidence).sum().item()
                )
                + 1,
                "raw_global_candidate_count": int(global_scores.numel()),
                "raw_region_confidence_rank": int(
                    (region_scores > confidence).sum().item()
                )
                + 1,
                "raw_region_candidate_count": int(region_scores.numel()),
                "region_clock_before": region.clock,
                "region_schedule_step_before": region.schedule_step,
                "region_commit_count": len(committed[region_index]),
            }
        )
    return details
