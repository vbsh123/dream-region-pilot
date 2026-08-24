from __future__ import annotations

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
