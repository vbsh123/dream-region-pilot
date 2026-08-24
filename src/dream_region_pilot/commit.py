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

    committed: dict[int, list[int]] = {}
    for region in regions:
        relative = torch.tensor(
            region.token_indices, dtype=torch.long, device=tokens.device
        )
        masked_relative = relative[response_mask[relative]]
        count = local_transfer_count(
            int(masked_relative.numel()),
            clock=region.schedule_step,
            local_steps=local_steps,
            eps=eps,
        )
        if count == 0:
            committed[region.index] = []
            continue
        absolute = masked_relative + prompt_length
        scores = full_confidence[0, absolute]
        if alg_temp is None or alg_temp == 0:
            chosen_local = torch.topk(scores, k=count).indices
        else:
            weights = F.softmax(scores / alg_temp, dim=-1)
            chosen_local = torch.multinomial(weights, num_samples=count)
        selected_absolute = absolute[chosen_local]
        tokens[0, selected_absolute] = full_predictions[0, selected_absolute]
        committed[region.index] = [
            int(position.item() - prompt_length) for position in selected_absolute
        ]
    return committed
