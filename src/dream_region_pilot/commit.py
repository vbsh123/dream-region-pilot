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
    schedule_step: int,
    local_steps: int,
    eps: float,
) -> int:
    """Dream's linear transfer quota applied to one region's local schedule."""
    if remaining_masks <= 0:
        return 0
    if not 0 <= schedule_step < local_steps:
        raise ValueError("schedule_step must index an unfinished local schedule")
    if schedule_step == local_steps - 1:
        return remaining_masks
    delta = (1.0 - eps) / local_steps
    current_time = 1.0 - schedule_step * delta
    next_time = 1.0 - (schedule_step + 1) * delta
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
    deferral_confidence_threshold: float | None = None,
    max_region_deferrals: int | None = 0,
    region_deferral_counts: dict[int, int] | None = None,
    deferral_decisions: list[dict[str, Any]] | None = None,
    force_region_reasons: dict[int, str] | None = None,
    stop_protection_mode: str | None = None,
    stop_token_ids: set[int] | None = None,
    stop_protected_regions: set[int] | None = None,
    stop_protection_decisions: list[dict[str, Any]] | None = None,
    stop_filter_confidence_threshold: float = 0.4,
    max_response_position_exclusive: int | None = None,
) -> dict[int, list[int]]:
    if stop_protection_mode not in {None, "filter", "defer"}:
        raise ValueError("stop_protection_mode must be None, filter, or defer")
    if stop_protection_mode is not None and not stop_token_ids:
        raise ValueError("stop_token_ids is required for stop protection")
    if not 0.0 <= stop_filter_confidence_threshold <= 1.0:
        raise ValueError("stop_filter_confidence_threshold must be in [0, 1]")
    if deferral_confidence_threshold is not None:
        if not 0.0 <= deferral_confidence_threshold <= 1.0:
            raise ValueError("deferral_confidence_threshold must be in [0, 1]")
        if max_region_deferrals is not None and max_region_deferrals < 0:
            raise ValueError("max_region_deferrals must be non-negative")
        if region_deferral_counts is None:
            raise ValueError(
                "region_deferral_counts is required when deferral is enabled"
            )
    response_mask = tokens[0, prompt_length:] == mask_token_id
    absolute_mask = tokens == mask_token_id
    absolute_mask[:, :prompt_length] = False
    masked_logits = logits[absolute_mask]
    confidence, predictions = sample_tokens(
        masked_logits,
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
    full_raw_top1_predictions = tokens.clone()
    if stop_protection_mode == "defer":
        full_raw_top1_predictions[absolute_mask] = masked_logits.argmax(dim=-1)
    full_raw_proposed_probability = torch.zeros(
        tokens.shape,
        dtype=torch.float32,
        device=tokens.device,
    )
    if stop_protection_mode == "filter":
        raw_logits = masked_logits.float()
        proposed_logits = raw_logits.gather(
            -1, predictions.unsqueeze(-1)
        ).squeeze(-1)
        raw_proposed_probability = (
            proposed_logits - torch.logsumexp(raw_logits, dim=-1)
        ).exp()
        full_raw_proposed_probability[absolute_mask] = raw_proposed_probability
    full_raw_top1_probability = torch.zeros(
        tokens.shape,
        dtype=torch.float32,
        device=tokens.device,
    )
    if deferral_confidence_threshold is not None:
        raw_logits = masked_logits.float()
        raw_maximum = raw_logits.max(dim=-1).values
        raw_top1_probability = (
            raw_maximum - torch.logsumexp(raw_logits, dim=-1)
        ).exp()
        full_raw_top1_probability[absolute_mask] = raw_top1_probability

    committed: dict[int, list[int]] = {region.index: [] for region in regions}
    for region in regions:
        relative = torch.tensor(
            region.token_indices, dtype=torch.long, device=tokens.device
        )
        masked_relative = relative[response_mask[relative]]
        if max_response_position_exclusive is not None:
            masked_relative = masked_relative[
                masked_relative < max_response_position_exclusive
            ]
        count = local_transfer_count(
            int(masked_relative.numel()),
            schedule_step=region.schedule_step,
            local_steps=local_steps,
            eps=eps,
        )
        if count == 0:
            if deferral_confidence_threshold is not None:
                if deferral_decisions is not None:
                    deferral_decisions.append(
                        {
                            "region": region.index,
                            "action": "natural_zero_quota",
                            "scheduled_quota": 0,
                            "minimum_scheduled_raw_top1_probability": None,
                            "consecutive_deferrals": region_deferral_counts.get(
                                region.index, 0
                            ),
                        }
                    )
            continue
        count = min(count, int(masked_relative.numel()))
        candidate_absolute = masked_relative + prompt_length
        scores = full_confidence[0, candidate_absolute]
        candidate_predictions = full_predictions[0, candidate_absolute]
        stop_test_predictions = (
            full_raw_top1_predictions[0, candidate_absolute]
            if stop_protection_mode == "defer"
            else candidate_predictions
        )
        protected = (
            stop_protection_mode is not None
            and region.index in (stop_protected_regions or set())
        )
        stop_candidate_mask = torch.zeros_like(
            candidate_predictions, dtype=torch.bool
        )
        if protected:
            for stop_id in stop_token_ids or set():
                stop_candidate_mask |= stop_test_predictions == stop_id
                if stop_protection_mode == "defer":
                    # A low-temperature sample can still differ from raw
                    # top-1. Never let such a sampled stop bypass protection.
                    stop_candidate_mask |= candidate_predictions == stop_id
        low_confidence_candidate_mask = torch.zeros_like(
            candidate_predictions, dtype=torch.bool
        )
        if protected and stop_protection_mode == "filter":
            candidate_probabilities = full_raw_proposed_probability[
                0, candidate_absolute
            ]
            low_confidence_candidate_mask = (
                candidate_probabilities < stop_filter_confidence_threshold
            )

        selection_pool = torch.arange(
            masked_relative.numel(), device=tokens.device
        )
        if protected and stop_protection_mode == "filter":
            selection_pool = selection_pool[
                ~stop_candidate_mask & ~low_confidence_candidate_mask
            ]
        selected_count = min(count, int(selection_pool.numel()))
        if selected_count > 0:
            pool_scores = scores[selection_pool]
            if alg_temp is None or alg_temp == 0:
                chosen_in_pool = torch.topk(
                    pool_scores, k=selected_count
                ).indices
            else:
                weights = F.softmax(pool_scores / alg_temp, dim=-1)
                chosen_in_pool = torch.multinomial(
                    weights, num_samples=selected_count
                )
            chosen_local = selection_pool[chosen_in_pool]
            selected_relative = masked_relative[chosen_local]
        else:
            chosen_local = selection_pool
            selected_relative = masked_relative[selection_pool]

        if protected and stop_protection_mode == "defer":
            selected_has_stop = bool(
                stop_candidate_mask[chosen_local].any().item()
            )
            if selected_has_stop:
                if stop_protection_decisions is not None:
                    stop_protection_decisions.append(
                        {
                            "region": region.index,
                            "action": "stop_deferred",
                            "scheduled_quota": count,
                            "committed_quota": 0,
                            "stop_candidates": int(
                                stop_candidate_mask.sum().item()
                            ),
                            "hold_schedule": True,
                        }
                    )
                continue

        if protected and stop_protection_mode == "filter" and bool(
            stop_candidate_mask.any().item()
            or low_confidence_candidate_mask.any().item()
        ):
            hold_schedule = selected_count < count
            if stop_protection_decisions is not None:
                stop_protection_decisions.append(
                    {
                        "region": region.index,
                        "action": (
                            "stop_filtered"
                            if not hold_schedule
                            else (
                                "stop_filtered_partial"
                                if selected_count > 0
                                else "stop_filtered_empty"
                            )
                        ),
                        "scheduled_quota": count,
                        "committed_quota": selected_count,
                        "stop_candidates": int(
                            stop_candidate_mask.sum().item()
                        ),
                        "low_confidence_candidates": int(
                            low_confidence_candidate_mask.sum().item()
                        ),
                        "confidence_threshold": (
                            stop_filter_confidence_threshold
                        ),
                        "hold_schedule": hold_schedule,
                    }
                )
            if selected_count == 0:
                continue

        if deferral_confidence_threshold is not None:
            selected_absolute = selected_relative + prompt_length
            minimum_probability = float(
                full_raw_top1_probability[0, selected_absolute].min().item()
            )
            previous_deferrals = region_deferral_counts.get(region.index, 0)
            force_reason = (force_region_reasons or {}).get(region.index)
            if (
                minimum_probability < deferral_confidence_threshold
                and force_reason is None
                and (
                    max_region_deferrals is None
                    or previous_deferrals < max_region_deferrals
                )
            ):
                region_deferral_counts[region.index] = previous_deferrals + 1
                if deferral_decisions is not None:
                    deferral_decisions.append(
                        {
                            "region": region.index,
                            "action": "deferred",
                            "scheduled_quota": int(selected_relative.numel()),
                            "minimum_scheduled_raw_top1_probability": (
                                minimum_probability
                            ),
                            "consecutive_deferrals": previous_deferrals + 1,
                        }
                    )
                continue
            forced = minimum_probability < deferral_confidence_threshold
            region_deferral_counts[region.index] = 0
            if deferral_decisions is not None:
                deferral_decisions.append(
                    {
                        "region": region.index,
                        "action": (
                            f"{force_reason}_forced"
                            if forced and force_reason is not None
                            else ("forced" if forced else "threshold_pass")
                        ),
                        "scheduled_quota": int(selected_relative.numel()),
                        "minimum_scheduled_raw_top1_probability": (
                            minimum_probability
                        ),
                        "consecutive_deferrals": 0,
                    }
                )
        selected_absolute = selected_relative + prompt_length
        tokens[0, selected_absolute] = full_predictions[0, selected_absolute]
        committed[region.index] = [
            int(position.item()) for position in selected_relative
        ]
    return committed


def _dawn_greedy_independent_set(
    edge_mask: torch.Tensor,
    node_mask: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    """DAWN's confidence-ordered greedy conflict-independent selection.

    This follows the intended MIS procedure in the public DAWN implementation,
    but stops once no eligible node remains.  The released helper loops for the
    original candidate count even after neighbors have been suppressed, which
    can select ``-inf`` non-candidates and even a prompt position.
    """
    available = node_mask.clone()
    conflict = edge_mask | edge_mask.T
    selected: list[int] = []
    while bool(available.any()):
        scores = torch.where(available, confidence, -torch.inf)
        best = int(torch.argmax(scores).item())
        selected.append(best)
        available[best] = False
        available[conflict[best]] = False
    return torch.tensor(selected, dtype=torch.long, device=node_mask.device)


def dawn_region_transfer_mask(
    attention: torch.Tensor,
    mask_index: torch.Tensor,
    confidence: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    sink_threshold: float,
    edge_threshold: float,
    high_confidence_threshold: float,
    induce_threshold: float,
    candidate_confidence_threshold: float,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    """Apply DAWN's anchor/conflict selector to one regional candidate set.

    Attention and anchors remain full-canvas.  Only the positions eligible for
    transfer are restricted to ``candidate_mask``; therefore this changes the
    commitment decision without changing model visibility.
    """
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError("attention must be a square token-token matrix")
    if not (
        attention.shape[0]
        == mask_index.numel()
        == confidence.numel()
        == candidate_mask.numel()
    ):
        raise ValueError("DAWN selector tensors must cover the same canvas")
    for name, value in (
        ("sink_threshold", sink_threshold),
        ("edge_threshold", edge_threshold),
        ("high_confidence_threshold", high_confidence_threshold),
        ("induce_threshold", induce_threshold),
        ("candidate_confidence_threshold", candidate_confidence_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    scores = attention.float().clone()
    sink_mask = scores.mean(dim=0) > sink_threshold
    scores.masked_fill_(sink_mask.unsqueeze(0), 0.0)
    scores.diagonal().zero_()
    dependency = (scores >= edge_threshold).T

    transfer_confident = candidate_mask & (
        confidence >= high_confidence_threshold
    )

    # Match DAWN's current-forward decoded anchors.  Prompt and already
    # revealed generated tokens may act as anchors, but masked tokens cannot.
    decoded_anchor = (~mask_index & (
        confidence >= high_confidence_threshold
    )).unsqueeze(-1)
    decoded_edge = dependency & decoded_anchor
    dependent_nodes = decoded_edge.any(dim=0) & candidate_mask
    transfer_induced = dependent_nodes & (confidence >= induce_threshold)

    adjacent_transfer = (
        dependency
        & transfer_induced.unsqueeze(-1)
        & transfer_confident.unsqueeze(-1)
    ).any(dim=0)
    conflict_candidates = (
        candidate_mask
        & (confidence >= candidate_confidence_threshold)
        & (confidence < high_confidence_threshold)
        & ~transfer_induced
        & ~adjacent_transfer
    )
    candidate_pairs = (
        conflict_candidates.unsqueeze(1) & conflict_candidates.unsqueeze(0)
    )
    selected_mis = _dawn_greedy_independent_set(
        candidate_pairs & dependency,
        conflict_candidates,
        confidence,
    )
    transfer_conflict = torch.zeros_like(candidate_mask)
    if selected_mis.numel():
        transfer_conflict[selected_mis] = True

    transfer = transfer_confident | transfer_induced | transfer_conflict
    used_fallback = False
    if not bool(transfer.any()) and bool(candidate_mask.any()):
        fallback_scores = torch.where(candidate_mask, confidence, -torch.inf)
        transfer[int(torch.argmax(fallback_scores).item())] = True
        used_fallback = True
    return transfer, {
        "confident": int(transfer_confident.sum().item()),
        "induced": int(transfer_induced.sum().item()),
        "conflict_mis": int(transfer_conflict.sum().item()),
        "fallback": used_fallback,
        "selected": int(transfer.sum().item()),
    }


def commit_active_regions_dawn(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    attention: torch.Tensor,
    *,
    prompt_length: int,
    mask_token_id: int,
    regions: list[Region],
    temperature: float,
    top_p: float | None,
    top_k: int | None,
    dawn_config: dict[str, Any],
    deferral_confidence_threshold: float | None = None,
    region_deferral_counts: dict[int, int] | None = None,
    deferral_decisions: list[dict[str, Any]] | None = None,
    force_region_reasons: dict[int, str] | None = None,
    selector_stats: list[dict[str, Any]] | None = None,
) -> dict[int, list[int]]:
    """Run the official DAWN selection rule independently inside each region."""
    if tokens.shape[0] != 1:
        raise ValueError("Regional DAWN pilot currently requires batch size one")
    if attention.shape != (1, tokens.shape[1], tokens.shape[1]):
        raise ValueError("DAWN attention must have shape [1, sequence, sequence]")
    if deferral_confidence_threshold is not None and region_deferral_counts is None:
        raise ValueError("region_deferral_counts is required when deferral is enabled")

    # Preserve the experiment's Dream prediction sampler, but evaluate DAWN's
    # released thresholds on raw, untempered model probabilities.  DAWN's
    # released Dream commands use temperature=0 and no top-p filtering; using
    # this pilot's temperature=0.1 probabilities would divide logits by 0.1,
    # make almost every distribution artificially sharp, and collapse the
    # regional decoder to a handful of forwards.
    _, predictions = sample_tokens(
        logits[0],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        policy="maskgit_plus",
    )
    raw_logits = logits[0].float()
    raw_maximum = raw_logits.max(dim=-1).values
    confidence = (
        raw_maximum - torch.logsumexp(raw_logits, dim=-1)
    ).exp()
    mask_index = tokens[0] == mask_token_id
    committed: dict[int, list[int]] = {region.index: [] for region in regions}

    for region in regions:
        candidate_mask = torch.zeros_like(mask_index)
        relative = torch.tensor(
            region.remaining_mask_indices,
            dtype=torch.long,
            device=tokens.device,
        )
        if relative.numel() == 0:
            continue
        candidate_mask[relative + prompt_length] = True
        transfer, stats = dawn_region_transfer_mask(
            attention[0],
            mask_index,
            confidence,
            candidate_mask,
            sink_threshold=float(dawn_config.get("sink_threshold", 0.03)),
            edge_threshold=float(dawn_config.get("edge_threshold", 0.10)),
            high_confidence_threshold=float(
                dawn_config.get("high_confidence_threshold", 0.90)
            ),
            induce_threshold=float(dawn_config.get("induce_threshold", 0.75)),
            candidate_confidence_threshold=float(
                dawn_config.get("candidate_confidence_threshold", 0.80)
            ),
        )
        selected_absolute = torch.nonzero(transfer, as_tuple=True)[0]
        selected_relative = selected_absolute - prompt_length

        if deferral_confidence_threshold is not None:
            selected_logits = logits[0, selected_absolute].float()
            raw_top = selected_logits.max(dim=-1).values
            raw_probability = (
                raw_top - torch.logsumexp(selected_logits, dim=-1)
            ).exp()
            minimum_probability = float(raw_probability.min().item())
            force_reason = (force_region_reasons or {}).get(region.index)
            if (
                minimum_probability < deferral_confidence_threshold
                and force_reason is None
            ):
                region_deferral_counts[region.index] = (
                    region_deferral_counts.get(region.index, 0) + 1
                )
                if deferral_decisions is not None:
                    deferral_decisions.append(
                        {
                            "region": region.index,
                            "action": "deferred",
                            "scheduled_quota": int(selected_relative.numel()),
                            "minimum_scheduled_raw_top1_probability": (
                                minimum_probability
                            ),
                            "consecutive_deferrals": region_deferral_counts[
                                region.index
                            ],
                        }
                    )
                stats["deferred"] = True
                if selector_stats is not None:
                    selector_stats.append({"region": region.index, **stats})
                continue
            forced = minimum_probability < deferral_confidence_threshold
            region_deferral_counts[region.index] = 0
            if deferral_decisions is not None:
                deferral_decisions.append(
                    {
                        "region": region.index,
                        "action": (
                            f"{force_reason}_forced"
                            if forced and force_reason is not None
                            else ("forced" if forced else "threshold_pass")
                        ),
                        "scheduled_quota": int(selected_relative.numel()),
                        "minimum_scheduled_raw_top1_probability": (
                            minimum_probability
                        ),
                        "consecutive_deferrals": 0,
                    }
                )

        tokens[0, selected_absolute] = predictions[selected_absolute]
        committed[region.index] = [
            int(position.item()) for position in selected_relative
        ]
        stats["deferred"] = False
        if selector_stats is not None:
            selector_stats.append({"region": region.index, **stats})
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
