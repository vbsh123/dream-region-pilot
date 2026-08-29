from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .commit import (
    commit_active_regions,
    commit_active_regions_dawn,
    describe_commitments,
)
from .diagnostics import ExampleDiagnostics
from .mean_field import mean_field_commit_indices
from .model_adapter import DreamModelAdapter
from .regions import build_fixed_regions, refresh_remaining_masks
from .scheduler import RegionScheduler, parse_strategy


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mask_token_id(model, tokenizer) -> int:
    if isinstance(getattr(tokenizer, "mask_token_id", None), int):
        return int(tokenizer.mask_token_id)
    generation_config = getattr(model, "generation_config", None)
    value = getattr(generation_config, "mask_token_id", None)
    if isinstance(value, int):
        return value
    value = getattr(model.config, "mask_token_id", None)
    if isinstance(value, int):
        return value
    raise ValueError("Dream model/tokenizer does not define mask_token_id")


def stop_token_ids(tokenizer) -> set[int]:
    result = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        result.add(eos)
    vocabulary = tokenizer.get_vocab()
    for marker in ("<|eot_id|>", "<|im_end|>"):
        if marker in vocabulary:
            result.add(int(vocabulary[marker]))
    return result


def regional_approximation_name(mode: str) -> str:
    names = {
        "always_on_dawn_tail_guard": "regional_dawn_selector_plus_tail_guard",
        "always_on_coupled_defer_dawn_tail_guard": (
            "regional_dawn_selector_plus_coupled_startup_deferral_and_tail_guard"
        ),
        "flowblock_proxy": "flowblock_admission_proxy_plus_per_region_linear_time",
        "loose_wavefront": "loose_confidence_admission_plus_per_region_linear_time",
        "controlled_position": "positional_bounded_progress_skew",
        "controlled_position_tail_guard": (
            "predicted_terminal_region_guard_plus_positional_bounded_progress_skew"
        ),
        "always_on_tail_guard": (
            "predicted_terminal_region_guard_plus_always_on_regions"
        ),
        "always_on_bounded_defer": "bounded_confidence_deferral",
        "always_on_bounded_defer_tail_guard": (
            "bounded_confidence_deferral_plus_predicted_terminal_region_guard"
        ),
        "always_on_coupled_defer": (
            "confidence_deferral_plus_positional_bounded_staleness"
        ),
        "always_on_coupled_defer_tail_guard": (
            "confidence_deferral_plus_positional_bounded_staleness_plus_"
            "predicted_terminal_region_guard"
        ),
        "always_on_coupled_defer_stop_filter": (
            "confidence_deferral_plus_positional_bounded_staleness_plus_"
            "prefix_conditioned_stop_filtering"
        ),
        "always_on_coupled_defer_stop_defer": (
            "confidence_deferral_plus_positional_bounded_staleness_plus_"
            "prefix_conditioned_stop_deferral"
        ),
    }
    return names.get(mode, "per_region_linear_time")


def decode_response(
    tokenizer,
    response: torch.Tensor,
    until: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, int]:
    values = [int(value) for value in response.detach().cpu().tolist()]
    stops = stop_token_ids(tokenizer)
    stop = next(
        (index for index, token_id in enumerate(values) if token_id in stops),
        len(values),
    )
    text = tokenizer.decode(values[:stop], skip_special_tokens=True)
    for marker in until or ():
        text = text.split(marker, 1)[0]
    return text.strip(), stop


def region_readiness_from_logits(
    logits: torch.Tensor,
    *,
    prompt_length: int,
    regions,
    confidence_threshold: float,
) -> dict[int, float]:
    """FlowBlock-style acceptance ratio on each region's remaining masks."""
    response_logits = logits[0, prompt_length:].float()
    maximum = response_logits.max(dim=-1).values
    maximum_probability = (maximum - torch.logsumexp(response_logits, dim=-1)).exp()
    readiness: dict[int, float] = {}
    for region in regions:
        if region.done:
            readiness[region.index] = 1.0
            continue
        indices = torch.tensor(
            region.remaining_mask_indices,
            dtype=torch.long,
            device=response_logits.device,
        )
        readiness[region.index] = float(
            (maximum_probability.index_select(0, indices) >= confidence_threshold)
            .float()
            .mean()
            .item()
        )
    return readiness


def predicted_tail_region(
    logits: torch.Tensor,
    *,
    prompt_length: int,
    response_mask: torch.Tensor,
    regions,
    stop_ids: set[int],
) -> int | None:
    """Return the region containing the earliest masked top-1 stop prediction."""
    if not stop_ids:
        return None
    predictions = logits[0, prompt_length:].argmax(dim=-1)
    is_stop = torch.zeros_like(predictions, dtype=torch.bool)
    for token_id in stop_ids:
        is_stop |= predictions == token_id
    candidates = torch.nonzero(response_mask[0] & is_stop, as_tuple=False)
    if candidates.numel() == 0:
        return None
    position = int(candidates[0, 0].item())
    for region in regions:
        if region.token_indices[0] <= position <= region.token_indices[-1]:
            return region.index
    raise RuntimeError(f"Predicted tail position {position} is outside all regions")


@torch.no_grad()
def decode_vanilla(
    model,
    tokenizer,
    prompt: torch.Tensor,
    generation: dict[str, Any],
) -> dict[str, Any]:
    device = prompt.device
    synchronize(device)
    started = time.perf_counter()
    attention_mask = prompt.ne(tokenizer.pad_token_id)
    generation_kwargs = {
        "attention_mask": attention_mask,
        "max_new_tokens": int(generation["max_new_tokens"]),
        "steps": int(generation["steps"]),
        "alg": str(generation["commit_policy"]),
        "alg_temp": generation.get("alg_temp"),
        "temperature": float(generation.get("temperature", 0.0)),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "return_dict_in_generate": True,
    }
    is_dawn_release = model.__class__.__module__ == "model.modeling_dream"
    if is_dawn_release:
        # DAWN's released Original baseline relies on this fork's sequential
        # block sampler and its default 32-token blocks. Keep it explicit so a
        # future upstream default change cannot silently alter the protocol.
        generation_kwargs["block_length"] = 32
    output = model.diffusion_generate(prompt, **generation_kwargs)
    synchronize(device)
    elapsed = time.perf_counter() - started

    reported_nfe = None
    if isinstance(output, tuple) and len(output) == 2:
        output, reported_nfe = output
    sequences = output.sequences if hasattr(output, "sequences") else output
    nfe = int(reported_nfe) if reported_nfe is not None else int(generation["steps"])
    response = sequences[0, prompt.shape[1] :]
    text, effective_tokens = decode_response(
        tokenizer, response, generation.get("until")
    )
    canvas_tokens = int(response.numel())
    canvas_tokens_per_second = canvas_tokens / elapsed if elapsed > 0 else None
    effective_tokens_per_second = (
        effective_tokens / elapsed if elapsed > 0 else None
    )
    return {
        "generation": text,
        "response_token_ids": [int(value) for value in response.detach().cpu().tolist()],
        "nfe": nfe,
        "configured_steps": int(generation["steps"]),
        "global_forward_passes": nfe,
        "average_tokens_committed_per_forward": canvas_tokens / nfe,
        "canvas_tokens": canvas_tokens,
        "effective_generated_tokens": effective_tokens,
        "wall_clock_seconds": elapsed,
        "canvas_tokens_per_second": canvas_tokens_per_second,
        "effective_tokens_per_second": effective_tokens_per_second,
        "forward_passes_per_second": nfe / elapsed if elapsed > 0 else None,
        "mean_field_seconds": 0.0,
        "diagnostic_seconds_excluded_from_wall_clock": 0.0,
        "schedule_approximation": False,
        "decoder_implementation": (
            "dawn_release_original_sequential_blocks"
            if is_dawn_release
            else "huggingface_dream_global"
        ),
    }


@torch.no_grad()
def decode_official_dawn(
    model,
    tokenizer,
    prompt: torch.Tensor,
    generation: dict[str, Any],
    dawn_config: dict[str, Any],
) -> dict[str, Any]:
    """Run DAWN's released sequential-block Dream decoder unchanged.

    The dataset prompt, checkpoint, scoring, seed policy, GPU, and outer timing
    harness remain those of this pilot. Decoder settings default to DAWN's
    released Dream GSM8K command rather than the regional hybrid settings.
    """
    device = prompt.device
    block_length = int(dawn_config.get("block_length", 32))
    generation_length = int(generation["max_new_tokens"])
    if generation_length % block_length:
        raise ValueError(
            "Official DAWN requires max_new_tokens divisible by block_length"
        )
    configured_steps = generation_length // block_length
    attention_mask = prompt.ne(tokenizer.pad_token_id)
    synchronize(device)
    started = time.perf_counter()
    output = model.diffusion_generate(
        prompt,
        attention_mask=attention_mask,
        max_new_tokens=generation_length,
        steps=configured_steps,
        alg="dawn",
        temperature=float(dawn_config.get("official_temperature", 0.0)),
        top_p=dawn_config.get("official_top_p"),
        top_k=dawn_config.get("official_top_k"),
        block_length=block_length,
        conf_threshold=float(
            dawn_config.get("candidate_confidence_threshold", 0.80)
        ),
        tau_induce=float(dawn_config.get("induce_threshold", 0.75)),
        tau_sink=float(dawn_config.get("sink_threshold", 0.03)),
        tau_edge=float(dawn_config.get("edge_threshold", 0.10)),
        return_dict_in_generate=True,
    )
    synchronize(device)
    elapsed = time.perf_counter() - started

    reported_nfe = None
    if isinstance(output, tuple) and len(output) == 2:
        output, reported_nfe = output
    if reported_nfe is None:
        raise RuntimeError("Official DAWN decoder did not report its actual NFE")
    sequences = output.sequences if hasattr(output, "sequences") else output
    response = sequences[0, prompt.shape[1] :]
    text, effective_tokens = decode_response(
        tokenizer, response, generation.get("until")
    )
    canvas_tokens = int(response.numel())
    nfe = int(reported_nfe)
    return {
        "generation": text,
        "response_token_ids": [
            int(value) for value in response.detach().cpu().tolist()
        ],
        "nfe": nfe,
        "configured_steps": configured_steps,
        "global_forward_passes": nfe,
        "average_tokens_committed_per_forward": canvas_tokens / nfe,
        "canvas_tokens": canvas_tokens,
        "effective_generated_tokens": effective_tokens,
        "wall_clock_seconds": elapsed,
        "canvas_tokens_per_second": (
            canvas_tokens / elapsed if elapsed > 0 else None
        ),
        "effective_tokens_per_second": (
            effective_tokens / elapsed if elapsed > 0 else None
        ),
        "forward_passes_per_second": nfe / elapsed if elapsed > 0 else None,
        "mean_field_seconds": 0.0,
        "dawn_selection_seconds": 0.0,
        "diagnostic_seconds_excluded_from_wall_clock": 0.0,
        "official_dawn": True,
        "dawn_config": {
            "block_length": block_length,
            "temperature": float(
                dawn_config.get("official_temperature", 0.0)
            ),
            "top_p": dawn_config.get("official_top_p"),
            "top_k": dawn_config.get("official_top_k"),
            "sink_threshold": float(dawn_config.get("sink_threshold", 0.03)),
            "edge_threshold": float(dawn_config.get("edge_threshold", 0.10)),
            "induce_threshold": float(
                dawn_config.get("induce_threshold", 0.75)
            ),
            "candidate_confidence_threshold": float(
                dawn_config.get("candidate_confidence_threshold", 0.80)
            ),
        },
        "schedule_approximation": False,
        "schedule_approximation_name": "official_dawn_sequential_blocks",
    }


@torch.no_grad()
def decode_mean_field_repro(
    model,
    tokenizer,
    prompt: torch.Tensor,
    *,
    generation: dict[str, Any],
    adapter: DreamModelAdapter,
    mean_field: dict[str, Any],
) -> dict[str, Any]:
    """Paper-faithful Algorithm 1 reproduction over sequential active blocks.

    The paper's linked implementation is unavailable (GitHub returns 404), so
    this is deliberately named a reproduction rather than an official run.
    """
    device = prompt.device
    prompt_length = prompt.shape[1]
    generation_length = int(generation["max_new_tokens"])
    block_size = int(mean_field.get("block_size", 32))
    threshold = float(mean_field.get("threshold", 0.9))
    iterations = int(mean_field.get("iterations", 2))
    pair_chunk_size = int(mean_field.get("pair_chunk_size", 16))
    if block_size <= 0:
        raise ValueError("mean_field_baseline.block_size must be positive")

    mask_id = mask_token_id(model, tokenizer)
    tokens = F.pad(prompt, (0, generation_length), value=mask_id)
    blocks = build_fixed_regions(generation_length, block_size)
    nfe = 0
    fallback_count = 0
    committed_per_forward: list[int] = []
    selection_seconds = 0.0

    synchronize(device)
    started = time.perf_counter()
    for block in blocks:
        relative = torch.tensor(
            block.token_indices, dtype=torch.long, device=device
        )
        while bool((tokens[0, prompt_length + relative] == mask_id).any()):
            logits = adapter.forward_logits(model, tokens)
            nfe += 1
            masked_relative = relative[
                tokens[0, prompt_length + relative] == mask_id
            ]
            active_logits = logits[0, prompt_length + masked_relative]
            synchronize(device)
            selection_started = time.perf_counter()
            selected_rows, _, used_fallback = mean_field_commit_indices(
                active_logits,
                threshold=threshold,
                iterations=iterations,
                pair_chunk_size=pair_chunk_size,
            )
            synchronize(device)
            selection_seconds += time.perf_counter() - selection_started
            chosen_relative = masked_relative.index_select(0, selected_rows)
            predictions = active_logits.argmax(dim=-1).index_select(
                0, selected_rows
            )
            tokens[0, prompt_length + chosen_relative] = predictions
            count = int(selected_rows.numel())
            committed_per_forward.append(count)
            fallback_count += int(used_fallback)
            if nfe > generation_length:
                raise RuntimeError(
                    "Mean-Field reproduction exceeded one forced-progress "
                    "forward per canvas token"
                )

    synchronize(device)
    elapsed = time.perf_counter() - started
    response = tokens[0, prompt_length:]
    text, effective_tokens = decode_response(
        tokenizer, response, generation.get("until")
    )
    canvas_tokens = int(response.numel())
    return {
        "generation": text,
        "response_token_ids": [
            int(value) for value in response.detach().cpu().tolist()
        ],
        "nfe": nfe,
        "global_forward_passes": nfe,
        "average_tokens_committed_per_forward": canvas_tokens / nfe,
        "tokens_committed_per_forward": committed_per_forward,
        "canvas_tokens": canvas_tokens,
        "effective_generated_tokens": effective_tokens,
        "wall_clock_seconds": elapsed,
        "canvas_tokens_per_second": canvas_tokens / elapsed if elapsed > 0 else None,
        "effective_tokens_per_second": (
            effective_tokens / elapsed if elapsed > 0 else None
        ),
        "forward_passes_per_second": nfe / elapsed if elapsed > 0 else None,
        "mean_field_seconds": selection_seconds,
        "diagnostic_seconds_excluded_from_wall_clock": 0.0,
        "mean_field_threshold": threshold,
        "mean_field_iterations": iterations,
        "mean_field_block_size": block_size,
        "mean_field_exact_jsd": True,
        "mean_field_forced_progress_events": fallback_count,
        "schedule_approximation": True,
        "schedule_approximation_name": (
            "mean_field_algorithm1_reproduction_sequential_blocks"
        ),
    }


@torch.no_grad()
def decode_regional(
    model,
    tokenizer,
    prompt: torch.Tensor,
    *,
    strategy: str,
    generation: dict[str, Any],
    adapter: DreamModelAdapter,
    diagnostics_dir: Path | None,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = parse_strategy(strategy)
    device = prompt.device
    prompt_length = prompt.shape[1]
    generation_length = int(generation["max_new_tokens"])
    region_size = int(generation["region_size"])
    local_steps = int(generation.get("local_steps") or region_size)
    if local_steps <= 0:
        raise ValueError("local_steps must be positive")
    mask_id = mask_token_id(model, tokenizer)
    termination_ids = stop_token_ids(tokenizer)
    tokens = F.pad(prompt, (0, generation_length), value=mask_id)
    regions = build_fixed_regions(generation_length, region_size)
    refresh_remaining_masks(regions, [True] * generation_length)
    probe = probe or {}
    uses_dawn_selector = mode in {
        "always_on_dawn_tail_guard",
        "always_on_coupled_defer_dawn_tail_guard",
    }
    # DAWN guarantees at least one transfer for every scheduled nonempty
    # region.  Let a region consume up to one scheduling point per position so
    # region-size ablations cannot exhaust a shorter Dream quota prematurely.
    scheduler_local_steps = (
        max(local_steps, region_size) if uses_dawn_selector else local_steps
    )
    dawn_config = probe.get("dawn", {})
    strategy_probe = probe
    if mode == "flowblock_proxy":
        flowblock_proxy = probe.get("flowblock_proxy", {})
        strategy_probe = {
            **probe,
            "max_active_regions": int(
                flowblock_proxy.get("max_active_regions", 2)
            ),
            "spawn_readiness": float(
                flowblock_proxy.get("spawn_readiness", 0.60)
            ),
            "readiness_confidence_threshold": float(
                flowblock_proxy.get("readiness_confidence_threshold", 0.50)
            ),
        }
    scheduler = RegionScheduler(
        regions,
        mode=mode,
        max_active_regions=int(
            strategy_probe.get("max_active_regions", len(regions))
        ),
        spawn_readiness=float(strategy_probe.get("spawn_readiness", 0.15)),
        max_progress_gap=int(strategy_probe.get("max_progress_gap", 8)),
    )
    initial_admitted_region_count = scheduler.admitted_count
    diagnostics = (
        ExampleDiagnostics(diagnostics_dir)
        if diagnostics_dir is not None
        else None
    )

    nfe = 0
    diagnostic_seconds = 0.0
    tokens_committed_per_forward: list[int] = []
    admission_events: list[dict[str, Any]] = []
    control_timeline: list[dict[str, Any]] = []
    tail_guard_timeline: list[dict[str, Any]] = []
    deferral_timeline: list[dict[str, Any]] = []
    stop_protection_timeline: list[dict[str, Any]] = []
    bounded_deferral_modes = {
        "always_on_bounded_defer",
        "always_on_bounded_defer_tail_guard",
    }
    coupled_deferral_modes = {
        "always_on_coupled_defer",
        "always_on_coupled_defer_tail_guard",
        "always_on_coupled_defer_stop_filter",
        "always_on_coupled_defer_stop_defer",
        "always_on_coupled_defer_dawn_tail_guard",
    }
    stop_protection_modes = {
        "always_on_coupled_defer_stop_filter": "filter",
        "always_on_coupled_defer_stop_defer": "defer",
    }
    deferral_modes = bounded_deferral_modes | coupled_deferral_modes
    uses_bounded_deferral = mode in deferral_modes
    deferral_confidence_threshold = float(
        strategy_probe.get("deferral_confidence_threshold", 0.50)
    )
    max_region_deferrals = int(
        strategy_probe.get("max_region_deferrals", 4)
    )
    max_global_deferral_iterations = int(
        strategy_probe.get("max_global_deferral_iterations", 4)
    )
    configured_deferral_cutoff = strategy_probe.get(
        "deferral_until_revealed_tokens"
    )
    deferral_until_revealed_tokens = (
        int(configured_deferral_cutoff)
        if configured_deferral_cutoff is not None
        else None
    )
    if not 0.0 <= deferral_confidence_threshold <= 1.0:
        raise ValueError("probe.deferral_confidence_threshold must be in [0, 1]")
    if max_region_deferrals < 0:
        raise ValueError("probe.max_region_deferrals must be non-negative")
    if max_global_deferral_iterations <= 0:
        raise ValueError(
            "probe.max_global_deferral_iterations must be positive"
        )
    if (
        deferral_until_revealed_tokens is not None
        and deferral_until_revealed_tokens < 0
    ):
        raise ValueError(
            "probe.deferral_until_revealed_tokens must be non-negative"
        )
    region_deferral_counts = {region.index: 0 for region in regions}
    global_empty_deferral_streak = 0
    dawn_selection_seconds = 0.0
    dawn_selector_timeline: list[dict[str, Any]] = []
    stop_stalled_regions: set[int] = set()
    accepted_stop_position: int | None = None
    accepted_stop_iteration: int | None = None
    stop_filter_confidence_threshold = float(
        strategy_probe.get("stop_filter_confidence_threshold", 0.4)
    )
    if not 0.0 <= stop_filter_confidence_threshold <= 1.0:
        raise ValueError(
            "probe.stop_filter_confidence_threshold must be in [0, 1]"
        )

    synchronize(device)
    started = time.perf_counter()
    maximum_iterations = scheduler_local_steps * (len(regions) + 2)
    if mode in coupled_deferral_modes:
        maximum_iterations *= max_global_deferral_iterations + 1
    while True:
        response_mask = tokens[:, prompt_length:] == mask_id
        relevant_mask = (
            response_mask
            if accepted_stop_position is None
            else response_mask[:, :accepted_stop_position]
        )
        if not bool(relevant_mask.any()):
            break
        dawn_attention = None
        if uses_dawn_selector:
            logits, dawn_attention = adapter.forward_with_dawn_attention(
                model, tokens
            )
        else:
            logits = adapter.forward_logits(model, tokens)
        nfe += 1

        readiness_by_region = (
            region_readiness_from_logits(
                logits,
                prompt_length=prompt_length,
                regions=regions,
                confidence_threshold=float(
                    strategy_probe.get("readiness_confidence_threshold", 0.5)
                ),
            )
            if scheduler.is_wavefront
            else {}
        )

        provisional_tail = None
        guarded_tail = None
        tail_detection_modes = {
            "always_on_tail_guard",
            "always_on_dawn_tail_guard",
            "always_on_bounded_defer_tail_guard",
            "always_on_coupled_defer_tail_guard",
            "always_on_coupled_defer_dawn_tail_guard",
            "controlled_position_tail_guard",
        } | set(stop_protection_modes)
        if mode in tail_detection_modes:
            provisional_tail = predicted_tail_region(
                logits,
                prompt_length=prompt_length,
                response_mask=response_mask,
                regions=regions,
                stop_ids=termination_ids,
            )
            if provisional_tail is not None and any(
                not region.done for region in regions[:provisional_tail]
            ):
                guarded_tail = provisional_tail
        progress_tokens_before = {
            region.index: scheduler.revealed_tokens(region) for region in regions
        }
        gap_exempt_children_this_iteration = set(stop_stalled_regions)
        max_region_exclusive = guarded_tail
        if mode in stop_protection_modes and guarded_tail is not None:
            # Allow only the predicted terminal region to make protected
            # progress. Everything strictly after it remains excluded exactly
            # as in the coarse tail guard.
            max_region_exclusive = guarded_tail + 1
        if accepted_stop_position is not None:
            endpoint_region_exclusive = next(
                region.index + 1
                for region in regions
                if accepted_stop_position in region.token_indices
            )
            max_region_exclusive = (
                endpoint_region_exclusive
                if max_region_exclusive is None
                else min(max_region_exclusive, endpoint_region_exclusive)
            )
        active = scheduler.regions_allowed_to_advance(
            scheduler_local_steps,
            max_region_exclusive=max_region_exclusive,
            progress_gap_exempt_children=gap_exempt_children_this_iteration,
        )
        if not active:
            clocks = {region.index: region.clock for region in regions}
            raise RuntimeError(
                "Regional scheduler deadlocked with masks remaining; "
                f"strategy={strategy}, clocks={clocks}"
            )
        deferral_decisions: list[dict[str, Any]] = []
        force_region_reasons: dict[int, str] = {}
        if mode in coupled_deferral_modes:
            force_region_reasons.update(
                {
                    region_index: "gap"
                    for region_index in scheduler.last_urgent_regions
                }
            )
            if deferral_until_revealed_tokens is not None:
                for region in active:
                    if (
                        scheduler.revealed_tokens(region)
                        >= deferral_until_revealed_tokens
                    ):
                        force_region_reasons.setdefault(
                            region.index, "deferral_window_closed"
                        )
            if (
                global_empty_deferral_streak
                >= max_global_deferral_iterations
                and not force_region_reasons
            ):
                force_region_reasons[active[0].index] = "global_deadlock"
        dawn_selector_stats: list[dict[str, Any]] = []
        stop_protection_decisions: list[dict[str, Any]] = []
        stop_protected_regions: set[int] = set()
        if mode in stop_protection_modes and guarded_tail is not None:
            stop_protected_regions.add(guarded_tail)
        if uses_dawn_selector:
            if dawn_attention is None:
                raise RuntimeError("DAWN selector has no attention matrix")
            synchronize(device)
            dawn_selection_started = time.perf_counter()
            committed = commit_active_regions_dawn(
                tokens,
                logits,
                dawn_attention,
                prompt_length=prompt_length,
                mask_token_id=mask_id,
                regions=active,
                temperature=float(generation.get("temperature", 0.0)),
                top_p=generation.get("top_p"),
                top_k=generation.get("top_k"),
                dawn_config=dawn_config,
                deferral_confidence_threshold=(
                    deferral_confidence_threshold
                    if uses_bounded_deferral
                    else None
                ),
                region_deferral_counts=(
                    region_deferral_counts if uses_bounded_deferral else None
                ),
                deferral_decisions=deferral_decisions,
                force_region_reasons=force_region_reasons,
                selector_stats=dawn_selector_stats,
            )
            synchronize(device)
            dawn_selection_seconds += (
                time.perf_counter() - dawn_selection_started
            )
        else:
            committed = commit_active_regions(
                tokens,
                logits,
                prompt_length=prompt_length,
                mask_token_id=mask_id,
                regions=active,
                local_steps=local_steps,
                eps=float(generation["eps"]),
                temperature=float(generation.get("temperature", 0.0)),
                top_p=generation.get("top_p"),
                top_k=generation.get("top_k"),
                policy=str(generation["commit_policy"]),
                alg_temp=generation.get("alg_temp"),
                deferral_confidence_threshold=(
                    deferral_confidence_threshold
                    if uses_bounded_deferral
                    else None
                ),
                max_region_deferrals=(
                    max_region_deferrals
                    if mode in bounded_deferral_modes
                    else None
                ),
                region_deferral_counts=(
                    region_deferral_counts if uses_bounded_deferral else None
                ),
                deferral_decisions=deferral_decisions,
                force_region_reasons=force_region_reasons,
                stop_protection_mode=stop_protection_modes.get(mode),
                stop_token_ids=termination_ids,
                stop_protected_regions=stop_protected_regions,
                stop_protection_decisions=stop_protection_decisions,
                stop_filter_confidence_threshold=(
                    stop_filter_confidence_threshold
                ),
                max_response_position_exclusive=accepted_stop_position,
            )
        commitment_details: list[dict[str, Any]] = []
        if diagnostics is not None:
            synchronize(device)
            diagnostic_started = time.perf_counter()
            commitment_details = describe_commitments(
                logits,
                tokens,
                prompt_length=prompt_length,
                regions=regions,
                committed=committed,
                temperature=float(generation.get("temperature", 0.0)),
                top_p=generation.get("top_p"),
                top_k=generation.get("top_k"),
                policy=str(generation["commit_policy"]),
            )
            for detail in commitment_details:
                for key in (
                    "token_id",
                    "raw_top1_token_id",
                    "raw_top2_token_id",
                    "sampling_top1_token_id",
                    "sampling_top2_token_id",
                ):
                    detail[key.removesuffix("_id")] = (
                        tokenizer.convert_ids_to_tokens(detail[key])
                    )
            synchronize(device)
            diagnostic_seconds += time.perf_counter() - diagnostic_started
        deferred_regions = {
            int(item["region"])
            for item in deferral_decisions
            if item["action"] == "deferred"
        }
        stop_schedule_held_regions = {
            int(item["region"])
            for item in stop_protection_decisions
            if bool(item["hold_schedule"])
        }
        stop_stalled_regions = set(stop_schedule_held_regions)
        schedule_advanced_regions = [
            region
            for region in active
            if region.index not in deferred_regions
            and region.index not in stop_schedule_held_regions
        ]
        advanced = scheduler.apply_updates(schedule_advanced_regions, committed)
        committed_this_forward = sum(
            len(value) for value in committed.values()
        )
        committed_stop_positions = (
            sorted(
                position
                for positions in committed.values()
                for position in positions
                if int(tokens[0, prompt_length + position].item())
                in termination_ids
            )
            if mode in stop_protection_modes
            else []
        )
        if committed_stop_positions:
            earliest_new_stop = committed_stop_positions[0]
            if (
                accepted_stop_position is None
                or earliest_new_stop < accepted_stop_position
            ):
                accepted_stop_position = earliest_new_stop
                accepted_stop_iteration = nfe
        tokens_committed_per_forward.append(committed_this_forward)
        if mode in coupled_deferral_modes:
            if committed_this_forward == 0 and deferred_regions:
                global_empty_deferral_streak += 1
            else:
                global_empty_deferral_streak = 0
        effective_remaining_mask = tokens[0, prompt_length:] == mask_id
        if accepted_stop_position is not None:
            effective_remaining_mask = effective_remaining_mask.clone()
            effective_remaining_mask[accepted_stop_position:] = False
        refresh_remaining_masks(
            regions, effective_remaining_mask.detach().cpu().tolist()
        )
        newly_admitted = scheduler.maybe_admit_next(readiness_by_region)
        if newly_admitted:
            admission_events.append(
                {
                    "iteration": nfe,
                    "regions": newly_admitted,
                    "frontier_region": newly_admitted[0] - 1,
                    "frontier_readiness": readiness_by_region.get(
                        newly_admitted[0] - 1
                    ),
                }
            )
        control_state = {
            "iteration": nfe,
            "progress_tokens_before": progress_tokens_before,
            "progress_tokens_after": {
                region.index: scheduler.revealed_tokens(region) for region in regions
            },
            "control_edges": [
                list(pair) for pair in sorted(scheduler.last_control_edges)
            ],
            "blocked_regions": sorted(scheduler.last_blocked_regions),
            "urgent_regions": sorted(scheduler.last_urgent_regions),
            "predicted_tail_region": provisional_tail,
            "guarded_tail_region": guarded_tail,
            "deferral_decisions": deferral_decisions,
            "region_deferral_counts": dict(region_deferral_counts),
            "force_region_reasons": force_region_reasons,
            "global_empty_deferral_streak": global_empty_deferral_streak,
            "dawn_selector": dawn_selector_stats,
            "stop_protected_regions": sorted(stop_protected_regions),
            "stop_protection_decisions": stop_protection_decisions,
            "stop_gap_exempt_children": sorted(
                gap_exempt_children_this_iteration
            ),
            "next_stop_gap_exempt_children": sorted(stop_stalled_regions),
            "accepted_stop_position": accepted_stop_position,
            "accepted_stop_iteration": accepted_stop_iteration,
        }
        if scheduler.is_controlled:
            control_timeline.append(control_state)
        if mode in {
            "always_on_tail_guard",
            "always_on_dawn_tail_guard",
            "always_on_bounded_defer_tail_guard",
            "always_on_coupled_defer_tail_guard",
            "always_on_coupled_defer_dawn_tail_guard",
            "controlled_position_tail_guard",
        }:
            tail_guard_timeline.append(
                {
                    "iteration": nfe,
                    "predicted_tail_region": provisional_tail,
                    "guarded_tail_region": guarded_tail,
                }
            )
        if uses_dawn_selector:
            dawn_selector_timeline.append(
                {
                    "iteration": nfe,
                    "regions": dawn_selector_stats,
                }
            )
        if uses_bounded_deferral:
            deferral_timeline.append(
                {
                    "iteration": nfe,
                    "decisions": deferral_decisions,
                    "deferred_regions": sorted(deferred_regions),
                    "forced_regions": sorted(
                        int(item["region"])
                        for item in deferral_decisions
                        if item["action"] == "forced"
                        or str(item["action"]).endswith("_forced")
                    ),
                    "region_deferral_counts": dict(region_deferral_counts),
                    "force_region_reasons": force_region_reasons,
                    "global_empty_deferral_streak": (
                        global_empty_deferral_streak
                    ),
                }
            )
        if mode in stop_protection_modes:
            stop_protection_timeline.append(
                {
                    "iteration": nfe,
                    "mode": stop_protection_modes[mode],
                    "protected_regions": sorted(stop_protected_regions),
                    "decisions": stop_protection_decisions,
                    "schedule_held_regions": sorted(
                        stop_schedule_held_regions
                    ),
                }
            )
        if diagnostics is not None:
            diagnostic_started = time.perf_counter()
            diagnostics.record_iteration(
                iteration=nfe,
                regions=regions,
                scheduled=[region.index for region in active],
                advanced=[region.index for region in advanced],
                committed=committed,
                commitment_details=commitment_details,
                admitted_region_count=scheduler.admitted_count,
                newly_admitted=newly_admitted,
                readiness_by_region=readiness_by_region,
                control_state=control_state,
            )
            diagnostic_seconds += time.perf_counter() - diagnostic_started
        if nfe > maximum_iterations:
            raise RuntimeError(
                f"Regional decoder exceeded safety bound of {maximum_iterations} NFEs"
            )

    synchronize(device)
    elapsed_including_diagnostics = time.perf_counter() - started
    if diagnostics is not None:
        diagnostics.finalize()
    response = tokens[0, prompt_length:]
    text, effective_tokens = decode_response(
        tokenizer, response, generation.get("until")
    )
    canvas_tokens = int(response.numel())
    wall_clock_seconds = elapsed_including_diagnostics - diagnostic_seconds
    return {
        "generation": text,
        "response_token_ids": [int(value) for value in response.detach().cpu().tolist()],
        "nfe": nfe,
        "global_forward_passes": nfe,
        "average_tokens_committed_per_forward": (
            sum(tokens_committed_per_forward) / nfe
        ),
        "tokens_committed_per_forward": tokens_committed_per_forward,
        "canvas_tokens": canvas_tokens,
        "effective_generated_tokens": effective_tokens,
        "wall_clock_seconds": wall_clock_seconds,
        "canvas_tokens_per_second": (
            canvas_tokens / wall_clock_seconds if wall_clock_seconds > 0 else None
        ),
        "effective_tokens_per_second": (
            effective_tokens / wall_clock_seconds
            if wall_clock_seconds > 0
            else None
        ),
        "forward_passes_per_second": (
            nfe / wall_clock_seconds if wall_clock_seconds > 0 else None
        ),
        "diagnostic_seconds_excluded_from_wall_clock": diagnostic_seconds,
        "mean_field_seconds": 0.0,
        "dawn_selection_seconds": dawn_selection_seconds,
        "dawn_selector_timeline": dawn_selector_timeline,
        "dawn_fallback_region_events": sum(
            int(bool(region_item.get("fallback")))
            for item in dawn_selector_timeline
            for region_item in item["regions"]
        ),
        "dawn_selected_tokens": sum(
            int(region_item.get("selected", 0))
            for item in dawn_selector_timeline
            for region_item in item["regions"]
            if not bool(region_item.get("deferred"))
        ),
        "dawn_config": dawn_config if uses_dawn_selector else None,
        "region_clocks": {region.index: region.clock for region in regions},
        "region_schedule_steps": {
            region.index: region.schedule_step for region in regions
        },
        "region_size": region_size,
        "local_steps": local_steps,
        "admission_events": admission_events,
        "initial_admitted_region_count": initial_admitted_region_count,
        "final_admitted_region_count": scheduler.admitted_count,
        "max_active_regions": scheduler.max_active_regions,
        "max_progress_gap": scheduler.max_progress_gap,
        "spawn_readiness": scheduler.spawn_readiness,
        "readiness_confidence_threshold": float(
            strategy_probe.get("readiness_confidence_threshold", 0.5)
        ),
        "control_timeline": control_timeline,
        "tail_guard_timeline": tail_guard_timeline,
        "deferral_timeline": deferral_timeline,
        "stop_protection_timeline": stop_protection_timeline,
        "stop_filter_confidence_threshold": (
            stop_filter_confidence_threshold
            if mode == "always_on_coupled_defer_stop_filter"
            else None
        ),
        "accepted_stop_position": accepted_stop_position,
        "accepted_stop_iteration": accepted_stop_iteration,
        "early_stop_terminated": accepted_stop_position is not None,
        "ignored_suffix_tokens": (
            generation_length - accepted_stop_position - 1
            if accepted_stop_position is not None
            else 0
        ),
        "deferral_confidence_threshold": (
            deferral_confidence_threshold if uses_bounded_deferral else None
        ),
        "max_region_deferrals": (
            max_region_deferrals if mode in bounded_deferral_modes else None
        ),
        "max_global_deferral_iterations": (
            max_global_deferral_iterations
            if mode in coupled_deferral_modes
            else None
        ),
        "deferral_until_revealed_tokens": (
            deferral_until_revealed_tokens
            if mode in coupled_deferral_modes
            else None
        ),
        "iterations_with_tail_guard": sum(
            item["guarded_tail_region"] is not None
            for item in tail_guard_timeline
        ),
        "iterations_with_stop_protection": sum(
            bool(item["decisions"]) for item in stop_protection_timeline
        ),
        "stop_protection_region_events": sum(
            len(item["decisions"]) for item in stop_protection_timeline
        ),
        "stop_protection_schedule_hold_events": sum(
            len(item["schedule_held_regions"])
            for item in stop_protection_timeline
        ),
        "stop_filtered_candidate_events": sum(
            int(decision["stop_candidates"])
            for item in stop_protection_timeline
            for decision in item["decisions"]
        ),
        "stop_low_confidence_candidate_events": sum(
            int(decision.get("low_confidence_candidates", 0))
            for item in stop_protection_timeline
            for decision in item["decisions"]
        ),
        "iterations_with_deferral": sum(
            bool(item["deferred_regions"]) for item in deferral_timeline
        ),
        "deferred_region_events": sum(
            len(item["deferred_regions"]) for item in deferral_timeline
        ),
        "forced_region_events": sum(
            len(item["forced_regions"]) for item in deferral_timeline
        ),
        "gap_forced_region_events": sum(
            sum(
                item["action"] == "gap_forced"
                for item in timeline_item["decisions"]
            )
            for timeline_item in deferral_timeline
        ),
        "global_deadlock_forced_region_events": sum(
            sum(
                item["action"] == "global_deadlock_forced"
                for item in timeline_item["decisions"]
            )
            for timeline_item in deferral_timeline
        ),
        "deferral_window_closed_region_events": sum(
            sum(
                item["action"] == "deferral_window_closed_forced"
                for item in timeline_item["decisions"]
            )
            for timeline_item in deferral_timeline
        ),
        "iterations_with_blocking": sum(
            bool(item["blocked_regions"]) for item in control_timeline
        ),
        "blocked_region_events": sum(
            len(item["blocked_regions"]) for item in control_timeline
        ),
        "schedule_approximation": True,
        "schedule_approximation_name": regional_approximation_name(mode),
    }
