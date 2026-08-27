from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .commit import commit_active_regions, describe_commitments
from .dependencies import DAPDDreamAdapter, graph_summary, make_dependency_snapshot
from .diagnostics import ExampleDiagnostics
from .mean_field import mean_field_commit_indices, topk_tail_jsd_dependency
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
    output = model.diffusion_generate(
        prompt,
        attention_mask=attention_mask,
        max_new_tokens=int(generation["max_new_tokens"]),
        steps=int(generation["steps"]),
        alg=str(generation["commit_policy"]),
        alg_temp=generation.get("alg_temp"),
        temperature=float(generation.get("temperature", 0.0)),
        top_p=generation.get("top_p"),
        top_k=generation.get("top_k"),
        return_dict_in_generate=True,
    )
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
        "dependency_recomputations": 0,
        "dependency_seconds": 0.0,
        "mean_field_seconds": 0.0,
        "dependency_signal_seconds": 0.0,
        "diagnostic_seconds_excluded_from_wall_clock": 0.0,
        "schedule_approximation": False,
    }


@torch.no_grad()
def decode_mean_field_repro(
    model,
    tokenizer,
    prompt: torch.Tensor,
    *,
    generation: dict[str, Any],
    adapter: DAPDDreamAdapter,
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
        "dependency_recomputations": nfe,
        "dependency_seconds": 0.0,
        "mean_field_seconds": selection_seconds,
        "dependency_signal_seconds": selection_seconds,
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
    dependency: dict[str, Any],
    adapter: DAPDDreamAdapter,
    release_completed_parents: bool,
    diagnostics_dir: Path | None,
    diagnostic_snapshot_interval: int,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode, lag = parse_strategy(strategy)
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
        lag=lag,
        release_completed_parents=release_completed_parents,
        max_active_regions=int(
            strategy_probe.get("max_active_regions", len(regions))
        ),
        spawn_readiness=float(strategy_probe.get("spawn_readiness", 0.15)),
        max_progress_gap=int(strategy_probe.get("max_progress_gap", 8)),
        edge_persistence=int(strategy_probe.get("edge_persistence", 2)),
    )
    diagnostics = (
        ExampleDiagnostics(diagnostics_dir, diagnostic_snapshot_interval)
        if diagnostics_dir is not None
        else None
    )

    recompute_interval = int(dependency["recompute_interval"])
    if recompute_interval <= 0:
        raise ValueError("dependency.recompute_interval must be positive")
    graph_controlled_modes = {
        "controlled_dapd",
        "controlled_jsd",
        "controlled_combo",
        "controlled_dapd_dynamic",
        "controlled_jsd_dynamic",
        "controlled_combo_dynamic",
    }
    use_graph = mode in {"fixed_lag", "wavefront_probe"} | graph_controlled_modes
    needs_dapd = mode in {
        "fixed_lag",
        "wavefront_probe",
        "controlled_dapd",
        "controlled_combo",
        "controlled_dapd_dynamic",
        "controlled_combo_dynamic",
    }
    mean_field_config = probe.get("mean_field", {})
    use_mean_field = mode in {
        "wavefront_probe",
        "controlled_jsd",
        "controlled_combo",
        "controlled_jsd_dynamic",
        "controlled_combo_dynamic",
    } and bool(mean_field_config.get("enabled", False))
    mean_field_thresholds = [
        float(value)
        for value in mean_field_config.get(
            "thresholds", [0.5, 0.7, 0.8, 0.9, 0.95]
        )
    ]
    combination_threshold = float(
        mean_field_config.get("combination_threshold", 0.9)
    )
    if use_mean_field and combination_threshold not in mean_field_thresholds:
        mean_field_thresholds.append(combination_threshold)
    latest_snapshot = None
    nfe = 0
    dependency_recomputations = 0
    dependency_seconds = 0.0
    diagnostic_seconds = 0.0
    mean_field_seconds = 0.0
    graph_summaries: list[dict[str, Any]] = []
    tokens_committed_per_forward: list[int] = []
    admission_events: list[dict[str, Any]] = []
    control_timeline: list[dict[str, Any]] = []
    tail_guard_timeline: list[dict[str, Any]] = []
    deferral_timeline: list[dict[str, Any]] = []
    bounded_deferral_modes = {
        "always_on_bounded_defer",
        "always_on_bounded_defer_tail_guard",
    }
    coupled_deferral_modes = {
        "always_on_coupled_defer",
        "always_on_coupled_defer_tail_guard",
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

    synchronize(device)
    started = time.perf_counter()
    maximum_iterations = local_steps * (len(regions) + lag + 2)
    if mode in coupled_deferral_modes:
        maximum_iterations *= max_global_deferral_iterations + 1
    while bool((tokens[:, prompt_length:] == mask_id).any()):
        response_mask = tokens[:, prompt_length:] == mask_id
        should_recompute = use_graph and (
            latest_snapshot is None or nfe % recompute_interval == 0
        )
        if should_recompute:
            if needs_dapd:
                synchronize(device)
                dependency_started = time.perf_counter()
                logits, raw_token, normalized_token = (
                    adapter.forward_with_dependencies(
                        model,
                        tokens,
                        response_mask,
                        prompt_length,
                    )
                )
                synchronize(device)
                dependency_seconds += time.perf_counter() - dependency_started
            else:
                # JSD uses the ordinary logits from this forward and does not
                # need DAPD's Q/K capture or attention reconstruction.
                logits = adapter.forward_logits(model, tokens)
                synchronize(device)
                raw_token = torch.zeros(
                    (generation_length, generation_length),
                    dtype=torch.float32,
                    device=device,
                )
                normalized_token = raw_token.clone()
            dependency_recomputations += 1
            additional_dependencies = None
            dependency_metadata: dict[str, Any] = {
                "dapd_computed": needs_dapd,
            }
            if use_mean_field:
                mean_field_started = time.perf_counter()
                (
                    mean_field_raw_token,
                    mean_field_token,
                    mean_field_metadata,
                ) = topk_tail_jsd_dependency(
                    logits[:, prompt_length:],
                    response_mask,
                    topk=int(mean_field_config.get("topk", 256)),
                    pair_chunk_size=int(
                        mean_field_config.get("pair_chunk_size", 128)
                    ),
                )
                synchronize(device)
                mean_field_seconds += time.perf_counter() - mean_field_started
                additional_dependencies = {
                    "mean_field_raw": mean_field_raw_token,
                    "mean_field": mean_field_token,
                }
                dependency_metadata["mean_field"] = mean_field_metadata
            latest_snapshot = make_dependency_snapshot(
                raw_token=raw_token,
                normalized_token=normalized_token,
                regions=regions,
                matrix=str(dependency["matrix"]),
                aggregator=str(dependency["aggregator"]),
                topk_percent=float(dependency["topk_percent"]),
                gamma=float(dependency["gamma"]),
                threshold=float(dependency["threshold"]),
                additional_token_dependencies=additional_dependencies,
                additional_thresholds=(
                    {"mean_field": mean_field_thresholds}
                    if use_mean_field
                    else None
                ),
                additional_aggregators=(
                    {"mean_field": "mean", "mean_field_raw": "mean"}
                    if use_mean_field
                    else None
                ),
                combination_threshold=(
                    combination_threshold
                    if use_mean_field and needs_dapd
                    else None
                ),
                metadata=dependency_metadata,
            )
            primary_graph_key = "dapd"
            if mode == "fixed_lag":
                scheduler.set_parents(latest_snapshot.parents)
            elif mode in graph_controlled_modes:
                graph_key = {
                    "controlled_dapd": "dapd",
                    "controlled_dapd_dynamic": "dapd",
                    "controlled_jsd": (
                        f"mean_field_t{combination_threshold:.2f}"
                    ),
                    "controlled_jsd_dynamic": (
                        f"mean_field_t{combination_threshold:.2f}"
                    ),
                    "controlled_combo": (
                        f"combined_intersection_t{combination_threshold:.2f}"
                    ),
                    "controlled_combo_dynamic": (
                        f"combined_intersection_t{combination_threshold:.2f}"
                    ),
                }[mode]
                primary_graph_key = graph_key
                if graph_key not in latest_snapshot.signal_graphs:
                    raise RuntimeError(
                        f"Required dependency graph {graph_key!r} was not constructed"
                    )
                scheduler.observe_dependency_edges(
                    latest_snapshot.signal_graphs[graph_key].edges
                )
            graph_summaries.append(
                {
                    "iteration": nfe,
                    "primary_signal": primary_graph_key,
                    **graph_summary(
                        latest_snapshot,
                        latest_snapshot.signal_graphs[primary_graph_key],
                    ),
                    "signals": {
                        name: {
                            "threshold": signal_graph.threshold,
                            **graph_summary(latest_snapshot, signal_graph),
                        }
                        for name, signal_graph in latest_snapshot.signal_graphs.items()
                    },
                    "metadata": latest_snapshot.metadata,
                }
            )
            if diagnostics is not None:
                diagnostic_started = time.perf_counter()
                diagnostics.record_dependency(nfe, latest_snapshot)
                diagnostic_seconds += time.perf_counter() - diagnostic_started
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
        if mode in {
            "always_on_tail_guard",
            "always_on_bounded_defer_tail_guard",
            "always_on_coupled_defer_tail_guard",
            "controlled_position_tail_guard",
        }:
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
        active = scheduler.regions_allowed_to_advance(
            local_steps,
            max_region_exclusive=guarded_tail,
        )
        if not active:
            clocks = {region.index: region.clock for region in regions}
            raise RuntimeError(
                "Regional scheduler deadlocked with masks remaining; "
                f"strategy={strategy}, clocks={clocks}, "
                f"release_completed_parents={release_completed_parents}"
            )
        commit_groups = scheduler.commitment_groups(active)
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
            region_groups=commit_groups,
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
        schedule_advanced_regions = [
            region for region in active if region.index not in deferred_regions
        ]
        advanced = scheduler.apply_updates(schedule_advanced_regions, committed)
        committed_this_forward = sum(
            len(value) for value in committed.values()
        )
        tokens_committed_per_forward.append(committed_this_forward)
        if mode in coupled_deferral_modes:
            if committed_this_forward == 0 and deferred_regions:
                global_empty_deferral_streak += 1
            else:
                global_empty_deferral_streak = 0
        refresh_remaining_masks(
            regions,
            (tokens[0, prompt_length:] == mask_id).detach().cpu().tolist(),
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
            "progress_tokens": {
                region.index: scheduler.revealed_tokens(region) for region in regions
            },
            "active_dependency_edges": [
                list(pair) for pair in sorted(scheduler.active_dependency_edges)
            ],
            "control_edges": [
                list(pair) for pair in sorted(scheduler.last_control_edges)
            ],
            "blocked_regions": sorted(scheduler.last_blocked_regions),
            "urgent_regions": sorted(scheduler.last_urgent_regions),
            "commit_groups": [
                [region.index for region in group]
                for group in commit_groups
            ],
            "predicted_tail_region": provisional_tail,
            "guarded_tail_region": guarded_tail,
            "deferral_decisions": deferral_decisions,
            "region_deferral_counts": dict(region_deferral_counts),
            "force_region_reasons": force_region_reasons,
            "global_empty_deferral_streak": global_empty_deferral_streak,
        }
        if scheduler.is_controlled:
            control_timeline.append(control_state)
        if mode in {
            "always_on_tail_guard",
            "always_on_bounded_defer_tail_guard",
            "always_on_coupled_defer_tail_guard",
            "controlled_position_tail_guard",
        }:
            tail_guard_timeline.append(
                {
                    "iteration": nfe,
                    "predicted_tail_region": provisional_tail,
                    "guarded_tail_region": guarded_tail,
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
        if diagnostics is not None:
            diagnostic_started = time.perf_counter()
            diagnostics.record_iteration(
                iteration=nfe,
                regions=regions,
                scheduled=[region.index for region in active],
                advanced=[region.index for region in advanced],
                committed=committed,
                commitment_details=commitment_details,
                edges=latest_snapshot.edges if latest_snapshot is not None else [],
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
        "dependency_recomputations": dependency_recomputations,
        "dependency_seconds": dependency_seconds,
        "mean_field_seconds": mean_field_seconds,
        "dependency_signal_seconds": dependency_seconds + mean_field_seconds,
        "graph_summaries": graph_summaries,
        "mean_graph_edge_density": (
            sum(item["edge_density"] for item in graph_summaries)
            / len(graph_summaries)
            if graph_summaries
            else None
        ),
        "region_clocks": {region.index: region.clock for region in regions},
        "region_schedule_steps": {
            region.index: region.schedule_step for region in regions
        },
        "admission_events": admission_events,
        "final_admitted_region_count": scheduler.admitted_count,
        "max_active_regions": scheduler.max_active_regions,
        "spawn_readiness": scheduler.spawn_readiness,
        "readiness_confidence_threshold": float(
            strategy_probe.get("readiness_confidence_threshold", 0.5)
        ),
        "control_timeline": control_timeline,
        "tail_guard_timeline": tail_guard_timeline,
        "deferral_timeline": deferral_timeline,
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
        "mean_active_dependency_edge_count": (
            sum(len(item["active_dependency_edges"]) for item in control_timeline)
            / len(control_timeline)
            if control_timeline
            else 0.0
        ),
        "iterations_with_pooled_commit_groups": sum(
            any(len(group) > 1 for group in item.get("commit_groups", []))
            for item in control_timeline
        ),
        "mean_max_commit_group_size": (
            sum(
                max(
                    (len(group) for group in item.get("commit_groups", [])),
                    default=0,
                )
                for item in control_timeline
            )
            / len(control_timeline)
            if control_timeline
            else 0.0
        ),
        "final_active_dependency_edges": [
            list(pair) for pair in sorted(scheduler.active_dependency_edges)
        ],
        "schedule_approximation": True,
        "schedule_approximation_name": (
            "per_region_linear_time_with_completed_parent_release"
            if mode == "fixed_lag" and release_completed_parents
            else (
                "flowblock_admission_proxy_plus_per_region_linear_time"
                if mode == "flowblock_proxy"
                else (
                    "loose_confidence_admission_plus_per_region_linear_time"
                    if mode in {"wavefront_probe", "loose_wavefront"}
                    else (
                        "persistent_dependency_bounded_progress_skew"
                        if mode in graph_controlled_modes
                        and not scheduler.uses_dynamic_commit_groups
                        else (
                            "dynamic_dependency_components_pooled_commitment"
                            if scheduler.uses_dynamic_commit_groups
                            else (
                                "positional_bounded_progress_skew"
                                if mode == "controlled_position"
                                else (
                                    "predicted_terminal_region_guard_plus_positional_bounded_skew"
                                    if mode == "controlled_position_tail_guard"
                                    else (
                                        "predicted_terminal_region_guard_plus_always_on_regions"
                                        if mode == "always_on_tail_guard"
                                        else (
                                            "bounded_confidence_deferral_plus_predicted_terminal_region_guard"
                                            if mode
                                            == "always_on_bounded_defer_tail_guard"
                                            else (
                                                "bounded_confidence_deferral"
                                                if mode
                                                == "always_on_bounded_defer"
                                                else (
                                                    "confidence_deferral_plus_positional_bounded_staleness"
                                                    if mode
                                                    == "always_on_coupled_defer"
                                                    else (
                                                        "confidence_deferral_plus_positional_bounded_staleness_plus_predicted_terminal_region_guard"
                                                        if mode
                                                        == "always_on_coupled_defer_tail_guard"
                                                        else "per_region_linear_time"
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        ),
    }
