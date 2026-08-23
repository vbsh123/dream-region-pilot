from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .commit import commit_active_regions
from .dependencies import DAPDDreamAdapter, graph_summary, make_dependency_snapshot
from .diagnostics import ExampleDiagnostics
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
    if "<|eot_id|>" in vocabulary:
        result.add(int(vocabulary["<|eot_id|>"]))
    return result


def decode_response(tokenizer, response: torch.Tensor) -> tuple[str, int]:
    values = [int(value) for value in response.detach().cpu().tolist()]
    stops = stop_token_ids(tokenizer)
    stop = next(
        (index for index, token_id in enumerate(values) if token_id in stops),
        len(values),
    )
    return tokenizer.decode(values[:stop], skip_special_tokens=True).strip(), stop


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
    output = model.diffusion_generate(
        prompt,
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
    text, effective_tokens = decode_response(tokenizer, response)
    canvas_tokens = int(response.numel())
    return {
        "generation": text,
        "response_token_ids": [int(value) for value in response.detach().cpu().tolist()],
        "nfe": nfe,
        "global_forward_passes": nfe,
        "average_tokens_committed_per_forward": canvas_tokens / nfe,
        "canvas_tokens": canvas_tokens,
        "effective_generated_tokens": effective_tokens,
        "wall_clock_seconds": elapsed,
        "dependency_recomputations": 0,
        "dependency_seconds": 0.0,
        "schedule_approximation": False,
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
    tokens = F.pad(prompt, (0, generation_length), value=mask_id)
    regions = build_fixed_regions(generation_length, region_size)
    refresh_remaining_masks(regions, [True] * generation_length)
    scheduler = RegionScheduler(
        regions,
        mode=mode,
        lag=lag,
        release_completed_parents=release_completed_parents,
    )
    diagnostics = (
        ExampleDiagnostics(diagnostics_dir, diagnostic_snapshot_interval)
        if diagnostics_dir is not None
        else None
    )

    recompute_interval = int(dependency["recompute_interval"])
    if recompute_interval <= 0:
        raise ValueError("dependency.recompute_interval must be positive")
    use_graph = mode == "fixed_lag"
    latest_snapshot = None
    nfe = 0
    dependency_recomputations = 0
    dependency_seconds = 0.0
    diagnostic_seconds = 0.0
    graph_summaries: list[dict[str, Any]] = []
    tokens_committed_per_forward: list[int] = []

    synchronize(device)
    started = time.perf_counter()
    maximum_iterations = local_steps * (len(regions) + lag + 2)
    while bool((tokens[:, prompt_length:] == mask_id).any()):
        response_mask = tokens[:, prompt_length:] == mask_id
        should_recompute = use_graph and (
            latest_snapshot is None or nfe % recompute_interval == 0
        )
        if should_recompute:
            synchronize(device)
            dependency_started = time.perf_counter()
            logits, raw_token, normalized_token = adapter.forward_with_dependencies(
                model,
                tokens,
                response_mask,
                prompt_length,
            )
            synchronize(device)
            dependency_seconds += time.perf_counter() - dependency_started
            dependency_recomputations += 1
            latest_snapshot = make_dependency_snapshot(
                raw_token=raw_token,
                normalized_token=normalized_token,
                regions=regions,
                matrix=str(dependency["matrix"]),
                aggregator=str(dependency["aggregator"]),
                topk_percent=float(dependency["topk_percent"]),
                gamma=float(dependency["gamma"]),
                threshold=float(dependency["threshold"]),
            )
            scheduler.set_parents(latest_snapshot.parents)
            graph_summaries.append({"iteration": nfe, **graph_summary(latest_snapshot)})
            if diagnostics is not None:
                diagnostic_started = time.perf_counter()
                diagnostics.record_dependency(nfe, latest_snapshot)
                diagnostic_seconds += time.perf_counter() - diagnostic_started
        else:
            logits = adapter.forward_logits(model, tokens)
        nfe += 1

        active = scheduler.regions_allowed_to_advance(local_steps)
        if not active:
            clocks = {region.index: region.clock for region in regions}
            raise RuntimeError(
                "Regional scheduler deadlocked with masks remaining; "
                f"strategy={strategy}, clocks={clocks}, "
                f"release_completed_parents={release_completed_parents}"
            )
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
        )
        advanced = scheduler.apply_updates(active, committed)
        tokens_committed_per_forward.append(sum(len(value) for value in committed.values()))
        refresh_remaining_masks(
            regions,
            (tokens[0, prompt_length:] == mask_id).detach().cpu().tolist(),
        )
        if diagnostics is not None:
            diagnostic_started = time.perf_counter()
            diagnostics.record_iteration(
                iteration=nfe,
                regions=regions,
                scheduled=[region.index for region in active],
                advanced=[region.index for region in advanced],
                committed=committed,
                edges=latest_snapshot.edges if latest_snapshot is not None else [],
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
    text, effective_tokens = decode_response(tokenizer, response)
    canvas_tokens = int(response.numel())
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
        "wall_clock_seconds": elapsed_including_diagnostics - diagnostic_seconds,
        "diagnostic_seconds_excluded_from_wall_clock": diagnostic_seconds,
        "dependency_recomputations": dependency_recomputations,
        "dependency_seconds": dependency_seconds,
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
        "schedule_approximation": True,
        "schedule_approximation_name": (
            "per_region_linear_time_with_completed_parent_release"
            if mode == "fixed_lag" and release_completed_parents
            else "per_region_linear_time"
        ),
    }
