from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .benchmarks import (
    gsm8k_cot_score_details,
    load_benchmark,
    prepare_example,
    score_generation,
)
from .config import load_config
from .decoding import decode_mean_field_repro, decode_regional, decode_vanilla
from .dependencies import DAPDDreamAdapter
from .evaluation import write_summary


VANILLA_STEP_OVERRIDES = {
    "vanilla": None,
    "vanilla_steps128": 128,
    "vanilla_steps96": 96,
    "vanilla_steps72": 72,
    "vanilla_steps64": 64,
    "vanilla_steps32": 32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dream fixed-region pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example-indices", type=int, nargs="+")
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--region-size", type=int, choices=(16, 32, 64))
    parser.add_argument("--local-steps", type=int)
    parser.add_argument(
        "--dependency-aggregator",
        choices=("mean", "topk_percent", "power_mean"),
    )
    parser.add_argument("--dependency-matrix", choices=("raw", "normalized"))
    parser.add_argument("--dependency-threshold", type=float)
    parser.add_argument("--dependency-recompute-interval", type=int)
    parser.add_argument("--topk-percent", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--diagnostic-examples", type=int)
    parser.add_argument("--diagnostic-snapshot-interval", type=int)
    parser.add_argument("--probe-window", type=int)
    parser.add_argument("--spawn-readiness", type=float)
    parser.add_argument("--readiness-confidence-threshold", type=float)
    parser.add_argument("--mean-field-topk", type=int)
    parser.add_argument("--mean-field-thresholds", type=float, nargs="+")
    parser.add_argument("--mean-field-combination-threshold", type=float)
    parser.add_argument("--max-progress-gap", type=int)
    parser.add_argument("--edge-persistence", type=int)
    parser.add_argument("--deferral-confidence-threshold", type=float)
    parser.add_argument("--max-region-deferrals", type=int)
    parser.add_argument("--max-global-deferral-iterations", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str):
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in choices:
        raise ValueError(f"Unsupported dtype {name!r}")
    return choices[name]


def load_model(model_config: dict[str, Any]):
    kwargs = {
        "trust_remote_code": bool(model_config.get("trust_remote_code", True)),
        "torch_dtype": torch_dtype(str(model_config.get("dtype", "bfloat16"))),
        "low_cpu_mem_usage": True,
    }
    if model_config.get("revision"):
        kwargs["revision"] = str(model_config["revision"])
    name = str(model_config["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=kwargs["trust_remote_code"],
        revision=kwargs.get("revision"),
    )
    model = AutoModel.from_pretrained(name, **kwargs)
    device = torch.device(str(model_config.get("device", "cuda")))
    return model.to(device).eval(), tokenizer


def encode_prompt(
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    add_special_tokens: bool = False,
) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(
        rendered,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    ).to(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.region_size is not None:
        config["generation"]["region_size"] = args.region_size
        if args.local_steps is None:
            config["generation"]["local_steps"] = args.region_size
    if args.local_steps is not None:
        config["generation"]["local_steps"] = args.local_steps
    dependency_overrides = {
        "aggregator": args.dependency_aggregator,
        "matrix": args.dependency_matrix,
        "threshold": args.dependency_threshold,
        "recompute_interval": args.dependency_recompute_interval,
        "topk_percent": args.topk_percent,
        "gamma": args.gamma,
    }
    for key, value in dependency_overrides.items():
        if value is not None:
            config["dependency"][key] = value
    if args.diagnostic_examples is not None:
        config["experiment"]["diagnostic_examples"] = args.diagnostic_examples
    if args.diagnostic_snapshot_interval is not None:
        config["experiment"][
            "diagnostic_snapshot_interval"
        ] = args.diagnostic_snapshot_interval
    probe_config = config.setdefault("probe", {})
    mean_field_config = probe_config.setdefault("mean_field", {})
    probe_overrides = {
        "max_active_regions": args.probe_window,
        "spawn_readiness": args.spawn_readiness,
        "readiness_confidence_threshold": args.readiness_confidence_threshold,
        "max_progress_gap": args.max_progress_gap,
        "edge_persistence": args.edge_persistence,
        "deferral_confidence_threshold": args.deferral_confidence_threshold,
        "max_region_deferrals": args.max_region_deferrals,
        "max_global_deferral_iterations": args.max_global_deferral_iterations,
    }
    for key, value in probe_overrides.items():
        if value is not None:
            probe_config[key] = value
    if args.mean_field_topk is not None:
        mean_field_config["topk"] = args.mean_field_topk
    if args.mean_field_thresholds is not None:
        mean_field_config["thresholds"] = args.mean_field_thresholds
    if args.mean_field_combination_threshold is not None:
        mean_field_config["combination_threshold"] = (
            args.mean_field_combination_threshold
        )
    strategies = args.strategies or list(config["experiment"]["strategies"])
    allowed = set(VANILLA_STEP_OVERRIDES) | {
        "fixed_sequential",
        "always_on",
        "always_on_tail_guard",
        "always_on_bounded_defer",
        "always_on_bounded_defer_tail_guard",
        "always_on_coupled_defer",
        "always_on_coupled_defer_tail_guard",
        "async_lag0",
        "async_lag1",
        "async_lag2",
        "async_lag4",
        "wavefront_probe",
        "loose_wavefront",
        "flowblock_proxy",
        "mean_field_repro",
        "controlled_position",
        "controlled_position_tail_guard",
        "controlled_dapd",
        "controlled_jsd",
        "controlled_combo",
        "controlled_dapd_dynamic",
        "controlled_jsd_dynamic",
        "controlled_combo_dynamic",
    }
    unknown = set(strategies) - allowed
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    completed: set[tuple[int, str]] = set()
    rows: list[dict[str, Any]] = []
    if results_path.exists():
        if not args.resume:
            raise FileExistsError(f"{results_path} exists; pass --resume")
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed.add((int(row["example_index"]), str(row["strategy"])))

    if args.example_indices is not None:
        if any(index < 0 for index in args.example_indices):
            raise ValueError("--example-indices must be non-negative")
        requested_indices = list(dict.fromkeys(args.example_indices))
        dataset = load_benchmark(config["data"], max(requested_indices) + 1)
        dataset_items = [(index, dataset[index]) for index in requested_indices]
    else:
        dataset = load_benchmark(config["data"], args.limit)
        dataset_items = list(enumerate(dataset))
    task = str(config["data"].get("task", "gsm8k"))
    model, tokenizer = load_model(config["model"])
    dependency_config = config["dependency"]
    adapter = DAPDDreamAdapter(
        Path(dependency_config["dapd_repo"]),
        str(dependency_config["dapd_revision"]),
        float(dependency_config["layer_ratio"]),
    )
    base_seed = int(config["generation"]["seed"])
    diagnostics_count = int(config["experiment"]["diagnostic_examples"])

    total = sum(
        (example_index, strategy) not in completed
        for example_index, _ in dataset_items
        for strategy in strategies
    )
    with results_path.open("a", encoding="utf-8") as handle:
        progress = tqdm(total=total, desc="Dream region pilot")
        for run_position, (example_index, source) in enumerate(dataset_items):
            example = prepare_example(config["data"], source)
            question = example.question
            prompt_text = str(config["data"]["prompt_template"]).format(
                question=question
            )
            prompt = encode_prompt(
                tokenizer,
                prompt_text,
                model.device,
                add_special_tokens=bool(
                    config["data"].get("add_special_tokens", False)
                ),
            )
            reference = example.reference_answer
            ordered = list(strategies)
            if bool(config["experiment"].get("rotate_strategy_order", True)):
                offset = example_index % len(ordered)
                ordered = ordered[offset:] + ordered[:offset]

            for strategy in ordered:
                if (example_index, strategy) in completed:
                    continue
                seed = base_seed + example_index
                seed_everything(seed)
                diagnostics_dir = None
                if run_position < diagnostics_count and (
                    strategy.startswith("async_")
                    or strategy in {
                        "always_on",
                        "always_on_tail_guard",
                        "always_on_bounded_defer",
                        "always_on_bounded_defer_tail_guard",
                        "always_on_coupled_defer",
                        "always_on_coupled_defer_tail_guard",
                    }
                    or strategy == "wavefront_probe"
                    or strategy == "loose_wavefront"
                    or strategy == "flowblock_proxy"
                    or strategy.startswith("controlled_")
                ):
                    diagnostics_dir = (
                        output_dir
                        / "diagnostics"
                        / f"example_{example_index:03d}"
                        / strategy
                    )
                if strategy in VANILLA_STEP_OVERRIDES:
                    vanilla_generation = dict(config["generation"])
                    step_override = VANILLA_STEP_OVERRIDES[strategy]
                    if step_override is not None:
                        vanilla_generation["steps"] = step_override
                    generated = decode_vanilla(
                        model, tokenizer, prompt, vanilla_generation
                    )
                elif strategy == "mean_field_repro":
                    generated = decode_mean_field_repro(
                        model,
                        tokenizer,
                        prompt,
                        generation=config["generation"],
                        adapter=adapter,
                        mean_field=config.get("mean_field_baseline", {}),
                    )
                else:
                    generated = decode_regional(
                        model,
                        tokenizer,
                        prompt,
                        strategy=strategy,
                        generation=config["generation"],
                        dependency=dependency_config,
                        adapter=adapter,
                        release_completed_parents=bool(
                            config["experiment"]["release_completed_parents"]
                        ),
                        diagnostics_dir=diagnostics_dir,
                        diagnostic_snapshot_interval=int(
                            config["experiment"]["diagnostic_snapshot_interval"]
                        ),
                        probe=probe_config,
                    )
                prediction, correct, scoring_method = score_generation(
                    config["data"], generated["generation"], reference, source
                )
                scoring_details = {}
                if config["data"].get("protocol") == "lm_eval_gsm8k_cot":
                    scoring_details = gsm8k_cot_score_details(
                        generated["generation"], reference
                    )
                row = {
                    "example_index": example_index,
                    "task": task,
                    "strategy": strategy,
                    "seed": seed,
                    "question": question,
                    "reference_answer": reference,
                    "predicted_answer": prediction,
                    "correct": correct,
                    "scoring_method": scoring_method,
                    **scoring_details,
                    **generated,
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                rows.append(row)
                progress.update(1)
        progress.close()

    metadata = {
        "artifact_type": "dream_regional_async_reasoning_pilot",
        "config": config,
        "examples": len(dataset_items),
        "example_indices": [index for index, _ in dataset_items],
        "task": task,
        "strategies": strategies,
        "model_resolved_commit": getattr(model.config, "_commit_hash", None),
        "transformers_version": transformers.__version__,
        "dapd_revision": adapter.revision,
        "implementation_notes": [
            "Attention visibility is unchanged from Dream.",
            "DAPD post-RoPE Q/K extraction and token dependency construction are imported from the pinned public checkout.",
            "Regional modes use one linear Dream timestep schedule per fixed region.",
            "Backpressure clocks count non-empty commitment events; a separate schedule cursor consumes zero-transfer Dream schedule points.",
            "Completed parents release children because strict positive lag otherwise deadlocks terminal child steps.",
            "wavefront_probe admits at most one positional region per forward, uses W as a maximum active-region count, and uses a FlowBlock-style confidence acceptance ratio only for admission.",
            "flowblock_proxy transfers only W=2, theta_spawn=0.60, and token-readiness probability 0.50 into the Dream regional admission harness; it is not an implementation of FlowBlock's T2T editing, block-causal KV cache, or threshold commit policy.",
            "The 15% spawn threshold is theta_spawn; the separate token confidence threshold defaults to 0.5, matching FlowBlock's reported math setting but not Dream's entropy commit rule.",
            "Mean-Field JSD is diagnostic-only in wavefront_probe; controlled_jsd and controlled_combo use its persistent region edges for pausing decisions but never change admission or token scoring.",
            "loose_wavefront is the graph-free W=8, theta_spawn=0.15 readiness-only ablation; unlike controlled_position it has no permanent positional bounded-skew edges.",
            "vanilla_steps32/64/72/96/128 call the official Dream diffusion_generate path with only the configured global step count changed; max_new_tokens remains 256.",
            "vanilla_steps72 is the primary compute-matched control for regional strategies that previously averaged about 71 NFEs.",
            "mean_field_repro implements Algorithm 1 from arXiv:2606.15805 with exact JSD inside sequential active blocks. The paper's linked GitHub repository currently returns 404, so this is not labelled official code.",
            "The Mean-Field pseudocode does not define an empty-commit fallback. This reproduction commits the maximum-intensity token to guarantee progress and logs every such event.",
            "The all-canvas Mean-Field signal uses a top-k-union plus shared-tail approximation; retained probability mass is logged and paper_exact is false unless top-k equals the vocabulary size.",
            "Controlled modes activate an edge only after both endpoints reveal a token and the edge persists for two graph observations.",
            "Controlled modes normally advance every unblocked admitted region; they do not round-robin or graph-color components.",
            "The controlled_*_dynamic modes rebuild connected dependency components after every graph update and pool the ordinary Dream commitment budget across each component. They do not remask or alter attention visibility.",
            "Dynamic component merges and splits preserve each fixed region's revealed tokens, schedule cursor, and real-progress clock; no synthetic clock progress or equalization is introduced.",
            "controlled_position uses the identical admission and bounded-skew controller without dynamic DAPD/JSD edges, isolating whether graph control helps beyond positional staggering.",
            "controlled_position_tail_guard is a coarse terminal-region ablation: while an earlier fixed region remains unfinished, it withholds the region containing the earliest currently masked top-1 stop-token prediction and every later region from forced progress.",
            "always_on_tail_guard applies the same coarse terminal-region guard with all regions admitted from iteration zero and no positional or dependency backpressure.",
            "always_on_bounded_defer keeps every region active from iteration zero but withholds a region's ordinary local update when the least-confident token in that update's quota is below the raw, untempered top-1 probability threshold. The local schedule cursor does not advance on a confidence deferral.",
            "Bounded deferral forces the ordinary regional update after the configured number of consecutive skips. Natural zero-token points in Dream's transfer schedule advance the schedule cursor and do not count as confidence deferrals.",
            "always_on_bounded_defer_tail_guard combines bounded regional deferral with the identical predicted terminal-region guard; it has no admission window, parent graph, or positional backpressure.",
            "always_on_coupled_defer starts every region immediately, permits low-confidence regional skips, and uses adjacent positional edges to bound revealed-token staleness. When an endpoint is paused at the gap, its lagging neighbor's ordinary update bypasses the confidence gate.",
            "Coupled deferral has no per-region wall-clock skip deadline: if a parent also skips, the child does not consume positional slack. Four globally empty confidence-deferral iterations trigger one forced leftmost active update solely to prevent a total fixed-point deadlock.",
            "The first nonzero local update in every coupled-deferral region bypasses the confidence gate, ensuring that a fully masked region receives one seed token before it may defer. Dream's preceding natural zero-quota schedule point remains zero-cost in deferral accounting.",
            "Progress is actual revealed-token count. A higher-position child is paused if it outruns its parent, while a parent is paused only when its lead exceeds the configurable eight-token default.",
            "Urgent service never forces an extra low-confidence token; it schedules the lagging region's next ordinary Dream local update.",
            "GSM8K uses numeric final-answer scoring. ASDiv handles both numeric and categorical gold answers. MATH-500 uses math-verify 0.9.0 symbolic scoring.",
            "HumanEval reports deterministic pass@1 from the official OpenAI tests. Generated Python is executed in a restricted child process; the Vast worker should still be treated as disposable rather than as a security sandbox.",
            "The OpenAI HumanEval evaluator is imported from the pinned source-only checkout at external/HumanEval; its legacy setup.py is intentionally not installed.",
            "configs/gsm8k_cot_official_50.yaml mirrors Dream's released zero-shot lm-eval GSM8K-CoT prompt, temperature 0.1, top-p 0.9, entropy policy, stop strings, chat template, and strict/flexible answer filters.",
            "For paired strategy attribution, every strategy is reseeded with 1234 plus the example index. Upstream lm-eval seeds once and consumes one continuous RNG stream, so exact sample identity is not expected even when aggregate vanilla accuracy agrees.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(rows, output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
