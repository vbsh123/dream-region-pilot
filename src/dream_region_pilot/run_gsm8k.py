from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .benchmarks import (
    build_fewshot_contexts,
    gsm8k_cot_score_details,
    load_benchmark,
    prepare_example,
    score_generation,
)
from .config import load_config
from .decoding import (
    decode_mean_field_repro,
    decode_official_dawn,
    decode_regional,
    decode_vanilla,
)
from .evaluation import write_summary
from .model_adapter import DreamModelAdapter, verify_dawn_checkout


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
    parser.add_argument(
        "--region-size", type=int, choices=(16, 20, 25, 32, 40, 64)
    )
    parser.add_argument("--local-steps", type=int)
    parser.add_argument("--diagnostic-examples", type=int)
    parser.add_argument("--probe-window", type=int)
    parser.add_argument("--spawn-readiness", type=float)
    parser.add_argument("--readiness-confidence-threshold", type=float)
    parser.add_argument("--max-progress-gap", type=int)
    parser.add_argument("--deferral-confidence-threshold", type=float)
    parser.add_argument("--stop-filter-confidence-threshold", type=float)
    parser.add_argument("--max-region-deferrals", type=int)
    parser.add_argument("--max-global-deferral-iterations", type=int)
    parser.add_argument("--deferral-until-revealed-tokens", type=int)
    parser.add_argument("--dawn-sink-threshold", type=float)
    parser.add_argument("--dawn-edge-threshold", type=float)
    parser.add_argument("--dawn-high-confidence-threshold", type=float)
    parser.add_argument("--dawn-induce-threshold", type=float)
    parser.add_argument("--dawn-candidate-confidence-threshold", type=float)
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


def load_model(
    model_config: dict[str, Any],
    *,
    use_dawn_model: bool = False,
    dawn_repo: Path | None = None,
    dawn_revision: str | None = None,
):
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
    model_class = AutoModel
    if use_dawn_model:
        if dawn_repo is None or dawn_revision is None:
            raise ValueError("DAWN model requires a pinned repository and revision")
        verify_dawn_checkout(dawn_repo, dawn_revision)
        dawn_python_root = str(dawn_repo.resolve() / "dream")
        if dawn_python_root not in sys.path:
            sys.path.insert(0, dawn_python_root)
        from model.modeling_dream import DreamModel

        model_class = DreamModel
    model = model_class.from_pretrained(name, **kwargs)
    device = torch.device(str(model_config.get("device", "cuda")))
    return model.to(device).eval(), tokenizer


def encode_prompt(
    tokenizer,
    prompt: str,
    device: torch.device,
    *,
    add_special_tokens: bool = False,
    apply_chat_template: bool = True,
    prepend_bos_token: bool = False,
) -> torch.Tensor:
    if apply_chat_template:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered = prompt
        if prepend_bos_token:
            if tokenizer.bos_token is None:
                raise ValueError("Tokenizer has no BOS token to prepend")
            rendered = tokenizer.bos_token + rendered
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
    if args.diagnostic_examples is not None:
        config["experiment"]["diagnostic_examples"] = args.diagnostic_examples
    probe_config = config.setdefault("probe", {})
    probe_overrides = {
        "max_active_regions": args.probe_window,
        "spawn_readiness": args.spawn_readiness,
        "readiness_confidence_threshold": args.readiness_confidence_threshold,
        "max_progress_gap": args.max_progress_gap,
        "deferral_confidence_threshold": args.deferral_confidence_threshold,
        "stop_filter_confidence_threshold": (
            args.stop_filter_confidence_threshold
        ),
        "max_region_deferrals": args.max_region_deferrals,
        "max_global_deferral_iterations": args.max_global_deferral_iterations,
        "deferral_until_revealed_tokens": (
            args.deferral_until_revealed_tokens
        ),
    }
    for key, value in probe_overrides.items():
        if value is not None:
            probe_config[key] = value
    dawn_config = probe_config.setdefault("dawn", {})
    dawn_overrides = {
        "sink_threshold": args.dawn_sink_threshold,
        "edge_threshold": args.dawn_edge_threshold,
        "high_confidence_threshold": args.dawn_high_confidence_threshold,
        "induce_threshold": args.dawn_induce_threshold,
        "candidate_confidence_threshold": (
            args.dawn_candidate_confidence_threshold
        ),
    }
    for key, value in dawn_overrides.items():
        if value is not None:
            dawn_config[key] = value
    strategies = args.strategies or list(config["experiment"]["strategies"])
    allowed = set(VANILLA_STEP_OVERRIDES) | {
        "official_dawn",
        "fixed_sequential",
        "always_on",
        "always_on_tail_guard",
        "always_on_dawn_tail_guard",
        "always_on_bounded_defer",
        "always_on_bounded_defer_tail_guard",
        "always_on_coupled_defer",
        "always_on_coupled_defer_tail_guard",
        "always_on_coupled_defer_stop_filter",
        "always_on_coupled_defer_stop_defer",
        "always_on_coupled_defer_dawn_tail_guard",
        "loose_wavefront",
        "flowblock_proxy",
        "mean_field_repro",
        "controlled_position",
        "controlled_position_tail_guard",
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
    fewshot_contexts = build_fewshot_contexts(
        config["data"], [index for index, _ in dataset_items]
    )
    source_config = config["sources"]
    model_implementation = str(
        config["model"].get("implementation", "huggingface_remote")
    )
    # Backend selection must be explicit for released-protocol reproductions.
    # Inferring it only from strategy names silently ran DAWN's five-shot
    # prompt through Hugging Face Dream's different global generation backend
    # whenever the requested comparison omitted a strategy named "dawn".
    uses_dawn = (
        model_implementation == "dawn_release"
        or any("dawn" in strategy for strategy in strategies)
    )
    resolved_model_implementation = (
        "dawn_release" if uses_dawn else "huggingface_remote"
    )
    dawn_repo = Path(source_config.get("dawn_repo", "external/DAWN"))
    dawn_revision = str(
        source_config.get(
            "dawn_revision", "19c32c28b5bf0475ccdfad853c74fc885f6410cd"
        )
    )
    task = str(config["data"].get("task", "gsm8k"))
    model, tokenizer = load_model(
        config["model"],
        use_dawn_model=uses_dawn,
        dawn_repo=dawn_repo if uses_dawn else None,
        dawn_revision=dawn_revision if uses_dawn else None,
    )
    adapter = DreamModelAdapter()
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
            prompt_text = fewshot_contexts.get(example_index, "") + str(
                config["data"]["prompt_template"]
            ).format(question=question)
            prompt = encode_prompt(
                tokenizer,
                prompt_text,
                model.device,
                add_special_tokens=bool(
                    config["data"].get("add_special_tokens", False)
                ),
                apply_chat_template=bool(
                    config["data"].get("apply_chat_template", True)
                ),
                prepend_bos_token=bool(
                    config["data"].get("prepend_bos_token", False)
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
                    strategy in {
                        "always_on",
                        "always_on_tail_guard",
                        "always_on_bounded_defer",
                        "always_on_bounded_defer_tail_guard",
                        "always_on_coupled_defer",
                        "always_on_coupled_defer_tail_guard",
                        "always_on_coupled_defer_stop_filter",
                        "always_on_coupled_defer_stop_defer",
                        "always_on_dawn_tail_guard",
                        "always_on_coupled_defer_dawn_tail_guard",
                    }
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
                elif strategy == "official_dawn":
                    generated = decode_official_dawn(
                        model,
                        tokenizer,
                        prompt,
                        generation=config["generation"],
                        dawn_config=probe_config.get("dawn", {}),
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
                        adapter=adapter,
                        diagnostics_dir=diagnostics_dir,
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
                    "model_implementation": resolved_model_implementation,
                    "seed": seed,
                    "question": question,
                    "reference_answer": reference,
                    "num_fewshot": int(config["data"].get("num_fewshot", 0)),
                    "apply_chat_template": bool(
                        config["data"].get("apply_chat_template", True)
                    ),
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
        "model_implementation": resolved_model_implementation,
        "transformers_version": transformers.__version__,
        "dawn_revision": dawn_revision if uses_dawn else None,
        "implementation_notes": [
            "Attention visibility is unchanged from Dream.",
            "Regional modes use one linear Dream timestep schedule per fixed region.",
            "Backpressure clocks count non-empty commitment events; a separate schedule cursor consumes zero-transfer Dream schedule points.",
            "flowblock_proxy transfers only W=2, theta_spawn=0.60, and token-readiness probability 0.50 into the Dream regional admission harness; it is not an implementation of FlowBlock's T2T editing, block-causal KV cache, or threshold commit policy.",
            "The 15% spawn threshold is theta_spawn; the separate token confidence threshold defaults to 0.5, matching FlowBlock's reported math setting but not Dream's entropy commit rule.",
            "loose_wavefront is the graph-free W=8, theta_spawn=0.15 readiness-only ablation; unlike controlled_position it has no permanent positional bounded-skew edges.",
            "vanilla_steps32/64/72/96/128 call the official Dream diffusion_generate path with only the configured global step count changed; max_new_tokens remains 256.",
            "vanilla_steps72 is the primary compute-matched control for regional strategies that previously averaged about 71 NFEs.",
            "mean_field_repro implements Algorithm 1 from arXiv:2606.15805 with exact JSD inside sequential active blocks. The paper's linked GitHub repository currently returns 404, so this is not labelled official code.",
            "The Mean-Field pseudocode does not define an empty-commit fallback. This reproduction commits the maximum-intensity token to guarantee progress and logs every such event.",
            "Controlled modes normally advance every unblocked admitted region; they do not round-robin.",
            "controlled_position uses loose confidence admission plus adjacent positional bounded-skew control.",
            "controlled_position_tail_guard is a coarse terminal-region ablation: while an earlier fixed region remains unfinished, it withholds the region containing the earliest currently masked top-1 stop-token prediction and every later region from forced progress.",
            "always_on_tail_guard applies the same coarse terminal-region guard with all regions admitted from iteration zero and no positional backpressure.",
            "always_on_bounded_defer keeps every region active from iteration zero but withholds a region's ordinary local update when the least-confident token in that update's quota is below the raw, untempered top-1 probability threshold. The local schedule cursor does not advance on a confidence deferral.",
            "Bounded deferral forces the ordinary regional update after the configured number of consecutive skips. Natural zero-token points in Dream's transfer schedule advance the schedule cursor and do not count as confidence deferrals.",
            "always_on_bounded_defer_tail_guard combines bounded regional deferral with the identical predicted terminal-region guard; it has no admission window or positional backpressure.",
            "always_on_coupled_defer starts every region immediately, permits low-confidence regional skips, and uses adjacent positional edges to bound revealed-token staleness. When an endpoint is paused at the gap, its lagging neighbor's ordinary update bypasses the confidence gate.",
            "Coupled deferral has no per-region wall-clock skip deadline: if a parent also skips, the child does not consume positional slack. Four globally empty confidence-deferral iterations trigger one forced leftmost active update solely to prevent a total fixed-point deadlock.",
            "When deferral_until_revealed_tokens is set, confidence deferral is used only until each region reaches that many actually revealed tokens. Positional gap backpressure and the optional tail guard continue for the rest of decoding.",
            "The two stop-aware strategies retain the coarse guard for every region strictly after the earliest predicted terminal region, but allow that terminal region itself to participate while its left prefix is unfinished.",
            "always_on_coupled_defer_stop_filter excludes sampled EOS/EOT/IM_END proposals inside that terminal region and backfills the ordinary quota only with non-stop proposals whose raw proposed-token probability clears stop_filter_confidence_threshold.",
            "always_on_coupled_defer_stop_defer instead preserves the ordinary regional position selection and defers its entire local transition whenever a selected position's raw top-1 or sampled token is EOS/EOT/IM_END. It does not backfill from lower-ranked positions.",
            "Stop protection has priority over confidence and gap forcing. A terminal region stalled by stop protection temporarily exempts its left neighbor from the maximum-lead pause, allowing the unfinished prefix to complete; the stalled child is still prevented from outrunning its parent.",
            "For the two stop-aware strategies only, the first actually committed stop token latches the endpoint. Positions after it are ignored, holes before it continue decoding, and generation terminates once that prefix contains no masks. Synthetic suffix completion is not counted as model commitment.",
            "Regional DAWN strategies load the pinned official DAWN Dream model fork and use its one-forward late-layer averaged attention scores. The model still has full-canvas visibility.",
            "official_dawn calls the pinned released DAWN sequential 32-token-block decoder directly with its released Dream GSM8K thresholds, temperature zero, and no top-p/top-k filtering. It changes no regional scheduling code and reports DAWN's returned actual NFE.",
            "When model.implementation is dawn_release, every strategy uses the pinned DAWN Dream model fork. In that configuration, vanilla reproduces DAWN's released Original entropy path: eight sequential 32-token blocks, 32 local steps per block, and 256 actual forward passes.",
            "always_on_dawn_tail_guard applies DAWN's anchor/conflict token selector independently to every unguarded region; always_on_coupled_defer_dawn_tail_guard adds the existing startup deferral and adjacent revealed-token backpressure.",
            "The regional DAWN selector uses the official GSM8K thresholds (sink 0.03, edge 0.10, induced 0.75, candidate confidence 0.80, high confidence 0.90) unless overridden.",
            "DAWN threshold confidence is computed from raw untempered logits, matching DAWN's released temperature-zero operating point. Token predictions retain this experiment's configured Dream sampler so the ablation changes selection rather than silently changing the benchmark sampling protocol.",
            "The public DAWN MIS helper continues for the original candidate count after suppressing conflicts and can therefore select non-candidates. This pilot implements the intended greedy MIS termination when no eligible node remains and records that deviation explicitly.",
            "Progress is actual revealed-token count. A higher-position child is paused if it outruns its parent. In coupled-deferral modes, a parent is paused when its lead reaches the resolved max_progress_gap; the resolved value is stored in metadata.config.probe and in every regional result row.",
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
