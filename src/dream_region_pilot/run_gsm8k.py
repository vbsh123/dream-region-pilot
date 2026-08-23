from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .config import load_config
from .decoding import decode_regional, decode_vanilla
from .dependencies import DAPDDreamAdapter
from .evaluation import extract_answer, write_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Dream fixed-region pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
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


def encode_prompt(tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(
        rendered, add_special_tokens=False, return_tensors="pt"
    ).to(device)


def load_gsm8k(config: dict[str, Any], limit_override: int | None):
    data = config["data"]
    dataset = load_dataset(data["dataset"], data["subset"], split=data["split"])
    limit = int(limit_override if limit_override is not None else data["limit"])
    return dataset.select(range(min(limit, len(dataset))))


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
    allowed = {
        "vanilla",
        "fixed_sequential",
        "always_on",
        "async_lag0",
        "async_lag1",
        "async_lag2",
        "async_lag4",
        "wavefront_probe",
        "controlled_dapd",
        "controlled_jsd",
        "controlled_combo",
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

    dataset = load_gsm8k(config, args.limit)
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
        for example_index in range(len(dataset))
        for strategy in strategies
    )
    with results_path.open("a", encoding="utf-8") as handle:
        progress = tqdm(total=total, desc="Dream region pilot")
        for example_index, source in enumerate(dataset):
            question = str(source["question"])
            prompt_text = str(config["data"]["prompt_template"]).format(
                question=question
            )
            prompt = encode_prompt(tokenizer, prompt_text, model.device)
            reference = extract_answer(str(source["answer"]))
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
                if example_index < diagnostics_count and (
                    strategy.startswith("async_")
                    or strategy == "wavefront_probe"
                    or strategy.startswith("controlled_")
                ):
                    diagnostics_dir = (
                        output_dir
                        / "diagnostics"
                        / f"example_{example_index:03d}"
                        / strategy
                    )
                if strategy == "vanilla":
                    generated = decode_vanilla(
                        model, tokenizer, prompt, config["generation"]
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
                prediction = extract_answer(generated["generation"])
                row = {
                    "example_index": example_index,
                    "strategy": strategy,
                    "seed": seed,
                    "question": question,
                    "reference_answer": reference,
                    "predicted_answer": prediction,
                    "correct": prediction == reference,
                    **generated,
                }
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
                rows.append(row)
                progress.update(1)
        progress.close()

    metadata = {
        "artifact_type": "dream_fixed_region_async_pilot",
        "config": config,
        "examples": len(dataset),
        "strategies": strategies,
        "model_resolved_commit": getattr(model.config, "_commit_hash", None),
        "dapd_revision": adapter.revision,
        "implementation_notes": [
            "Attention visibility is unchanged from Dream.",
            "DAPD post-RoPE Q/K extraction and token dependency construction are imported from the pinned public checkout.",
            "Regional modes use one linear Dream timestep schedule per fixed region.",
            "Backpressure clocks count non-empty commitment events; a separate schedule cursor consumes zero-transfer Dream schedule points.",
            "Completed parents release children because strict positive lag otherwise deadlocks terminal child steps.",
            "wavefront_probe admits at most one positional region per forward, uses W as a maximum active-region count, and uses a FlowBlock-style confidence acceptance ratio only for admission.",
            "The 15% spawn threshold is theta_spawn; the separate token confidence threshold defaults to 0.5, matching FlowBlock's reported math setting but not Dream's entropy commit rule.",
            "Mean-Field JSD is diagnostic-only in wavefront_probe; controlled_jsd and controlled_combo use its persistent region edges for pausing decisions but never change admission or token scoring.",
            "The all-canvas Mean-Field signal uses a top-k-union plus shared-tail approximation; retained probability mass is logged and paper_exact is false unless top-k equals the vocabulary size.",
            "Controlled modes activate an edge only after both endpoints reveal a token and the edge persists for two graph observations.",
            "Controlled modes normally advance every unblocked admitted region; they do not round-robin or graph-color components.",
            "Progress is actual revealed-token count. A higher-position child is paused if it outruns its parent, while a parent is paused only when its lead exceeds the configurable eight-token default.",
            "Urgent service never forces an extra low-confidence token; it schedules the lagging region's next ordinary Dream local update.",
            "GSM8K uses a zero-shot chat prompt requesting a #### numeric answer; this is not claimed to reproduce DAPD paper evaluation prompting.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(rows, output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
