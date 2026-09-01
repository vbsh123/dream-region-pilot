#!/usr/bin/env python3
"""Export pilot HumanEval results in Fast-dLLM's lm-eval sample format.

This intentionally performs no code cleanup. The exported files can be passed
to Fast-dLLM's ``postprocess_code.py`` so both implementations are evaluated
with the same sanitizer and code-eval scorer.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = load_dataset("openai/openai_humaneval", split="test")
    exported: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        example_index = int(row["example_index"])
        document = dict(dataset[example_index])
        exported[str(row["strategy"])].append(
            {
                "doc_id": example_index,
                "doc": document,
                "target": document["test"],
                "resps": [[str(row["generation"])]],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for strategy, samples in exported.items():
        samples.sort(key=lambda sample: int(sample["doc_id"]))
        destination = args.output_dir / f"{strategy}.jsonl"
        destination.write_text(
            "".join(json.dumps(sample) + "\n" for sample in samples),
            encoding="utf-8",
        )
        print(f"{strategy}: {len(samples)} samples -> {destination}")


if __name__ == "__main__":
    main()
