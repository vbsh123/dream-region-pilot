from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

from datasets import load_dataset

from .evaluation import extract_answer


@dataclass(frozen=True)
class BenchmarkExample:
    question: str
    reference_answer: str


def load_benchmark(data: dict[str, Any], limit_override: int | None):
    subset = data.get("subset")
    if subset in (None, ""):
        dataset = load_dataset(data["dataset"], split=data["split"])
    else:
        dataset = load_dataset(
            data["dataset"], str(subset), split=data["split"]
        )
    limit = int(limit_override if limit_override is not None else data["limit"])
    return dataset.select(range(min(limit, len(dataset))))


def prepare_example(data: dict[str, Any], source: dict[str, Any]) -> BenchmarkExample:
    task = str(data.get("task", "gsm8k"))
    if task == "gsm8k":
        reference = extract_answer(str(source["answer"]))
        if reference is None:
            raise ValueError("GSM8K example has no numeric reference answer")
        return BenchmarkExample(str(source["question"]), reference)
    if task == "asdiv":
        problem = "\n".join(
            part.strip()
            for part in (str(source.get("body", "")), str(source["question"]))
            if part.strip()
        )
        return BenchmarkExample(problem, str(source["answer"]))
    if task == "math500":
        return BenchmarkExample(str(source["problem"]), str(source["answer"]))
    raise ValueError(f"Unsupported benchmark task {task!r}")


def _canonical_numeric(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.removesuffix("%").strip()
    try:
        if "/" in cleaned and re.fullmatch(r"-?\d+/\d+", cleaned):
            fraction = Fraction(cleaned)
        else:
            fraction = Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return cleaned
    return (
        str(fraction.numerator)
        if fraction.denominator == 1
        else f"{fraction.numerator}/{fraction.denominator}"
    )


def _last_boxed(text: str) -> str | None:
    last = None
    start = 0
    while True:
        command = text.find("\\boxed", start)
        if command < 0:
            return last


def _final_marked_text(text: str) -> str | None:
    marked = re.findall(r"(?im)^\s*####\s*(.+?)\s*$", text)
    if marked:
        return marked[-1].strip()
    boxed = _last_boxed(text)
    if boxed:
        return boxed
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _canonical_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\([^)]*\)", "", value)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
    return cleaned or None
        opening = text.find("{", command + len("\\boxed"))
        if opening < 0:
            return last
        depth = 0
        for index in range(opening, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    last = text[opening + 1 : index].strip()
                    start = index + 1
                    break
        else:
            return last


def score_generation(
    data: dict[str, Any],
    generation: str,
    reference: str,
) -> tuple[str | None, bool, str]:
    task = str(data.get("task", "gsm8k"))
    if task == "gsm8k":
        prediction = extract_answer(generation)
        correct = _canonical_numeric(prediction) == _canonical_numeric(reference)
        return prediction, correct, "numeric_final_answer"

    if task == "asdiv":
        reference_number = extract_answer(reference)
        if reference_number is not None:
            prediction = extract_answer(generation)
            correct = _canonical_numeric(prediction) == _canonical_numeric(
                reference_number
            )
            return prediction, correct, "asdiv_numeric_final_answer"
        prediction = _final_marked_text(generation)
        correct = _canonical_text(prediction) == _canonical_text(reference)
        return prediction, correct, "asdiv_text_final_answer"

    if task == "math500":
        try:
            from math_verify import parse, verify
        except ImportError as exc:
            raise RuntimeError(
                "MATH-500 scoring requires the evaluation extra installed by "
                "scripts/setup_vast.sh"
            ) from exc
        gold = parse(f"${reference}$")
        predicted = parse(generation)
        correct = bool(gold and predicted and verify(gold, predicted))
        display = _last_boxed(generation)
        if display is None:
            lines = [line.strip() for line in generation.splitlines() if line.strip()]
            display = lines[-1] if lines else None
        return display, correct, "math_verify_0.9.0"

    raise ValueError(f"Unsupported benchmark task {task!r}")
