from __future__ import annotations

import re
import textwrap
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
    if task == "humaneval":
        return BenchmarkExample(str(source["prompt"]), str(source["task_id"]))
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


def _python_code_from_generation(generation: str) -> str:
    fenced = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```",
        generation,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return (fenced[0] if fenced else generation).strip()


def _humaneval_solution(generation: str, source: dict[str, Any]) -> str:
    """Convert a chat-model answer into one complete HumanEval solution."""
    code = _python_code_from_generation(generation)
    prompt = str(source["prompt"])
    entry_point = re.escape(str(source["entry_point"]))
    definition = re.search(
        rf"(?m)^[ \t]*(?:async\s+)?def\s+{entry_point}\s*\(", code
    )
    if definition is not None:
        # HumanEval prompts sometimes carry imports before the target definition.
        prompt_definition = re.search(r"(?m)^\s*(?:async\s+)?def\s+", prompt)
        prefix = prompt[: prompt_definition.start()] if prompt_definition else ""
        generated_prefix = code[: definition.start()]
        generated_import = re.search(r"(?m)^(?:from|import)\s+", generated_prefix)
        start = generated_import.start() if generated_import else definition.start()
        return prefix + textwrap.dedent(code[start:])

    # A completion-only response lost its first-line indentation when Dream's
    # decoded response was stripped. Normalize it and append it to the prompt.
    lines = code.splitlines()
    body = "    " + lines[0] if lines else "    pass"
    if len(lines) > 1:
        body += "\n" + "\n".join(lines[1:])
    return prompt + body + "\n"


def score_generation(
    data: dict[str, Any],
    generation: str,
    reference: str,
    source: dict[str, Any] | None = None,
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

    if task == "humaneval":
        if not bool(data.get("allow_code_execution", False)):
            raise RuntimeError(
                "HumanEval executes model-generated Python. Set "
                "data.allow_code_execution=true only inside an isolated worker."
            )
        if source is None:
            raise ValueError("HumanEval scoring requires the original dataset row")
        try:
            from human_eval.execution import check_correctness
        except ImportError as exc:
            raise RuntimeError(
                "HumanEval scoring requires the evaluation extra installed by "
                "scripts/setup_vast.sh"
            ) from exc
        problem = {
            "task_id": str(source["task_id"]),
            "prompt": "",
            "canonical_solution": str(source["canonical_solution"]),
            "test": str(source["test"]),
            "entry_point": str(source["entry_point"]),
        }
        solution = _humaneval_solution(generation, source)
        outcome = check_correctness(
            problem,
            solution,
            timeout=float(data.get("execution_timeout_seconds", 3.0)),
            completion_id=0,
        )
        return (
            str(source["entry_point"]),
            bool(outcome["passed"]),
            "openai_humaneval_pass_at_1",
        )

    raise ValueError(f"Unsupported benchmark task {task!r}")
