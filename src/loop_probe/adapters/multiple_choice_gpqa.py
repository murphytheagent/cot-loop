from __future__ import annotations

import os
import random

from datasets import load_dataset

from ._common import (
    extract_answer_letter_from_last_lines,
    load_local_rows,
    resolve_sample_id,
)
from ..types import DatasetSpec, SampleRecord

GPQA_LETTERS = ("A", "B", "C", "D")


def _require_config(spec: DatasetSpec) -> None:
    if os.path.isfile(spec.dataset):
        return
    if spec.dataset == "Idavidrein/gpqa" and not spec.config:
        raise SystemExit(
            "GPQA requires an explicit --dataset-config, e.g. gpqa_main or gpqa_diamond."
        )


def _required_row_fields(row: dict[str, object], idx: int) -> tuple[str, str, list[str]]:
    required = (
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise SystemExit(f"Row {idx} is missing GPQA column(s): {missing}.")

    question = row["Question"]
    correct = row["Correct Answer"]
    incorrect = [row[f"Incorrect Answer {i}"] for i in range(1, 4)]
    if question is None or correct is None or any(value is None for value in incorrect):
        raise SystemExit(f"Row {idx} contains a null GPQA field.")
    return str(question), str(correct), [str(value) for value in incorrect]


def load_and_shuffle(
    spec: DatasetSpec,
    seed: int,
) -> list[tuple[SampleRecord, list[str], str]]:
    _require_config(spec)
    if spec.max_samples is not None and spec.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")

    if os.path.isfile(spec.dataset):
        rows = load_local_rows(spec.dataset)
        if spec.max_samples is not None:
            rows = rows[: spec.max_samples]
    else:
        ds = load_dataset(spec.dataset, spec.config, split=spec.split)
        if spec.max_samples is not None:
            ds = ds.select(range(min(len(ds), spec.max_samples)))
        rows = list(ds)

    samples: list[tuple[SampleRecord, list[str], str]] = []
    for idx, row in enumerate(rows):
        question, correct, incorrect = _required_row_fields(row, idx)
        sample_id = resolve_sample_id(row.get("_source_sample_id", idx), idx)
        shuffled = [(correct, True)] + [(option, False) for option in incorrect]
        rng = random.Random(seed ^ sample_id)
        rng.shuffle(shuffled)
        options = [option for option, _ in shuffled]
        gold_index = next(i for i, (_, is_gold) in enumerate(shuffled) if is_gold)
        gold_letter = GPQA_LETTERS[gold_index]
        samples.append(
            (
                SampleRecord(
                    sample_id=sample_id,
                    prompt=question,
                    source_split=spec.split,
                ),
                options,
                gold_letter,
            )
        )
    return samples


def build_mcq_prompt(tokenizer, question: str, options: list[str]) -> str:
    if len(options) != 4:
        raise ValueError(f"GPQA expects 4 options, got {len(options)}.")
    option_block = "\n".join(
        f"{letter}. {option}" for letter, option in zip(GPQA_LETTERS, options)
    )
    user_msg = (
        f"{question}\n\n"
        f"Answer choices:\n{option_block}\n\n"
        "Think through the problem carefully if needed. "
        "The final non-empty line must be exactly `Answer: X`, "
        "where X is one of A, B, C, or D."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def grade(response: str, gold_letter: str) -> bool:
    predicted = extract_answer_letter_from_last_lines(
        response,
        GPQA_LETTERS,
    )
    if predicted is None:
        return False
    return predicted == gold_letter
