from __future__ import annotations

import json
import os
from collections.abc import Sequence

from datasets import load_dataset

from ._common import (
    extract_answer_letter_from_last_lines,
    load_local_rows,
    resolve_sample_id,
)
from ..types import DatasetSpec, SampleRecord

MMLUPRO_LETTERS = tuple(chr(ord("A") + idx) for idx in range(10))


def _normalize_options(value: object, idx: int) -> list[str]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            raise SystemExit(
                f"Row {idx} has string 'options' but it is not valid JSON."
            ) from None
        value = decoded

    if not isinstance(value, Sequence) or isinstance(value, (bytes, str)):
        raise SystemExit(f"Row {idx} has invalid MMLU-Pro options: {value!r}")

    options = [str(option) for option in value]
    if len(options) > len(MMLUPRO_LETTERS):
        raise SystemExit(
            f"Row {idx} has {len(options)} options; max supported is {len(MMLUPRO_LETTERS)}."
        )
    return options


def load_samples(
    spec: DatasetSpec,
) -> list[tuple[SampleRecord, list[str], str, int | None]]:
    if spec.max_samples is not None and spec.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")

    if os.path.isfile(spec.dataset):
        rows = load_local_rows(spec.dataset)
        if spec.max_samples is not None:
            rows = rows[: spec.max_samples]
    else:
        ds = load_dataset(spec.dataset, spec.config, split=spec.split)
        required = {"question", "options"}
        if not required.issubset(set(ds.column_names)):
            raise SystemExit(
                f"MMLU-Pro dataset must include {sorted(required)}; found {list(ds.column_names)}."
            )
        if spec.max_samples is not None:
            ds = ds.select(range(min(len(ds), spec.max_samples)))
        rows = list(ds)

    samples: list[tuple[SampleRecord, list[str], str, int | None]] = []
    for idx, row in enumerate(rows):
        if "question" not in row or "options" not in row:
            raise SystemExit(f"Row {idx} is missing 'question' or 'options'.")
        question = row["question"]
        if question is None:
            continue
        options = _normalize_options(row["options"], idx)
        gold_answer = row.get("answer")
        gold_index = row.get("answer_index")
        if gold_answer is None and gold_index is None:
            raise SystemExit(
                f"Row {idx} must include either 'answer' or 'answer_index'."
            )
        gold_answer_str = str(gold_answer) if gold_answer is not None else ""
        try:
            gold_index_int = int(gold_index) if gold_index is not None else None
        except Exception as exc:
            raise SystemExit(f"Row {idx} has invalid answer_index={gold_index!r}.") from exc
        sample_id = resolve_sample_id(row.get("_source_sample_id", idx), idx)
        samples.append(
            (
                SampleRecord(
                    sample_id=sample_id,
                    prompt=str(question),
                    source_split=spec.split,
                ),
                options,
                gold_answer_str,
                gold_index_int,
            )
        )
    return samples


def build_mcq_prompt(tokenizer, question: str, options: list[str]) -> str:
    if len(options) > len(MMLUPRO_LETTERS):
        raise ValueError(
            f"MMLU-Pro expects at most {len(MMLUPRO_LETTERS)} options, got {len(options)}."
        )
    valid_letters = MMLUPRO_LETTERS[: len(options)]
    option_block = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(valid_letters, options)
    )
    user_msg = (
        f"{question}\n\n"
        f"Answer choices:\n{option_block}\n\n"
        "Think through the problem carefully if needed. "
        "The final non-empty line must be exactly `Answer: X`, "
        f"where X is one of {', '.join(valid_letters)}."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


def grade(
    response: str,
    gold_answer: str,
    gold_index: int | None,
) -> bool:
    predicted = extract_answer_letter_from_last_lines(
        response,
        MMLUPRO_LETTERS,
    )
    if predicted is None:
        return False

    gold_candidates = set()
    gold_answer = gold_answer.strip().upper()
    if gold_answer:
        parsed_gold = extract_answer_letter_from_last_lines(
            gold_answer,
            MMLUPRO_LETTERS,
            max_lines=1,
        )
        gold_candidates.add(parsed_gold or gold_answer)
    if gold_index is not None:
        if gold_index < 0 or gold_index >= len(MMLUPRO_LETTERS):
            raise ValueError(
                f"gold_index must be in [0, {len(MMLUPRO_LETTERS)}), got {gold_index}."
            )
        gold_candidates.add(MMLUPRO_LETTERS[gold_index])
    return predicted in gold_candidates
