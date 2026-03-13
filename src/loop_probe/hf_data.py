import json
import os
import random
from collections.abc import Sequence
from dataclasses import asdict

from datasets import load_dataset

from .types import DatasetSpec, SampleRecord


def specs_equal(a: DatasetSpec, b: DatasetSpec) -> bool:
    return asdict(a) == asdict(b)


def _is_default_aime_jsonl(dataset_path: str) -> bool:
    return os.path.basename(dataset_path) == "aime_2024_2025.jsonl"


def _load_local_jsonl_records(spec: DatasetSpec, prompt_field: str) -> list[SampleRecord]:
    if spec.max_samples is not None and spec.max_samples < 1:
        raise SystemExit("--*-max-samples must be >= 1 when provided.")

    rows: list[dict] = []
    with open(spec.dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if spec.max_samples is not None:
        rows = rows[: spec.max_samples]

    if not rows:
        return []

    if _is_default_aime_jsonl(spec.dataset):
        required = ("question", "answer")
        missing = [key for key in required if key not in rows[0]]
        if missing:
            raise SystemExit(
                f"Default AIME dataset '{spec.dataset}' is missing required key(s): {missing}"
            )
        actual_prompt_field = "question"
    else:
        if prompt_field not in rows[0]:
            raise SystemExit(
                f"Prompt field '{prompt_field}' not found in local dataset '{spec.dataset}'. "
                f"Available keys include: {sorted(rows[0].keys())}"
            )
        actual_prompt_field = prompt_field

    records: list[SampleRecord] = []
    for idx, row in enumerate(rows):
        if _is_default_aime_jsonl(spec.dataset):
            if "question" not in row or "answer" not in row:
                raise SystemExit(
                    f"Row {idx} in '{spec.dataset}' must include both 'question' and 'answer'."
                )
        prompt = row.get(actual_prompt_field)
        if prompt is None:
            continue
        sample_id = row.get("_source_sample_id", idx)
        try:
            sample_id = int(sample_id)
        except Exception as exc:
            raise SystemExit(
                f"Row {idx} in '{spec.dataset}' has invalid _source_sample_id={sample_id!r}."
            ) from exc
        if sample_id < 0:
            raise SystemExit(
                f"Row {idx} in '{spec.dataset}' has negative _source_sample_id={sample_id}."
            )
        records.append(
            SampleRecord(sample_id=sample_id, prompt=str(prompt), source_split=spec.split)
        )
    return records


def load_prompt_records(spec: DatasetSpec, prompt_field: str) -> list[SampleRecord]:
    if os.path.isfile(spec.dataset):
        return _load_local_jsonl_records(spec, prompt_field)

    ds = load_dataset(spec.dataset, spec.config, split=spec.split)
    if prompt_field not in ds.column_names:
        raise SystemExit(
            f"Prompt field '{prompt_field}' not found in dataset columns: {ds.column_names}"
        )

    if spec.max_samples is not None:
        if spec.max_samples < 1:
            raise SystemExit("--*-max-samples must be >= 1 when provided.")
        limit = min(len(ds), spec.max_samples)
        ds = ds.select(range(limit))

    records: list[SampleRecord] = []
    for idx, row in enumerate(ds):
        prompt = row[prompt_field]
        if prompt is None:
            continue
        records.append(
            SampleRecord(sample_id=idx, prompt=str(prompt), source_split=spec.split)
        )

    return records


def split_records(
    records: Sequence[SampleRecord],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if not 0.0 < test_ratio < 1.0:
        raise SystemExit("test_ratio must be in (0, 1).")

    work = list(records)
    if len(work) < 2:
        raise SystemExit("Need at least 2 rows to split train/test from a single dataset.")

    rng = random.Random(seed)
    rng.shuffle(work)

    test_size = max(1, int(round(len(work) * test_ratio)))
    if test_size >= len(work):
        test_size = len(work) - 1

    test_records = work[:test_size]
    train_records = work[test_size:]
    return train_records, test_records
