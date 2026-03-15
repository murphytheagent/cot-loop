from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable

_ANSWER_LINE_PATTERN = re.compile(r"^Answer: ([A-Z])$")


def resolve_sample_id(value: object, default: int) -> int:
    try:
        sample_id = int(value)
    except Exception:
        sample_id = default
    if sample_id < 0:
        return default
    return sample_id


def load_local_rows(path: str) -> list[dict[str, object]]:
    if path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_answer_letter_from_last_lines(
    response: str,
    valid_letters: Iterable[str],
    *,
    max_lines: int = 6,
) -> str | None:
    allowed = {
        str(letter).strip().upper()
        for letter in valid_letters
        if str(letter).strip()
    }
    if not allowed:
        raise ValueError("valid_letters must contain at least one non-empty letter.")

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1]
    match = _ANSWER_LINE_PATTERN.match(candidate)
    if not match:
        return None
    letter = match.group(1).upper()
    if letter in allowed:
        return letter
    return None
