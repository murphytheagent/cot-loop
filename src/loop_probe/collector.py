from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .configs import RolloutConfig

ALL_STATISTICS = (
    "success_fraction",
    "loop_fraction",
    "avg_generation_length",
    "avg_loop_generation_length",
    "avg_first_loop_prefix_length",
    "max_length_hit_fraction",
    "loop_max_length_hit_fraction",
    "max_length_hit_loop_fraction",
    "generation_length_variance",
    "max_length_hit_success_fraction",
    "loop_success_fraction",
    "avg_correct_generation_length",
    "avg_wrong_generation_length",
)


@dataclass(frozen=True)
class CollectorConfig:
    rollout_cfg: RolloutConfig
    seed: int
    task_kind: str
    statistics: list[str]
    livecodebench_repo: str | None = None
    release_version: str = "release_v6"
    lm_style_override: str | None = None


@dataclass(frozen=True)
class LcbSampleRecord:
    question_id: str
    generation_index: int
    code_output: str
    token_count: int
    prompt_token_count: int
    total_token_count: int
    effective_max_tokens: int
    max_model_len: int
    loop_flag: bool
    max_length_hit: bool
    finish_reason: str
    prompt_too_long: bool


@dataclass
class WorkerAggregator:
    num_samples_seen: int = 0
    num_generated: int = 0
    num_graded: int = 0
    num_correct: int = 0
    num_wrong: int = 0
    num_looped: int = 0
    num_max_length_hits: int = 0
    num_prompt_too_long: int = 0
    num_looped_and_max_length_hit: int = 0
    num_correct_and_looped: int = 0
    num_correct_and_max_length_hit: int = 0
    length_sum: int = 0
    length_sq_sum: int = 0
    loop_length_sum: int = 0
    correct_length_sum: int = 0
    wrong_length_sum: int = 0
    first_loop_prefix_sum: int = 0
    prompt_length_sum: int = 0
    prompt_length_min: int | None = None
    prompt_length_max: int | None = None
    lcb_sample_records: list[LcbSampleRecord] = field(default_factory=list)


def merge_aggregators(aggregators: Iterable[WorkerAggregator]) -> WorkerAggregator:
    merged = WorkerAggregator()
    for agg in aggregators:
        merged.num_samples_seen += int(agg.num_samples_seen)
        merged.num_generated += int(agg.num_generated)
        merged.num_graded += int(agg.num_graded)
        merged.num_correct += int(agg.num_correct)
        merged.num_wrong += int(agg.num_wrong)
        merged.num_looped += int(agg.num_looped)
        merged.num_max_length_hits += int(agg.num_max_length_hits)
        merged.num_prompt_too_long += int(agg.num_prompt_too_long)
        merged.num_looped_and_max_length_hit += int(agg.num_looped_and_max_length_hit)
        merged.num_correct_and_looped += int(agg.num_correct_and_looped)
        merged.num_correct_and_max_length_hit += int(agg.num_correct_and_max_length_hit)
        merged.length_sum += int(agg.length_sum)
        merged.length_sq_sum += int(agg.length_sq_sum)
        merged.loop_length_sum += int(agg.loop_length_sum)
        merged.correct_length_sum += int(agg.correct_length_sum)
        merged.wrong_length_sum += int(agg.wrong_length_sum)
        merged.first_loop_prefix_sum += int(agg.first_loop_prefix_sum)
        merged.prompt_length_sum += int(agg.prompt_length_sum)
        if agg.prompt_length_min is not None:
            merged.prompt_length_min = (
                agg.prompt_length_min
                if merged.prompt_length_min is None
                else min(merged.prompt_length_min, agg.prompt_length_min)
            )
        if agg.prompt_length_max is not None:
            merged.prompt_length_max = (
                agg.prompt_length_max
                if merged.prompt_length_max is None
                else max(merged.prompt_length_max, agg.prompt_length_max)
            )
        merged.lcb_sample_records.extend(agg.lcb_sample_records)
    return merged


def _safe_div(num: int | float, denom: int | float) -> float | None:
    if denom == 0:
        return None
    return float(num) / float(denom)


def compute_metrics(
    agg: WorkerAggregator,
    requested: Iterable[str],
) -> dict[str, float | None]:
    requested_list = list(requested)
    unknown = sorted(set(requested_list) - set(ALL_STATISTICS))
    if unknown:
        raise ValueError(
            f"Unknown statistic(s): {unknown}. Valid choices: {list(ALL_STATISTICS)}"
        )

    metrics: dict[str, float | None] = {}
    n = agg.num_generated

    for name in requested_list:
        if name == "success_fraction":
            metrics[name] = _safe_div(agg.num_correct, agg.num_graded)
        elif name == "loop_fraction":
            metrics[name] = _safe_div(agg.num_looped, agg.num_generated)
        elif name == "avg_generation_length":
            metrics[name] = _safe_div(agg.length_sum, agg.num_generated)
        elif name == "avg_loop_generation_length":
            metrics[name] = _safe_div(agg.loop_length_sum, agg.num_looped)
        elif name == "avg_first_loop_prefix_length":
            metrics[name] = _safe_div(agg.first_loop_prefix_sum, agg.num_looped)
        elif name == "max_length_hit_fraction":
            metrics[name] = _safe_div(agg.num_max_length_hits, agg.num_generated)
        elif name == "loop_max_length_hit_fraction":
            metrics[name] = _safe_div(
                agg.num_looped_and_max_length_hit,
                agg.num_looped,
            )
        elif name == "max_length_hit_loop_fraction":
            metrics[name] = _safe_div(
                agg.num_looped_and_max_length_hit,
                agg.num_max_length_hits,
            )
        elif name == "generation_length_variance":
            if n == 0:
                metrics[name] = None
            else:
                mean = agg.length_sum / n
                variance = (agg.length_sq_sum / n) - (mean * mean)
                if variance < 0.0 and abs(variance) < 1e-12:
                    variance = 0.0
                metrics[name] = float(variance)
        elif name == "max_length_hit_success_fraction":
            metrics[name] = _safe_div(
                agg.num_correct_and_max_length_hit,
                agg.num_max_length_hits,
            )
        elif name == "loop_success_fraction":
            metrics[name] = _safe_div(agg.num_correct_and_looped, agg.num_looped)
        elif name == "avg_correct_generation_length":
            metrics[name] = _safe_div(agg.correct_length_sum, agg.num_correct)
        elif name == "avg_wrong_generation_length":
            metrics[name] = _safe_div(agg.wrong_length_sum, agg.num_wrong)

    return metrics
