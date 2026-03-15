#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_STATS_DIR = Path("outputs/qwen3_1p7b_cross_dataset_rollout_stats")
FIGURES_DIRNAME = "figures"
EXPECTED_STATS_CONTRACT_VERSION = "rollout_stats_v2"


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    filename: str
    display_name: str
    short_name: str
    task_kind: str
    expected_dataset: str
    description: str
    chat_format: str
    expected_dataset_config: str | None = None
    expected_split: str = "test"
    expected_model_id: str = "Qwen/Qwen3-1.7B"
    sample_note: str | None = None


DATASETS: tuple[DatasetInfo, ...] = (
    DatasetInfo(
        key="math500",
        filename="HuggingFaceH4_MATH-500__test__Qwen_Qwen3-1.7B.json",
        display_name="MATH-500",
        short_name="MATH-500",
        task_kind="math_freeform",
        expected_dataset="HuggingFaceH4/MATH-500",
        description=(
            "500 free-form math problems from the HuggingFaceH4 MATH-500 test split."
        ),
        chat_format=(
            "Tokenizer chat template with one user turn: the raw problem text "
            "followed by 'You must put your final answer within \\\\boxed{}.'"
        ),
    ),
    DatasetInfo(
        key="aime",
        filename="aime_2024_2025.jsonl__test__Qwen_Qwen3-1.7B.json",
        display_name="AIME 2024/2025",
        short_name="AIME",
        task_kind="math_freeform",
        expected_dataset="data/aime_2024_2025.jsonl",
        description=(
            "A local 60-row JSONL made by concatenating the AIME 2024 and 2025 test "
            "splits into question/answer records."
        ),
        chat_format=(
            "The same math-freeform Qwen chat template as MATH-500: one user turn "
            "with the problem plus a final-answer-in-box instruction."
        ),
    ),
    DatasetInfo(
        key="gpqa",
        filename="gpqa_diamond.csv__test__Qwen_Qwen3-1.7B.json",
        display_name="GPQA Diamond",
        short_name="GPQA",
        task_kind="multiple_choice_gpqa",
        expected_dataset="data/gpqa_diamond.csv",
        description=(
            "A local staged GPQA Diamond CSV with 198 graduate-level science "
            "questions and shuffled four-way answer options."
        ),
        chat_format=(
            "Tokenizer chat template with one user turn containing the question, "
            "a shuffled A-D answer block, and the instruction that the final "
            "non-empty line must be exactly 'Answer: X'."
        ),
    ),
    DatasetInfo(
        key="mmlu_pro",
        filename="TIGER-Lab_MMLU-Pro__test__Qwen_Qwen3-1.7B.json",
        display_name="MMLU-Pro",
        short_name="MMLU-Pro",
        task_kind="multiple_choice_mmlupro",
        expected_dataset="TIGER-Lab/MMLU-Pro",
        description="The TIGER-Lab MMLU-Pro test split for this rollout pass.",
        chat_format=(
            "Tokenizer chat template with one user turn containing the question, "
            "an A-J answer list, and the instruction that the final non-empty line "
            "must be exactly 'Answer: X'."
        ),
    ),
    DatasetInfo(
        key="livecodebench",
        filename="livecodebench_release_v6__test__Qwen_Qwen3-1.7B.json",
        display_name="LiveCodeBench release_v6",
        short_name="LiveCodeBench",
        task_kind="livecodebench_codegen",
        expected_dataset="livecodebench_release_v6",
        description=(
            "A question-id-sorted slice of the LiveCodeBench release_v6 "
            "code-generation benchmark."
        ),
        chat_format=(
            "LiveCodeBench's format_prompt_generation pipeline with LM style "
            "CodeQwenInstruct, producing raw string prompts rather than a tokenizer "
            "chat-template wrapper."
        ),
    ),
)


COUNT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("num_samples", "samples"),
    ("num_generated", "generated"),
    ("num_graded", "graded"),
    ("num_correct", "correct"),
    ("num_wrong", "wrong"),
    ("num_looped", "looped"),
    ("num_max_length_hits", "max_length_hits"),
    ("num_looped_and_max_length_hit", "loop_and_max_hits"),
    ("num_correct_and_looped", "correct_and_looped"),
    ("num_correct_and_max_length_hit", "correct_and_max_hits"),
    ("num_prompt_too_long", "prompt_too_long"),
)

METRIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ("success_fraction", "success_fraction"),
    ("loop_fraction", "loop_fraction"),
    ("max_length_hit_fraction", "max_length_hit_fraction"),
    ("loop_max_length_hit_fraction", "loop_max_length_hit_fraction"),
    ("max_length_hit_loop_fraction", "max_length_hit_loop_fraction"),
    ("max_length_hit_success_fraction", "max_length_hit_success_fraction"),
    ("loop_success_fraction", "loop_success_fraction"),
    ("avg_generation_length", "avg_generation_length"),
    ("avg_loop_generation_length", "avg_loop_generation_length"),
    ("avg_first_loop_prefix_length", "avg_first_loop_prefix_length"),
    ("avg_correct_generation_length", "avg_correct_generation_length"),
    ("avg_wrong_generation_length", "avg_wrong_generation_length"),
    ("generation_length_variance", "generation_length_variance"),
)

STAT_LABELS: dict[str, str] = {
    "success_fraction": "Fraction of graded rollouts with the correct answer.",
    "loop_fraction": "Fraction of generated rollouts flagged by the n-gram loop detector.",
    "avg_generation_length": "Average number of generated tokens per rollout.",
    "avg_loop_generation_length": "Average generated length restricted to looped rollouts.",
    "avg_first_loop_prefix_length": "Average prefix length before the first detected loop.",
    "max_length_hit_fraction": "Fraction of generated rollouts whose prompt-plus-generation length hit max_model_len.",
    "loop_max_length_hit_fraction": "Among looped rollouts, fraction whose prompt-plus-generation length also hit max_model_len.",
    "max_length_hit_loop_fraction": "Among max-length-hit rollouts, fraction that were also looped.",
    "generation_length_variance": "Variance of generated token counts across generated rollouts.",
    "max_length_hit_success_fraction": "Among max-length-hit rollouts, fraction that were correct.",
    "loop_success_fraction": "Among looped rollouts, fraction that were still correct.",
    "avg_correct_generation_length": "Average generated length restricted to correct rollouts.",
    "avg_wrong_generation_length": "Average generated length restricted to wrong rollouts.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--report-stem",
        default="qwen3_1p7b_cross_dataset_rollout_report",
        help="Base filename for the generated .tex and .pdf artifacts.",
    )
    parser.add_argument(
        "--pdflatex",
        default="pdflatex",
        help="pdflatex executable to use for report compilation.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write summary files and TeX but do not compile the PDF.",
    )
    return parser.parse_args()


def _normalize_local_dataset_id(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    suffix = Path(value).suffix.lower()
    if suffix in {".csv", ".json", ".jsonl", ".parquet", ".tsv"}:
        return Path(value).name
    return value


def _dataset_description(
    info: DatasetInfo,
    *,
    max_samples: Any,
) -> str:
    if info.key == "mmlu_pro" and max_samples is not None:
        return (
            "The TIGER-Lab MMLU-Pro test split, capped to "
            f"{int(max_samples)} examples for this rollout pass."
        )
    if info.key == "livecodebench" and max_samples is not None:
        return (
            "A question-id-sorted slice of the LiveCodeBench release_v6 "
            f"code-generation benchmark, capped to {int(max_samples)} "
            "problems for this rollout pass."
        )
    return info.description


def _dataset_sample_note(
    info: DatasetInfo,
    *,
    max_samples: Any,
) -> str | None:
    if max_samples is not None:
        return f"Requested cap: at most {int(max_samples)} samples."
    return info.sample_note


def load_dataset_record(stats_dir: Path, info: DatasetInfo) -> dict[str, Any]:
    path = stats_dir / info.filename
    if not path.exists():
        raise FileNotFoundError(f"Missing stats JSON: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}")

    metadata = payload.get("metadata", {})
    counts = payload.get("counts", {})
    metrics = payload.get("metrics", {})
    generation = metadata.get("generation_config", {})
    max_samples = metadata.get("max_samples")
    lcb_native_metrics = metadata.get("lcb_native_metrics", {})
    legacy_lcb_pass_at_1 = metadata.get("lcb_native_pass_at_1")
    if not lcb_native_metrics and legacy_lcb_pass_at_1 is not None:
        lcb_native_metrics = {"pass@1": legacy_lcb_pass_at_1}

    expected_pairs = {
        "task_kind": info.task_kind,
        "config": info.expected_dataset_config,
        "split": info.expected_split,
        "model_id": info.expected_model_id,
    }
    for key, expected in expected_pairs.items():
        actual = metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"{path} has unexpected metadata.{key}={actual!r}; expected {expected!r}."
            )
    actual_dataset = metadata.get("dataset")
    normalized_actual_dataset = _normalize_local_dataset_id(actual_dataset)
    normalized_expected_dataset = _normalize_local_dataset_id(info.expected_dataset)
    if normalized_actual_dataset != normalized_expected_dataset:
        raise ValueError(
            f"{path} has unexpected metadata.dataset={actual_dataset!r}; "
            f"expected {info.expected_dataset!r}."
        )

    row: dict[str, Any] = {
        "key": info.key,
        "filename": info.filename,
        "display_name": info.display_name,
        "short_name": info.short_name,
        "task_kind": info.task_kind,
        "description": _dataset_description(info, max_samples=max_samples),
        "chat_format": info.chat_format,
        "sample_note": _dataset_sample_note(info, max_samples=max_samples),
        "dataset": metadata.get("dataset"),
        "dataset_config": metadata.get("config"),
        "split": metadata.get("split"),
        "model_id": metadata.get("model_id"),
        "timestamp": metadata.get("timestamp"),
        "max_samples": max_samples,
        "statistics": list(metadata.get("statistics", [])),
        "loop_detector": metadata.get("loop_detector", {}),
        "generation_config": generation,
        "prompt_token_summary": metadata.get("prompt_token_summary", {}),
        "lcb_native_metrics": lcb_native_metrics,
        "lcb_native_pass_at_1": lcb_native_metrics.get("pass@1"),
        "lcb_native_pass_at_5": lcb_native_metrics.get("pass@5"),
        "lcb_native_pass_at_10": lcb_native_metrics.get("pass@10"),
        "stats_contract_version": metadata.get("stats_contract_version"),
        "seed": metadata.get("seed"),
        "release_version": metadata.get("release_version"),
        "lm_style": metadata.get("lm_style"),
    }
    for source_name, out_name in COUNT_COLUMNS:
        row[out_name] = counts.get(source_name)
    for source_name, out_name in METRIC_COLUMNS:
        row[out_name] = metrics.get(source_name)
    return row


def format_percent(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.{decimals}f}\\%"


def format_float(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{decimals}f}"


def format_int(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{int(value)}"


def latex_escape(text: Any) -> str:
    raw = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in raw)


def summarize_generation_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    common: dict[str, Any] = {}
    all_keys = set()
    for row in rows:
        all_keys.update(row["generation_config"].keys())
    for key in sorted(all_keys):
        values = [row["generation_config"].get(key) for row in rows]
        first = values[0]
        if all(value == first for value in values):
            common[key] = first
    return common


def validate_bundle_contract(rows: list[dict[str, Any]]) -> None:
    required_generation_keys = (
        "temperature",
        "num_generations",
        "max_tokens",
        "max_model_len",
        "dtype",
        "trust_remote_code",
    )
    mismatches = []
    baseline = rows[0]
    baseline_contract_version = baseline.get("stats_contract_version")
    if baseline_contract_version not in (None, EXPECTED_STATS_CONTRACT_VERSION):
        raise ValueError(
            "Cross-dataset rollout report requires stats_contract_version to be "
            f"either legacy None or {EXPECTED_STATS_CONTRACT_VERSION!r}, got "
            f"{baseline_contract_version!r} for {baseline['display_name']}."
        )
    for row in rows[1:]:
        if row.get("stats_contract_version") != baseline_contract_version:
            mismatches.append(
                (
                    "stats_contract_version",
                    baseline["display_name"],
                    baseline_contract_version,
                    row["display_name"],
                    row.get("stats_contract_version"),
                )
            )

    for key in required_generation_keys:
        expected = baseline["generation_config"].get(key)
        for row in rows[1:]:
            actual = row["generation_config"].get(key)
            if actual != expected:
                mismatches.append(
                    (
                        key,
                        baseline["display_name"],
                        expected,
                        row["display_name"],
                        actual,
                    )
                )

    loop_n = baseline["loop_detector"].get("n")
    loop_k = baseline["loop_detector"].get("k")
    for row in rows[1:]:
        other_n = row["loop_detector"].get("n")
        other_k = row["loop_detector"].get("k")
        if (other_n, other_k) != (loop_n, loop_k):
            mismatches.append(
                (
                    "loop_detector",
                    baseline["display_name"],
                    {"n": loop_n, "k": loop_k},
                    row["display_name"],
                    {"n": other_n, "k": other_k},
                )
            )

    for metadata_key in ("seed", "statistics"):
        expected = baseline.get(metadata_key)
        for row in rows[1:]:
            actual = row.get(metadata_key)
            if metadata_key == "statistics":
                expected_value = sorted(expected or [])
                actual_value = sorted(actual or [])
            else:
                expected_value = expected
                actual_value = actual
            if actual_value != expected_value:
                mismatches.append(
                    (
                        metadata_key,
                        baseline["display_name"],
                        expected_value,
                        row["display_name"],
                        actual_value,
                    )
                )

    for row in rows[1:]:
        if (
            row["task_kind"] == "livecodebench_codegen"
            and row.get("stats_contract_version") == EXPECTED_STATS_CONTRACT_VERSION
            and row.get("release_version") != "release_v6"
        ):
            mismatches.append(
                (
                    "release_version",
                    row["display_name"],
                    row.get("release_version"),
                    row["display_name"],
                    "release_v6",
                )
            )
        if row["task_kind"] == "livecodebench_codegen" and row.get("lm_style") is None:
            mismatches.append(
                (
                    "lm_style",
                    row["display_name"],
                    row.get("lm_style"),
                    row["display_name"],
                    "non-null required",
                )
            )

    if mismatches:
        lines = [
            "Cross-dataset rollout report requires a consistent measurement contract.",
            "Found mismatched settings:",
        ]
        for key, left_name, left_value, right_name, right_value in mismatches:
            lines.append(
                f"- {key}: {left_name}={left_value!r} vs {right_name}={right_value!r}"
            )
        raise ValueError("\n".join(lines))


def write_summary_files(rows: list[dict[str, Any]], out_dir: Path) -> None:
    summary_json = out_dir / "cross_dataset_rollout_summary.json"
    summary_csv = out_dir / "cross_dataset_rollout_summary.csv"
    common_generation = summarize_generation_config(rows)
    tracked_stats = sorted({stat for row in rows for stat in row["statistics"]})
    payload = {
        "datasets": rows,
        "common_generation_config": common_generation,
        "tracked_statistics": tracked_stats,
    }
    summary_json.write_text(json.dumps(payload, indent=2))

    fieldnames = [
        "display_name",
        "task_kind",
        "dataset",
        "dataset_config",
        "split",
        "model_id",
        "timestamp",
        "sample_note",
        "lcb_native_pass_at_1",
        "lcb_native_pass_at_5",
        "lcb_native_pass_at_10",
    ] + [column for _, column in COUNT_COLUMNS] + [column for _, column in METRIC_COLUMNS]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _nan_or_float(value: Any) -> float:
    return float(value) if value is not None else math.nan


def _annotate_bars(ax, bars, suffix: str) -> None:
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.annotate(
            f"{height:.1f}{suffix}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _finite_max(*series: list[float]) -> float:
    finite = [value for seq in series for value in seq if not math.isnan(value)]
    return max(finite) if finite else 1.0


def build_rate_plot(rows: list[dict[str, Any]], figures_dir: Path) -> Path:
    labels = [row["short_name"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    correct = [100.0 * _nan_or_float(row["success_fraction"]) for row in rows]
    looped = [100.0 * _nan_or_float(row["loop_fraction"]) for row in rows]
    max_hit = [100.0 * _nan_or_float(row["max_length_hit_fraction"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars1 = ax.bar([i - width for i in x], correct, width=width, label="Rollout success", color="#2a9d8f")
    bars2 = ax.bar(x, looped, width=width, label="Looped", color="#e76f51")
    bars3 = ax.bar([i + width for i in x], max_hit, width=width, label="Hit max length", color="#264653")
    _annotate_bars(ax, bars1, "%")
    _annotate_bars(ax, bars2, "%")
    _annotate_bars(ax, bars3, "%")
    ax.set_ylabel("Percent of rollouts")
    ax.set_title("Cross-dataset rollout success, loop rate, and max-length hits")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, _finite_max(correct, looped, max_hit) * 1.22)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "cross_dataset_rates.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_overlap_plot(rows: list[dict[str, Any]], figures_dir: Path) -> Path:
    labels = [row["short_name"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    loop_to_max = [100.0 * _nan_or_float(row["loop_max_length_hit_fraction"]) for row in rows]
    max_to_loop = [100.0 * _nan_or_float(row["max_length_hit_loop_fraction"]) for row in rows]
    loop_to_correct = [100.0 * _nan_or_float(row["loop_success_fraction"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars1 = ax.bar([i - width for i in x], loop_to_max, width=width, label="Looped -> max length", color="#f4a261")
    bars2 = ax.bar(x, max_to_loop, width=width, label="Max length -> looped", color="#457b9d")
    bars3 = ax.bar([i + width for i in x], loop_to_correct, width=width, label="Looped -> correct", color="#8ab17d")
    _annotate_bars(ax, bars1, "%")
    _annotate_bars(ax, bars2, "%")
    _annotate_bars(ax, bars3, "%")
    ax.set_ylabel("Conditional percent")
    ax.set_title("Overlap between looping, truncation, and correctness")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, _finite_max(loop_to_max, max_to_loop, loop_to_correct) * 1.18)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "cross_dataset_overlap.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_length_plot(rows: list[dict[str, Any]], figures_dir: Path) -> Path:
    labels = [row["short_name"] for row in rows]
    x = list(range(len(rows)))
    width = 0.24
    avg_len = [_nan_or_float(row["avg_generation_length"]) / 1000.0 for row in rows]
    avg_loop_len = [_nan_or_float(row["avg_loop_generation_length"]) / 1000.0 for row in rows]
    avg_prefix = [_nan_or_float(row["avg_first_loop_prefix_length"]) / 1000.0 for row in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars1 = ax.bar([i - width for i in x], avg_len, width=width, label="Average generation", color="#6d597a")
    bars2 = ax.bar(x, avg_loop_len, width=width, label="Average looped generation", color="#b56576")
    bars3 = ax.bar([i + width for i in x], avg_prefix, width=width, label="Average first-loop prefix", color="#355070")
    _annotate_bars(ax, bars1, "k")
    _annotate_bars(ax, bars2, "k")
    _annotate_bars(ax, bars3, "k")
    ax.set_ylabel("Tokens (thousands)")
    ax.set_title("Generation-length profile by dataset")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, _finite_max(avg_len, avg_loop_len, avg_prefix) * 1.18)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path = figures_dir / "cross_dataset_lengths.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_figures(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Path]:
    figures_dir = out_dir / FIGURES_DIRNAME
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "rates": build_rate_plot(rows, figures_dir),
        "overlap": build_overlap_plot(rows, figures_dir),
        "lengths": build_length_plot(rows, figures_dir),
    }


def _find_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return next(row for row in rows if row["key"] == key)


def build_results_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & Prompts & Rollouts & Rollout success & Looped & Max length & Loop+Max & Loop+Correct \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["display_name"]),
                    format_int(row["samples"]),
                    format_int(row["generated"]),
                    format_percent(row["success_fraction"]),
                    format_percent(row["loop_fraction"]),
                    format_percent(row["max_length_hit_fraction"]),
                    format_percent(row["loop_max_length_hit_fraction"]),
                    format_percent(row["loop_success_fraction"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _count_ratio(count: Any, total: Any, decimals: int = 1) -> str:
    if count is None or total in (None, 0):
        return "NA"
    return f"{int(count)}/{int(total)} ({100.0 * float(count) / float(total):.{decimals}f}\\%)"


def _native_lcb_metrics_text(row: dict[str, Any]) -> str:
    metrics = row.get("lcb_native_metrics") or {}
    ordered_keys = ("pass@1", "pass@5", "pass@10")
    parts = []
    for key in ordered_keys:
        if key in metrics and metrics[key] is not None:
            parts.append(f"{key} {format_percent(metrics[key])}")
    for key in sorted(metrics):
        if key in ordered_keys or metrics[key] is None:
            continue
        parts.append(f"{key} {format_percent(metrics[key])}")
    return "; ".join(parts)


def build_count_bullets(rows: list[dict[str, Any]]) -> str:
    bullets = []
    for row in rows:
        bullets.append(
            "\n".join(
                [
                    r"\item "
                    f"{latex_escape(row['display_name'])}: "
                    f"rollout success {_count_ratio(row['correct'], row['graded'])} graded rollouts; "
                    f"looped {_count_ratio(row['looped'], row['generated'])} generated rollouts; "
                    f"max-length hits {_count_ratio(row['max_length_hits'], row['generated'])} generated rollouts; "
                    f"looped and max-length {_count_ratio(row['loop_and_max_hits'], row['looped'])}; "
                    f"max-length and looped {_count_ratio(row['loop_and_max_hits'], row['max_length_hits'])}; "
                    f"looped and correct {_count_ratio(row['correct_and_looped'], row['looped'])}; "
                    f"prompt-too-long {_count_ratio(row['prompt_too_long'], row['samples'])} prompts."
                    + (
                        f" Native LiveCodeBench metrics: {latex_escape(_native_lcb_metrics_text(row))}."
                        if row["task_kind"] == "livecodebench_codegen"
                        and _native_lcb_metrics_text(row)
                        else ""
                    )
                ]
            )
        )
    return "\n".join(bullets)


def build_dataset_profiles(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for row in rows:
        notes = row["description"]
        if row["sample_note"]:
            notes = f"{notes} {row['sample_note']}"
        blocks.append(
            "\n".join(
                [
                    rf"\paragraph{{{latex_escape(row['display_name'])}}}",
                    rf"\textbf{{Task kind.}} \texttt{{{latex_escape(row['task_kind'])}}}",
                    rf"\textbf{{Dataset.}} {latex_escape(notes)}",
                    rf"\textbf{{Chat / prompt format.}} {latex_escape(row['chat_format'])}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_generation_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & Temp & Gens/prompt & Max tokens & Max model len & TP & DP & Max seqs & Batch toks \\",
        r"\midrule",
    ]
    for row in rows:
        cfg = row["generation_config"]
        lines.append(
            " & ".join(
                [
                    latex_escape(row["display_name"]),
                    format_float(cfg.get("temperature"), decimals=1),
                    format_int(cfg.get("num_generations")),
                    format_int(cfg.get("max_tokens")),
                    format_int(cfg.get("max_model_len")),
                    format_int(cfg.get("tp")),
                    format_int(cfg.get("dp")),
                    format_int(cfg.get("max_num_seqs")),
                    format_int(cfg.get("max_num_batched_tokens")),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def build_tracked_stats_list(rows: list[dict[str, Any]]) -> str:
    tracked = sorted({stat for row in rows for stat in row["statistics"]})
    items = []
    for name in tracked:
        label = STAT_LABELS.get(name, "Statistic recorded in the collector output.")
        items.append(rf"\item \texttt{{{latex_escape(name)}}}: {latex_escape(label)}")
    return "\n".join(items)


def build_key_findings(rows: list[dict[str, Any]]) -> str:
    loopiest = max(
        rows,
        key=lambda row: float("-inf") if row["loop_fraction"] is None else row["loop_fraction"],
    )
    steadiest = min(
        rows,
        key=lambda row: float("inf") if row["loop_fraction"] is None else row["loop_fraction"],
    )
    longest_loop = max(
        rows,
        key=lambda row: (
            float("-inf")
            if row["avg_loop_generation_length"] is None
            else row["avg_loop_generation_length"]
        ),
    )
    tightest_overlap = max(
        rows,
        key=lambda row: (
            float("-inf")
            if row["max_length_hit_loop_fraction"] is None
            else row["max_length_hit_loop_fraction"]
        ),
    )
    most_resilient = max(
        rows,
        key=lambda row: (
            float("-inf")
            if row["loop_success_fraction"] is None
            else row["loop_success_fraction"]
        ),
    )
    return "\n".join(
        [
            (
                f"The largest raw loop rate appears on {latex_escape(loopiest['display_name'])} at "
                f"{format_percent(loopiest['loop_fraction'])}, while "
                f"{latex_escape(steadiest['display_name'])} is the most stable dataset "
                f"in this bundle at {format_percent(steadiest['loop_fraction'])}."
            ),
            (
                f"The longest looped generations also appear on {latex_escape(longest_loop['display_name'])}: "
                f"{format_float(longest_loop['avg_loop_generation_length'])} tokens on average, "
                f"with the first detected loop not appearing until "
                f"{format_float(longest_loop['avg_first_loop_prefix_length'])} tokens."
            ),
            (
                f"Prompt-plus-generation max-length termination is almost synonymous with looping on the hardest long-form "
                f"settings: {latex_escape(tightest_overlap['display_name'])} reaches "
                f"{format_percent(tightest_overlap['max_length_hit_loop_fraction'])} for the "
                f"max-length-hit loop conditional."
            ),
            (
                f"The most forgiving loop regime is {latex_escape(most_resilient['display_name'])}, where "
                f"{format_percent(most_resilient['loop_success_fraction'])} of looped rollouts still "
                f"end in a correct answer."
            ),
        ]
    )


def build_tex(rows: list[dict[str, Any]], out_dir: Path, report_stem: str) -> Path:
    math_row = _find_row(rows, "math500")
    aime_row = _find_row(rows, "aime")
    stats_union = sorted({stat for row in rows for stat in row["statistics"]})
    common_cfg = summarize_generation_config(rows)
    loop_n = rows[0]["loop_detector"].get("n")
    loop_k = rows[0]["loop_detector"].get("k")
    figures = {
        "rates": f"{FIGURES_DIRNAME}/cross_dataset_rates.png",
        "overlap": f"{FIGURES_DIRNAME}/cross_dataset_overlap.png",
        "lengths": f"{FIGURES_DIRNAME}/cross_dataset_lengths.png",
    }

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{tabularx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{longtable}}
\usepackage{{array}}
\hypersetup{{colorlinks=true, linkcolor=blue, urlcolor=blue}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.6em}}

\title{{Cross-Dataset Rollout Statistics Report for Qwen/Qwen3-1.7B}}
\author{{Murphy}}
\date{{{latex_escape(rows[0]["timestamp"] or "Generated from collector outputs")}}}

\begin{{document}}
\maketitle

\section*{{Scope}}
This report consolidates the full math $\rightarrow$ GPQA $\rightarrow$ MMLU-Pro $\rightarrow$ LiveCodeBench rollout-stat sweep for \texttt{{Qwen/Qwen3-1.7B}}. The collector uses the n-gram loop detector with \texttt{{n={loop_n}}} and \texttt{{k={loop_k}}} and records both prompt-level counts and rollout-level event rates, including overlap statistics between looping, max-model-length termination, and correctness.

The math block was evaluated as two separate datasets under the same freeform prompt contract: \texttt{{MATH-500}} ({format_int(math_row["samples"])} samples) and \texttt{{AIME 2024/2025}} ({format_int(aime_row["samples"])} samples). The downstream multiple-choice and code-generation blocks were evaluated on \texttt{{GPQA Diamond}}, \texttt{{MMLU-Pro}}, and \texttt{{LiveCodeBench release\_v6}} in the same rollout pipeline.

\section*{{Datasets and Prompt Formats}}
{build_dataset_profiles(rows)}

\section*{{Generation Configuration}}
The common collector configuration across all runs is:
\begin{{itemize}}[leftmargin=1.5em]
\item Model: \texttt{{{latex_escape(rows[0]["model_id"])}}}
\item Seed: \texttt{{{latex_escape(rows[0].get("seed"))}}}
\item Loop detector: \texttt{{n={loop_n}, k={loop_k}}}
\item Common settings shared by every dataset JSON: \texttt{{temperature={latex_escape(common_cfg.get("temperature"))}}}, \texttt{{num\_generations={latex_escape(common_cfg.get("num_generations"))}}}, \texttt{{dtype={latex_escape(common_cfg.get("dtype"))}}}, and \texttt{{trust\_remote\_code={latex_escape(common_cfg.get("trust_remote_code"))}}}
\end{{itemize}}

Per-dataset runtime settings are:
{build_generation_table(rows)}

\section*{{Statistics Logged}}
The refreshed collector contract tracks the following metrics across the bundle ({latex_escape(str(len(stats_union)))} named metrics in the union):
\begin{{itemize}}[leftmargin=1.5em]
{build_tracked_stats_list(rows)}
\end{{itemize}}

The JSON payloads also retain the raw event counts for prompts, graded/generated rollouts, looped rollouts, max-length hits, prompt-too-long exclusions, and their pairwise intersections. For \texttt{{LiveCodeBench}}, the collector also stores the native benchmark metrics separately from rollout-level success.

\section*{{Results Summary}}
{build_results_table(rows)}

\begin{{itemize}}[leftmargin=1.5em]
{build_count_bullets(rows)}
\end{{itemize}}

\subsection*{{Headline observations}}
{build_key_findings(rows)}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{figures["rates"]}}}
\caption{{Rollout-level success, loop rate, and max-length-hit rate for each evaluated dataset.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{figures["overlap"]}}}
\caption{{Overlap statistics: among looped rollouts, how many hit max model length; among max-length-hit rollouts, how many looped; and among looped rollouts, how many still answered correctly.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\textwidth]{{{figures["lengths"]}}}
\caption{{Average generation lengths, average looped-generation lengths, and average first-loop-prefix lengths in thousands of tokens.}}
\end{{figure}}

\section*{{Interpretation}}
Across the math, science, broad-knowledge, and coding settings, the same pattern repeats: long generation tails are where looping lives. The strongest evidence is in the overlap panel: once a rollout's prompt-plus-generation length reaches \texttt{{max\_model\_len}}, it is often also a detected loop. The converse is weaker but still substantial: in several datasets, a large fraction of looped rollouts also terminate at that full-context ceiling.

The length panel shows that looped generations are substantially longer than the dataset-wide average everywhere in the bundle. The average first-loop-prefix length also varies meaningfully by dataset, which indicates that not every loop is an immediate degeneration: some settings enter the repeated n-gram regime only after a long reasoning or coding prefix.

Rollout success under looping is dataset-dependent rather than uniform. Some tasks retain a noticeable fraction of correct answers even inside looped rollouts, which suggests the model can sometimes reach the right answer before it starts repeating. For \texttt{{LiveCodeBench}}, the native benchmark metrics should be read from the separate \texttt{{pass@k}} values rather than from the rollout-success column.

\end{{document}}
"""
    tex_path = out_dir / f"{report_stem}.tex"
    tex_path.write_text(tex)
    return tex_path


def compile_pdf(tex_path: Path, pdflatex: str) -> Path:
    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        subprocess.run(
            cmd,
            cwd=tex_path.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    args = parse_args()
    stats_dir = args.stats_dir.resolve()
    out_dir = (args.out_dir or args.stats_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [load_dataset_record(stats_dir, info) for info in DATASETS]
    validate_bundle_contract(rows)
    write_summary_files(rows, out_dir)
    build_figures(rows, out_dir)
    tex_path = build_tex(rows, out_dir, args.report_stem)
    if not args.skip_pdf:
        compile_pdf(tex_path, args.pdflatex)


if __name__ == "__main__":
    main()
