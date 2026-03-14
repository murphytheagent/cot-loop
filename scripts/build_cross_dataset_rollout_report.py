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


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    filename: str
    display_name: str
    short_name: str
    task_kind: str
    description: str
    chat_format: str
    sample_note: str | None = None


DATASETS: tuple[DatasetInfo, ...] = (
    DatasetInfo(
        key="math500",
        filename="HuggingFaceH4_MATH-500__test__Qwen_Qwen3-1.7B.json",
        display_name="MATH-500",
        short_name="MATH-500",
        task_kind="math_freeform",
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
        description=(
            "A local staged GPQA Diamond CSV with 198 graduate-level science "
            "questions and shuffled four-way answer options."
        ),
        chat_format=(
            "Tokenizer chat template with one user turn containing the question, "
            "a shuffled A-D answer block, and the instruction to respond with a "
            "single best answer letter."
        ),
    ),
    DatasetInfo(
        key="mmlu_pro",
        filename="TIGER-Lab_MMLU-Pro__test__Qwen_Qwen3-1.7B.json",
        display_name="MMLU-Pro",
        short_name="MMLU-Pro",
        task_kind="multiple_choice_mmlupro",
        description=(
            "The TIGER-Lab MMLU-Pro test split, capped to 2000 examples for this "
            "rollout pass."
        ),
        chat_format=(
            "Tokenizer chat template with one user turn containing the question, "
            "an A-J answer list, and the instruction to return only the best answer "
            "letter."
        ),
        sample_note="Requested cap: at most 2000 samples.",
    ),
    DatasetInfo(
        key="livecodebench",
        filename="livecodebench_release_v6__test__Qwen_Qwen3-1.7B.json",
        display_name="LiveCodeBench release_v6",
        short_name="LiveCodeBench",
        task_kind="livecodebench_codegen",
        description=(
            "1055 LiveCodeBench code-generation problems formed by concatenating "
            "test.jsonl through test6.jsonl and sorting by question_id."
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
    "loop_fraction": "Fraction of graded rollouts flagged by the n-gram loop detector.",
    "avg_generation_length": "Average number of generated tokens per rollout.",
    "avg_loop_generation_length": "Average generated length restricted to looped rollouts.",
    "avg_first_loop_prefix_length": "Average prefix length before the first detected loop.",
    "max_length_hit_fraction": "Fraction of rollouts that terminated at the model-length limit.",
    "loop_max_length_hit_fraction": "Among looped rollouts, fraction that also hit max model length.",
    "max_length_hit_loop_fraction": "Among max-length-hit rollouts, fraction that were also looped.",
    "generation_length_variance": "Variance of generated token counts across graded rollouts.",
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

    row: dict[str, Any] = {
        "key": info.key,
        "filename": info.filename,
        "display_name": info.display_name,
        "short_name": info.short_name,
        "task_kind": info.task_kind,
        "description": info.description,
        "chat_format": info.chat_format,
        "sample_note": info.sample_note,
        "dataset": metadata.get("dataset"),
        "dataset_config": metadata.get("config"),
        "split": metadata.get("split"),
        "model_id": metadata.get("model_id"),
        "timestamp": metadata.get("timestamp"),
        "statistics": list(metadata.get("statistics", [])),
        "loop_detector": metadata.get("loop_detector", {}),
        "generation_config": generation,
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
    bars1 = ax.bar([i - width for i in x], correct, width=width, label="Correct", color="#2a9d8f")
    bars2 = ax.bar(x, looped, width=width, label="Looped", color="#e76f51")
    bars3 = ax.bar([i + width for i in x], max_hit, width=width, label="Hit max length", color="#264653")
    _annotate_bars(ax, bars1, "%")
    _annotate_bars(ax, bars2, "%")
    _annotate_bars(ax, bars3, "%")
    ax.set_ylabel("Percent of evaluated samples")
    ax.set_title("Cross-dataset correctness, loop rate, and max-length hits")
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
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Dataset & Samples & Correct & Looped & Max length & Loop+Max & Loop+Correct \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["display_name"]),
                    format_int(row["samples"]),
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


def build_count_bullets(rows: list[dict[str, Any]]) -> str:
    bullets = []
    for row in rows:
        bullets.append(
            "\n".join(
                [
                    r"\item "
                    f"{latex_escape(row['display_name'])}: "
                    f"correct {_count_ratio(row['correct'], row['samples'])}; "
                    f"looped {_count_ratio(row['looped'], row['samples'])}; "
                    f"max-length hits {_count_ratio(row['max_length_hits'], row['samples'])}; "
                    f"looped and max-length {_count_ratio(row['loop_and_max_hits'], row['looped'])}; "
                    f"max-length and looped {_count_ratio(row['loop_and_max_hits'], row['max_length_hits'])}; "
                    f"looped and correct {_count_ratio(row['correct_and_looped'], row['looped'])}."
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
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Dataset & Temp & Max tokens & Max model len & TP & DP & Max seqs & Batch toks \\",
        r"\midrule",
    ]
    for row in rows:
        cfg = row["generation_config"]
        lines.append(
            " & ".join(
                [
                    latex_escape(row["display_name"]),
                    format_float(cfg.get("temperature"), decimals=1),
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
    loopiest = max(rows, key=lambda row: row["loop_fraction"] or float("-inf"))
    longest_loop = max(rows, key=lambda row: row["avg_loop_generation_length"] or float("-inf"))
    tightest_overlap = max(rows, key=lambda row: row["max_length_hit_loop_fraction"] or float("-inf"))
    most_resilient = max(rows, key=lambda row: row["loop_success_fraction"] or float("-inf"))
    math_row = _find_row(rows, "math500")
    return "\n".join(
        [
            (
                f"The largest raw loop rate appears on {latex_escape(loopiest['display_name'])} at "
                f"{format_percent(loopiest['loop_fraction'])}, while "
                f"{latex_escape(math_row['display_name'])} is the most stable dataset "
                f"in this bundle at {format_percent(math_row['loop_fraction'])}."
            ),
            (
                f"The longest looped generations also appear on {latex_escape(longest_loop['display_name'])}: "
                f"{format_float(longest_loop['avg_loop_generation_length'])} tokens on average, "
                f"with the first detected loop not appearing until "
                f"{format_float(longest_loop['avg_first_loop_prefix_length'])} tokens."
            ),
            (
                f"Max-length termination is almost synonymous with looping on the hardest long-form "
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
This report consolidates the full math $\rightarrow$ GPQA $\rightarrow$ MMLU-Pro $\rightarrow$ LiveCodeBench rollout-stat sweep for \texttt{{Qwen/Qwen3-1.7B}}. The collector uses the n-gram loop detector with \texttt{{n={loop_n}}} and \texttt{{k={loop_k}}} and records both top-level event rates and overlap statistics between looping, max-model-length termination, and correctness.

The math block was evaluated as two separate datasets under the same freeform prompt contract: \texttt{{MATH-500}} ({format_int(math_row["samples"])} samples) and \texttt{{AIME 2024/2025}} ({format_int(aime_row["samples"])} samples). The downstream multiple-choice and code-generation blocks were evaluated on \texttt{{GPQA Diamond}}, \texttt{{MMLU-Pro}}, and \texttt{{LiveCodeBench release\_v6}} in the same rollout pipeline.

\section*{{Datasets and Prompt Formats}}
{build_dataset_profiles(rows)}

\section*{{Generation Configuration}}
The common collector configuration across all runs is:
\begin{{itemize}}[leftmargin=1.5em]
\item Model: \texttt{{{latex_escape(rows[0]["model_id"])}}}
\item Seed: \texttt{{0}}
\item Loop detector: \texttt{{n={loop_n}, k={loop_k}}}
\item Common settings shared by every dataset JSON: \texttt{{temperature={latex_escape(common_cfg.get("temperature"))}}}, \texttt{{dtype={latex_escape(common_cfg.get("dtype"))}}}, and \texttt{{trust\_remote\_code={latex_escape(common_cfg.get("trust_remote_code"))}}}
\end{{itemize}}

Per-dataset runtime settings are:
{build_generation_table(rows)}

\section*{{Statistics Logged}}
The refreshed collector contract tracks the following metrics across the bundle ({latex_escape(str(len(stats_union)))} named metrics in the union):
\begin{{itemize}}[leftmargin=1.5em]
{build_tracked_stats_list(rows)}
\end{{itemize}}

The JSON payloads also retain the raw event counts for samples, correct vs. wrong rollouts, looped rollouts, max-length hits, and their pairwise intersections.

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
\caption{{Correctness, loop rate, and max-length-hit rate for each evaluated dataset.}}
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
Across the math, science, broad-knowledge, and coding settings, the same pattern repeats: long generation tails are where looping lives. The strongest evidence is in the overlap panel: once a rollout hits max model length, it is almost always also a detected loop, especially on \texttt{{GPQA}}, \texttt{{MMLU-Pro}}, and \texttt{{LiveCodeBench}}. The converse is weaker but still substantial: a large fraction of looped rollouts on the non-math datasets also terminate at the model-length ceiling.

The length panel shows that looped generations are dramatically longer than the dataset-wide average everywhere. That gap is modestly above \texttt{{+16k}} tokens on \texttt{{GPQA}} and grows to roughly \texttt{{+20k}} tokens on both \texttt{{MATH-500}} and \texttt{{LiveCodeBench}}. The average first-loop-prefix length also differs by task family: \texttt{{GPQA}} loops are typically detected earlier than the long-form math and coding loops, while \texttt{{AIME}} and \texttt{{LiveCodeBench}} often burn more than \texttt{{10k}} tokens before the repeated n-gram pattern is first visible.

Correctness under looping is dataset-dependent. \texttt{{GPQA}} and \texttt{{MMLU-Pro}} still retain a non-trivial fraction of correct answers inside looped rollouts, which suggests some looping trajectories occur after the model has effectively committed to the right option. \texttt{{LiveCodeBench}} is qualitatively different: looped rollouts there are rarely correct, so looping aligns more directly with task failure than with benign verbosity.

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
    write_summary_files(rows, out_dir)
    build_figures(rows, out_dir)
    tex_path = build_tex(rows, out_dir, args.report_stem)
    if not args.skip_pdf:
        compile_pdf(tex_path, args.pdflatex)


if __name__ == "__main__":
    main()
