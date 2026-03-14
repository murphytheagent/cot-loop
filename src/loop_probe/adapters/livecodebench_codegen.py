from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..collector import LcbSampleRecord

_LCB_DATASET_REPO = "livecodebench/code_generation_lite"
_LCB_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
    "release_latest": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
        "test6.jsonl",
    ],
}


def _require_repo_path(repo_path: str) -> None:
    if not repo_path:
        raise SystemExit("--livecodebench-repo is required for livecodebench_codegen.")
    if not os.path.isdir(repo_path):
        raise SystemExit(f"LiveCodeBench repo path does not exist: {repo_path}")
    if not os.path.isdir(os.path.join(repo_path, "lcb_runner")):
        raise SystemExit(
            f"LiveCodeBench repo path must contain lcb_runner/: {repo_path}"
        )


def _ensure_repo_on_path(repo_path: str) -> None:
    _require_repo_path(repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)


@contextmanager
def _repo_cwd(repo_path: str):
    _require_repo_path(repo_path)
    prev_cwd = os.getcwd()
    os.chdir(repo_path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def _import_lcb_symbols(repo_path: str) -> dict[str, Any]:
    _ensure_repo_on_path(repo_path)
    with _repo_cwd(repo_path):
        try:
            from lcb_runner.evaluation import extract_instance_results
            from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
            from lcb_runner.lm_styles import LMStyle
            from lcb_runner.prompts import format_prompt_generation
            from lcb_runner.runner.scenario_router import (
                get_metrics,
                sort_and_extract_save_results,
            )
            from lcb_runner.utils.extraction_utils import extract_code
            from lcb_runner.utils.scenarios import Scenario
        except Exception as exc:
            raise ImportError(
                "Failed to import LiveCodeBench helpers from the provided checkout."
            ) from exc

    return {
        "extract_instance_results": extract_instance_results,
        "CodeGenerationProblem": CodeGenerationProblem,
        "LMStyle": LMStyle,
        "format_prompt_generation": format_prompt_generation,
        "get_metrics": get_metrics,
        "sort_and_extract_save_results": sort_and_extract_save_results,
        "extract_code": extract_code,
        "Scenario": Scenario,
    }


def build_lcb_args(release_version: str, repo_path: str) -> SimpleNamespace:
    symbols = _import_lcb_symbols(repo_path)
    scenario = symbols["Scenario"].codegeneration
    return SimpleNamespace(
        scenario=scenario,
        release_version=release_version,
        not_fast=False,
        start_date=None,
        end_date=None,
        num_process_evaluate=4,
        timeout=5.0,
    )


def _release_files(release_version: str) -> list[str]:
    if release_version in _LCB_RELEASE_FILES:
        return list(_LCB_RELEASE_FILES[release_version])

    single = re.fullmatch(r"v([1-6])", release_version)
    if single:
        idx = int(single.group(1))
        return [f"test{idx}.jsonl" if idx != 1 else "test.jsonl"]

    pair = re.fullmatch(r"v([1-6])_v([1-6])", release_version)
    if pair:
        start = int(pair.group(1))
        end = int(pair.group(2))
        if start > end:
            raise ValueError(
                f"LiveCodeBench release range must be ordered, got {release_version!r}."
            )
        return [
            f"test{idx}.jsonl" if idx != 1 else "test.jsonl"
            for idx in range(start, end + 1)
        ]

    raise ValueError(f"Unsupported LiveCodeBench release_version {release_version!r}.")


def _load_codegen_benchmark(
    repo_path: str,
    release_version: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
):
    symbols = _import_lcb_symbols(repo_path)
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError(
            "Failed to import huggingface_hub for LiveCodeBench dataset downloads."
        ) from exc

    benchmark = []
    problem_cls = symbols["CodeGenerationProblem"]
    for filename in _release_files(release_version):
        path = hf_hub_download(_LCB_DATASET_REPO, filename, repo_type="dataset")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                benchmark.append(problem_cls(**json.loads(line)))

    if start_date is not None:
        lower = datetime.strptime(start_date, "%Y-%m-%d")
        benchmark = [row for row in benchmark if lower <= row.contest_date]
    if end_date is not None:
        upper = datetime.strptime(end_date, "%Y-%m-%d")
        benchmark = [row for row in benchmark if row.contest_date <= upper]

    benchmark = sorted(benchmark, key=lambda row: row.question_id)
    return benchmark, symbols["format_prompt_generation"]


def _get_lm_style(model_id: str, override: str | None = None, repo_path: str = ""):
    symbols = _import_lcb_symbols(repo_path)
    lm_style_cls = symbols["LMStyle"]
    if override:
        try:
            return lm_style_cls(override)
        except ValueError:
            if hasattr(lm_style_cls, override):
                return getattr(lm_style_cls, override)
            raise
    if model_id.lower().startswith("qwen/"):
        return lm_style_cls.CodeQwenInstruct
    return lm_style_cls.CodeQwenInstruct


def preflight(repo_path: str, release_version: str) -> None:
    _load_codegen_benchmark(repo_path, release_version)


def load_benchmark(
    repo_path: str,
    release_version: str,
):
    return _load_codegen_benchmark(repo_path, release_version)


def build_prompts(
    benchmark,
    format_prompt,
    *,
    repo_path: str,
    model_id: str,
    lm_style_override: str | None = None,
    max_samples: int | None = None,
) -> tuple[list[tuple[str, str]], str]:
    lm_style = _get_lm_style(
        model_id,
        override=lm_style_override,
        repo_path=repo_path,
    )
    prompt_records: list[tuple[str, str]] = []
    selected = benchmark if max_samples is None else benchmark[:max_samples]
    with _repo_cwd(repo_path):
        for instance in selected:
            prompt = format_prompt(instance, lm_style)
            if not isinstance(prompt, str):
                raise SystemExit(
                    "LiveCodeBench prompt builder returned a non-string prompt. "
                    "Use an --lm-style-override that produces raw string prompts."
                )
            prompt_records.append((str(instance.question_id), prompt))
    return prompt_records, lm_style.value


def extract_code_output(
    response_text: str,
    *,
    repo_path: str,
    model_id: str,
    lm_style_override: str | None = None,
) -> str:
    symbols = _import_lcb_symbols(repo_path)
    lm_style = _get_lm_style(
        model_id,
        override=lm_style_override,
        repo_path=repo_path,
    )
    with _repo_cwd(repo_path):
        extracted = symbols["extract_code"](response_text, lm_style)
    return "" if extracted is None else str(extracted)


def evaluate_records(
    benchmark,
    records: list[LcbSampleRecord],
    *,
    repo_path: str,
    release_version: str,
) -> tuple[float | None, dict[str, bool]]:
    symbols = _import_lcb_symbols(repo_path)
    args_ns = build_lcb_args(release_version, repo_path)

    by_question_id = {record.question_id: record for record in records}
    missing = [
        str(instance.question_id)
        for instance in benchmark
        if str(instance.question_id) not in by_question_id
    ]
    if missing:
        raise RuntimeError(
            f"Missing LiveCodeBench records for question_id(s): {missing[:10]}"
        )

    save_results = [
        instance.insert_output(
            [by_question_id[str(instance.question_id)].code_output],
            [by_question_id[str(instance.question_id)].code_output],
        )
        for instance in benchmark
    ]
    with _repo_cwd(repo_path):
        save_results, combined_results = symbols["sort_and_extract_save_results"](
            symbols["Scenario"].codegeneration,
            save_results,
        )
        metrics = symbols["get_metrics"](
            symbols["Scenario"].codegeneration,
            args_ns,
            benchmark,
            combined_results,
        )
        graded = symbols["extract_instance_results"](metrics[1])

    pass_at_1 = metrics[0].get("pass@1")
    grading_by_question_id: dict[str, bool] = {}
    for instance, instance_grades in zip(benchmark, graded):
        passed = bool(instance_grades[0]) if instance_grades else False
        grading_by_question_id[str(instance.question_id)] = passed

    return float(pass_at_1) if pass_at_1 is not None else None, grading_by_question_id
