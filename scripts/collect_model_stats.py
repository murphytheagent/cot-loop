#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_module
import re
import signal
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from transformers import AutoTokenizer

from loop_probe.adapters import (
    livecodebench_codegen,
    math_freeform,
    multiple_choice_gpqa,
    multiple_choice_mmlupro,
)
from loop_probe.collector import (
    ALL_STATISTICS,
    CollectorConfig,
    LcbSampleRecord,
    WorkerAggregator,
    compute_metrics,
    merge_aggregators,
)
from loop_probe.configs import RolloutConfig
from loop_probe.labeling import first_ngram_loop_prefix_length
from loop_probe.prompt_builder import build_math_prompt
from loop_probe.rollout import resolve_sampling_defaults
from loop_probe.types import DatasetSpec
from utils import get_visible_devices, suppress_sem_unlink_errors

TASK_CHOICES = (
    "math_freeform",
    "multiple_choice_gpqa",
    "multiple_choice_mmlupro",
    "livecodebench_codegen",
)
STATS_CONTRACT_VERSION = "rollout_stats_v2"


@dataclass(frozen=True)
class PromptWorkItem:
    sample_id: int
    prompt: str
    gold_answer: str | None = None
    gold_index: int | None = None
    question_id: str | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-kind", required=True, choices=TASK_CHOICES)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-generations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=81920)
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--statistics",
        default=",".join(ALL_STATISTICS),
        help="Comma-separated statistics to compute, or 'all'.",
    )
    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument("--out", default="")
    parser.add_argument("--livecodebench-repo", default="")
    parser.add_argument("--release-version", default="release_v6")
    parser.add_argument("--lm-style-override", default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    return parser.parse_args()


def _parse_statistics(raw: str) -> list[str]:
    value = raw.strip()
    if not value or value.lower() == "all":
        return list(ALL_STATISTICS)
    seen: set[str] = set()
    stats: list[str] = []
    for name in (part.strip() for part in value.split(",")):
        if not name or name in seen:
            continue
        seen.add(name)
        stats.append(name)
    if not stats:
        return list(ALL_STATISTICS)
    unknown = sorted(set(stats) - set(ALL_STATISTICS))
    if unknown:
        raise SystemExit(
            f"Unknown statistic(s): {unknown}. Valid choices: {list(ALL_STATISTICS)}"
        )
    return stats


def _slugify(value: str) -> str:
    text = value.strip()
    if not text:
        return "default"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_") or "default"


def _derive_output_path(args: argparse.Namespace) -> str:
    dataset_label = os.path.basename(args.dataset) if os.path.isfile(args.dataset) else args.dataset
    parts = [_slugify(dataset_label)]
    if args.dataset_config:
        parts.append(_slugify(args.dataset_config))
    parts.append(_slugify(args.split))
    parts.append(_slugify(args.model_id))
    filename = "__".join(parts) + ".json"
    return os.path.join(ROOT, "outputs", "model_stats", filename)


def _normalize_finish_reason(reason: object) -> str:
    if hasattr(reason, "value"):
        reason = getattr(reason, "value")
    if reason is None:
        return "unknown"
    text = str(reason).strip()
    if not text:
        return "unknown"
    return text.split(".")[-1].lower()


def _effective_max_tokens(prompt_len: int, rollout_cfg: RolloutConfig) -> int:
    return min(rollout_cfg.max_tokens, max(0, rollout_cfg.max_model_len - prompt_len))


def _total_token_count(prompt_len: int, token_count: int) -> int:
    return prompt_len + token_count


def _hit_max_model_len(
    *,
    prompt_len: int,
    token_count: int,
    finish_reason: str,
    rollout_cfg: RolloutConfig,
) -> bool:
    if finish_reason != "length":
        return False
    return _total_token_count(prompt_len, token_count) >= rollout_cfg.max_model_len


def _run_dataset_preflight(args: argparse.Namespace) -> None:
    if (
        not os.path.exists(args.dataset)
        and args.dataset.lower().endswith((".jsonl", ".json", ".csv"))
    ):
        raise SystemExit(f"Dataset path does not exist: {args.dataset}")

    if args.task_kind == "multiple_choice_gpqa" and not os.path.isfile(args.dataset):
        if not os.environ.get("HF_TOKEN", "").strip():
            raise SystemExit(
                "multiple_choice_gpqa requires HF_TOKEN for gated dataset access."
            )

    if args.task_kind == "livecodebench_codegen":
        if not os.environ.get("HF_DATASETS_CACHE", "").strip():
            raise SystemExit(
                "livecodebench_codegen requires HF_DATASETS_CACHE to be set before startup."
            )
        if not args.livecodebench_repo:
            raise SystemExit(
                "livecodebench_codegen requires --livecodebench-repo."
            )
        if not os.path.isdir(args.livecodebench_repo):
            raise SystemExit(
                f"LiveCodeBench repo path does not exist: {args.livecodebench_repo}"
            )


def _run_dependency_preflight(args: argparse.Namespace) -> None:
    try:
        import vllm  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "vLLM is required for collect_model_stats. Install vLLM first."
        ) from exc

    if args.task_kind == "math_freeform":
        math_freeform.preflight()
    elif args.task_kind == "livecodebench_codegen":
        try:
            livecodebench_codegen.preflight(
                args.livecodebench_repo,
                args.release_version,
            )
        except Exception as exc:
            raise SystemExit(
                "LiveCodeBench preflight failed before model init."
            ) from exc


def _build_rollout_config(args: argparse.Namespace) -> RolloutConfig:
    if args.dp < 1:
        raise SystemExit("--dp must be >= 1.")
    if args.tp < 1:
        raise SystemExit("--tp must be >= 1.")
    if args.dp > 1 and args.tp != 1:
        raise SystemExit("Data-parallel collection requires --tp 1.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be >= 1 when provided.")
    if args.loop_n < 1:
        raise SystemExit("--loop-n must be >= 1.")
    if args.loop_k < 2:
        raise SystemExit("--loop-k must be >= 2.")
    if args.num_generations < 1:
        raise SystemExit("--num-generations must be >= 1.")

    return RolloutConfig(
        model_id=args.model_id,
        temperature=args.temperature,
        num_generations=args.num_generations,
        max_tokens=args.max_tokens,
        tp=args.tp,
        dp=args.dp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
    )


def _load_prompt_items(
    args: argparse.Namespace,
    tokenizer,
) -> tuple[list[PromptWorkItem], dict[str, object], Any]:
    spec = DatasetSpec(
        dataset=args.dataset,
        config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
    )

    if args.task_kind == "math_freeform":
        samples = math_freeform.load_samples(
            spec,
            question_field=args.question_field,
            answer_field=args.answer_field,
        )
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=build_math_prompt(tokenizer, record.prompt),
                gold_answer=gold_answer,
            )
            for record, gold_answer in samples
        ]
        return items, {}, None

    if args.task_kind == "multiple_choice_gpqa":
        samples = multiple_choice_gpqa.load_and_shuffle(spec, args.seed)
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=multiple_choice_gpqa.build_mcq_prompt(
                    tokenizer,
                    record.prompt,
                    options,
                ),
                gold_answer=gold_letter,
            )
            for record, options, gold_letter in samples
        ]
        metadata = {
            "shuffle_policy": {
                "kind": "seed_xor_sample_id",
                "base_seed": args.seed,
            }
        }
        return items, metadata, None

    if args.task_kind == "multiple_choice_mmlupro":
        samples = multiple_choice_mmlupro.load_samples(spec)
        items = [
            PromptWorkItem(
                sample_id=record.sample_id,
                prompt=multiple_choice_mmlupro.build_mcq_prompt(
                    tokenizer,
                    record.prompt,
                    options,
                ),
                gold_answer=gold_answer,
                gold_index=gold_index,
            )
            for record, options, gold_answer, gold_index in samples
        ]
        return items, {}, None

    benchmark, format_prompt = livecodebench_codegen.load_benchmark(
        args.livecodebench_repo,
        args.release_version,
    )
    prompt_records, lm_style = livecodebench_codegen.build_prompts(
        benchmark,
        format_prompt,
        repo_path=args.livecodebench_repo,
        model_id=args.model_id,
        lm_style_override=args.lm_style_override,
        max_samples=args.max_samples,
    )
    selected_benchmark = benchmark if args.max_samples is None else benchmark[: args.max_samples]
    items = [
        PromptWorkItem(
            sample_id=idx,
            prompt=prompt,
            question_id=question_id,
        )
        for idx, (question_id, prompt) in enumerate(prompt_records)
    ]
    return items, {"lm_style": lm_style}, selected_benchmark


def _update_qa_stats(
    agg: WorkerAggregator,
    *,
    task_kind: str,
    item: PromptWorkItem,
    response_text: str,
    token_count: int,
    loop_flag: bool,
    max_length_hit: bool,
) -> None:
    if task_kind == "math_freeform":
        correct = math_freeform.grade(response_text, item.gold_answer or "")
    elif task_kind == "multiple_choice_gpqa":
        correct = multiple_choice_gpqa.grade(response_text, item.gold_answer or "")
    elif task_kind == "multiple_choice_mmlupro":
        correct = multiple_choice_mmlupro.grade(
            response_text,
            item.gold_answer or "",
            item.gold_index,
        )
    else:
        raise ValueError(f"Unsupported QA task_kind '{task_kind}'.")

    agg.num_graded += 1
    if correct:
        agg.num_correct += 1
        agg.correct_length_sum += token_count
        if loop_flag:
            agg.num_correct_and_looped += 1
        if max_length_hit:
            agg.num_correct_and_max_length_hit += 1
    else:
        agg.num_wrong += 1
        agg.wrong_length_sum += token_count


def _collect_worker_stats(
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    *,
    loop_n: int,
    loop_k: int,
    rank: int = 0,
) -> WorkerAggregator:
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(
            "vLLM is required for collect_model_stats. Install vLLM first."
        ) from exc

    agg = WorkerAggregator()
    if not items:
        return agg

    rollout_cfg = collector_cfg.rollout_cfg
    if rollout_cfg.max_num_seqs is not None and rollout_cfg.max_num_seqs < 1:
        raise SystemExit("--max-num-seqs must be >= 1 when provided.")
    if (
        rollout_cfg.max_num_seqs is not None
        and rollout_cfg.max_num_seqs < rollout_cfg.num_generations
    ):
        raise SystemExit(
            "--max-num-seqs must be >= --num-generations when sampling multiple "
            "generations per prompt."
        )

    top_p, top_k = resolve_sampling_defaults(rollout_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
        use_fast=True,
    )

    llm_kwargs = {
        "model": rollout_cfg.model_id,
        "tensor_parallel_size": rollout_cfg.tp,
        "dtype": rollout_cfg.dtype,
        "max_model_len": rollout_cfg.max_model_len,
        "trust_remote_code": rollout_cfg.trust_remote_code,
    }
    gpu_mem_util = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "").strip()
    if gpu_mem_util:
        try:
            gpu_mem_value = float(gpu_mem_util)
        except Exception as exc:
            raise SystemExit(
                "VLLM_GPU_MEMORY_UTILIZATION must be a float in (0, 1]."
            ) from exc
        if not (0.0 < gpu_mem_value <= 1.0):
            raise SystemExit("VLLM_GPU_MEMORY_UTILIZATION must be in (0, 1].")
        llm_kwargs["gpu_memory_utilization"] = gpu_mem_value
    if rollout_cfg.max_num_seqs is not None:
        llm_kwargs["max_num_seqs"] = rollout_cfg.max_num_seqs
    if rollout_cfg.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = rollout_cfg.max_num_batched_tokens

    llm = LLM(**llm_kwargs)

    if rollout_cfg.max_num_seqs is not None:
        chunk_size = max(1, rollout_cfg.max_num_seqs // rollout_cfg.num_generations)
    else:
        chunk_size = len(items)
    for start in range(0, len(items), chunk_size):
        end = min(start + chunk_size, len(items))
        batch_items = items[start:end]
        batch_prompts = [item.prompt for item in batch_items]
        prompt_input_ids = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        valid_items: list[tuple[PromptWorkItem, int, int]] = []
        for item, input_ids in zip(batch_items, prompt_input_ids):
            agg.num_samples_seen += 1
            prompt_len = len(input_ids)
            agg.prompt_length_sum += prompt_len
            agg.prompt_length_min = (
                prompt_len
                if agg.prompt_length_min is None
                else min(agg.prompt_length_min, prompt_len)
            )
            agg.prompt_length_max = (
                prompt_len
                if agg.prompt_length_max is None
                else max(agg.prompt_length_max, prompt_len)
            )
            effective_max = _effective_max_tokens(prompt_len, rollout_cfg)
            if effective_max < 1:
                agg.num_prompt_too_long += 1
                if collector_cfg.task_kind == "livecodebench_codegen":
                    agg.lcb_sample_records.append(
                        LcbSampleRecord(
                            question_id=item.question_id or "",
                            generation_index=-1,
                            code_output="",
                            token_count=0,
                            prompt_token_count=prompt_len,
                            total_token_count=prompt_len,
                            effective_max_tokens=effective_max,
                            max_model_len=rollout_cfg.max_model_len,
                            loop_flag=False,
                            max_length_hit=False,
                            finish_reason="prompt_too_long",
                            prompt_too_long=True,
                        )
                    )
                continue
            valid_items.append((item, prompt_len, effective_max))

        if valid_items:
            for item, prompt_len, effective_max in valid_items:
                sampling_params = SamplingParams(
                    temperature=rollout_cfg.temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=rollout_cfg.max_tokens,
                    n=rollout_cfg.num_generations,
                    repetition_penalty=1.0,
                    seed=collector_cfg.seed + item.sample_id,
                )
                outputs = llm.generate([item.prompt], sampling_params)
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"Expected 1 prompt output, got {len(outputs)}."
                    )
                output = outputs[0]
                if len(output.outputs) != rollout_cfg.num_generations:
                    raise RuntimeError(
                        "Expected "
                        f"{rollout_cfg.num_generations} output(s) per prompt, got "
                        f"{len(output.outputs)}."
                    )
                for generation_index, sample in enumerate(output.outputs):
                    text = str(getattr(sample, "text", ""))
                    token_ids = getattr(sample, "token_ids", None)
                    if not token_ids:
                        token_ids = tokenizer.encode(text, add_special_tokens=False)
                    token_count = len(token_ids)
                    total_token_count = _total_token_count(prompt_len, token_count)
                    finish_reason = _normalize_finish_reason(
                        getattr(sample, "finish_reason", None)
                        or getattr(output, "finish_reason", None)
                        or getattr(sample, "stop_reason", None)
                    )
                    first_loop_prefix = first_ngram_loop_prefix_length(
                        token_ids,
                        n=loop_n,
                        k=loop_k,
                    )
                    loop_flag = first_loop_prefix is not None
                    max_length_hit = _hit_max_model_len(
                        prompt_len=prompt_len,
                        token_count=token_count,
                        finish_reason=finish_reason,
                        rollout_cfg=rollout_cfg,
                    )

                    agg.num_generated += 1
                    agg.length_sum += token_count
                    agg.length_sq_sum += token_count * token_count
                    if loop_flag and first_loop_prefix is not None:
                        agg.num_looped += 1
                        agg.loop_length_sum += token_count
                        agg.first_loop_prefix_sum += first_loop_prefix
                    if max_length_hit:
                        agg.num_max_length_hits += 1
                    if loop_flag and max_length_hit:
                        agg.num_looped_and_max_length_hit += 1

                    if collector_cfg.task_kind == "livecodebench_codegen":
                        agg.lcb_sample_records.append(
                            LcbSampleRecord(
                                question_id=item.question_id or "",
                                generation_index=generation_index,
                                code_output=livecodebench_codegen.extract_code_output(
                                    text,
                                    repo_path=collector_cfg.livecodebench_repo or "",
                                    model_id=rollout_cfg.model_id,
                                    lm_style_override=collector_cfg.lm_style_override,
                                ),
                                token_count=token_count,
                                prompt_token_count=prompt_len,
                                total_token_count=total_token_count,
                                effective_max_tokens=effective_max,
                                max_model_len=rollout_cfg.max_model_len,
                                loop_flag=loop_flag,
                                max_length_hit=max_length_hit,
                                finish_reason=finish_reason,
                                prompt_too_long=False,
                            )
                        )
                    else:
                        _update_qa_stats(
                            agg,
                            task_kind=collector_cfg.task_kind,
                            item=item,
                            response_text=text,
                            token_count=token_count,
                            loop_flag=loop_flag,
                            max_length_hit=max_length_hit,
                        )

        print(
            f"[collect-dp-rank {rank}] processed {end}/{len(items)} prompts",
            flush=True,
        )

    return agg


def _worker_main(
    rank: int,
    device: str,
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    loop_n: int,
    loop_k: int,
    out_queue,
) -> None:
    suppress_sem_unlink_errors()
    try:
        os.setsid()
    except OSError:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    try:
        agg = _collect_worker_stats(
            items,
            collector_cfg,
            loop_n=loop_n,
            loop_k=loop_k,
            rank=rank,
        )
    except Exception:
        tb = traceback.format_exc()
        try:
            out_queue.put((rank, None, tb))
        except Exception:
            pass
        raise
    out_queue.put((rank, agg, None))


def _cleanup_worker_group(worker_pid: int, rank: int) -> None:
    if worker_pid <= 0:
        return
    try:
        os.killpg(worker_pid, 0)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        print(
            f"[collect-dp-rank {rank}] warning: cannot probe process group {worker_pid}: {exc}",
            flush=True,
        )
        return

    for sig, wait_s in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 0.5)):
        try:
            os.killpg(worker_pid, sig)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            print(
                f"[collect-dp-rank {rank}] warning: cannot signal process group {worker_pid}: {exc}",
                flush=True,
            )
            return
        time.sleep(wait_s)
        try:
            os.killpg(worker_pid, 0)
        except ProcessLookupError:
            return


def _run_collection(
    items: list[PromptWorkItem],
    collector_cfg: CollectorConfig,
    *,
    loop_n: int,
    loop_k: int,
) -> WorkerAggregator:
    if collector_cfg.rollout_cfg.dp == 1:
        return _collect_worker_stats(
            items,
            collector_cfg,
            loop_n=loop_n,
            loop_k=loop_k,
        )

    devices = get_visible_devices()
    worker_count = min(collector_cfg.rollout_cfg.dp, len(items))
    if len(devices) < worker_count:
        raise SystemExit(
            f"Requested dp={collector_cfg.rollout_cfg.dp}, but only {len(devices)} visible GPU(s)."
        )

    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue()
    processes = []
    for rank in range(worker_count):
        shard_items = items[rank::worker_count]
        p = ctx.Process(
            target=_worker_main,
            args=(
                rank,
                devices[rank],
                shard_items,
                collector_cfg,
                loop_n,
                loop_k,
                out_queue,
            ),
        )
        p.start()
        processes.append(p)

    aggregators: dict[int, WorkerAggregator] = {}
    failures: list[tuple[int, str | int]] = []
    try:
        while len(aggregators) < worker_count:
            try:
                worker_rank, agg, error = out_queue.get(timeout=30)
            except queue_module.Empty:
                dead_missing = []
                for proc_rank, process in enumerate(processes):
                    if proc_rank in aggregators:
                        continue
                    if process.exitcode is not None:
                        dead_missing.append((proc_rank, process.exitcode))
                if dead_missing:
                    raise SystemExit(
                        f"DP worker(s) exited before reporting stats: {dead_missing}"
                    )
                continue

            worker_rank = int(worker_rank)
            if error is not None:
                raise SystemExit(f"DP worker {worker_rank} failed:\n{error}")
            if agg is None:
                raise SystemExit(f"DP worker {worker_rank} failed: missing aggregator")
            aggregators[worker_rank] = agg

        for proc_rank, process in enumerate(processes):
            process.join(timeout=30)
            if process.is_alive():
                if proc_rank in aggregators:
                    process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    print(
                        f"[collect-dp-rank {proc_rank}] terminated after stats were received (teardown hang)",
                        flush=True,
                    )
                else:
                    process.terminate()
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    failures.append((proc_rank, "alive_without_stats"))

            if process.exitcode not in (0, None) and proc_rank not in aggregators:
                failures.append((proc_rank, process.exitcode))

        if failures:
            failure_lines = []
            for rank, detail in failures:
                failure_lines.append(f"rank={rank}: {detail}")
            raise SystemExit("DP worker(s) failed:\n" + "\n".join(failure_lines))

        merged = merge_aggregators(aggregators[rank] for rank in range(worker_count))
        return merged
    finally:
        for rank, process in enumerate(processes):
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
            _cleanup_worker_group(process.pid, rank)
        try:
            out_queue.close()
            out_queue.join_thread()
        except Exception:
            pass


def _apply_lcb_grades(
    agg: WorkerAggregator,
    benchmark,
    *,
    repo_path: str,
    release_version: str,
) -> dict[str, float | None]:
    native_metrics, grading_by_record_key = livecodebench_codegen.evaluate_records(
        benchmark,
        agg.lcb_sample_records,
        repo_path=repo_path,
        release_version=release_version,
    )
    records_by_key = {}
    for record in agg.lcb_sample_records:
        if record.prompt_too_long:
            continue
        record_key = (record.question_id, record.generation_index)
        if record_key in records_by_key:
            raise RuntimeError(f"Duplicate LiveCodeBench record key: {record_key}")
        records_by_key[record_key] = record

    missing_keys = sorted(set(records_by_key) - set(grading_by_record_key))
    if missing_keys:
        raise RuntimeError(
            "Missing LiveCodeBench grades for generated records: "
            f"{missing_keys[:10]}"
        )

    agg.num_graded = 0
    agg.num_correct = 0
    agg.num_wrong = 0
    agg.correct_length_sum = 0
    agg.wrong_length_sum = 0
    agg.num_correct_and_looped = 0
    agg.num_correct_and_max_length_hit = 0

    for record_key, passed in grading_by_record_key.items():
        record = records_by_key[record_key]
        agg.num_graded += 1
        if passed:
            agg.num_correct += 1
            agg.correct_length_sum += record.token_count
            if record.loop_flag:
                agg.num_correct_and_looped += 1
            if record.max_length_hit:
                agg.num_correct_and_max_length_hit += 1
        else:
            agg.num_wrong += 1
            agg.wrong_length_sum += record.token_count
    return native_metrics


def _write_lcb_records_checkpoint(agg: WorkerAggregator, out_path: str) -> str:
    base, ext = os.path.splitext(out_path)
    checkpoint_path = f"{base}__lcb_records{ext or '.json'}"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "num_samples": agg.num_samples_seen,
        "num_generated": agg.num_generated,
        "num_prompt_too_long": agg.num_prompt_too_long,
        "records": [asdict(record) for record in agg.lcb_sample_records],
    }
    with open(checkpoint_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Wrote LiveCodeBench records checkpoint to {checkpoint_path}", flush=True)
    return checkpoint_path


def _prompt_token_summary(agg: WorkerAggregator) -> dict[str, float | int | None]:
    avg_prompt_length = (
        float(agg.prompt_length_sum) / float(agg.num_samples_seen)
        if agg.num_samples_seen
        else None
    )
    return {
        "avg_prompt_token_count": avg_prompt_length,
        "min_prompt_token_count": agg.prompt_length_min,
        "max_prompt_token_count": agg.prompt_length_max,
    }


def main() -> None:
    args = _parse_args()
    _run_dataset_preflight(args)
    _run_dependency_preflight(args)
    rollout_cfg = _build_rollout_config(args)
    statistics = _parse_statistics(args.statistics)

    tokenizer = AutoTokenizer.from_pretrained(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
        use_fast=True,
    )
    items, task_metadata, lcb_benchmark = _load_prompt_items(args, tokenizer)
    if not items:
        raise SystemExit("No prompt items were loaded for collection.")

    collector_cfg = CollectorConfig(
        rollout_cfg=rollout_cfg,
        seed=args.seed,
        task_kind=args.task_kind,
        statistics=statistics,
        livecodebench_repo=args.livecodebench_repo or None,
        release_version=args.release_version,
        lm_style_override=args.lm_style_override,
    )

    agg = _run_collection(
        items,
        collector_cfg,
        loop_n=args.loop_n,
        loop_k=args.loop_k,
    )
    out_path = args.out or _derive_output_path(args)
    lcb_native_metrics: dict[str, float | None] = {}
    if args.task_kind == "livecodebench_codegen":
        _write_lcb_records_checkpoint(agg, out_path)
        lcb_native_metrics = _apply_lcb_grades(
            agg,
            lcb_benchmark,
            repo_path=args.livecodebench_repo,
            release_version=args.release_version,
        )

    metrics = compute_metrics(agg, statistics)

    payload = {
        "metadata": {
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "task_kind": args.task_kind,
            "model_id": rollout_cfg.model_id,
            "generation_config": rollout_cfg.to_dict(),
            "stats_contract_version": STATS_CONTRACT_VERSION,
            "seed": args.seed,
            "statistics": statistics,
            "loop_detector": {"n": args.loop_n, "k": args.loop_k},
            "prompt_token_summary": _prompt_token_summary(agg),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **(
                {"release_version": args.release_version}
                if args.task_kind == "livecodebench_codegen"
                else {}
            ),
            **task_metadata,
        },
        "counts": {
            "num_samples": agg.num_samples_seen,
            "num_generated": agg.num_generated,
            "num_graded": agg.num_graded,
            "num_correct": agg.num_correct,
            "num_wrong": agg.num_wrong,
            "num_looped": agg.num_looped,
            "num_max_length_hits": agg.num_max_length_hits,
            "num_prompt_too_long": agg.num_prompt_too_long,
            "num_looped_and_max_length_hit": agg.num_looped_and_max_length_hit,
            "num_correct_and_looped": agg.num_correct_and_looped,
            "num_correct_and_max_length_hit": agg.num_correct_and_max_length_hit,
        },
        "metrics": metrics,
    }
    if lcb_native_metrics:
        payload["metadata"]["lcb_native_metrics"] = lcb_native_metrics

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote model stats to {out_path}", flush=True)


if __name__ == "__main__":
    main()
