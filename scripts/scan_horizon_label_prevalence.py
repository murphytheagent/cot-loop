#!/usr/bin/env python3
"""Estimate onset-horizon label prevalence from one rollout pass."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from transformers import AutoTokenizer

from loop_probe.configs import get_rollout_config, preset_choices
from loop_probe.labeling import first_ngram_loop_prefix_length
from loop_probe.rollout import generate_rollout_token_ids
from utils import build_prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Local JSONL prompt file.")
    parser.add_argument("--prompt-field", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--num-length-bins", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", required=True)

    parser.add_argument("--model-preset", choices=preset_choices(), default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)

    parser.set_defaults(trust_remote_code=None)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")

    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    return parser.parse_args()


def _load_rows(path: str, prompt_field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise SystemExit(f"Expected object row at {path}:{line_num}")
            if prompt_field not in row:
                raise SystemExit(f"Missing prompt field '{prompt_field}' at {path}:{line_num}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    return rows


def _assign_length_bins(
    rows: list[dict[str, object]],
    *,
    tokenizer,
    prompt_field: str,
    num_length_bins: int,
) -> list[tuple[str, int]]:
    if num_length_bins < 1:
        raise SystemExit("--num-length-bins must be >= 1.")

    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        source = str(row.get("source", "unknown"))
        prompt = str(row[prompt_field])
        token_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        by_source[source].append((idx, token_len))

    out: list[tuple[str, int] | None] = [None] * len(rows)
    for source, items in by_source.items():
        lengths = sorted(length for _, length in items)
        if len(items) == 1:
            idx, _ = items[0]
            out[idx] = (source, 0)
            continue

        edges = []
        for q_idx in range(1, num_length_bins):
            quantile = q_idx / num_length_bins
            pos = min(max(int(math.floor(quantile * (len(lengths) - 1))), 0), len(lengths) - 1)
            edges.append(lengths[pos])
        for idx, token_len in items:
            bin_id = 0
            while bin_id < len(edges) and token_len > edges[bin_id]:
                bin_id += 1
            out[idx] = (source, bin_id)

    return [bucket for bucket in out if bucket is not None]


def _sample_indices(
    rows: list[dict[str, object]],
    *,
    tokenizer,
    prompt_field: str,
    sample_size: int,
    sample_seed: int,
    num_length_bins: int,
) -> list[int]:
    total = len(rows)
    if sample_size <= 0 or sample_size >= total:
        return list(range(total))

    buckets = _assign_length_bins(
        rows,
        tokenizer=tokenizer,
        prompt_field=prompt_field,
        num_length_bins=num_length_bins,
    )
    by_bucket: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, bucket in enumerate(buckets):
        by_bucket[bucket].append(idx)

    rng = random.Random(sample_seed)
    for indices in by_bucket.values():
        rng.shuffle(indices)

    allocations: dict[tuple[str, int], int] = {}
    remainders: list[tuple[float, tuple[str, int]]] = []
    remaining = sample_size
    for bucket, indices in by_bucket.items():
        exact = (len(indices) * sample_size) / total
        take = int(math.floor(exact))
        if take == 0 and remaining > 0:
            take = 1
        take = min(take, len(indices))
        allocations[bucket] = take
        remaining -= take
        remainders.append((exact - math.floor(exact), bucket))

    if remaining < 0:
        for _, bucket in sorted(remainders):
            while remaining < 0 and allocations[bucket] > 1:
                allocations[bucket] -= 1
                remaining += 1
    elif remaining > 0:
        for _, bucket in sorted(remainders, reverse=True):
            capacity = len(by_bucket[bucket]) - allocations[bucket]
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            allocations[bucket] += add
            remaining -= add
            if remaining == 0:
                break

    selected: list[int] = []
    for bucket, indices in by_bucket.items():
        selected.extend(indices[: allocations[bucket]])

    if len(selected) > sample_size:
        rng.shuffle(selected)
        selected = selected[:sample_size]
    elif len(selected) < sample_size:
        missing = [idx for idx in range(total) if idx not in set(selected)]
        rng.shuffle(missing)
        selected.extend(missing[: sample_size - len(selected)])

    selected.sort()
    return selected


def _quantile(values: list[int], q: float) -> int | None:
    if not values:
        return None
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    pos = int(round((len(ordered) - 1) * q))
    return ordered[pos]


def main() -> None:
    args = _parse_args()

    rows = _load_rows(args.dataset, args.prompt_field)
    rollout_cfg = get_rollout_config(
        args.model_preset,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tp=args.tp,
        dp=args.dp,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
        use_fast=True,
    )
    selected_indices = _sample_indices(
        rows,
        tokenizer=tokenizer,
        prompt_field=args.prompt_field,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        num_length_bins=args.num_length_bins,
    )
    sampled_rows = [rows[idx] for idx in selected_indices]
    prompts = [
        build_prompt(tokenizer, str(row[args.prompt_field]), num_repetition=1)
        for row in sampled_rows
    ]

    rollout_token_ids = generate_rollout_token_ids(
        prompts,
        rollout_cfg,
        seed=args.sample_seed,
    )
    first_hits = [
        first_ngram_loop_prefix_length(token_ids, n=args.loop_n, k=args.loop_k)
        for token_ids in rollout_token_ids
    ]

    horizons = sorted(set(int(h) for h in args.horizons))
    if not horizons or horizons[0] < 1:
        raise SystemExit("--horizons must contain positive integers.")

    sampled_sources: dict[str, int] = defaultdict(int)
    for row in sampled_rows:
        sampled_sources[str(row.get("source", "unknown"))] += 1

    onset_values = [int(hit) for hit in first_hits if hit is not None]
    prevalence = {}
    for horizon in horizons:
        positives = sum(1 for hit in first_hits if hit is not None and hit <= horizon)
        prevalence[str(horizon)] = {
            "num_positive": positives,
            "num_negative": len(first_hits) - positives,
            "prevalence": positives / len(first_hits),
        }

    payload = {
        "dataset": args.dataset,
        "prompt_field": args.prompt_field,
        "sample_size": len(sampled_rows),
        "sample_seed": args.sample_seed,
        "num_length_bins": args.num_length_bins,
        "sources": dict(sorted(sampled_sources.items())),
        "rollout_config": rollout_cfg.to_dict(),
        "loop_detector": {"n": args.loop_n, "k": args.loop_k},
        "horizons": horizons,
        "prevalence": prevalence,
        "eventual_loop_within_max_tokens": {
            "num_positive": len(onset_values),
            "num_negative": len(first_hits) - len(onset_values),
            "prevalence": len(onset_values) / len(first_hits),
        },
        "first_hit_summary": {
            "min": min(onset_values) if onset_values else None,
            "p25": _quantile(onset_values, 0.25),
            "median": _quantile(onset_values, 0.5),
            "p75": _quantile(onset_values, 0.75),
            "p90": _quantile(onset_values, 0.9),
            "max": max(onset_values) if onset_values else None,
        },
    }

    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
