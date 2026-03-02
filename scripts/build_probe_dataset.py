#!/usr/bin/env python3
"""Build loop-probe train/test datasets from Hugging Face datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from dataclasses import asdict

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.configs import get_rollout_config, preset_choices
from loop_probe.hf_data import load_prompt_records, specs_equal, split_records
from loop_probe.labeling import labels_from_rollouts
from loop_probe.prefill import (
    FEATURE_POOLING_CHOICES,
    extract_prefill_features,
    load_prefill_model_and_tokenizer,
)
from loop_probe.rollout import generate_rollout_token_ids
from loop_probe.serialization import save_split_shards, write_manifest
from loop_probe.types import DatasetSpec, SampleRecord
from utils import build_prompt

DEFAULT_TEST_DATASET = "data/aime_2024_2025.jsonl"

RATIO_SPLIT_SOURCES = {
    "single_dataset_split",
    "same_train_test_spec_split",
    "same_train_test_source_ratio_split",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--train-max-samples", type=int, default=None)

    parser.add_argument(
        "--test-dataset",
        default="",
        help=(
            "Optional test dataset (HF dataset id or local JSONL path). "
            f"If omitted, defaults to '{DEFAULT_TEST_DATASET}'."
        ),
    )
    parser.add_argument("--test-config", default=None)
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--test-max-samples", type=int, default=None)

    parser.add_argument("--prompt-field", required=True)
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.1,
        help=(
            "Used for ratio-based splitting when train/test come from one source."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)

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
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )

    parser.add_argument("--loop-n", type=int, default=30)
    parser.add_argument("--loop-k", type=int, default=20)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument(
        "--feature-pooling",
        choices=FEATURE_POOLING_CHOICES,
        default="last_token",
        help="How to pool token activations into one vector per prompt.",
    )
    parser.add_argument(
        "--feature-layer",
        type=int,
        default=-1,
        help=(
            "Transformer layer index for features "
            "(0 = first layer, -1 = final layer)."
        ),
    )
    parser.add_argument(
        "--reuse-if-compatible",
        action="store_true",
        help="Skip rebuilding when out-dir has a compatible manifest and shard files.",
    )
    parser.add_argument("--out-dir", required=True)

    return parser.parse_args()


def _sample_ids(records: list[SampleRecord]) -> list[int]:
    return [rec.sample_id for rec in records]


def _prompts(records: list[SampleRecord]) -> list[str]:
    return [rec.prompt for rec in records]


def _apply_chat_prompt(
    tokenizer,
    records: list[SampleRecord],
    *,
    num_repetition: int = 1,
) -> list[SampleRecord]:
    formatted: list[SampleRecord] = []
    for rec in records:
        formatted.append(
            SampleRecord(
                sample_id=rec.sample_id,
                prompt=build_prompt(tokenizer, rec.prompt, num_repetition),
                source_split=rec.source_split,
            )
        )
    return formatted


def _same_data_source(a: DatasetSpec, b: DatasetSpec) -> bool:
    if os.path.isfile(a.dataset) and os.path.isfile(b.dataset):
        try:
            return os.path.samefile(a.dataset, b.dataset)
        except OSError:
            return os.path.abspath(a.dataset) == os.path.abspath(b.dataset)

    return (
        a.dataset == b.dataset
        and a.config == b.config
        and a.split == b.split
    )


def _with_max_samples(spec: DatasetSpec, max_samples: int | None) -> DatasetSpec:
    return DatasetSpec(
        dataset=spec.dataset,
        config=spec.config,
        split=spec.split,
        max_samples=max_samples,
    )


def _split_source_uses_ratio(split_source: str) -> bool:
    return split_source in RATIO_SPLIT_SOURCES


def _make_specs(args: argparse.Namespace) -> tuple[DatasetSpec, DatasetSpec | None]:
    train_spec = DatasetSpec(
        dataset=args.train_dataset,
        config=args.train_config,
        split=args.train_split,
        max_samples=args.train_max_samples,
    )

    test_dataset = args.test_dataset or DEFAULT_TEST_DATASET
    if not args.test_dataset and not os.path.isfile(test_dataset):
        raise SystemExit(
            f"Default test dataset '{test_dataset}' was not found. "
            "Pass --test-dataset explicitly or create the default file."
        )

    test_spec = DatasetSpec(
        dataset=test_dataset,
        config=args.test_config,
        split=args.test_split,
        max_samples=args.test_max_samples,
    )
    return train_spec, test_spec


def _resolve_splits(
    args: argparse.Namespace,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec | None,
) -> tuple[list[SampleRecord], list[SampleRecord], str]:
    if test_spec is None:
        merged_records = load_prompt_records(train_spec, args.prompt_field)
        train_records, test_records = split_records(
            merged_records,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
        split_source = "single_dataset_split"
        return train_records, test_records, split_source

    if specs_equal(train_spec, test_spec):
        merged_records = load_prompt_records(train_spec, args.prompt_field)
        train_records, test_records = split_records(
            merged_records,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
        split_source = "same_train_test_spec_split"
        return train_records, test_records, split_source

    if _same_data_source(train_spec, test_spec):
        train_max = train_spec.max_samples
        test_max = test_spec.max_samples

        # When both split sizes are explicitly requested from the same source,
        # build a single shuffled pool and carve out disjoint train/test subsets.
        if train_max is not None and test_max is not None:
            requested_total = train_max + test_max
            merged_spec = _with_max_samples(train_spec, requested_total)
            merged_records = load_prompt_records(merged_spec, args.prompt_field)
            if len(merged_records) < requested_total:
                raise SystemExit(
                    "Requested disjoint split sizes exceed available rows: "
                    f"need train_max_samples + test_max_samples = {requested_total}, "
                    f"got {len(merged_records)}."
                )
            work = list(merged_records)
            rng = random.Random(args.seed)
            rng.shuffle(work)
            test_records = work[:test_max]
            train_records = work[test_max : test_max + train_max]
            split_source = "same_train_test_source_count_split"
            return train_records, test_records, split_source

        merged_spec = _with_max_samples(train_spec, None)
        merged_records = load_prompt_records(merged_spec, args.prompt_field)
        train_records, test_records = split_records(
            merged_records,
            test_ratio=args.split_ratio,
            seed=args.seed,
        )
        if train_max is not None:
            train_records = train_records[:train_max]
        if test_max is not None:
            test_records = test_records[:test_max]
        if not train_records:
            raise SystemExit("Train split is empty.")
        if not test_records:
            raise SystemExit("Test split is empty.")
        split_source = "same_train_test_source_ratio_split"
        return train_records, test_records, split_source

    train_records = load_prompt_records(train_spec, args.prompt_field)
    test_records = load_prompt_records(test_spec, args.prompt_field)
    if not train_records:
        raise SystemExit("Train split is empty.")
    if not test_records:
        raise SystemExit("Test split is empty.")
    split_source = "separate_specs"
    return train_records, test_records, split_source


def _resolve_split_source(train_spec: DatasetSpec, test_spec: DatasetSpec | None) -> str:
    if test_spec is None:
        return "single_dataset_split"
    if specs_equal(train_spec, test_spec):
        return "same_train_test_spec_split"
    if _same_data_source(train_spec, test_spec):
        if train_spec.max_samples is not None and test_spec.max_samples is not None:
            return "same_train_test_source_count_split"
        return "same_train_test_source_ratio_split"
    return "separate_specs"


def _split_shards_exist(out_dir: str, split_info: object) -> bool:
    if not isinstance(split_info, dict):
        return False
    shard_paths = split_info.get("shards")
    if not isinstance(shard_paths, list) or not shard_paths:
        return False
    for rel_path in shard_paths:
        if not isinstance(rel_path, str):
            return False
        if not os.path.exists(os.path.join(out_dir, rel_path)):
            return False
    return True


def _probe_cache_status(
    out_dir: str,
    *,
    train_spec: DatasetSpec,
    test_spec: DatasetSpec | None,
    prompt_field: str,
    split_source: str,
    split_ratio: float,
    loop_n: int,
    loop_k: int,
    rollout_config: dict[str, object],
    feature_pooling: str,
    feature_layer: int,
    seed: int,
) -> tuple[bool, str]:
    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return False, "manifest.json not found"

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:
        return False, f"failed to read manifest.json: {exc}"

    expected = {
        "prompt_field": prompt_field,
        "split_source": split_source,
        "loop_detector": {"n": loop_n, "k": loop_k},
        "rollout_config": rollout_config,
        "feature_extraction": {
            "pooling": feature_pooling,
            "layer": feature_layer,
        },
        "train_spec": asdict(train_spec),
        "test_spec": asdict(test_spec) if test_spec else None,
    }
    if _split_source_uses_ratio(split_source):
        expected["split_ratio"] = split_ratio
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"manifest mismatch on '{key}'"

    manifest_seed = manifest.get("seed", None)
    if manifest_seed is not None:
        try:
            seed_value = int(manifest_seed)
        except Exception:
            return False, "manifest seed is not an integer"
        if seed_value != seed:
            return False, f"manifest seed={seed_value} != requested seed={seed}"

    if not _split_shards_exist(out_dir, manifest.get("train")):
        return False, "train shards are missing"
    if not _split_shards_exist(out_dir, manifest.get("test")):
        return False, "test shards are missing"

    if manifest_seed is None:
        return True, "compatible legacy manifest match (no seed recorded)"
    return True, "manifest and shards match requested config"


def _build_split(
    split_name: str,
    records: list[SampleRecord],
    *,
    model,
    tokenizer,
    device,
    prefill_batch_size: int,
    feature_pooling: str,
    feature_layer: int,
):
    prompt_count = len(records)
    sample_ids = _sample_ids(records)
    print(f"[{split_name}] extracting prefill features for {prompt_count} prompts", flush=True)
    features = extract_prefill_features(
        model,
        tokenizer,
        device,
        records,
        log_prefix=split_name,
        batch_size=prefill_batch_size,
        feature_pooling=feature_pooling,
        feature_layer=feature_layer,
    )
    return features, sample_ids


def _label_split(
    split_name: str,
    prompts: list[str],
    *,
    rollout_cfg,
    seed: int,
    loop_n: int,
    loop_k: int,
) -> list[int]:
    print(f"[{split_name}] running rollouts for {len(prompts)} prompts", flush=True)
    rollout_token_ids = generate_rollout_token_ids(
        prompts,
        rollout_cfg,
        seed=seed,
    )
    return labels_from_rollouts(
        rollout_token_ids,
        loop_n=loop_n,
        loop_k=loop_k,
    )


def main() -> None:
    args = _parse_args()

    try:
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
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    train_spec, test_spec = _make_specs(args)
    split_source = _resolve_split_source(train_spec, test_spec)

    if args.reuse_if_compatible:
        cache_hit, reason = _probe_cache_status(
            args.out_dir,
            train_spec=train_spec,
            test_spec=test_spec,
            prompt_field=args.prompt_field,
            split_source=split_source,
            split_ratio=args.split_ratio,
            loop_n=args.loop_n,
            loop_k=args.loop_k,
            rollout_config=rollout_cfg.to_dict(),
            feature_pooling=args.feature_pooling,
            feature_layer=args.feature_layer,
            seed=args.seed,
        )
        if cache_hit:
            print(
                f"Reusing cached probe dataset at {args.out_dir}: {reason}",
                flush=True,
            )
            return
        print(
            f"Cache miss at {args.out_dir}: {reason}. Rebuilding dataset.",
            flush=True,
        )

    train_records, test_records, split_source = _resolve_splits(args, train_spec, test_spec)

    print(
        f"Building probe dataset with model={rollout_cfg.model_id}, "
        f"train={len(train_records)}, test={len(test_records)}",
        flush=True,
    )

    model, tokenizer, device = load_prefill_model_and_tokenizer(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
    )

    # Use a shared chat-prompt construction path across detector builders and eval scripts.
    train_records = _apply_chat_prompt(tokenizer, train_records, num_repetition=1)
    test_records = _apply_chat_prompt(tokenizer, test_records, num_repetition=1)

    train_features, train_ids = _build_split(
        "train",
        train_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefill_batch_size=args.prefill_batch_size,
        feature_pooling=args.feature_pooling,
        feature_layer=args.feature_layer,
    )

    test_features, test_ids = _build_split(
        "test",
        test_records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prefill_batch_size=args.prefill_batch_size,
        feature_pooling=args.feature_pooling,
        feature_layer=args.feature_layer,
    )

    input_dim = int(train_features.size(1))
    if int(test_features.size(1)) != input_dim:
        raise SystemExit(
            f"Input dim mismatch between train/test: {input_dim} vs {int(test_features.size(1))}"
        )

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    train_prompts = _prompts(train_records)
    test_prompts = _prompts(test_records)
    all_prompts = train_prompts + test_prompts
    all_labels = _label_split(
        "all",
        all_prompts,
        rollout_cfg=rollout_cfg,
        seed=args.seed,
        loop_n=args.loop_n,
        loop_k=args.loop_k,
    )
    split_at = len(train_prompts)
    train_labels = all_labels[:split_at]
    test_labels = all_labels[split_at:]

    if train_features.size(0) != len(train_labels):
        raise SystemExit(
            f"Mismatched feature/label counts for split 'train': "
            f"{train_features.size(0)} vs {len(train_labels)}"
        )
    if test_features.size(0) != len(test_labels):
        raise SystemExit(
            f"Mismatched feature/label counts for split 'test': "
            f"{test_features.size(0)} vs {len(test_labels)}"
        )

    train_meta = save_split_shards(
        args.out_dir,
        "train",
        train_features,
        train_labels,
        train_ids,
        shard_size=args.shard_size,
    )
    test_meta = save_split_shards(
        args.out_dir,
        "test",
        test_features,
        test_labels,
        test_ids,
        shard_size=args.shard_size,
    )

    manifest = {
        "version": 2,
        "input_dim": input_dim,
        "feature_extraction": {
            "pooling": args.feature_pooling,
            "layer": args.feature_layer,
        },
        "prompt_field": args.prompt_field,
        "prompt_template": {
            "source": "utils.build_prompt",
            "num_repetition": 1,
            "chat_template": True,
        },
        "split_source": split_source,
        "split_ratio": args.split_ratio if _split_source_uses_ratio(split_source) else None,
        "seed": args.seed,
        "loop_detector": {
            "n": args.loop_n,
            "k": args.loop_k,
        },
        "rollout_config": rollout_cfg.to_dict(),
        "train_spec": asdict(train_spec),
        "test_spec": asdict(test_spec) if test_spec else None,
        "train": train_meta,
        "test": test_meta,
    }
    write_manifest(args.out_dir, manifest)

    print(f"Wrote probe dataset to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
