#!/usr/bin/env python3
"""Build loop-probe train/test datasets from Hugging Face datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
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
    extract_prefill_features_multi,
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
FEATURE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
BALANCE_CHOICES = ("none", "downsample")
ROLLOUT_LAST_TOKEN_ALL_LAYERS_MEAN = "rollout_last_token_all_layers_mean"
COMPLETION_POOLING_CHOICES = (ROLLOUT_LAST_TOKEN_ALL_LAYERS_MEAN,)
ALL_FEATURE_POOLING_CHOICES = FEATURE_POOLING_CHOICES + COMPLETION_POOLING_CHOICES


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
        "--completion-batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for rollout-completion feature extraction. "
            "Keep small for long generated trajectories."
        ),
    )
    parser.add_argument(
        "--feature-pooling",
        choices=ALL_FEATURE_POOLING_CHOICES,
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
        "--feature-key",
        default=None,
        help=(
            "Optional key for the primary feature view. "
            "If omitted, it is derived from pooling/layer."
        ),
    )
    parser.add_argument(
        "--extra-feature-view",
        action="append",
        default=[],
        metavar="KEY:POOLING:LAYER",
        help=(
            "Additional feature view to materialize into the same dataset. "
            "Repeat this flag for multiple views."
        ),
    )
    parser.add_argument(
        "--reuse-if-compatible",
        action="store_true",
        help="Skip rebuilding when out-dir has a compatible manifest and shard files.",
    )
    parser.add_argument(
        "--balance-train",
        choices=BALANCE_CHOICES,
        default="none",
        help=(
            "Optional balancing policy for the train split after loop labels are built. "
            "'downsample' keeps equal positives/negatives by random majority-class subsampling."
        ),
    )
    parser.add_argument(
        "--balance-test",
        choices=BALANCE_CHOICES,
        default="none",
        help=(
            "Optional balancing policy for the test split after loop labels are built. "
            "'downsample' keeps equal positives/negatives by random majority-class subsampling."
        ),
    )
    parser.add_argument(
        "--balance-seed",
        type=int,
        default=None,
        help="Optional balancing RNG seed. Defaults to --seed when omitted.",
    )
    parser.add_argument("--out-dir", required=True)

    return parser.parse_args()


def _layer_tag(feature_layer: int) -> str:
    if feature_layer == -1:
        return "final"
    if feature_layer >= 0:
        return f"layer{feature_layer}"
    return f"layer_m{abs(feature_layer)}"


def _default_feature_key(*, pooling: str, feature_layer: int) -> str:
    return f"{pooling}_{_layer_tag(feature_layer)}"


def _parse_feature_view(raw: str) -> tuple[str, dict[str, object]]:
    key, sep, remainder = raw.partition(":")
    if sep == "":
        raise SystemExit(
            "--extra-feature-view must match KEY:POOLING:LAYER, "
            f"got '{raw}'."
        )
    if ":" not in remainder:
        raise SystemExit(
            "--extra-feature-view must match KEY:POOLING:LAYER, "
            f"got '{raw}'."
        )

    pooling, _, layer_text = remainder.rpartition(":")
    key = key.strip()
    pooling = pooling.strip()
    layer_text = layer_text.strip()

    if not key:
        raise SystemExit("Feature view key cannot be empty.")
    if not FEATURE_KEY_PATTERN.fullmatch(key):
        raise SystemExit(
            "Feature view key must match [A-Za-z0-9_.-]+, "
            f"got '{key}'."
        )
    if pooling not in ALL_FEATURE_POOLING_CHOICES:
        raise SystemExit(
            f"Unknown feature pooling '{pooling}' for view '{key}'. "
            f"Valid: {ALL_FEATURE_POOLING_CHOICES}"
        )
    try:
        layer = int(layer_text)
    except Exception as exc:
        raise SystemExit(
            f"Feature layer for view '{key}' is not an integer: '{layer_text}'."
        ) from exc
    return key, {"pooling": pooling, "layer": layer}


def _feature_stage(pooling: str) -> str:
    if pooling in COMPLETION_POOLING_CHOICES:
        return "completion"
    return "prefill"


def _validate_layer_for_pooling(*, pooling: str, layer: int, key: str) -> None:
    if pooling in COMPLETION_POOLING_CHOICES and layer != -1:
        raise SystemExit(
            "Rollout-completion all-layer pooling ignores --feature-layer and "
            f"requires -1 for feature view '{key}', got {layer}."
        )
    if pooling in ("last_token_all_layers_mean", "last_token_all_layers_concat") and layer != -1:
        raise SystemExit(
            "All-layer prefill pooling ignores --feature-layer and requires -1 for "
            f"feature view '{key}', got {layer}."
        )


def _resolve_feature_views(args: argparse.Namespace) -> tuple[str, dict[str, dict[str, object]]]:
    if args.feature_key is not None:
        primary_key = args.feature_key.strip()
    else:
        primary_key = ""
    if not primary_key:
        primary_key = _default_feature_key(
            pooling=args.feature_pooling,
            feature_layer=args.feature_layer,
        )
    if not FEATURE_KEY_PATTERN.fullmatch(primary_key):
        raise SystemExit(
            "Primary feature key must match [A-Za-z0-9_.-]+, "
            f"got '{primary_key}'."
        )

    primary_pooling = args.feature_pooling
    primary_layer = int(args.feature_layer)
    _validate_layer_for_pooling(
        pooling=primary_pooling,
        layer=primary_layer,
        key=primary_key,
    )

    feature_views: dict[str, dict[str, object]] = {
        primary_key: {
            "pooling": primary_pooling,
            "layer": primary_layer,
            "stage": _feature_stage(primary_pooling),
        }
    }

    for raw in args.extra_feature_view:
        key, spec = _parse_feature_view(raw)
        spec["stage"] = _feature_stage(str(spec["pooling"]))
        _validate_layer_for_pooling(
            pooling=str(spec["pooling"]),
            layer=int(spec["layer"]),
            key=key,
        )
        prior = feature_views.get(key)
        if prior is not None and prior != spec:
            raise SystemExit(
                f"Duplicate feature view key '{key}' has conflicting settings."
            )
        feature_views[key] = spec

    return primary_key, feature_views


def _split_feature_views_by_stage(
    feature_views: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    prefill: dict[str, dict[str, object]] = {}
    completion: dict[str, dict[str, object]] = {}
    for key, spec in feature_views.items():
        stage = str(spec.get("stage", "prefill"))
        if stage == "completion":
            completion[key] = spec
        else:
            prefill[key] = spec
    return prefill, completion


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


def _view_shards_exist(
    manifest: dict[str, object],
    out_dir: str,
    *,
    feature_key: str,
) -> bool:
    feature_views = manifest.get("feature_views")
    if not isinstance(feature_views, dict):
        return False
    view_info = feature_views.get(feature_key)
    if not isinstance(view_info, dict):
        return False
    return _split_shards_exist(out_dir, view_info.get("train")) and _split_shards_exist(
        out_dir, view_info.get("test")
    )


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
    requested_primary_feature_key: str,
    requested_feature_views: dict[str, dict[str, object]],
    seed: int,
    balance_train: str,
    balance_test: str,
    balance_seed: int,
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
        "train_spec": asdict(train_spec),
        "test_spec": asdict(test_spec) if test_spec else None,
    }
    if _split_source_uses_ratio(split_source):
        expected["split_ratio"] = split_ratio
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False, f"manifest mismatch on '{key}'"

    expected_balancing = {
        "train": balance_train,
        "test": balance_test,
        "seed": balance_seed,
    }
    manifest_balancing = manifest.get("balancing")
    if manifest_balancing is None:
        if balance_train != "none" or balance_test != "none":
            return False, "manifest missing balancing metadata"
    elif manifest_balancing != expected_balancing:
        return False, "manifest mismatch on 'balancing'"

    manifest_seed = manifest.get("seed", None)
    if manifest_seed is not None:
        try:
            seed_value = int(manifest_seed)
        except Exception:
            return False, "manifest seed is not an integer"
        if seed_value != seed:
            return False, f"manifest seed={seed_value} != requested seed={seed}"

    feature_views = manifest.get("feature_views")
    if isinstance(feature_views, dict):
        manifest_default_feature_key = manifest.get("default_feature_key")
        if not isinstance(manifest_default_feature_key, str) or not manifest_default_feature_key:
            return False, "manifest missing default_feature_key for multi-view dataset"
        if manifest_default_feature_key != requested_primary_feature_key:
            return (
                False,
                "manifest mismatch on 'default_feature_key' "
                f"(cached='{manifest_default_feature_key}', requested='{requested_primary_feature_key}')",
            )

        for key, requested in requested_feature_views.items():
            view_info = feature_views.get(key)
            if not isinstance(view_info, dict):
                return False, f"missing requested feature view '{key}'"
            expected_stage = str(requested.get("stage", "prefill"))
            cached_stage = view_info.get("stage")
            if cached_stage is None:
                if expected_stage != "prefill":
                    return (
                        False,
                        f"feature view '{key}' missing stage metadata (expected '{expected_stage}')",
                    )
            elif cached_stage != expected_stage:
                return False, f"feature view '{key}' mismatch on 'stage'"
            expected_view = {
                "pooling": requested["pooling"],
                "layer": requested["layer"],
            }
            for view_field, expected_value in expected_view.items():
                if view_info.get(view_field) != expected_value:
                    return (
                        False,
                        f"feature view '{key}' mismatch on '{view_field}'",
                    )
            if not _view_shards_exist(manifest, out_dir, feature_key=key):
                return False, f"feature view '{key}' shards are missing"
    else:
        if len(requested_feature_views) != 1:
            return False, "manifest lacks multi-view data for requested feature set"
        only_view = next(iter(requested_feature_views.values()))
        if str(only_view.get("stage", "prefill")) != "prefill":
            return False, "legacy single-view manifest cannot satisfy completion-stage feature request"
        expected_feature = {
            "pooling": only_view["pooling"],
            "layer": only_view["layer"],
        }
        manifest_feature = manifest.get("feature_extraction")
        resolved_feature = expected_feature
        if manifest_feature is None:
            legacy_default_feature = {
                "pooling": "last_token",
                "layer": -1,
            }
            if expected_feature != legacy_default_feature:
                return (
                    False,
                    "legacy manifest missing feature_extraction metadata cannot satisfy "
                    "non-default feature view request",
                )
            resolved_feature = legacy_default_feature
        elif manifest_feature != expected_feature:
            return False, "manifest mismatch on 'feature_extraction'"
        expected_primary_key = _default_feature_key(
            pooling=resolved_feature["pooling"],
            feature_layer=resolved_feature["layer"],
        )
        if requested_primary_feature_key != expected_primary_key:
            return (
                False,
                "legacy manifest cannot satisfy "
                f"primary feature key '{requested_primary_feature_key}'",
            )
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
    feature_views: dict[str, dict[str, object]],
):
    prompt_count = len(records)
    sample_ids = _sample_ids(records)
    print(f"[{split_name}] extracting prefill features for {prompt_count} prompts", flush=True)
    view_specs = {
        key: (
            str(spec["pooling"]),
            int(spec["layer"]),
        )
        for key, spec in feature_views.items()
    }
    features_by_key = extract_prefill_features_multi(
        model,
        tokenizer,
        device,
        records,
        feature_views=view_specs,
        log_prefix=split_name,
        batch_size=prefill_batch_size,
    )
    return features_by_key, sample_ids


def _label_split(
    split_name: str,
    prompts: list[str],
    *,
    rollout_cfg,
    seed: int,
    loop_n: int,
    loop_k: int,
    return_rollout_token_ids: bool = True,
) -> tuple[list[int], list[list[int]] | None]:
    print(f"[{split_name}] running rollouts for {len(prompts)} prompts", flush=True)
    rollout_token_ids = generate_rollout_token_ids(
        prompts,
        rollout_cfg,
        seed=seed,
    )
    labels = labels_from_rollouts(
        rollout_token_ids,
        loop_n=loop_n,
        loop_k=loop_k,
    )
    if not return_rollout_token_ids:
        return labels, None
    return labels, rollout_token_ids


def _extract_completion_features(
    split_name: str,
    records: list[SampleRecord],
    rollout_token_ids: list[list[int]],
    *,
    model,
    tokenizer,
    device,
    feature_views: dict[str, dict[str, object]],
    batch_size: int,
    max_model_len: int | None,
) -> dict[str, torch.Tensor]:
    if not feature_views:
        return {}
    if batch_size < 1:
        raise SystemExit("--completion-batch-size must be >= 1.")
    if len(records) != len(rollout_token_ids):
        raise SystemExit(
            "Completion feature extraction got mismatched record/rollout counts: "
            f"{len(records)} vs {len(rollout_token_ids)}."
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise SystemExit(
                "Tokenizer has no pad_token/eos_token; cannot batch completion prompts."
            )
        tokenizer.pad_token = tokenizer.eos_token

    prompt_token_ids = tokenizer(
        [rec.prompt for rec in records],
        add_special_tokens=False,
    )["input_ids"]
    if len(prompt_token_ids) != len(records):
        raise RuntimeError("Tokenizer returned a mismatched number of completion prompts.")

    merged_sequences: list[list[int]] = []
    for idx, (prompt_ids, gen_ids) in enumerate(zip(prompt_token_ids, rollout_token_ids)):
        merged = list(prompt_ids) + [int(tok) for tok in gen_ids]
        if max_model_len is not None and max_model_len > 0 and len(merged) > max_model_len:
            merged = merged[-max_model_len:]
        if not merged:
            raise RuntimeError(f"Encountered empty prompt+rollout sequence at index {idx}.")
        merged_sequences.append(merged)

    features_by_key: dict[str, list[torch.Tensor]] = {k: [] for k in feature_views}
    total = len(merged_sequences)
    pad_id = int(tokenizer.pad_token_id)

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_sequences = merged_sequences[start:end]
            max_len = max(len(seq) for seq in batch_sequences)

            input_ids = torch.full(
                (len(batch_sequences), max_len),
                fill_value=pad_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch_sequences), max_len),
                dtype=torch.long,
                device=device,
            )
            for row_idx, seq in enumerate(batch_sequences):
                seq_t = torch.tensor(seq, dtype=torch.long, device=device)
                seq_len = int(seq_t.numel())
                input_ids[row_idx, :seq_len] = seq_t
                attention_mask[row_idx, :seq_len] = 1

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            if out.hidden_states is None:
                raise RuntimeError(
                    "Model did not return hidden states during completion feature extraction."
                )
            num_hidden_layers = len(out.hidden_states) - 1
            if num_hidden_layers < 1:
                raise RuntimeError("Model returned no hidden layers during completion extraction.")

            last_token_idx = attention_mask.sum(dim=1) - 1
            if torch.any(last_token_idx < 0):
                raise RuntimeError(
                    "Found an empty completion sequence during hidden-state extraction."
                )
            batch_idx = torch.arange(input_ids.size(0), device=device)
            per_layer_last_token = torch.stack(
                [
                    out.hidden_states[layer_idx + 1][batch_idx, last_token_idx]
                    for layer_idx in range(num_hidden_layers)
                ],
                dim=1,
            )

            for key, spec in feature_views.items():
                pooling = str(spec["pooling"])
                if pooling == ROLLOUT_LAST_TOKEN_ALL_LAYERS_MEAN:
                    batch_vecs = per_layer_last_token.mean(dim=1).float().cpu()
                else:
                    raise SystemExit(
                        f"Unsupported completion feature pooling '{pooling}' for view '{key}'. "
                        f"Valid completion poolings: {COMPLETION_POOLING_CHOICES}"
                    )
                features_by_key[key].extend(batch_vecs.unbind(dim=0))

            if end == total or start == 0 or end % 50 == 0:
                print(f"[{split_name}] completion {end}/{total}", flush=True)

    return {
        key: torch.stack(view_features, dim=0)
        for key, view_features in features_by_key.items()
    }


def _balanced_indices(
    labels: list[int],
    *,
    split_name: str,
    mode: str,
    seed: int,
) -> list[int]:
    if mode == "none":
        return list(range(len(labels)))

    if mode != "downsample":
        raise SystemExit(f"Unsupported balance mode '{mode}' for split '{split_name}'.")

    positive_idx = [idx for idx, label in enumerate(labels) if int(label) == 1]
    negative_idx = [idx for idx, label in enumerate(labels) if int(label) == 0]
    if not positive_idx or not negative_idx:
        raise SystemExit(
            f"Cannot downsample-balance split '{split_name}' because one class is missing "
            f"(pos={len(positive_idx)}, neg={len(negative_idx)})."
        )

    target_per_class = min(len(positive_idx), len(negative_idx))
    rng = random.Random(seed)
    rng.shuffle(positive_idx)
    rng.shuffle(negative_idx)
    keep = positive_idx[:target_per_class] + negative_idx[:target_per_class]
    keep.sort()
    return keep


def _subset_list(values: list[int], keep_indices: list[int]) -> list[int]:
    return [values[idx] for idx in keep_indices]


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
    primary_feature_key, feature_views = _resolve_feature_views(args)
    prefill_feature_views, completion_feature_views = _split_feature_views_by_stage(
        feature_views
    )
    balance_seed = args.seed if args.balance_seed is None else args.balance_seed

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
            requested_primary_feature_key=primary_feature_key,
            requested_feature_views=feature_views,
            seed=args.seed,
            balance_train=args.balance_train,
            balance_test=args.balance_test,
            balance_seed=balance_seed,
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
        f"train={len(train_records)}, test={len(test_records)}, "
        f"feature_views={list(feature_views.keys())}",
        flush=True,
    )

    model, tokenizer, device = load_prefill_model_and_tokenizer(
        rollout_cfg.model_id,
        trust_remote_code=rollout_cfg.trust_remote_code,
    )

    # Use a shared chat-prompt construction path across detector builders and eval scripts.
    train_records = _apply_chat_prompt(tokenizer, train_records, num_repetition=1)
    test_records = _apply_chat_prompt(tokenizer, test_records, num_repetition=1)

    train_ids = _sample_ids(train_records)
    test_ids = _sample_ids(test_records)
    train_features_by_key: dict[str, torch.Tensor] = {}
    test_features_by_key: dict[str, torch.Tensor] = {}

    if prefill_feature_views:
        train_prefill_features_by_key, train_ids = _build_split(
            "train",
            train_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefill_batch_size=args.prefill_batch_size,
            feature_views=prefill_feature_views,
        )
        test_prefill_features_by_key, test_ids = _build_split(
            "test",
            test_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            prefill_batch_size=args.prefill_batch_size,
            feature_views=prefill_feature_views,
        )
        train_features_by_key.update(train_prefill_features_by_key)
        test_features_by_key.update(test_prefill_features_by_key)

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
    need_rollout_tokens = bool(completion_feature_views)
    all_labels, all_rollout_token_ids = _label_split(
        "all",
        all_prompts,
        rollout_cfg=rollout_cfg,
        seed=args.seed,
        loop_n=args.loop_n,
        loop_k=args.loop_k,
        return_rollout_token_ids=need_rollout_tokens,
    )
    split_at = len(train_prompts)
    train_labels = all_labels[:split_at]
    test_labels = all_labels[split_at:]

    if completion_feature_views:
        if all_rollout_token_ids is None:
            raise RuntimeError(
                "Completion views requested but rollout token IDs were not retained."
            )
        train_rollout_token_ids = all_rollout_token_ids[:split_at]
        test_rollout_token_ids = all_rollout_token_ids[split_at:]
        completion_model, completion_tokenizer, completion_device = (
            load_prefill_model_and_tokenizer(
                rollout_cfg.model_id,
                trust_remote_code=rollout_cfg.trust_remote_code,
            )
        )
        train_completion_features_by_key = _extract_completion_features(
            "train",
            train_records,
            train_rollout_token_ids,
            model=completion_model,
            tokenizer=completion_tokenizer,
            device=completion_device,
            feature_views=completion_feature_views,
            batch_size=args.completion_batch_size,
            max_model_len=rollout_cfg.max_model_len,
        )
        test_completion_features_by_key = _extract_completion_features(
            "test",
            test_records,
            test_rollout_token_ids,
            model=completion_model,
            tokenizer=completion_tokenizer,
            device=completion_device,
            feature_views=completion_feature_views,
            batch_size=args.completion_batch_size,
            max_model_len=rollout_cfg.max_model_len,
        )
        train_features_by_key.update(train_completion_features_by_key)
        test_features_by_key.update(test_completion_features_by_key)

        del train_rollout_token_ids
        del test_rollout_token_ids
        del all_rollout_token_ids

        del completion_model
        del completion_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    train_keep_idx = _balanced_indices(
        train_labels,
        split_name="train",
        mode=args.balance_train,
        seed=balance_seed,
    )
    test_keep_idx = _balanced_indices(
        test_labels,
        split_name="test",
        mode=args.balance_test,
        seed=balance_seed + 1,
    )
    train_labels = _subset_list(train_labels, train_keep_idx)
    test_labels = _subset_list(test_labels, test_keep_idx)
    train_ids = _subset_list(train_ids, train_keep_idx)
    test_ids = _subset_list(test_ids, test_keep_idx)
    train_keep_idx_t = torch.tensor(train_keep_idx, dtype=torch.long)
    test_keep_idx_t = torch.tensor(test_keep_idx, dtype=torch.long)

    feature_views_manifest: dict[str, dict[str, object]] = {}
    primary_train_meta: dict[str, object] | None = None
    primary_test_meta: dict[str, object] | None = None
    primary_input_dim: int | None = None

    for feature_key, feature_spec in feature_views.items():
        train_features = train_features_by_key.get(feature_key)
        test_features = test_features_by_key.get(feature_key)
        if train_features is None or test_features is None:
            raise SystemExit(
                f"Missing extracted features for requested view '{feature_key}'."
            )
        train_features = train_features.index_select(0, train_keep_idx_t)
        test_features = test_features.index_select(0, test_keep_idx_t)

        input_dim = int(train_features.size(1))
        if int(test_features.size(1)) != input_dim:
            raise SystemExit(
                "Input dim mismatch between train/test for feature "
                f"'{feature_key}': {input_dim} vs {int(test_features.size(1))}"
            )
        if train_features.size(0) != len(train_labels):
            raise SystemExit(
                "Mismatched feature/label counts for split 'train' "
                f"feature '{feature_key}': {train_features.size(0)} vs {len(train_labels)}"
            )
        if test_features.size(0) != len(test_labels):
            raise SystemExit(
                "Mismatched feature/label counts for split 'test' "
                f"feature '{feature_key}': {test_features.size(0)} vs {len(test_labels)}"
            )

        if feature_key == primary_feature_key:
            train_split_name = "train"
            test_split_name = "test"
        else:
            train_split_name = os.path.join("features", feature_key, "train")
            test_split_name = os.path.join("features", feature_key, "test")

        train_meta = save_split_shards(
            args.out_dir,
            train_split_name,
            train_features,
            train_labels,
            train_ids,
            shard_size=args.shard_size,
        )
        test_meta = save_split_shards(
            args.out_dir,
            test_split_name,
            test_features,
            test_labels,
            test_ids,
            shard_size=args.shard_size,
        )

        feature_views_manifest[feature_key] = {
            "stage": feature_spec.get("stage", "prefill"),
            "pooling": feature_spec["pooling"],
            "layer": feature_spec["layer"],
            "input_dim": input_dim,
            "train": train_meta,
            "test": test_meta,
        }

        if feature_key == primary_feature_key:
            primary_train_meta = train_meta
            primary_test_meta = test_meta
            primary_input_dim = input_dim

    if primary_train_meta is None or primary_test_meta is None or primary_input_dim is None:
        raise SystemExit(f"Primary feature view '{primary_feature_key}' was not materialized.")

    manifest = {
        "version": 3,
        "input_dim": primary_input_dim,
        "default_feature_key": primary_feature_key,
        "feature_extraction": {
            "stage": feature_views[primary_feature_key].get("stage", "prefill"),
            "pooling": feature_views[primary_feature_key]["pooling"],
            "layer": feature_views[primary_feature_key]["layer"],
        },
        "feature_views": feature_views_manifest,
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
        "balancing": {
            "train": args.balance_train,
            "test": args.balance_test,
            "seed": balance_seed,
        },
        "rollout_config": rollout_cfg.to_dict(),
        "train_spec": asdict(train_spec),
        "test_spec": asdict(test_spec) if test_spec else None,
        "train": primary_train_meta,
        "test": primary_test_meta,
    }
    write_manifest(args.out_dir, manifest)

    print(f"Wrote probe dataset to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
