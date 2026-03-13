#!/usr/bin/env python3
"""Re-extract prefill feature views on fixed labels/sample IDs from a reference dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loop_probe.prefill import (
    FEATURE_POOLING_CHOICES,
    extract_prefill_features_multi,
    load_prefill_model_and_tokenizer,
)
from loop_probe.serialization import save_split_shards, write_manifest
from loop_probe.types import SampleRecord
from utils import build_prompt

FEATURE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--train-pool-jsonl", required=True)
    parser.add_argument("--eval-pool-jsonl", required=True)
    parser.add_argument("--prompt-field", default=None)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true")
    parser.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
    )
    parser.set_defaults(trust_remote_code=None)
    parser.add_argument("--num-repetition", type=int, default=None)
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=None,
    )
    parser.add_argument("--feature-key", default=None)
    parser.add_argument(
        "--feature-pooling",
        choices=FEATURE_POOLING_CHOICES,
        required=True,
    )
    parser.add_argument("--feature-layer", type=int, default=-1)
    parser.add_argument(
        "--extra-feature-view",
        action="append",
        default=[],
        metavar="KEY:POOLING:LAYER",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _default_feature_key(*, pooling: str, feature_layer: int) -> str:
    if feature_layer == -1:
        layer_tag = "final"
    elif feature_layer >= 0:
        layer_tag = f"layer{feature_layer}"
    else:
        layer_tag = f"layer_m{abs(feature_layer)}"
    return f"{pooling}_{layer_tag}"


def _parse_feature_view(raw: str) -> tuple[str, dict[str, object]]:
    key, sep, remainder = raw.partition(":")
    if sep == "" or ":" not in remainder:
        raise SystemExit(
            "--extra-feature-view must match KEY:POOLING:LAYER, "
            f"got '{raw}'."
        )
    pooling, _, layer_text = remainder.rpartition(":")
    key = key.strip()
    pooling = pooling.strip()
    layer_text = layer_text.strip()
    if not key or not FEATURE_KEY_PATTERN.fullmatch(key):
        raise SystemExit(
            "Feature view key must match [A-Za-z0-9_.-]+, "
            f"got '{key}'."
        )
    if pooling not in FEATURE_POOLING_CHOICES:
        raise SystemExit(
            f"Unknown feature pooling '{pooling}' for view '{key}'. "
            f"Valid: {FEATURE_POOLING_CHOICES}"
        )
    try:
        layer = int(layer_text)
    except Exception as exc:
        raise SystemExit(
            f"Feature layer for view '{key}' is not an integer: '{layer_text}'."
        ) from exc
    return key, {"pooling": pooling, "layer": layer, "stage": "prefill"}


def _resolve_feature_views(args: argparse.Namespace) -> tuple[str, dict[str, dict[str, object]]]:
    primary_key = args.feature_key.strip() if args.feature_key else ""
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
    feature_views: dict[str, dict[str, object]] = {
        primary_key: {
            "pooling": args.feature_pooling,
            "layer": int(args.feature_layer),
            "stage": "prefill",
        }
    }
    for raw in args.extra_feature_view:
        key, spec = _parse_feature_view(raw)
        prior = feature_views.get(key)
        if prior is not None and prior != spec:
            raise SystemExit(
                f"Duplicate feature view key '{key}' has conflicting settings."
            )
        feature_views[key] = spec
    return primary_key, feature_views


def _load_jsonl_rows(path: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_num}") from exc
            if not isinstance(row, dict):
                raise SystemExit(f"Expected JSON object rows in {path}:{line_num}")
            rows.append(row)
    return rows


def _load_reference_manifest(reference_data_dir: str) -> dict[str, object]:
    manifest_path = os.path.join(reference_data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_built_splits(
    manifest: dict[str, object],
    requested_splits: list[str] | tuple[str, ...] | None,
) -> list[str]:
    available_splits = [
        split for split in ("train", "test") if isinstance(manifest.get(split), dict)
    ]
    if not available_splits:
        raise SystemExit(
            "Reference manifest does not expose any top-level train/test splits."
        )

    if not requested_splits:
        return available_splits

    built_splits = list(dict.fromkeys(requested_splits))
    missing = [split for split in built_splits if split not in available_splits]
    if missing:
        raise SystemExit(
            "Requested split(s) are unavailable in the reference manifest: "
            f"{missing}. Available splits: {available_splits}"
        )
    return built_splits


def _load_reference_split(
    *,
    reference_data_dir: str,
    manifest: dict[str, object],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    split_info = manifest.get(split)
    if not isinstance(split_info, dict):
        raise SystemExit(f"Reference manifest missing split '{split}'.")
    shard_paths = split_info.get("shards")
    if not isinstance(shard_paths, list) or not shard_paths:
        raise SystemExit(f"Reference split '{split}' has no shard files.")

    labels = []
    sample_ids = []
    for rel_path in shard_paths:
        if not isinstance(rel_path, str):
            raise SystemExit(f"Invalid shard path entry for split '{split}'.")
        shard = torch.load(os.path.join(reference_data_dir, rel_path), map_location="cpu")
        labels.append(shard["y"].to(dtype=torch.uint8))
        sample_ids.append(shard["sample_ids"].to(dtype=torch.int64))
    return torch.cat(labels, dim=0), torch.cat(sample_ids, dim=0)


def _build_records(
    *,
    pool_rows: list[dict[str, object]],
    sample_ids: torch.Tensor,
    prompt_field: str,
    tokenizer,
    num_repetition: int,
    split: str,
) -> list[SampleRecord]:
    records = []
    for sample_id in sample_ids.tolist():
        sample_id_int = int(sample_id)
        if sample_id_int < 0 or sample_id_int >= len(pool_rows):
            raise SystemExit(
                f"sample_id={sample_id_int} out of range for pool size {len(pool_rows)}"
            )
        row = pool_rows[sample_id_int]
        prompt = row.get(prompt_field)
        if not isinstance(prompt, str):
            raise SystemExit(
                f"Prompt field '{prompt_field}' missing/invalid for sample_id={sample_id_int}"
            )
        records.append(
            SampleRecord(
                sample_id=sample_id_int,
                prompt=build_prompt(tokenizer, prompt, num_repetition),
                source_split=split,
            )
        )
    return records


def _save_view_split(
    *,
    out_dir: str,
    primary_key: str,
    feature_key: str,
    split: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: torch.Tensor,
    shard_size: int,
) -> dict[str, object]:
    if feature_key == primary_key:
        view_dir = out_dir
        prefix = ""
    else:
        view_dir = os.path.join(out_dir, "features", feature_key)
        prefix = os.path.join("features", feature_key)
    split_meta = save_split_shards(
        view_dir,
        split,
        features,
        labels.tolist(),
        sample_ids.tolist(),
        shard_size=shard_size,
    )
    if prefix:
        split_meta["shards"] = [
            os.path.join(prefix, rel_path) for rel_path in split_meta["shards"]
        ]
    return split_meta


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    reference_manifest = _load_reference_manifest(args.reference_data_dir)
    prompt_field = args.prompt_field
    if prompt_field is None:
        prompt_field = reference_manifest.get("prompt_field")
    if not isinstance(prompt_field, str) or not prompt_field:
        raise SystemExit("Could not resolve prompt field; pass --prompt-field explicitly.")

    rollout_config = reference_manifest.get("rollout_config")
    model_id = args.model_id
    if model_id is None and isinstance(rollout_config, dict):
        candidate = rollout_config.get("model_id")
        if isinstance(candidate, str) and candidate:
            model_id = candidate
    if not isinstance(model_id, str) or not model_id:
        raise SystemExit("Could not resolve model id; pass --model-id explicitly.")

    trust_remote_code = args.trust_remote_code
    if trust_remote_code is None and isinstance(rollout_config, dict):
        candidate = rollout_config.get("trust_remote_code")
        if isinstance(candidate, bool):
            trust_remote_code = candidate
    if trust_remote_code is None:
        trust_remote_code = True

    prompt_template = reference_manifest.get("prompt_template")
    num_repetition = args.num_repetition
    if num_repetition is None and isinstance(prompt_template, dict):
        candidate = prompt_template.get("num_repetition")
        if isinstance(candidate, int) and candidate >= 1:
            num_repetition = candidate
    if num_repetition is None:
        num_repetition = 1

    primary_key, feature_views = _resolve_feature_views(args)
    feature_view_specs = {
        key: (str(spec["pooling"]), int(spec["layer"]))
        for key, spec in feature_views.items()
    }

    train_pool_rows = _load_jsonl_rows(args.train_pool_jsonl)
    eval_pool_rows = _load_jsonl_rows(args.eval_pool_jsonl)

    model, tokenizer, device = load_prefill_model_and_tokenizer(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
    )

    manifest_views: dict[str, dict[str, object]] = {
        key: {
            "layer": int(spec["layer"]),
            "pooling": str(spec["pooling"]),
            "stage": "prefill",
        }
        for key, spec in feature_views.items()
    }

    built_splits = _resolve_built_splits(reference_manifest, args.splits)
    print(
        f"Rebuilding splits {built_splits} from {args.reference_data_dir}",
        flush=True,
    )

    for split in built_splits:
        labels, sample_ids = _load_reference_split(
            reference_data_dir=args.reference_data_dir,
            manifest=reference_manifest,
            split=split,
        )
        pool_rows = train_pool_rows if split == "train" else eval_pool_rows
        records = _build_records(
            pool_rows=pool_rows,
            sample_ids=sample_ids,
            prompt_field=prompt_field,
            tokenizer=tokenizer,
            num_repetition=num_repetition,
            split=split,
        )
        features_by_key = extract_prefill_features_multi(
            model,
            tokenizer,
            device,
            records,
            feature_views=feature_view_specs,
            log_prefix=split,
            batch_size=args.prefill_batch_size,
        )
        for key, features in features_by_key.items():
            split_meta = _save_view_split(
                out_dir=args.out_dir,
                primary_key=primary_key,
                feature_key=key,
                split=split,
                features=features,
                labels=labels,
                sample_ids=sample_ids,
                shard_size=args.shard_size,
            )
            manifest_views[key][split] = split_meta
            manifest_views[key]["input_dim"] = int(features.size(1))

    payload = {
        "version": 4,
        "created_from": args.reference_data_dir,
        "default_feature_key": primary_key,
        "feature_key": primary_key,
        "feature_stage": "prefill",
        "feature_pooling": feature_views[primary_key]["pooling"],
        "input_dim": int(manifest_views[primary_key]["input_dim"]),
        "feature_views": manifest_views,
        "prompt_field": prompt_field,
        "prompt_template": {
            "chat_template": True,
            "num_repetition": num_repetition,
            "source": "utils.build_prompt",
        },
        "prefill_extraction": {
            "model_id": model_id,
            "trust_remote_code": trust_remote_code,
            "prefill_batch_size": args.prefill_batch_size,
        },
    }
    for split in built_splits:
        payload[split] = manifest_views[primary_key][split]
    for key in (
        "balancing",
        "label_spec",
        "loop_detector",
        "rollout_config",
        "seed",
        "selection",
        "split_ratio",
        "split_source",
        "test_spec",
        "train_spec",
    ):
        value = reference_manifest.get(key)
        if value is not None:
            payload[key] = value

    write_manifest(args.out_dir, payload)
    print(f"Wrote fixed-split prefill views to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
