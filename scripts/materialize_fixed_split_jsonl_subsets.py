#!/usr/bin/env python3
"""Write JSONL prompt subsets for the exact sample IDs stored in a reference dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data-dir", required=True)
    parser.add_argument("--train-pool-jsonl", required=True)
    parser.add_argument("--eval-pool-jsonl", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=None,
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _load_json(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _resolve_splits(
    manifest: dict[str, object],
    requested_splits: list[str] | tuple[str, ...] | None,
) -> list[str]:
    available = [
        split for split in ("train", "test") if isinstance(manifest.get(split), dict)
    ]
    if not available:
        raise SystemExit("Reference manifest exposes no top-level train/test splits.")
    if not requested_splits:
        return available
    splits = list(dict.fromkeys(requested_splits))
    missing = [split for split in splits if split not in available]
    if missing:
        raise SystemExit(
            f"Requested splits {missing} are unavailable. Available splits: {available}"
        )
    return splits


def _load_sample_ids(
    *,
    reference_data_dir: str,
    manifest: dict[str, object],
    split: str,
) -> list[int]:
    split_info = manifest.get(split)
    if not isinstance(split_info, dict):
        raise SystemExit(f"Reference manifest missing split '{split}'.")
    shard_paths = split_info.get("shards")
    if not isinstance(shard_paths, list) or not shard_paths:
        raise SystemExit(f"Reference split '{split}' has no shard files.")

    sample_ids = []
    for rel_path in shard_paths:
        if not isinstance(rel_path, str):
            raise SystemExit(f"Invalid shard path entry for split '{split}'.")
        shard = torch.load(os.path.join(reference_data_dir, rel_path), map_location="cpu")
        sample_ids.append(shard["sample_ids"].to(dtype=torch.int64))
    return [int(v) for v in torch.cat(sample_ids, dim=0).tolist()]


def _write_subset_jsonl(
    *,
    pool_rows: list[dict[str, object]],
    sample_ids: list[int],
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample_id in sample_ids:
            if sample_id < 0 or sample_id >= len(pool_rows):
                raise SystemExit(
                    f"sample_id={sample_id} out of range for pool size {len(pool_rows)}"
                )
            row = dict(pool_rows[sample_id])
            row["_source_sample_id"] = int(sample_id)
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def main() -> None:
    args = _parse_args()
    manifest = _load_json(os.path.join(args.reference_data_dir, "manifest.json"))
    splits = _resolve_splits(manifest, args.splits)
    train_pool_rows = _load_jsonl_rows(args.train_pool_jsonl)
    eval_pool_rows = _load_jsonl_rows(args.eval_pool_jsonl)

    os.makedirs(args.out_dir, exist_ok=True)
    summary: dict[str, object] = {
        "created_from": args.reference_data_dir,
        "splits": {},
    }

    for split in splits:
        sample_ids = _load_sample_ids(
            reference_data_dir=args.reference_data_dir,
            manifest=manifest,
            split=split,
        )
        pool_rows = train_pool_rows if split == "train" else eval_pool_rows
        out_path = os.path.join(args.out_dir, f"{split}.jsonl")
        _write_subset_jsonl(
            pool_rows=pool_rows,
            sample_ids=sample_ids,
            out_path=out_path,
        )
        summary["splits"][split] = {
            "num_rows": len(sample_ids),
            "path": os.path.relpath(out_path, args.out_dir),
        }
        print(f"Wrote {len(sample_ids)} rows to {out_path}", flush=True)

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
