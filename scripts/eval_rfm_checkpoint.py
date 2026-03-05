#!/usr/bin/env python3
"""Evaluate an RFM-lite checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.dataloader import ActivationDataset, read_manifest
from loop_probe.train_utils import evaluate_binary_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--feature-key",
        default=None,
        help=(
            "Optional feature view key from a multi-view dataset manifest. "
            "If omitted, uses manifest default or legacy single-view fields."
        ),
    )
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _rff_features(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    proj = x @ w.T
    proj += b[None, :]
    scale = np.sqrt(2.0 / float(w.shape[0]))
    return (scale * np.cos(proj)).astype(np.float32)


def _rff_gradients(
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    proj = x @ w.T
    proj += b[None, :]
    sin_term = np.sin(proj).astype(np.float32)
    weighted = sin_term * beta[None, :]
    scale = -np.sqrt(2.0 / float(w.shape[0]))
    return (scale * (weighted @ w)).astype(np.float32)


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def _metrics_from_logits(labels01: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    labels_t = torch.from_numpy(labels01.astype(np.float32))
    logits_t = torch.from_numpy(logits.astype(np.float32))
    return evaluate_binary_metrics(labels_t, logits_t)


def _to_numpy_checkpoint_entry(entry: dict[str, Any]) -> dict[str, np.ndarray]:
    out = {}
    for key in ("mean", "std", "w_rff", "b_rff", "beta"):
        value = entry.get(key)
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().cpu().numpy().astype(np.float32, copy=False)
        elif isinstance(value, np.ndarray):
            out[key] = value.astype(np.float32, copy=False)
        else:
            raise SystemExit(f"Checkpoint pipeline entry missing tensor array '{key}'.")
    return out


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise SystemExit("Invalid checkpoint payload.")
    if payload.get("algorithm") != "rfm_rff_lite":
        raise SystemExit(
            f"Unsupported checkpoint algorithm: {payload.get('algorithm')}"
        )
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, list) or not pipeline:
        raise SystemExit("Checkpoint pipeline is missing or empty.")
    grad_weight = float(payload.get("grad_weight", 1.0))
    grad_scale = float(np.sqrt(max(grad_weight, 0.0)))
    checkpoint_feature_key = payload.get("feature_key")
    manifest = read_manifest(args.data_dir)
    resolved_feature_key = args.feature_key
    if (
        (resolved_feature_key is None or resolved_feature_key == "")
        and isinstance(checkpoint_feature_key, str)
        and checkpoint_feature_key
        and isinstance(manifest.get("feature_views"), dict)
    ):
        resolved_feature_key = checkpoint_feature_key

    ds = ActivationDataset(
        data_dir=args.data_dir,
        split=args.split,
        feature_key=resolved_feature_key,
    )
    x_cur = ds.x.detach().cpu().numpy().astype(np.float32, copy=False)
    labels = ds.y.detach().cpu().numpy().astype(np.int64, copy=False)

    logits: np.ndarray | None = None
    for idx, raw_entry in enumerate(pipeline):
        if not isinstance(raw_entry, dict):
            raise SystemExit(f"Invalid pipeline entry at index {idx}.")
        entry = _to_numpy_checkpoint_entry(raw_entry)
        x_std = _standardize_apply(x_cur, entry["mean"], entry["std"])
        phi = _rff_features(x_std, entry["w_rff"], entry["b_rff"])
        logits = (phi @ entry["beta"]).astype(np.float32)

        if idx < len(pipeline) - 1:
            if grad_weight > 0.0:
                grads = _rff_gradients(
                    x_std,
                    entry["w_rff"],
                    entry["b_rff"],
                    entry["beta"],
                )
                x_cur = np.concatenate(
                    [x_std, grad_scale * grads.astype(np.float32)],
                    axis=1,
                ).astype(np.float32)
            else:
                x_cur = x_std

    if logits is None:
        raise SystemExit("No logits produced from checkpoint pipeline.")

    metrics = _metrics_from_logits(labels, logits)
    out = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "feature_key": ds.feature_key,
        "num_pipeline_steps": len(pipeline),
        **metrics,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
