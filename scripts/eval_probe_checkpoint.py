#!/usr/bin/env python3
"""Evaluate a trained torch probe checkpoint on a dataset split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.configs import ProbeConfig, build_probe_model, get_probe_config, probe_preset_choices
from loop_probe.dataloader import make_dataloader, read_manifest, resolve_input_dim, resolve_split_info
from loop_probe.train_utils import choose_device, evaluate_binary_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
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
    parser.add_argument(
        "--probe-preset",
        choices=probe_preset_choices(),
        default=None,
        help="Fallback probe preset when checkpoint has no probe_config payload.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def _probe_cfg_from_checkpoint(
    payload: dict[str, Any],
    probe_preset: str | None,
) -> ProbeConfig:
    probe_cfg_raw = payload.get("probe_config")
    if isinstance(probe_cfg_raw, dict):
        probe_type = str(probe_cfg_raw.get("probe_type", "linear"))
        if probe_type == "linear":
            return ProbeConfig(probe_type="linear")
        if probe_type == "mlp":
            hidden_dim = int(probe_cfg_raw.get("hidden_dim", 128))
            dropout = float(probe_cfg_raw.get("dropout", 0.1))
            return ProbeConfig(
                probe_type="mlp",
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        raise SystemExit(
            f"Unsupported probe_type in checkpoint probe_config: '{probe_type}'"
        )

    try:
        return get_probe_config(probe_preset)
    except ValueError as exc:
        raise SystemExit(
            "Checkpoint has no probe_config payload; pass --probe-preset explicitly."
        ) from exc


def _evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    return evaluate_binary_metrics(labels_cat, logits_cat)


def main() -> None:
    args = _parse_args()
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    manifest = read_manifest(args.data_dir)
    split_info, resolved_feature_key = resolve_split_info(
        manifest,
        split=args.split,
        feature_key=args.feature_key,
    )
    num_rows = int(split_info.get("num_rows", 0))
    if num_rows < 1:
        raise SystemExit(f"Split '{args.split}' in {args.data_dir} has no rows.")

    payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"Checkpoint payload missing state_dict: {args.checkpoint}")

    probe_cfg = _probe_cfg_from_checkpoint(payload, args.probe_preset)
    input_dim = resolve_input_dim(manifest, resolved_feature_key)
    model = build_probe_model(input_dim=input_dim, probe_cfg=probe_cfg)
    model.load_state_dict(payload["state_dict"], strict=True)

    device = choose_device(args.device)
    model = model.to(device)
    loader = make_dataloader(
        args.data_dir,
        args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        feature_key=resolved_feature_key,
    )
    metrics = _evaluate(model, loader, device)

    out = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "split": args.split,
        "feature_key": resolved_feature_key,
        "input_dim": input_dim,
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
