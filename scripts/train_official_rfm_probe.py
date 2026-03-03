#!/usr/bin/env python3
"""Train/evaluate the official Recursive Feature Machines (RFM) implementation."""

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

from loop_probe.dataloader import ActivationDataset
from loop_probe.train_utils import evaluate_binary_metrics, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="ID dataset directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--feature-key", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--method", choices=("eigenpro", "lstsq"), default="eigenpro")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--diag", action="store_true", default=False)
    parser.add_argument("--mem-gb", type=float, default=None)
    parser.add_argument(
        "--ood-data-dir",
        default="",
        help="Optional OOD dataset directory for extra test evaluation.",
    )
    return parser.parse_args()


def _load_split(data_dir: str, split: str, feature_key: str | None):
    ds = ActivationDataset(data_dir=data_dir, split=split, feature_key=feature_key)
    x = ds.x.detach().cpu().float()
    y = ds.y.detach().cpu().float().view(-1, 1)
    return x, y


def _to_metrics(y: torch.Tensor, scores: torch.Tensor) -> dict[str, float]:
    return evaluate_binary_metrics(y.view(-1), scores.view(-1))


def _write_json(path: str, payload: dict[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from rfm import LaplaceRFM
    except Exception as exc:
        raise SystemExit(
            "Official RFM package is required. Install via "
            "`pip install git+https://github.com/aradha/recursive_feature_machines.git@pip_install` "
            "and ensure `torchmetrics` + `scikit-learn` are installed."
        ) from exc

    x_train, y_train = _load_split(args.data_dir, "train", args.feature_key)
    x_test, y_test = _load_split(args.data_dir, "test", args.feature_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mem_gb is not None:
        mem_gb = float(args.mem_gb)
    elif device.type == "cuda":
        mem_gb = max(float(torch.cuda.get_device_properties(device).total_memory / (1024**3) - 1.0), 1.0)
    else:
        mem_gb = 8.0

    model = LaplaceRFM(
        bandwidth=float(args.bandwidth),
        device=device,
        mem_gb=mem_gb,
        diag=bool(args.diag),
        reg=float(args.reg),
        iters=int(args.iters),
        bandwidth_mode="constant",
    )
    model.fit(
        (x_train, y_train),
        (x_test, y_test),
        iters=int(args.iters),
        method=args.method,
        classification=True,
        epochs=int(args.epochs),
        verbose=True,
    )

    with torch.no_grad():
        train_scores = model.predict(x_train.to(device)).float().detach().cpu()
        test_scores = model.predict(x_test.to(device)).float().detach().cpu()

    train_metrics = _to_metrics(y_train, train_scores)
    test_metrics = _to_metrics(y_test, test_scores)

    metrics_row = {
        "epoch": 1,
        "step": 1,
        "seed": args.seed,
        "selection_rule": "official_rfm_single_fit",
        "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "train_positive_precision": train_metrics["positive_precision"],
        "train_positive_recall": train_metrics["positive_recall"],
        "train_positive_f1": train_metrics["positive_f1"],
        "train_prevalence": train_metrics["prevalence"],
        "train_roc_auc": train_metrics["roc_auc"],
        "train_pr_auc": train_metrics["pr_auc"],
        **test_metrics,
    }
    _write_json(os.path.join(args.out_dir, "best_metrics.json"), metrics_row)
    _write_json(
        os.path.join(args.out_dir, "model_config.json"),
        {
            "algorithm": "official_rfm",
            "source_repo": "https://github.com/aradha/recursive_feature_machines/tree/pip_install",
            "seed": args.seed,
            "feature_key": args.feature_key,
            "bandwidth": args.bandwidth,
            "reg": args.reg,
            "iters": args.iters,
            "method": args.method,
            "epochs": args.epochs,
            "diag": args.diag,
            "mem_gb": mem_gb,
        },
    )

    with open(os.path.join(args.out_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps(metrics_row, sort_keys=True) + "\n")

    if args.ood_data_dir:
        x_ood, y_ood = _load_split(args.ood_data_dir, "test", args.feature_key)
        with torch.no_grad():
            ood_scores = model.predict(x_ood.to(device)).float().detach().cpu()
        ood_metrics = _to_metrics(y_ood, ood_scores)
        _write_json(
            os.path.join(args.out_dir, "ood_metrics.json"),
            {
                "seed": args.seed,
                "feature_key": args.feature_key,
                "data_dir": args.ood_data_dir,
                **ood_metrics,
            },
        )

    print(f"Saved official RFM outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
