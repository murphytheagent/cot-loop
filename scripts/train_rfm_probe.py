#!/usr/bin/env python3
"""Train an RFF-based RFM-lite probe on prefill activations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.dataloader import ActivationDataset
from loop_probe.train_utils import evaluate_binary_metrics, set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--feature-key",
        default=None,
        help=(
            "Optional feature view key from a multi-view dataset manifest. "
            "If omitted, uses manifest default or legacy single-view fields."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-features", type=int, default=2048)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument(
        "--rfm-steps",
        type=int,
        default=1,
        help="Number of recursive gradient-feature augmentation steps.",
    )
    parser.add_argument(
        "--grad-weight",
        type=float,
        default=1.0,
        help="Scale factor for recursive gradient feature concatenation.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "pr_auc", "macro_f1", "positive_f1", "accuracy"),
        default="roc_auc",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=("roc_auc", "pr_auc", "macro_f1", "positive_f1", "accuracy"),
        default="macro_f1",
    )
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def _to_numpy(ds: ActivationDataset) -> tuple[np.ndarray, np.ndarray]:
    x = ds.x.detach().cpu().numpy().astype(np.float32, copy=False)
    y = ds.y.detach().cpu().numpy().astype(np.int64, copy=False)
    return x, y


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def _draw_rff(
    *,
    input_dim: int,
    random_features: int,
    bandwidth: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if bandwidth <= 0:
        raise SystemExit("--bandwidth must be > 0.")
    scale = 1.0 / bandwidth
    w = rng.normal(
        loc=0.0,
        scale=scale,
        size=(random_features, input_dim),
    ).astype(np.float32)
    b = rng.uniform(
        low=0.0,
        high=2.0 * np.pi,
        size=(random_features,),
    ).astype(np.float32)
    return w, b


def _rff_features(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    proj = x @ w.T
    proj += b[None, :]
    scale = np.sqrt(2.0 / float(w.shape[0]))
    return (scale * np.cos(proj)).astype(np.float32)


def _fit_ridge(phi: np.ndarray, labels01: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        raise SystemExit("--ridge must be > 0.")
    y = (2.0 * labels01.astype(np.float32)) - 1.0
    m = int(phi.shape[1])
    lhs = phi.T @ phi
    lhs += ridge * np.eye(m, dtype=np.float32)
    rhs = phi.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return beta.astype(np.float32)


def _rff_logits(phi: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return (phi @ beta).astype(np.float32)


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
    grad = scale * (weighted @ w)
    return grad.astype(np.float32)


def _rank_metric(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(out):
        return float("-inf")
    return out


def _write_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _metrics_from_logits(labels01: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    y_t = torch.from_numpy(labels01.astype(np.float32))
    logits_t = torch.from_numpy(logits.astype(np.float32))
    return evaluate_binary_metrics(y_t, logits_t)


def _maybe_init_wandb(args: argparse.Namespace):
    if not args.wandb_project:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        return None
    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "data_dir": args.data_dir,
            "feature_key": args.feature_key,
            "seed": args.seed,
            "random_features": args.random_features,
            "bandwidth": args.bandwidth,
            "ridge": args.ridge,
            "rfm_steps": args.rfm_steps,
            "grad_weight": args.grad_weight,
            "selection_metric": args.selection_metric,
            "tie_breaker": args.tie_breaker,
        },
    )


def main() -> None:
    args = _parse_args()
    if args.random_features < 1:
        raise SystemExit("--random-features must be >= 1.")
    if args.rfm_steps < 0:
        raise SystemExit("--rfm-steps must be >= 0.")
    if args.grad_weight < 0:
        raise SystemExit("--grad-weight must be >= 0.")

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = ActivationDataset(
        data_dir=args.data_dir,
        split="train",
        feature_key=args.feature_key,
    )
    test_ds = ActivationDataset(
        data_dir=args.data_dir,
        split="test",
        feature_key=args.feature_key,
    )
    x_train, y_train = _to_numpy(train_ds)
    x_test, y_test = _to_numpy(test_ds)

    metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
    best_metrics_json = os.path.join(args.out_dir, "best_metrics.json")
    best_model_pt = os.path.join(args.out_dir, "best_model.pt")
    model_summary_json = os.path.join(args.out_dir, "model_summary.json")
    with open(metrics_jsonl, "w", encoding="utf-8"):
        pass

    run = _maybe_init_wandb(args)

    x_train_cur = x_train.astype(np.float32, copy=True)
    x_test_cur = x_test.astype(np.float32, copy=True)
    best_key = (float("-inf"), float("-inf"))
    best_row: dict[str, Any] | None = None
    best_state: dict[str, Any] | None = None
    pipeline_states: list[dict[str, np.ndarray]] = []

    for step in range(args.rfm_steps + 1):
        mean, std = _standardize_fit(x_train_cur)
        x_train_std = _standardize_apply(x_train_cur, mean, std)
        x_test_std = _standardize_apply(x_test_cur, mean, std)

        w_rff, b_rff = _draw_rff(
            input_dim=int(x_train_std.shape[1]),
            random_features=args.random_features,
            bandwidth=args.bandwidth,
            rng=rng,
        )
        phi_train = _rff_features(x_train_std, w_rff, b_rff)
        phi_test = _rff_features(x_test_std, w_rff, b_rff)
        beta = _fit_ridge(phi_train, y_train, ridge=args.ridge)

        logits_train = _rff_logits(phi_train, beta)
        logits_test = _rff_logits(phi_test, beta)
        train_metrics = _metrics_from_logits(y_train, logits_train)
        eval_metrics = _metrics_from_logits(y_test, logits_test)

        row = {
            "epoch": step + 1,
            "step": step + 1,
            "seed": args.seed,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_positive_precision": train_metrics["positive_precision"],
            "train_positive_recall": train_metrics["positive_recall"],
            "train_positive_f1": train_metrics["positive_f1"],
            "train_prevalence": train_metrics["prevalence"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            **eval_metrics,
            "rfm_step": step,
            "random_features": args.random_features,
            "bandwidth": args.bandwidth,
            "ridge": args.ridge,
            "grad_weight": args.grad_weight,
            "input_dim": int(x_train_std.shape[1]),
        }
        _write_jsonl(metrics_jsonl, row)

        pipeline_states.append(
            {
                "mean": mean.copy(),
                "std": std.copy(),
                "w_rff": w_rff.copy(),
                "b_rff": b_rff.copy(),
                "beta": beta.copy(),
            }
        )

        if run is not None:
            run.log(
                {
                    "epoch": step + 1,
                    "rfm_step": step,
                    "eval/accuracy": eval_metrics["accuracy"],
                    "eval/macro_f1": eval_metrics["macro_f1"],
                    "eval/positive_precision": eval_metrics["positive_precision"],
                    "eval/positive_recall": eval_metrics["positive_recall"],
                    "eval/positive_f1": eval_metrics["positive_f1"],
                    "eval/prevalence": eval_metrics["prevalence"],
                    "eval/roc_auc": eval_metrics["roc_auc"],
                    "eval/pr_auc": eval_metrics["pr_auc"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/macro_f1": train_metrics["macro_f1"],
                    "train/roc_auc": train_metrics["roc_auc"],
                    "train/pr_auc": train_metrics["pr_auc"],
                },
                step=step + 1,
            )

        rank = (
            _rank_metric(row.get(args.selection_metric)),
            _rank_metric(row.get(args.tie_breaker)),
        )
        if best_row is None or rank > best_key:
            best_key = rank
            best_row = dict(row)
            best_state = {
                "step": step,
                "input_dim": int(x_train_std.shape[1]),
                "pipeline": [
                    {
                        "mean": entry["mean"].copy(),
                        "std": entry["std"].copy(),
                        "w_rff": entry["w_rff"].copy(),
                        "b_rff": entry["b_rff"].copy(),
                        "beta": entry["beta"].copy(),
                    }
                    for entry in pipeline_states
                ],
            }

        print(
            " ".join(
                [
                    f"step={step}",
                    f"eval_acc={eval_metrics['accuracy']:.4f}",
                    f"eval_pos_f1={eval_metrics['positive_f1']:.4f}",
                    f"eval_macro_f1={eval_metrics['macro_f1']:.4f}",
                    f"eval_auc={eval_metrics['roc_auc']:.4f}",
                    f"eval_pr_auc={eval_metrics['pr_auc']:.4f}",
                ]
            ),
            flush=True,
        )

        if step < args.rfm_steps and args.grad_weight > 0.0:
            grad_train = _rff_gradients(x_train_std, w_rff, b_rff, beta)
            grad_test = _rff_gradients(x_test_std, w_rff, b_rff, beta)
            grad_scale = float(np.sqrt(args.grad_weight))
            x_train_cur = np.concatenate(
                [x_train_std, grad_scale * grad_train.astype(np.float32)],
                axis=1,
            ).astype(np.float32)
            x_test_cur = np.concatenate(
                [x_test_std, grad_scale * grad_test.astype(np.float32)],
                axis=1,
            ).astype(np.float32)
        elif step < args.rfm_steps:
            x_train_cur = x_train_std
            x_test_cur = x_test_std

    if run is not None:
        run.finish()

    if best_row is None or best_state is None:
        raise SystemExit("No RFM metrics were produced.")

    payload = {
        "selection_rule": f"max({args.selection_metric}), tie_break=max({args.tie_breaker})",
        **best_row,
    }
    with open(best_metrics_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    torch.save(
        {
            "algorithm": "rfm_rff_lite",
            "feature_key": train_ds.feature_key,
            "grad_weight": float(args.grad_weight),
            "pipeline": [
                {
                    "mean": torch.from_numpy(entry["mean"].astype(np.float32)),
                    "std": torch.from_numpy(entry["std"].astype(np.float32)),
                    "w_rff": torch.from_numpy(entry["w_rff"].astype(np.float32)),
                    "b_rff": torch.from_numpy(entry["b_rff"].astype(np.float32)),
                    "beta": torch.from_numpy(entry["beta"].astype(np.float32)),
                }
                for entry in best_state["pipeline"]
            ],
        },
        best_model_pt,
    )
    with open(model_summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "algorithm": "rfm_rff_lite",
                "seed": args.seed,
                "feature_key": train_ds.feature_key,
                "random_features": args.random_features,
                "bandwidth": args.bandwidth,
                "ridge": args.ridge,
                "rfm_steps": args.rfm_steps,
                "grad_weight": args.grad_weight,
                "best_step": int(best_state["step"]),
                "input_dim": int(best_state["input_dim"]),
                "model_checkpoint": "best_model.pt",
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print(f"Saved RFM probe outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
