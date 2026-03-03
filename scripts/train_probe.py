#!/usr/bin/env python3
"""Train a probe on last-token prefill activations."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.dataloader import (
    make_dataloader,
    read_manifest,
    resolve_input_dim,
    resolve_split_info,
)
from loop_probe.configs import (
    build_probe_model,
    get_probe_config,
    probe_preset_choices,
)
from loop_probe.train_utils import (
    choose_device,
    evaluate_binary_metrics,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument(
        "--probe-preset",
        choices=probe_preset_choices(),
        default="linear",
    )
    parser.add_argument(
        "--feature-key",
        default=None,
        help=(
            "Optional feature view key from a multi-view dataset manifest. "
            "If omitted, uses manifest default or legacy single-view fields."
        ),
    )

    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", default=None)

    return parser.parse_args()


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


def _write_jsonl(path: str, row: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _checkpoint_payload(
    model,
    epoch: int,
    step: int,
    metrics: dict[str, float],
    probe_config: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "state_dict": model.state_dict(),
    }
    if probe_config is not None:
        payload["probe_config"] = probe_config
    return payload


def _cosine_lr_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    if total_steps <= 0:
        return 1.0

    clamped_step = min(max(step, 1), total_steps)
    if warmup_steps > 0 and clamped_step <= warmup_steps:
        return float(clamped_step) / float(warmup_steps)

    decay_steps = max(total_steps - warmup_steps, 1)
    decay_step = max(clamped_step - warmup_steps, 0)
    progress = min(max(decay_step / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def main() -> None:
    args = _parse_args()
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be > 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be in [0, 1).")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min-lr-ratio must be in [0, 1].")

    set_seed(args.seed)
    device = choose_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from dotenv import load_dotenv
    except Exception as exc:
        raise SystemExit(
            "python-dotenv is required. Install dependencies with `uv sync`."
        ) from exc

    load_dotenv()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not found. Put it in .env or environment.")

    import wandb

    manifest = read_manifest(args.data_dir)
    train_info, resolved_feature_key = resolve_split_info(
        manifest,
        split="train",
        feature_key=args.feature_key,
    )
    input_dim = resolve_input_dim(manifest, resolved_feature_key)
    try:
        probe_cfg = get_probe_config(args.probe_preset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    train_loader = make_dataloader(
        args.data_dir,
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        feature_key=resolved_feature_key,
    )
    test_loader = make_dataloader(
        args.data_dir,
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        feature_key=resolved_feature_key,
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise SystemExit("Training split is empty.")
    total_steps = args.epochs * steps_per_epoch

    num_pos = int(train_info.get("num_positive", 0))
    num_neg = int(train_info.get("num_negative", 0))
    if num_pos > 0 and num_neg > 0:
        pos_weight = torch.tensor(float(num_neg / num_pos), dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

    model = build_probe_model(input_dim=input_dim, probe_cfg=probe_cfg).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_steps = 0
    if args.lr_scheduler == "cosine":
        warmup_steps = int(round(args.warmup_ratio * total_steps))
        if args.warmup_ratio > 0.0 and warmup_steps < 1:
            warmup_steps = 1
        warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "data_dir": args.data_dir,
            "feature_key": resolved_feature_key,
            "input_dim": input_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "warmup_ratio": args.warmup_ratio,
            "warmup_steps": warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "pos_weight": float(pos_weight.item()),
            "probe_config": probe_cfg.to_dict(),
        },
    )

    metrics_jsonl = os.path.join(args.out_dir, "metrics.jsonl")
    best_ckpt = os.path.join(args.out_dir, "best.pt")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    best_metrics_json = os.path.join(args.out_dir, "best_metrics.json")

    with open(metrics_jsonl, "w", encoding="utf-8"):
        pass

    best_auc = float("-inf")
    best_f1 = float("-inf")
    best_eval_row: dict[str, object] | None = None
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        epoch_logits = []
        epoch_labels = []
        lr_now = args.lr

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            if args.lr_scheduler == "cosine":
                scale = _cosine_lr_factor(
                    global_step + 1,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
                lr_now = args.lr * scale
                for group in optimizer.param_groups:
                    group["lr"] = lr_now

            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(y.size(0))
            running_loss += float(loss.item()) * batch_size
            seen += batch_size
            global_step += 1

            # Store for epoch-level metrics
            epoch_logits.append(logits.detach().cpu())
            epoch_labels.append(y.detach().cpu())

            if batch_idx % args.log_every == 0:
                avg_loss = running_loss / max(seen, 1)
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr_now,
                        "epoch": epoch,
                        "step": global_step,
                    },
                    step=global_step,
                )
                print(
                    f"epoch={epoch} batch={batch_idx} step={global_step} "
                    f"train_loss={avg_loss:.6f} lr={lr_now:.6e}",
                    flush=True,
                )

        train_loss_epoch = running_loss / max(seen, 1)
        
        # Compute full epoch training metrics (only once per epoch)
        epoch_logits_cat = torch.cat(epoch_logits, dim=0)
        epoch_labels_cat = torch.cat(epoch_labels, dim=0)
        train_metrics_epoch = evaluate_binary_metrics(epoch_labels_cat, epoch_logits_cat)

        # Always log training metrics to wandb (every epoch)
        wandb.log(
            {
                "train/loss_epoch": train_loss_epoch,
                "train/accuracy_epoch": train_metrics_epoch["accuracy"],
                "train/macro_f1_epoch": train_metrics_epoch["macro_f1"],
                "train/roc_auc_epoch": train_metrics_epoch["roc_auc"],
                "train/lr": lr_now,
                "epoch": epoch,
                "step": global_step,
            },
            step=global_step,
        )

        # Evaluate on test set based on eval_every
        if epoch % args.eval_every == 0:
            eval_metrics = _evaluate(model, test_loader, device)
            
            # Log to metrics.jsonl
            log_row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss_epoch,
                "train_accuracy": train_metrics_epoch["accuracy"],
                "train_macro_f1": train_metrics_epoch["macro_f1"],
                "train_roc_auc": train_metrics_epoch["roc_auc"],
                "seed": args.seed,
                "lr": lr_now,
                **eval_metrics,
            }
            _write_jsonl(metrics_jsonl, log_row)

            # Log eval metrics to wandb
            wandb.log(
                {
                    "eval/accuracy": eval_metrics["accuracy"],
                    "eval/macro_f1": eval_metrics["macro_f1"],
                    "eval/roc_auc": eval_metrics["roc_auc"],
                    "epoch": epoch,
                    "step": global_step,
                },
                step=global_step,
            )

            # Update best checkpoint
            auc = eval_metrics["roc_auc"]
            auc_rank = auc if not math.isnan(auc) else float("-inf")
            is_better = (auc_rank > best_auc) or (
                auc_rank == best_auc and eval_metrics["macro_f1"] > best_f1
            )
            if is_better:
                best_auc = auc_rank
                best_f1 = eval_metrics["macro_f1"]
                best_eval_row = dict(log_row)
                torch.save(
                    _checkpoint_payload(
                        model,
                        epoch,
                        global_step,
                        eval_metrics,
                        probe_config=probe_cfg.to_dict(),
                    ),
                    best_ckpt,
                )

            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"train_loss={train_loss_epoch:.6f}",
                        f"train_acc={train_metrics_epoch['accuracy']:.4f}",
                        f"eval_acc={eval_metrics['accuracy']:.4f}",
                        f"eval_f1={eval_metrics['macro_f1']:.4f}",
                        f"eval_auc={eval_metrics['roc_auc']:.4f}",
                    ]
                ),
                flush=True,
            )
        else:
            # Print training metrics only
            print(
                " ".join(
                    [
                        f"epoch={epoch}",
                        f"train_loss={train_loss_epoch:.6f}",
                        f"train_acc={train_metrics_epoch['accuracy']:.4f}",
                        f"train_f1={train_metrics_epoch['macro_f1']:.4f}",
                        f"train_auc={train_metrics_epoch['roc_auc']:.4f}",
                    ]
                ),
                flush=True,
            )

        torch.save(
            _checkpoint_payload(
                model,
                epoch,
                global_step,
                {
                    "train_loss": train_loss_epoch,
                    "train_accuracy": train_metrics_epoch["accuracy"],
                    "train_macro_f1": train_metrics_epoch["macro_f1"],
                    "train_roc_auc": train_metrics_epoch["roc_auc"],
                },
                probe_config=probe_cfg.to_dict(),
            ),
            last_ckpt,
        )

    run.finish()
    if best_eval_row is not None:
        best_payload = {
            "selection_rule": "max(roc_auc), tie_break=max(macro_f1)",
            **best_eval_row,
        }
        with open(best_metrics_json, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, indent=2, sort_keys=True)
            f.write("\n")
    else:
        print(
            "No eval checkpoints were logged (check --eval-every); best_metrics.json not written.",
            flush=True,
        )
    print(f"Saved checkpoints to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
