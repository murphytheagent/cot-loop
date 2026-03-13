#!/usr/bin/env python3
"""Train a hidden-state probe as an additive correction over prompt metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from loop_probe.configs import build_probe_model, get_probe_config, probe_preset_choices
from loop_probe.dataloader import ActivationDataset, read_manifest, resolve_feature_key
from loop_probe.train_utils import choose_device, evaluate_binary_metrics, set_seed

SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "prevalence",
    "roc_auc",
    "pr_auc",
    "bin_roc_auc",
    "bin_pr_auc",
)

SELECTION_METRIC_CHOICES = (
    "accuracy",
    "macro_f1",
    "positive_f1",
    "roc_auc",
    "pr_auc",
    "bin_roc_auc",
    "bin_pr_auc",
)


@dataclass(frozen=True)
class PromptMetadata:
    sample_id: int
    source: str
    prompt: str
    char_length: int
    token_length: int
    log_token_length: float
    newline_count: int
    digit_count: int
    dollar_count: int


class ResidualDataset(Dataset):
    def __init__(
        self,
        hidden_x: torch.Tensor,
        meta_logits: torch.Tensor,
        labels: torch.Tensor,
        group_ids: torch.Tensor | None = None,
    ) -> None:
        if hidden_x.ndim != 2:
            raise SystemExit(f"Expected 2D hidden features, got {tuple(hidden_x.shape)}")
        if meta_logits.ndim != 1:
            raise SystemExit(f"Expected 1D metadata logits, got {tuple(meta_logits.shape)}")
        if labels.ndim != 1:
            raise SystemExit(f"Expected 1D labels, got {tuple(labels.shape)}")
        num_rows = int(hidden_x.size(0))
        if num_rows != int(meta_logits.size(0)) or num_rows != int(labels.size(0)):
            raise SystemExit("ResidualDataset length mismatch.")
        if group_ids is not None:
            if group_ids.ndim != 1:
                raise SystemExit(f"Expected 1D group ids, got {tuple(group_ids.shape)}")
            if num_rows != int(group_ids.size(0)):
                raise SystemExit("ResidualDataset group_ids length mismatch.")
        self.hidden_x = hidden_x
        self.meta_logits = meta_logits
        self.labels = labels
        self.group_ids = group_ids

    def __len__(self) -> int:
        return int(self.labels.size(0))

    def __getitem__(self, idx: int):
        if self.group_ids is None:
            return self.hidden_x[idx], self.meta_logits[idx], self.labels[idx]
        return self.hidden_x[idx], self.meta_logits[idx], self.labels[idx], self.group_ids[idx]


class MetadataModel:
    def __init__(self, feature_set: str) -> None:
        self.feature_set = feature_set
        self.source_vocab: list[str] = []
        self.source_reference: str | None = None
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=0,
        )

    def fit(self, rows: list[PromptMetadata], labels: np.ndarray) -> None:
        self.source_vocab = sorted({row.source for row in rows})
        self.source_reference = self.source_vocab[0] if self.source_vocab else None
        x = self._design_matrix(rows, fit_scaler=True)
        self.model.fit(x, labels.astype(int))

    def decision_function(self, rows: list[PromptMetadata]) -> np.ndarray:
        x = self._design_matrix(rows, fit_scaler=False)
        return self.model.decision_function(x)

    def to_json(self) -> dict[str, object]:
        return {
            "feature_set": self.feature_set,
            "source_vocab": self.source_vocab,
            "source_reference": self.source_reference,
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }

    def _design_matrix(
        self,
        rows: list[PromptMetadata],
        *,
        fit_scaler: bool,
    ) -> np.ndarray:
        if not rows:
            raise SystemExit("Cannot build metadata design matrix for empty rows.")
        numeric = np.asarray(
            [
                [
                    float(row.char_length),
                    float(row.token_length),
                    float(row.log_token_length),
                    float(row.newline_count),
                    float(row.digit_count),
                    float(row.dollar_count),
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
        if fit_scaler:
            numeric_scaled = self.scaler.fit_transform(numeric)
        else:
            numeric_scaled = self.scaler.transform(numeric)

        source_features = self._source_one_hot(rows)

        if self.feature_set == "legacy":
            selected = numeric_scaled[:, [0, 3, 4, 5]]
            return np.concatenate([source_features, selected], axis=1)

        base_numeric = numeric_scaled[:, [1, 2, 3, 4, 5]]
        token_interactions = self._source_interactions(
            source_features=source_features,
            values=numeric_scaled[:, [1, 2]],
        )
        return np.concatenate([source_features, base_numeric, token_interactions], axis=1)

    def _source_one_hot(self, rows: list[PromptMetadata]) -> np.ndarray:
        if not self.source_vocab:
            raise SystemExit("MetadataModel source vocabulary is empty.")
        if len(self.source_vocab) <= 1:
            return np.zeros((len(rows), 0), dtype=np.float64)

        columns: list[np.ndarray] = []
        for source in self.source_vocab[1:]:
            columns.append(
                np.asarray([1.0 if row.source == source else 0.0 for row in rows], dtype=np.float64)
            )
        return np.stack(columns, axis=1)

    @staticmethod
    def _source_interactions(
        *,
        source_features: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        if source_features.size == 0:
            return np.zeros((values.shape[0], 0), dtype=np.float64)
        interactions = []
        for idx in range(source_features.shape[1]):
            source_col = source_features[:, idx : idx + 1]
            interactions.append(source_col * values)
        return np.concatenate(interactions, axis=1)


class LowRankMLPProbe(nn.Module):
    """Factor the first wide map through a fixed low-rank bottleneck."""

    def __init__(
        self,
        *,
        input_dim: int,
        low_rank_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if low_rank_dim < 1:
            raise ValueError("low_rank_dim must be >= 1.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        layers: list[nn.Module] = [
            nn.Linear(input_dim, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-dir", required=True)
    parser.add_argument("--matched-eval-data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-pool-jsonl", required=True)
    parser.add_argument("--eval-pool-jsonl", required=True)
    parser.add_argument("--prompt-field", default="problem")
    parser.add_argument("--feature-key", default=None)
    parser.add_argument("--extra-feature-key", action="append", default=[])
    parser.add_argument("--tokenizer-model-id", required=True)
    parser.add_argument(
        "--metadata-feature-set",
        choices=("legacy", "strong"),
        default="strong",
    )
    parser.add_argument("--num-length-bins", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.03)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])

    parser.add_argument(
        "--probe-preset",
        choices=probe_preset_choices(),
        default="mlp",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--mlp-dropout", type=float, default=0.03)

    parser.add_argument(
        "--selection-split",
        choices=("val", "natural", "matched"),
        default="natural",
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRIC_CHOICES,
        default="roc_auc",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=SELECTION_METRIC_CHOICES,
        default="macro_f1",
    )
    parser.add_argument("--low-rank-dim", type=int, default=0)
    parser.add_argument("--holdout-ratio", type=float, default=0.0)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument(
        "--train-objective",
        choices=("bce", "group_bce", "group_bce_rank"),
        default="bce",
    )
    parser.add_argument("--rank-lambda", type=float, default=0.25)
    parser.add_argument("--rank-tau", type=float, default=1.0)
    parser.add_argument("--rank-warmup-epochs", type=int, default=1)
    parser.add_argument("--rank-min-pos", type=int, default=1)
    parser.add_argument("--rank-min-neg", type=int, default=1)
    parser.add_argument("--rank-max-pairs-per-group", type=int, default=0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args.lr <= 0.0:
        raise SystemExit("--lr must be > 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be in [0, 1).")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min-lr-ratio must be in [0, 1].")
    if args.num_length_bins < 1:
        raise SystemExit("--num-length-bins must be >= 1.")
    if args.low_rank_dim < 0:
        raise SystemExit("--low-rank-dim must be >= 0.")
    if not 0.0 <= args.holdout_ratio < 1.0:
        raise SystemExit("--holdout-ratio must be in [0, 1).")
    if args.selection_split == "val" and args.holdout_ratio <= 0.0:
        raise SystemExit("--selection-split=val requires --holdout-ratio > 0.")
    if args.rank_lambda < 0.0:
        raise SystemExit("--rank-lambda must be >= 0.")
    if args.rank_tau <= 0.0:
        raise SystemExit("--rank-tau must be > 0.")
    if args.rank_warmup_epochs < 1:
        raise SystemExit("--rank-warmup-epochs must be >= 1.")
    if args.rank_min_pos < 1:
        raise SystemExit("--rank-min-pos must be >= 1.")
    if args.rank_min_neg < 1:
        raise SystemExit("--rank-min-neg must be >= 1.")
    if args.rank_max_pairs_per_group < 0:
        raise SystemExit("--rank-max-pairs-per-group must be >= 0.")
    if args.train_objective != "group_bce_rank" and args.rank_lambda > 0.0:
        print(
            f"Ignoring --rank-lambda={args.rank_lambda} because --train-objective={args.train_objective}.",
            file=sys.stderr,
            flush=True,
        )


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
                raise SystemExit(f"Expected object rows in {path}:{line_num}")
            rows.append(row)
    return rows


def _metadata_rows_from_ids(
    *,
    pool_rows: list[dict[str, object]],
    sample_ids: torch.Tensor,
    prompt_field: str,
    tokenizer,
) -> list[PromptMetadata]:
    sample_id_list = [int(v) for v in sample_ids.tolist()]
    prompts = []
    subset_rows = []
    for sample_id in sample_id_list:
        if sample_id < 0 or sample_id >= len(pool_rows):
            raise SystemExit(
                f"sample_id={sample_id} out of range for pool size {len(pool_rows)}"
            )
        row = pool_rows[sample_id]
        prompt = row.get(prompt_field)
        if not isinstance(prompt, str):
            raise SystemExit(
                f"Prompt field '{prompt_field}' missing/invalid for sample_id={sample_id}"
            )
        source = row.get("source")
        if not isinstance(source, str):
            raise SystemExit(f"Missing string 'source' field for sample_id={sample_id}")
        prompts.append(prompt)
        subset_rows.append((sample_id, source, prompt))

    tokenized = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    input_ids = tokenized["input_ids"]
    out = []
    for (sample_id, source, prompt), token_ids in zip(subset_rows, input_ids, strict=True):
        out.append(
            PromptMetadata(
                sample_id=sample_id,
                source=source,
                prompt=prompt,
                char_length=len(prompt),
                token_length=len(token_ids),
                log_token_length=math.log1p(len(token_ids)),
                newline_count=prompt.count("\n"),
                digit_count=sum(ch.isdigit() for ch in prompt),
                dollar_count=prompt.count("$"),
            )
        )
    return out


def _evaluate_logits(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    return evaluate_binary_metrics(labels, logits)


def _subset_meta_rows(rows: list[PromptMetadata], indices: torch.Tensor) -> list[PromptMetadata]:
    return [rows[int(idx)] for idx in indices.tolist()]


def _build_holdout_split(
    *,
    rows: list[PromptMetadata],
    labels: torch.Tensor,
    ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    total = int(labels.size(0))
    all_idx = torch.arange(total, dtype=torch.long)
    if ratio <= 0.0:
        return all_idx, None

    groups: dict[tuple[str, int], list[int]] = {}
    for idx, (row, label) in enumerate(zip(rows, labels.tolist(), strict=True)):
        key = (row.source, int(label))
        groups.setdefault(key, []).append(idx)

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    holdout_idx: list[int] = []
    for indices in groups.values():
        group_arr = np.asarray(indices, dtype=np.int64)
        rng.shuffle(group_arr)
        if group_arr.size <= 1:
            train_idx.extend(int(v) for v in group_arr.tolist())
            continue
        holdout_count = int(round(group_arr.size * ratio))
        if group_arr.size >= 4 and holdout_count < 1:
            holdout_count = 1
        holdout_count = min(holdout_count, group_arr.size - 1)
        if holdout_count > 0:
            holdout_idx.extend(int(v) for v in group_arr[:holdout_count].tolist())
            train_idx.extend(int(v) for v in group_arr[holdout_count:].tolist())
        else:
            train_idx.extend(int(v) for v in group_arr.tolist())

    if not holdout_idx:
        raise SystemExit("Requested --holdout-ratio > 0 but no holdout rows were created.")
    return (
        torch.tensor(sorted(train_idx), dtype=torch.long),
        torch.tensor(sorted(holdout_idx), dtype=torch.long),
    )


def _assign_source_length_groups(
    *,
    rows: list[PromptMetadata],
    num_length_bins: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    group_ids = np.full(len(rows), -1, dtype=np.int64)
    group_summaries: list[dict[str, object]] = []
    next_group_id = 0
    sources = sorted({row.source for row in rows})
    for source in sources:
        indices = np.asarray(
            [idx for idx, row in enumerate(rows) if row.source == source],
            dtype=np.int64,
        )
        if indices.size == 0:
            continue
        source_lengths = np.asarray([rows[int(idx)].token_length for idx in indices], dtype=np.float64)
        quantiles = np.linspace(0.0, 1.0, num_length_bins + 1)
        edges = np.quantile(source_lengths, quantiles)
        if np.unique(edges).size <= 1:
            bin_assignments = np.zeros(indices.size, dtype=np.int64)
            bin_ids = [0]
        else:
            bin_assignments = np.searchsorted(edges[1:-1], source_lengths, side="right")
            bin_ids = list(range(num_length_bins))
        for bin_id in bin_ids:
            subset_idx = indices[bin_assignments == bin_id]
            if subset_idx.size == 0:
                continue
            group_ids[subset_idx] = next_group_id
            group_summaries.append(
                {
                    "group_id": next_group_id,
                    "group_label": f"{source}::len_bin_{bin_id}",
                    "source": source,
                    "length_bin": int(bin_id),
                    "num_rows": int(subset_idx.size),
                    "token_length_min": int(source_lengths[bin_assignments == bin_id].min()),
                    "token_length_max": int(source_lengths[bin_assignments == bin_id].max()),
                }
            )
            next_group_id += 1
    if np.any(group_ids < 0):
        raise SystemExit("Some training rows were not assigned to a source×length group.")
    return torch.tensor(group_ids, dtype=torch.long), group_summaries


def _attach_group_label_stats(
    *,
    group_summaries: list[dict[str, object]],
    group_ids: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict[str, object]]:
    labels_np = labels.detach().cpu().numpy().astype(int)
    group_ids_np = group_ids.detach().cpu().numpy().astype(int)
    out: list[dict[str, object]] = []
    for summary in group_summaries:
        group_id = int(summary["group_id"])
        mask = group_ids_np == group_id
        pos = int(labels_np[mask].sum())
        total = int(mask.sum())
        row = dict(summary)
        row["num_pos"] = pos
        row["num_neg"] = total - pos
        out.append(row)
    return out


def _bin_metrics(
    *,
    rows: list[PromptMetadata],
    labels: torch.Tensor,
    logits: torch.Tensor,
    num_length_bins: int,
) -> dict[str, float]:
    label_np = labels.detach().cpu().numpy().astype(int)
    logits_np = logits.detach().cpu().numpy()
    lengths = np.asarray([row.token_length for row in rows], dtype=np.float64)
    sources = [row.source for row in rows]

    roc_vals = []
    pr_vals = []
    strata = 0
    for source in sorted(set(sources)):
        indices = np.asarray([idx for idx, value in enumerate(sources) if value == source], dtype=np.int64)
        if indices.size == 0:
            continue
        source_lengths = lengths[indices]
        quantiles = np.linspace(0.0, 1.0, num_length_bins + 1)
        edges = np.quantile(source_lengths, quantiles)
        if np.unique(edges).size <= 1:
            bin_assignments = np.zeros(indices.size, dtype=np.int64)
            bins = [0]
        else:
            bin_assignments = np.searchsorted(edges[1:-1], source_lengths, side="right")
            bins = range(num_length_bins)
        for bin_id in bins:
            mask = bin_assignments == bin_id
            subset_idx = indices[mask]
            if subset_idx.size < 2:
                continue
            y_subset = label_np[subset_idx]
            if np.unique(y_subset).size < 2:
                continue
            probs_subset = 1.0 / (1.0 + np.exp(-logits_np[subset_idx]))
            roc_vals.append(float(roc_auc_score(y_subset, probs_subset)))
            pr_vals.append(float(average_precision_score(y_subset, probs_subset)))
            strata += 1

    if not roc_vals:
        return {
            "bin_roc_auc": float("nan"),
            "bin_pr_auc": float("nan"),
            "bin_num_strata": 0.0,
        }
    return {
        "bin_roc_auc": float(statistics.fmean(roc_vals)),
        "bin_pr_auc": float(statistics.fmean(pr_vals)),
        "bin_num_strata": float(strata),
    }


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


def _build_loader(
    hidden_x: torch.Tensor,
    meta_logits: np.ndarray,
    labels: torch.Tensor,
    group_ids: torch.Tensor | None = None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    ds = ResidualDataset(
        hidden_x=hidden_x.to(dtype=torch.float32),
        meta_logits=torch.tensor(meta_logits, dtype=torch.float32),
        labels=labels.to(dtype=torch.float32),
        group_ids=None if group_ids is None else group_ids.to(dtype=torch.long),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _evaluate_loader(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 4:
                hidden_x, meta_logits, labels, _group_ids = batch
            else:
                hidden_x, meta_logits, labels = batch
            hidden_x = hidden_x.to(device)
            meta_logits = meta_logits.to(device)
            logits = model(hidden_x) + meta_logits
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_labels, dim=0), torch.cat(all_logits, dim=0)


def _equal_group_bce_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    per_example = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    group_losses = []
    unique_groups = torch.unique(group_ids, sorted=True)
    for group_id in unique_groups.tolist():
        mask = group_ids == int(group_id)
        if not torch.any(mask):
            continue
        group_losses.append(per_example[mask].mean())
    if not group_losses:
        return per_example.mean(), 0
    return torch.stack(group_losses).mean(), len(group_losses)


def _within_group_pairwise_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    tau: float,
    min_pos: int,
    min_neg: int,
    max_pairs_per_group: int,
) -> tuple[torch.Tensor, int, int]:
    if tau <= 0.0:
        raise ValueError("tau must be positive.")
    group_losses = []
    eligible_groups = 0
    total_pairs = 0
    unique_groups = torch.unique(group_ids, sorted=True)
    for group_id in unique_groups.tolist():
        mask = group_ids == int(group_id)
        if int(mask.sum().item()) < (min_pos + min_neg):
            continue
        group_logits = logits[mask]
        group_labels = labels[mask] > 0.5
        pos_logits = group_logits[group_labels]
        neg_logits = group_logits[~group_labels]
        if int(pos_logits.numel()) < min_pos or int(neg_logits.numel()) < min_neg:
            continue
        diffs = (pos_logits[:, None] - neg_logits[None, :]).reshape(-1) / tau
        if max_pairs_per_group > 0 and int(diffs.numel()) > max_pairs_per_group:
            pair_idx = torch.randperm(int(diffs.numel()), device=diffs.device)[:max_pairs_per_group]
            diffs = diffs.index_select(0, pair_idx)
        group_losses.append(F.softplus(-diffs).mean())
        eligible_groups += 1
        total_pairs += int(diffs.numel())
    if not group_losses:
        return logits.new_tensor(0.0), 0, 0
    return torch.stack(group_losses).mean(), eligible_groups, total_pairs


def _rank_weight_for_epoch(*, epoch: int, max_weight: float, warmup_epochs: int) -> float:
    if max_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 1:
        return max_weight
    progress = min(max((epoch - 1) / float(warmup_epochs - 1), 0.0), 1.0)
    return max_weight * progress


def _best_key(
    metrics: dict[str, float],
    *,
    split_name: str,
    selection_metric: str,
    tie_breaker: str,
) -> tuple[float, float]:
    primary = float(metrics[f"{split_name}_{selection_metric}"])
    secondary = float(metrics[f"{split_name}_{tie_breaker}"])
    if math.isnan(primary):
        primary = float("-inf")
    if math.isnan(secondary):
        secondary = float("-inf")
    return primary, secondary


def _write_json(path: str, payload: dict[str, object]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_jsonl(path: str, row: dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _build_hidden_probe(args: argparse.Namespace, *, input_dim: int) -> tuple[nn.Module, dict[str, object]]:
    probe_cfg = get_probe_config(
        args.probe_preset,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
        depth=args.mlp_depth,
    )
    if args.low_rank_dim > 0:
        if probe_cfg.probe_type != "mlp":
            raise SystemExit("--low-rank-dim currently requires --probe-preset=mlp.")
        model = LowRankMLPProbe(
            input_dim=input_dim,
            low_rank_dim=args.low_rank_dim,
            hidden_dim=probe_cfg.hidden_dim,
            dropout=probe_cfg.dropout,
            depth=probe_cfg.depth,
        )
        model_cfg = probe_cfg.to_dict()
        model_cfg["low_rank_dim"] = args.low_rank_dim
        model_cfg["probe_type"] = "low_rank_mlp"
        return model, model_cfg
    return build_probe_model(input_dim=input_dim, probe_cfg=probe_cfg), probe_cfg.to_dict()


def _resolved_feature_keys(
    manifest: dict[str, object],
    *,
    primary_key: str | None,
    extra_keys: list[str],
) -> list[str]:
    keys: list[str] = []
    if primary_key:
        keys.append(primary_key)
    keys.extend(key for key in extra_keys if key)
    if keys:
        return keys
    resolved = resolve_feature_key(manifest, None)
    if resolved is None:
        feature_key = manifest.get("feature_key")
        if isinstance(feature_key, str) and feature_key:
            return [feature_key]
        raise SystemExit("Could not resolve default feature key from manifest.")
    return [resolved]


def _load_feature_concat(
    data_dir: str,
    split: str,
    *,
    feature_keys: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    xs = []
    labels: torch.Tensor | None = None
    sample_ids: torch.Tensor | None = None
    resolved_keys: list[str] = []
    for feature_key in feature_keys:
        ds = ActivationDataset(data_dir=data_dir, split=split, feature_key=feature_key)
        if labels is None:
            labels = ds.y.clone()
            sample_ids = ds.sample_ids.clone()
        else:
            if not torch.equal(ds.y, labels):
                raise SystemExit(
                    f"Label mismatch while concatenating feature key '{feature_key}' on split '{split}'."
                )
            if sample_ids is None or not torch.equal(ds.sample_ids, sample_ids):
                raise SystemExit(
                    f"sample_id mismatch while concatenating feature key '{feature_key}' on split '{split}'."
                )
        xs.append(ds.x)
        resolved_keys.append(ds.feature_key or feature_key)
    if labels is None or sample_ids is None:
        raise SystemExit("No feature tensors were loaded.")
    return torch.cat(xs, dim=1), labels, sample_ids, resolved_keys


def _aggregate_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in SUMMARY_METRICS:
        values = [
            float(row[metric])
            for row in rows
            if metric in row and not math.isnan(float(row[metric]))
        ]
        if not values:
            summary[metric] = {"mean": float("nan"), "std": float("nan"), "n": 0.0}
            continue
        summary[metric] = {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "n": float(len(values)),
        }
    return summary


def _summary_to_csv(path: str, summary: dict[str, dict[str, float]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "std", "n"])
        writer.writeheader()
        for metric in SUMMARY_METRICS:
            stats = summary[metric]
            writer.writerow(
                {
                    "metric": metric,
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "n": int(stats["n"]),
                }
            )


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model_id,
        trust_remote_code=True,
    )

    train_manifest = read_manifest(args.train_data_dir)
    feature_keys = _resolved_feature_keys(
        train_manifest,
        primary_key=args.feature_key,
        extra_keys=args.extra_feature_key,
    )
    train_x, train_y, train_sample_ids, resolved_train_keys = _load_feature_concat(
        args.train_data_dir,
        "train",
        feature_keys=feature_keys,
    )
    natural_x, natural_y, natural_sample_ids, resolved_natural_keys = _load_feature_concat(
        args.train_data_dir,
        "test",
        feature_keys=feature_keys,
    )
    matched_manifest = read_manifest(args.matched_eval_data_dir)
    matched_feature_keys = _resolved_feature_keys(
        matched_manifest,
        primary_key=args.feature_key,
        extra_keys=args.extra_feature_key,
    )
    matched_x, matched_y, matched_sample_ids, resolved_matched_keys = _load_feature_concat(
        args.matched_eval_data_dir,
        "test",
        feature_keys=matched_feature_keys,
    )
    if resolved_train_keys != resolved_natural_keys or resolved_train_keys != resolved_matched_keys:
        raise SystemExit("Resolved feature keys do not match across train/natural/matched datasets.")
    feature_key = "+".join(resolved_train_keys)

    train_pool_rows = _load_jsonl_rows(args.train_pool_jsonl)
    eval_pool_rows = _load_jsonl_rows(args.eval_pool_jsonl)

    train_meta_rows = _metadata_rows_from_ids(
        pool_rows=train_pool_rows,
        sample_ids=train_sample_ids,
        prompt_field=args.prompt_field,
        tokenizer=tokenizer,
    )
    natural_meta_rows = _metadata_rows_from_ids(
        pool_rows=eval_pool_rows,
        sample_ids=natural_sample_ids,
        prompt_field=args.prompt_field,
        tokenizer=tokenizer,
    )
    matched_meta_rows = _metadata_rows_from_ids(
        pool_rows=eval_pool_rows,
        sample_ids=matched_sample_ids,
        prompt_field=args.prompt_field,
        tokenizer=tokenizer,
    )

    train_idx, holdout_idx = _build_holdout_split(
        rows=train_meta_rows,
        labels=train_y,
        ratio=args.holdout_ratio,
        seed=args.holdout_seed,
    )
    model_train_x = train_x.index_select(0, train_idx)
    model_train_y = train_y.index_select(0, train_idx)
    model_train_sample_ids = train_sample_ids.index_select(0, train_idx)
    model_train_meta_rows = _subset_meta_rows(train_meta_rows, train_idx)
    train_group_ids, train_group_summaries = _assign_source_length_groups(
        rows=model_train_meta_rows,
        num_length_bins=args.num_length_bins,
    )
    train_group_summaries = _attach_group_label_stats(
        group_summaries=train_group_summaries,
        group_ids=train_group_ids,
        labels=model_train_y,
    )
    _write_json(
        os.path.join(args.out_dir, "train_group_summary.json"),
        {
            "num_length_bins": args.num_length_bins,
            "groups": train_group_summaries,
        },
    )

    holdout_x: torch.Tensor | None = None
    holdout_y: torch.Tensor | None = None
    holdout_sample_ids: torch.Tensor | None = None
    holdout_meta_rows: list[PromptMetadata] | None = None
    if holdout_idx is not None:
        holdout_x = train_x.index_select(0, holdout_idx)
        holdout_y = train_y.index_select(0, holdout_idx)
        holdout_sample_ids = train_sample_ids.index_select(0, holdout_idx)
        holdout_meta_rows = _subset_meta_rows(train_meta_rows, holdout_idx)

    train_labels_np = model_train_y.numpy().astype(int)
    metadata_models: dict[str, MetadataModel] = {}
    metadata_baselines: dict[str, dict[str, object]] = {}
    for feature_set in ("legacy", "strong"):
        meta_model = MetadataModel(feature_set=feature_set)
        meta_model.fit(model_train_meta_rows, train_labels_np)
        metadata_models[feature_set] = meta_model
        baseline_payload: dict[str, object] = {
            "train": _evaluate_logits(
                model_train_y,
                torch.tensor(meta_model.decision_function(model_train_meta_rows), dtype=torch.float32),
            ),
            "natural": _evaluate_logits(
                natural_y,
                torch.tensor(meta_model.decision_function(natural_meta_rows), dtype=torch.float32),
            ),
            "matched": _evaluate_logits(
                matched_y,
                torch.tensor(meta_model.decision_function(matched_meta_rows), dtype=torch.float32),
            ),
            "params": meta_model.to_json(),
        }
        if holdout_y is not None and holdout_meta_rows is not None:
            baseline_payload["val"] = _evaluate_logits(
                holdout_y,
                torch.tensor(meta_model.decision_function(holdout_meta_rows), dtype=torch.float32),
            )
        metadata_baselines[feature_set] = baseline_payload
    _write_json(os.path.join(args.out_dir, "metadata_baselines.json"), metadata_baselines)

    active_meta_model = metadata_models[args.metadata_feature_set]
    train_meta_logits = active_meta_model.decision_function(model_train_meta_rows)
    holdout_meta_logits = None
    if holdout_meta_rows is not None:
        holdout_meta_logits = active_meta_model.decision_function(holdout_meta_rows)
    natural_meta_logits = active_meta_model.decision_function(natural_meta_rows)
    matched_meta_logits = active_meta_model.decision_function(matched_meta_rows)

    device = choose_device(args.device)
    train_loader = _build_loader(
        model_train_x,
        train_meta_logits,
        model_train_y,
        group_ids=train_group_ids,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    holdout_loader = None
    if holdout_x is not None and holdout_y is not None and holdout_meta_logits is not None:
        holdout_loader = _build_loader(
            holdout_x,
            holdout_meta_logits,
            holdout_y,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    natural_loader = _build_loader(
        natural_x,
        natural_meta_logits,
        natural_y,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    matched_loader = _build_loader(
        matched_x,
        matched_meta_logits,
        matched_y,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(train_loader)
    if steps_per_epoch < 1:
        raise SystemExit("Training split is empty.")
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = 0
    if args.lr_scheduler == "cosine":
        warmup_steps = int(round(args.warmup_ratio * total_steps))
        if args.warmup_ratio > 0.0 and warmup_steps < 1:
            warmup_steps = 1
        warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    num_pos = int((model_train_y == 1).sum().item())
    num_neg = int((model_train_y == 0).sum().item())
    if num_pos > 0 and num_neg > 0:
        pos_weight = torch.tensor(float(num_neg / num_pos), dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)

    seed_rows = []
    for seed in args.seeds:
        set_seed(seed)
        seed_dir = os.path.join(args.out_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        metrics_jsonl = os.path.join(seed_dir, "metrics.jsonl")
        with open(metrics_jsonl, "w", encoding="utf-8"):
            pass

        model, probe_config_payload = _build_hidden_probe(args, input_dim=int(model_train_x.size(1)))
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_rank = (float("-inf"), float("-inf"))
        best_row: dict[str, object] | None = None
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_pointwise_loss = 0.0
            running_group_bce_loss = 0.0
            running_rank_loss = 0.0
            seen = 0
            rank_weight_now = _rank_weight_for_epoch(
                epoch=epoch,
                max_weight=args.rank_lambda,
                warmup_epochs=args.rank_warmup_epochs,
            )
            epoch_group_count = 0
            epoch_rank_group_count = 0
            epoch_rank_pair_count = 0
            train_logits = []
            train_labels = []
            lr_now = args.lr

            for batch_idx, batch in enumerate(train_loader, start=1):
                if len(batch) == 4:
                    hidden_x, meta_logits, labels, batch_group_ids = batch
                else:
                    hidden_x, meta_logits, labels = batch
                    batch_group_ids = None
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

                hidden_x = hidden_x.to(device)
                meta_logits = meta_logits.to(device)
                labels = labels.to(device)
                if batch_group_ids is not None:
                    batch_group_ids = batch_group_ids.to(device)

                residual_logits = model(hidden_x)
                total_logits = residual_logits + meta_logits
                pointwise_loss = F.binary_cross_entropy_with_logits(
                    total_logits,
                    labels,
                    pos_weight=pos_weight,
                )
                group_bce_loss = pointwise_loss
                group_count = 0
                rank_loss = total_logits.new_tensor(0.0)
                rank_group_count = 0
                rank_pair_count = 0
                loss = pointwise_loss
                if args.train_objective != "bce":
                    if batch_group_ids is None:
                        raise SystemExit("Grouped objectives require group ids in the training loader.")
                    group_bce_loss, group_count = _equal_group_bce_loss(
                        logits=total_logits,
                        labels=labels,
                        group_ids=batch_group_ids,
                        pos_weight=pos_weight,
                    )
                    loss = group_bce_loss
                    if args.train_objective == "group_bce_rank":
                        rank_loss, rank_group_count, rank_pair_count = _within_group_pairwise_loss(
                            logits=total_logits,
                            labels=labels,
                            group_ids=batch_group_ids,
                            tau=args.rank_tau,
                            min_pos=args.rank_min_pos,
                            min_neg=args.rank_min_neg,
                            max_pairs_per_group=args.rank_max_pairs_per_group,
                        )
                        loss = loss + (rank_weight_now * rank_loss)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                batch_size = int(labels.size(0))
                running_loss += float(loss.item()) * batch_size
                running_pointwise_loss += float(pointwise_loss.item()) * batch_size
                running_group_bce_loss += float(group_bce_loss.item()) * batch_size
                running_rank_loss += float(rank_loss.item()) * batch_size
                seen += batch_size
                epoch_group_count += group_count
                epoch_rank_group_count += rank_group_count
                epoch_rank_pair_count += rank_pair_count
                global_step += 1
                train_logits.append(total_logits.detach().cpu())
                train_labels.append(labels.detach().cpu())

                if batch_idx % args.log_every == 0:
                    avg_loss = running_loss / max(seen, 1)
                    print(
                        f"seed={seed} epoch={epoch} batch={batch_idx} "
                        f"step={global_step} train_loss={avg_loss:.6f} "
                        f"pointwise_loss={running_pointwise_loss / max(seen, 1):.6f} "
                        f"group_bce_loss={running_group_bce_loss / max(seen, 1):.6f} "
                        f"rank_loss={running_rank_loss / max(seen, 1):.6f} "
                        f"rank_weight={rank_weight_now:.4f} lr={lr_now:.6e}",
                        flush=True,
                    )

            train_logits_cat = torch.cat(train_logits, dim=0)
            train_labels_cat = torch.cat(train_labels, dim=0)
            train_metrics = _evaluate_logits(train_labels_cat, train_logits_cat)
            train_loss_epoch = running_loss / max(seen, 1)
            train_pointwise_loss_epoch = running_pointwise_loss / max(seen, 1)
            train_group_bce_loss_epoch = running_group_bce_loss / max(seen, 1)
            train_rank_loss_epoch = running_rank_loss / max(seen, 1)

            if epoch % args.eval_every != 0:
                print(
                    f"seed={seed} epoch={epoch} train_loss={train_loss_epoch:.6f} "
                    f"train_auc={train_metrics['roc_auc']:.4f} train_pr_auc={train_metrics['pr_auc']:.4f} "
                    f"rank_weight={rank_weight_now:.4f}",
                    flush=True,
                )
                continue

            val_metrics: dict[str, float] | None = None
            if holdout_loader is not None and holdout_meta_rows is not None:
                val_labels, val_logits = _evaluate_loader(model, holdout_loader, device)
                val_metrics = _evaluate_logits(val_labels, val_logits)
                val_metrics.update(
                    _bin_metrics(
                        rows=holdout_meta_rows,
                        labels=val_labels,
                        logits=val_logits,
                        num_length_bins=args.num_length_bins,
                    )
                )
            natural_labels, natural_logits = _evaluate_loader(model, natural_loader, device)
            matched_labels, matched_logits = _evaluate_loader(model, matched_loader, device)

            natural_metrics = _evaluate_logits(natural_labels, natural_logits)
            natural_metrics.update(
                _bin_metrics(
                    rows=natural_meta_rows,
                    labels=natural_labels,
                    logits=natural_logits,
                    num_length_bins=args.num_length_bins,
                )
            )
            matched_metrics = _evaluate_logits(matched_labels, matched_logits)
            matched_metrics.update(
                _bin_metrics(
                    rows=matched_meta_rows,
                    labels=matched_labels,
                    logits=matched_logits,
                    num_length_bins=args.num_length_bins,
                )
            )

            row = {
                "seed": seed,
                "epoch": epoch,
                "step": global_step,
                "lr": lr_now,
                "train_loss": train_loss_epoch,
                "train_pointwise_loss": train_pointwise_loss_epoch,
                "train_group_bce_loss": train_group_bce_loss_epoch,
                "train_rank_loss": train_rank_loss_epoch,
                "train_rank_weight": rank_weight_now,
                "train_group_count": epoch_group_count,
                "train_rank_group_count": epoch_rank_group_count,
                "train_rank_pair_count": epoch_rank_pair_count,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_positive_precision": train_metrics["positive_precision"],
                "train_positive_recall": train_metrics["positive_recall"],
                "train_positive_f1": train_metrics["positive_f1"],
                "train_prevalence": train_metrics["prevalence"],
                "train_roc_auc": train_metrics["roc_auc"],
                "train_pr_auc": train_metrics["pr_auc"],
            }
            if val_metrics is not None:
                for key, value in val_metrics.items():
                    row[f"val_{key}"] = value
            for prefix, metrics in (("natural", natural_metrics), ("matched", matched_metrics)):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
            _write_jsonl(metrics_jsonl, row)

            rank = _best_key(
                row,
                split_name=args.selection_split,
                selection_metric=args.selection_metric,
                tie_breaker=args.tie_breaker,
            )
            if rank > best_rank:
                best_rank = rank
                best_row = dict(row)
                torch.save(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "state_dict": model.state_dict(),
                        "feature_key": feature_key,
                        "feature_keys": resolved_train_keys,
                        "probe_config": probe_config_payload,
                        "metadata_feature_set": args.metadata_feature_set,
                        "metadata_model": active_meta_model.to_json(),
                        "selection_split": args.selection_split,
                        "selection_metric": args.selection_metric,
                        "tie_breaker": args.tie_breaker,
                        "holdout_ratio": args.holdout_ratio,
                        "holdout_seed": args.holdout_seed,
                        "train_objective": args.train_objective,
                        "rank_lambda": args.rank_lambda,
                        "rank_tau": args.rank_tau,
                        "rank_warmup_epochs": args.rank_warmup_epochs,
                        "rank_min_pos": args.rank_min_pos,
                        "rank_min_neg": args.rank_min_neg,
                        "rank_max_pairs_per_group": args.rank_max_pairs_per_group,
                    },
                    os.path.join(seed_dir, "best.pt"),
                )

            print(
                " ".join(
                    [
                        f"seed={seed}",
                        f"epoch={epoch}",
                        f"natural_auc={natural_metrics['roc_auc']:.4f}",
                        f"natural_pr={natural_metrics['pr_auc']:.4f}",
                        f"matched_auc={matched_metrics['roc_auc']:.4f}",
                        f"matched_pr={matched_metrics['pr_auc']:.4f}",
                        f"matched_bin_auc={matched_metrics['bin_roc_auc']:.4f}",
                        f"matched_bin_pr={matched_metrics['bin_pr_auc']:.4f}",
                    ]
                ),
                flush=True,
            )

        if best_row is None:
            raise SystemExit("No evaluation rows were recorded; check --eval-every.")

        best_payload = {
            "selection_rule": (
                f"max({args.selection_split}_{args.selection_metric}), "
                f"tie_break=max({args.selection_split}_{args.tie_breaker})"
            ),
            **best_row,
        }
        _write_json(os.path.join(seed_dir, "best_metrics.json"), best_payload)
        seed_rows.append(
            {
                "seed": seed,
                "accuracy": float(best_row["matched_accuracy"]) if args.selection_split == "matched" else float(best_row["natural_accuracy"]),
                "macro_f1": float(best_row["matched_macro_f1"]) if args.selection_split == "matched" else float(best_row["natural_macro_f1"]),
                "positive_precision": float(best_row["matched_positive_precision"]) if args.selection_split == "matched" else float(best_row["natural_positive_precision"]),
                "positive_recall": float(best_row["matched_positive_recall"]) if args.selection_split == "matched" else float(best_row["natural_positive_recall"]),
                "positive_f1": float(best_row["matched_positive_f1"]) if args.selection_split == "matched" else float(best_row["natural_positive_f1"]),
                "prevalence": float(best_row["matched_prevalence"]) if args.selection_split == "matched" else float(best_row["natural_prevalence"]),
                "roc_auc": float(best_row["matched_roc_auc"]) if args.selection_split == "matched" else float(best_row["natural_roc_auc"]),
                "pr_auc": float(best_row["matched_pr_auc"]) if args.selection_split == "matched" else float(best_row["natural_pr_auc"]),
                "bin_roc_auc": float(best_row["matched_bin_roc_auc"]) if args.selection_split == "matched" else float(best_row["natural_bin_roc_auc"]),
                "bin_pr_auc": float(best_row["matched_bin_pr_auc"]) if args.selection_split == "matched" else float(best_row["natural_bin_pr_auc"]),
                "natural_accuracy": float(best_row["natural_accuracy"]),
                "natural_macro_f1": float(best_row["natural_macro_f1"]),
                "natural_positive_precision": float(best_row["natural_positive_precision"]),
                "natural_positive_recall": float(best_row["natural_positive_recall"]),
                "natural_positive_f1": float(best_row["natural_positive_f1"]),
                "natural_prevalence": float(best_row["natural_prevalence"]),
                "natural_roc_auc": float(best_row["natural_roc_auc"]),
                "natural_pr_auc": float(best_row["natural_pr_auc"]),
                "natural_bin_roc_auc": float(best_row["natural_bin_roc_auc"]),
                "natural_bin_pr_auc": float(best_row["natural_bin_pr_auc"]),
                "matched_accuracy": float(best_row["matched_accuracy"]),
                "matched_macro_f1": float(best_row["matched_macro_f1"]),
                "matched_positive_precision": float(best_row["matched_positive_precision"]),
                "matched_positive_recall": float(best_row["matched_positive_recall"]),
                "matched_positive_f1": float(best_row["matched_positive_f1"]),
                "matched_prevalence": float(best_row["matched_prevalence"]),
                "matched_roc_auc": float(best_row["matched_roc_auc"]),
                "matched_pr_auc": float(best_row["matched_pr_auc"]),
                "matched_bin_roc_auc": float(best_row["matched_bin_roc_auc"]),
                "matched_bin_pr_auc": float(best_row["matched_bin_pr_auc"]),
            }
        )

    summary_payload = {
        "selection_split": args.selection_split,
        "selection_metric": args.selection_metric,
        "tie_breaker": args.tie_breaker,
        "metadata_feature_set": args.metadata_feature_set,
        "train_objective": args.train_objective,
        "rank_lambda": args.rank_lambda,
        "rank_tau": args.rank_tau,
        "rank_warmup_epochs": args.rank_warmup_epochs,
        "rank_min_pos": args.rank_min_pos,
        "rank_min_neg": args.rank_min_neg,
        "rank_max_pairs_per_group": args.rank_max_pairs_per_group,
        "train_group_summary": train_group_summaries,
        "metadata_baselines": metadata_baselines,
        "runs": seed_rows,
        "natural_summary": _aggregate_rows(
            [
                {
                    "accuracy": row["natural_accuracy"],
                    "macro_f1": row["natural_macro_f1"],
                    "positive_precision": row["natural_positive_precision"],
                    "positive_recall": row["natural_positive_recall"],
                    "positive_f1": row["natural_positive_f1"],
                    "prevalence": row["natural_prevalence"],
                    "roc_auc": row["natural_roc_auc"],
                    "pr_auc": row["natural_pr_auc"],
                    "bin_roc_auc": row["natural_bin_roc_auc"],
                    "bin_pr_auc": row["natural_bin_pr_auc"],
                }
                for row in seed_rows
            ]
        ),
        "matched_summary": _aggregate_rows(
            [
                {
                    "accuracy": row["matched_accuracy"],
                    "macro_f1": row["matched_macro_f1"],
                    "positive_precision": row["matched_positive_precision"],
                    "positive_recall": row["matched_positive_recall"],
                    "positive_f1": row["matched_positive_f1"],
                    "prevalence": row["matched_prevalence"],
                    "roc_auc": row["matched_roc_auc"],
                    "pr_auc": row["matched_pr_auc"],
                    "bin_roc_auc": row["matched_bin_roc_auc"],
                    "bin_pr_auc": row["matched_bin_pr_auc"],
                }
                for row in seed_rows
            ]
        ),
    }
    _write_json(os.path.join(args.out_dir, "seed_summary.json"), summary_payload)
    _summary_to_csv(
        os.path.join(args.out_dir, "natural_seed_summary.csv"),
        summary_payload["natural_summary"],
    )
    _summary_to_csv(
        os.path.join(args.out_dir, "matched_seed_summary.csv"),
        summary_payload["matched_summary"],
    )
    print(f"Wrote metadata residual probe outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
