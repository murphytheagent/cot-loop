#!/usr/bin/env python3
"""Audit why prompt length predicts completion length on the natural regression surface."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from loop_probe.configs import ProbeConfig, build_probe_model
from loop_probe.dataloader import (
    ActivationDataset,
    read_manifest,
    resolve_input_dim,
    resolve_sample_shape,
    resolve_split_info,
)
from loop_probe.train_utils import (
    choose_device,
    probe_scores_and_predictions,
    resolve_classifier_layer,
)


DATASET_ORDER = ("gpqa", "aime", "math500", "mmlu_pro", "livecodebench")
DISPLAY_NAMES = {
    "gpqa": "GPQA",
    "aime": "AIME",
    "math500": "MATH-500",
    "mmlu_pro": "MMLU-Pro",
    "livecodebench": "LiveCodeBench",
}
FEATURE_NAMES = (
    "prompt_token_count",
    "log_token_length",
    "char_length",
    "newline_count",
    "digit_count",
    "dollar_count",
    "choice_count",
)
MODEL_ORDER = (
    "prompt_length",
    "shape_linear",
    "shape_tree",
    "last_layer",
    "ensemble",
    "shape_tree_plus_last_layer",
    "shape_tree_plus_ensemble",
)


@dataclass(frozen=True)
class PromptArchiveRow:
    sample_id: int
    split: str
    prompt: str
    prompt_token_count: float
    effective_max_tokens: float
    mean_relative_length: float
    p_loop: float
    p_cap: float
    prompt_style: str | None
    choice_count_value: int

    @property
    def log_token_length(self) -> float:
        return math.log1p(self.prompt_token_count)

    @property
    def char_length(self) -> float:
        return float(len(self.prompt))

    @property
    def newline_count(self) -> float:
        return float(self.prompt.count("\n"))

    @property
    def digit_count(self) -> float:
        return float(sum(ch.isdigit() for ch in self.prompt))

    @property
    def dollar_count(self) -> float:
        return float(self.prompt.count("$"))

    @property
    def choice_count(self) -> float:
        return float(self.choice_count_value)

    def feature_value(self, name: str) -> float:
        if name == "prompt_token_count":
            return float(self.prompt_token_count)
        if name == "log_token_length":
            return self.log_token_length
        if name == "char_length":
            return self.char_length
        if name == "newline_count":
            return self.newline_count
        if name == "digit_count":
            return self.digit_count
        if name == "dollar_count":
            return self.dollar_count
        if name == "choice_count":
            return self.choice_count
        raise KeyError(f"Unknown feature '{name}'.")


@dataclass(frozen=True)
class ScoredModelResult:
    model_name: str
    metrics: dict[str, float]
    train_predictions: dict[int, float]
    test_predictions: dict[int, float]
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--selection-kind",
        choices=("best_loss", "best_rank"),
        default="best_loss",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_ORDER))
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_archive_rows(data_dir: Path) -> list[PromptArchiveRow]:
    path = data_dir / "diagnostics" / "prompt_rollout_archive.jsonl"
    rows: list[PromptArchiveRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_num}") from exc
            prompt = payload.get("prompt")
            split = payload.get("split")
            sample_id = payload.get("sample_id")
            if not isinstance(prompt, str) or not isinstance(split, str) or not isinstance(sample_id, int):
                raise SystemExit(f"Missing prompt/split/sample_id at {path}:{line_num}")
            choices = payload.get("choices")
            choice_count = len(choices) if isinstance(choices, list) else 0
            rows.append(
                PromptArchiveRow(
                    sample_id=sample_id,
                    split=split,
                    prompt=prompt,
                    prompt_token_count=float(payload["prompt_token_count"]),
                    effective_max_tokens=float(payload["effective_max_tokens"]),
                    mean_relative_length=float(payload["mean_relative_length"]),
                    p_loop=float(payload.get("p_loop") or 0.0),
                    p_cap=float(payload.get("p_cap") or 0.0),
                    prompt_style=(
                        str(payload["prompt_style"])
                        if isinstance(payload.get("prompt_style"), str)
                        else None
                    ),
                    choice_count_value=choice_count,
                )
            )
    return rows


def rows_by_split(rows: list[PromptArchiveRow], split: str) -> list[PromptArchiveRow]:
    return [row for row in rows if row.split == split]


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    idx = 0
    while idx < len(ordered):
        stop = idx + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[idx][1]:
            stop += 1
        avg = (idx + stop - 1) / 2.0 + 1.0
        for pos in range(idx, stop):
            out[ordered[pos][0]] = avg
        idx = stop
    return out


def safe_spearman(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    if np.allclose(y_true, y_true[0]) or np.allclose(scores, scores[0]):
        return float("nan")
    rx = np.asarray(ranks(y_true.tolist()), dtype=np.float64)
    ry = np.asarray(ranks(scores.tolist()), dtype=np.float64)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def top_capture_fraction(y_true: np.ndarray, scores: np.ndarray, *, fraction: float) -> float:
    if y_true.size == 0:
        return float("nan")
    total_mass = float(np.sum(y_true))
    if total_mass <= 0.0:
        return float("nan")
    keep = max(1, int(math.ceil(float(y_true.size) * fraction)))
    order = np.argsort(-scores, kind="stable")
    captured = float(np.sum(y_true[order[:keep]]))
    return captured / total_mass


def regression_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    errors = scores - y_true
    return {
        "mse": float(np.mean(np.square(errors))),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "target_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(scores)),
        "pearson": safe_pearson(y_true, scores),
        "spearman": safe_spearman(y_true, scores),
        "top_10p_capture": top_capture_fraction(y_true, scores, fraction=0.10),
        "top_20p_capture": top_capture_fraction(y_true, scores, fraction=0.20),
    }


def feature_matrix(rows: list[PromptArchiveRow], feature_names: tuple[str, ...]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        values.append([row.feature_value(name) for name in feature_names])
    return np.asarray(values, dtype=np.float64)


class LinearPromptRegressor:
    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model: LinearRegression | None = None
        self.constant_value: float | None = None

    def fit(self, rows: list[PromptArchiveRow]) -> None:
        x = feature_matrix(rows, self.feature_names)
        x_scaled = self.scaler.fit_transform(x)
        y = np.asarray([row.mean_relative_length for row in rows], dtype=np.float64)
        if y.size == 0:
            raise SystemExit("Cannot fit a regression model on an empty split.")
        if np.allclose(y, y[0]):
            self.constant_value = float(y[0])
            self.model = None
            return
        self.constant_value = None
        self.model = LinearRegression()
        self.model.fit(x_scaled, y)

    def predict(self, rows: list[PromptArchiveRow]) -> np.ndarray:
        x = feature_matrix(rows, self.feature_names)
        if self.constant_value is not None:
            return np.full(x.shape[0], self.constant_value, dtype=np.float64)
        if self.model is None:
            raise SystemExit("LinearPromptRegressor used before fit().")
        x_scaled = self.scaler.transform(x)
        predictions = self.model.predict(x_scaled)
        return np.clip(predictions, 0.0, 1.0)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "constant_value": self.constant_value,
        }
        if self.model is not None:
            payload["coef"] = self.model.coef_.tolist()
            payload["intercept"] = float(self.model.intercept_)
        return payload


class TreePromptRegressor:
    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names
        self.model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_depth=3,
            max_iter=300,
            min_samples_leaf=8,
            random_state=0,
        )
        self.constant_value: float | None = None

    def fit(self, rows: list[PromptArchiveRow]) -> None:
        x = feature_matrix(rows, self.feature_names)
        y = np.asarray([row.mean_relative_length for row in rows], dtype=np.float64)
        if y.size == 0:
            raise SystemExit("Cannot fit a tree regressor on an empty split.")
        if np.allclose(y, y[0]):
            self.constant_value = float(y[0])
            return
        self.constant_value = None
        self.model.fit(x, y)

    def predict(self, rows: list[PromptArchiveRow]) -> np.ndarray:
        x = feature_matrix(rows, self.feature_names)
        if self.constant_value is not None:
            return np.full(x.shape[0], self.constant_value, dtype=np.float64)
        predictions = self.model.predict(x)
        return np.clip(predictions, 0.0, 1.0)

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "constant_value": self.constant_value,
            "params": self.model.get_params(),
        }


class StackedRegressor:
    def __init__(self, *, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.model: LinearRegression | None = None
        self.constant_value: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x)
        if y.size == 0:
            raise SystemExit("Cannot fit a stacked regressor on an empty split.")
        if np.allclose(y, y[0]):
            self.constant_value = float(y[0])
            self.model = None
            return
        self.constant_value = None
        self.model = LinearRegression()
        self.model.fit(x_scaled, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.constant_value is not None:
            return np.full(x.shape[0], self.constant_value, dtype=np.float64)
        if self.model is None:
            raise SystemExit("StackedRegressor used before fit().")
        predictions = self.model.predict(self.scaler.transform(x))
        return np.clip(predictions, 0.0, 1.0)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_names": list(self.feature_names),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "constant_value": self.constant_value,
        }
        if self.model is not None:
            payload["coef"] = self.model.coef_.tolist()
            payload["intercept"] = float(self.model.intercept_)
        return payload


def probe_cfg_from_checkpoint(payload: dict[str, Any]) -> ProbeConfig:
    probe_cfg_raw = payload.get("probe_config")
    if not isinstance(probe_cfg_raw, dict):
        raise SystemExit("Checkpoint is missing probe_config.")
    return ProbeConfig(
        probe_type=str(probe_cfg_raw.get("probe_type", "linear")),
        hidden_dim=int(probe_cfg_raw.get("hidden_dim", 128)),
        dropout=float(probe_cfg_raw.get("dropout", 0.0)),
        depth=int(probe_cfg_raw.get("depth", probe_cfg_raw.get("mlp_depth", 1))),
        classifier_mode=str(probe_cfg_raw.get("classifier_mode", "last_layer")),
        classifier_layer=int(probe_cfg_raw.get("classifier_layer", -1)),
        vote_rule=str(probe_cfg_raw.get("vote_rule", "majority")),
        score_rule=str(probe_cfg_raw.get("score_rule", "vote_fraction")),
    )


def prepare_model_inputs(
    x: torch.Tensor,
    *,
    classifier_mode: str,
    resolved_classifier_layer: int | None,
) -> torch.Tensor:
    if classifier_mode == "last_layer":
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            if resolved_classifier_layer is None:
                raise SystemExit("Missing resolved classifier layer for stacked last-layer inputs.")
            return x[:, resolved_classifier_layer, :]
        raise SystemExit(f"Unsupported last_layer input shape {tuple(x.shape)}.")
    if classifier_mode == "ensemble":
        if x.ndim != 3:
            raise SystemExit(f"ensemble mode expects stacked inputs, got {tuple(x.shape)}.")
        return x
    raise SystemExit(f"Unsupported classifier_mode '{classifier_mode}'.")


def score_checkpoint(
    *,
    checkpoint_path: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[int, float]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise SystemExit(f"Checkpoint payload missing state_dict: {checkpoint_path}")

    manifest = read_manifest(str(data_dir))
    checkpoint_feature_key = payload.get("feature_key")
    resolved_feature_key_arg = None
    if isinstance(checkpoint_feature_key, str) and checkpoint_feature_key:
        if isinstance(manifest.get("feature_views"), dict):
            resolved_feature_key_arg = checkpoint_feature_key

    split_info, resolved_feature_key = resolve_split_info(
        manifest,
        split=split,
        feature_key=resolved_feature_key_arg,
    )
    if int(split_info.get("num_rows", split_info.get("num_samples", 0))) < 1:
        raise SystemExit(f"Split '{split}' in {data_dir} has no rows.")

    probe_cfg = probe_cfg_from_checkpoint(payload)
    input_dim = resolve_input_dim(manifest, resolved_feature_key)
    sample_shape = resolve_sample_shape(manifest, resolved_feature_key)
    resolved_classifier_layer: int | None = None
    if probe_cfg.classifier_mode == "last_layer" and len(sample_shape) == 2:
        resolved_classifier_layer = resolve_classifier_layer(
            int(sample_shape[0]),
            probe_cfg.classifier_layer,
        )

    model = build_probe_model(
        input_dim=input_dim,
        probe_cfg=probe_cfg,
        sample_shape=sample_shape,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    dataset = ActivationDataset(str(data_dir), split=split, feature_key=resolved_feature_key)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for x, _y in dataloader:
            x = prepare_model_inputs(
                x.to(device),
                classifier_mode=probe_cfg.classifier_mode,
                resolved_classifier_layer=resolved_classifier_layer,
            )
            logits = model(x)
            all_logits.append(logits.detach().cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    scores, _predictions = probe_scores_and_predictions(
        logits_cat,
        classifier_mode=probe_cfg.classifier_mode,
        score_rule=probe_cfg.score_rule,
    )
    sample_ids = dataset.sample_ids.tolist()
    if len(sample_ids) != int(scores.numel()):
        raise SystemExit(f"Score/sample_id length mismatch for {checkpoint_path}.")
    return {int(sample_id): float(score) for sample_id, score in zip(sample_ids, scores.tolist())}


def average_score_maps(score_maps: list[dict[int, float]]) -> dict[int, float]:
    if not score_maps:
        raise SystemExit("Expected at least one score map to average.")
    sample_ids = sorted(score_maps[0].keys())
    for score_map in score_maps[1:]:
        if sorted(score_map.keys()) != sample_ids:
            raise SystemExit("Seed score maps do not share the same sample ids.")
    averaged: dict[int, float] = {}
    for sample_id in sample_ids:
        averaged[sample_id] = float(np.mean([score_map[sample_id] for score_map in score_maps]))
    return averaged


def score_view_dir(
    *,
    view_dir: Path,
    data_dir: Path,
    selection_kind: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[int, float], dict[int, float], list[str]]:
    seed_dirs = sorted(path for path in view_dir.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise SystemExit(f"No seed dirs found under {view_dir}")
    train_maps: list[dict[int, float]] = []
    test_maps: list[dict[int, float]] = []
    used_seed_dirs: list[str] = []
    for seed_dir in seed_dirs:
        checkpoint_path = seed_dir / f"{selection_kind}.pt"
        if not checkpoint_path.exists():
            checkpoint_path = seed_dir / "best.pt"
        if not checkpoint_path.exists():
            raise SystemExit(f"Missing checkpoint under {seed_dir}")
        train_maps.append(
            score_checkpoint(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                split="train",
                batch_size=batch_size,
                device=device,
            )
        )
        test_maps.append(
            score_checkpoint(
                checkpoint_path=checkpoint_path,
                data_dir=data_dir,
                split="test",
                batch_size=batch_size,
                device=device,
            )
        )
        used_seed_dirs.append(str(seed_dir))
    return average_score_maps(train_maps), average_score_maps(test_maps), used_seed_dirs


def score_array(rows: list[PromptArchiveRow], score_by_sample_id: dict[int, float]) -> np.ndarray:
    missing = [row.sample_id for row in rows if row.sample_id not in score_by_sample_id]
    if missing:
        raise SystemExit(f"Missing scores for sample ids: {missing[:10]}")
    return np.asarray([score_by_sample_id[row.sample_id] for row in rows], dtype=np.float64)


def quantile_edges(values: np.ndarray, bins: int) -> list[float]:
    if values.size == 0:
        return []
    edges = [float(np.min(values))]
    for bucket in range(1, bins):
        edges.append(float(np.quantile(values, bucket / bins)))
    edges.append(float(np.max(values)))
    return edges


def assign_bucket(value: float, edges: list[float]) -> int:
    if not edges:
        return 0
    for idx in range(1, len(edges)):
        upper = edges[idx]
        if idx == len(edges) - 1 or value <= upper:
            return idx - 1
    return len(edges) - 2


def summarize_prompt_length_bins(
    rows: list[PromptArchiveRow],
    *,
    shape_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    last_layer_scores: np.ndarray,
    bins: int = 4,
) -> list[dict[str, Any]]:
    lengths = np.asarray([row.prompt_token_count for row in rows], dtype=np.float64)
    targets = np.asarray([row.mean_relative_length for row in rows], dtype=np.float64)
    edges = quantile_edges(lengths, bins)
    summaries: list[dict[str, Any]] = []
    for bucket in range(max(1, bins)):
        bucket_rows = [idx for idx, value in enumerate(lengths) if assign_bucket(float(value), edges) == bucket]
        if not bucket_rows:
            continue
        idx_array = np.asarray(bucket_rows, dtype=np.int64)
        summaries.append(
            {
                "bucket": bucket + 1,
                "num_rows": int(idx_array.size),
                "prompt_token_min": float(np.min(lengths[idx_array])),
                "prompt_token_max": float(np.max(lengths[idx_array])),
                "prompt_token_mean": float(np.mean(lengths[idx_array])),
                "mean_relative_length_mean": float(np.mean(targets[idx_array])),
                "length_vs_target_spearman": safe_spearman(lengths[idx_array], targets[idx_array]),
                "shape_vs_target_spearman": safe_spearman(shape_scores[idx_array], targets[idx_array]),
                "ensemble_vs_target_spearman": safe_spearman(ensemble_scores[idx_array], targets[idx_array]),
                "last_layer_vs_target_spearman": safe_spearman(last_layer_scores[idx_array], targets[idx_array]),
                "mean_newline_count": float(
                    np.mean([rows[idx].newline_count for idx in idx_array.tolist()])
                ),
                "mean_dollar_count": float(
                    np.mean([rows[idx].dollar_count for idx in idx_array.tolist()])
                ),
                "mean_choice_count": float(
                    np.mean([rows[idx].choice_count for idx in idx_array.tolist()])
                ),
            }
        )
    return summaries


def model_summary_payload(result: ScoredModelResult) -> dict[str, Any]:
    return {
        "model_name": result.model_name,
        "metrics": result.metrics,
        "payload": result.payload,
    }


def main() -> None:
    args = parse_args()
    regression_root = Path(args.regression_root)
    out_dir = Path(args.out_dir)
    device = choose_device(args.device)

    summary: dict[str, Any] = {
        "selection_kind": args.selection_kind,
        "regression_root": str(regression_root),
        "datasets": {},
    }
    metrics_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for dataset in args.datasets:
        if dataset not in DISPLAY_NAMES:
            raise SystemExit(f"Unknown dataset '{dataset}'.")
        dataset_root = regression_root / dataset
        data_dir = dataset_root / "shared_archive"
        archive_rows = read_archive_rows(data_dir)
        train_rows = rows_by_split(archive_rows, "train")
        test_rows = rows_by_split(archive_rows, "test")
        if not train_rows or not test_rows:
            raise SystemExit(f"{dataset}: expected non-empty train and test rows.")

        y_train = np.asarray([row.mean_relative_length for row in train_rows], dtype=np.float64)
        y_test = np.asarray([row.mean_relative_length for row in test_rows], dtype=np.float64)

        prompt_length_model = LinearPromptRegressor(("prompt_token_count",))
        prompt_length_model.fit(train_rows)
        length_train_pred = prompt_length_model.predict(train_rows)
        length_test_pred = prompt_length_model.predict(test_rows)

        shape_linear_model = LinearPromptRegressor(FEATURE_NAMES)
        shape_linear_model.fit(train_rows)
        shape_linear_train_pred = shape_linear_model.predict(train_rows)
        shape_linear_test_pred = shape_linear_model.predict(test_rows)

        shape_tree_model = TreePromptRegressor(FEATURE_NAMES)
        shape_tree_model.fit(train_rows)
        shape_tree_train_pred = shape_tree_model.predict(train_rows)
        shape_tree_test_pred = shape_tree_model.predict(test_rows)

        last_layer_train_map, last_layer_test_map, last_layer_seed_dirs = score_view_dir(
            view_dir=dataset_root / "mean_relative_length" / "last_layer",
            data_dir=data_dir,
            selection_kind=args.selection_kind,
            batch_size=args.batch_size,
            device=device,
        )
        ensemble_train_map, ensemble_test_map, ensemble_seed_dirs = score_view_dir(
            view_dir=dataset_root / "mean_relative_length" / "ensemble",
            data_dir=data_dir,
            selection_kind=args.selection_kind,
            batch_size=args.batch_size,
            device=device,
        )
        last_layer_train_pred = score_array(train_rows, last_layer_train_map)
        last_layer_test_pred = score_array(test_rows, last_layer_test_map)
        ensemble_train_pred = score_array(train_rows, ensemble_train_map)
        ensemble_test_pred = score_array(test_rows, ensemble_test_map)

        shape_tree_plus_last_layer = StackedRegressor(
            feature_names=("shape_tree_prediction", "last_layer_score")
        )
        shape_tree_plus_last_layer.fit(
            np.column_stack([shape_tree_train_pred, last_layer_train_pred]),
            y_train,
        )
        shape_tree_plus_last_layer_train_pred = shape_tree_plus_last_layer.predict(
            np.column_stack([shape_tree_train_pred, last_layer_train_pred])
        )
        shape_tree_plus_last_layer_test_pred = shape_tree_plus_last_layer.predict(
            np.column_stack([shape_tree_test_pred, last_layer_test_pred])
        )

        shape_tree_plus_ensemble = StackedRegressor(
            feature_names=("shape_tree_prediction", "ensemble_score")
        )
        shape_tree_plus_ensemble.fit(
            np.column_stack([shape_tree_train_pred, ensemble_train_pred]),
            y_train,
        )
        shape_tree_plus_ensemble_train_pred = shape_tree_plus_ensemble.predict(
            np.column_stack([shape_tree_train_pred, ensemble_train_pred])
        )
        shape_tree_plus_ensemble_test_pred = shape_tree_plus_ensemble.predict(
            np.column_stack([shape_tree_test_pred, ensemble_test_pred])
        )

        model_results = [
            ScoredModelResult(
                model_name="prompt_length",
                metrics=regression_metrics(y_test, length_test_pred),
                train_predictions={row.sample_id: float(pred) for row, pred in zip(train_rows, length_train_pred, strict=True)},
                test_predictions={row.sample_id: float(pred) for row, pred in zip(test_rows, length_test_pred, strict=True)},
                payload=prompt_length_model.to_json(),
            ),
            ScoredModelResult(
                model_name="shape_linear",
                metrics=regression_metrics(y_test, shape_linear_test_pred),
                train_predictions={row.sample_id: float(pred) for row, pred in zip(train_rows, shape_linear_train_pred, strict=True)},
                test_predictions={row.sample_id: float(pred) for row, pred in zip(test_rows, shape_linear_test_pred, strict=True)},
                payload=shape_linear_model.to_json(),
            ),
            ScoredModelResult(
                model_name="shape_tree",
                metrics=regression_metrics(y_test, shape_tree_test_pred),
                train_predictions={row.sample_id: float(pred) for row, pred in zip(train_rows, shape_tree_train_pred, strict=True)},
                test_predictions={row.sample_id: float(pred) for row, pred in zip(test_rows, shape_tree_test_pred, strict=True)},
                payload=shape_tree_model.to_json(),
            ),
            ScoredModelResult(
                model_name="last_layer",
                metrics=regression_metrics(y_test, last_layer_test_pred),
                train_predictions=last_layer_train_map,
                test_predictions=last_layer_test_map,
                payload={"seed_dirs": last_layer_seed_dirs},
            ),
            ScoredModelResult(
                model_name="ensemble",
                metrics=regression_metrics(y_test, ensemble_test_pred),
                train_predictions=ensemble_train_map,
                test_predictions=ensemble_test_map,
                payload={"seed_dirs": ensemble_seed_dirs},
            ),
            ScoredModelResult(
                model_name="shape_tree_plus_last_layer",
                metrics=regression_metrics(y_test, shape_tree_plus_last_layer_test_pred),
                train_predictions={
                    row.sample_id: float(pred)
                    for row, pred in zip(train_rows, shape_tree_plus_last_layer_train_pred, strict=True)
                },
                test_predictions={
                    row.sample_id: float(pred)
                    for row, pred in zip(test_rows, shape_tree_plus_last_layer_test_pred, strict=True)
                },
                payload=shape_tree_plus_last_layer.to_json(),
            ),
            ScoredModelResult(
                model_name="shape_tree_plus_ensemble",
                metrics=regression_metrics(y_test, shape_tree_plus_ensemble_test_pred),
                train_predictions={
                    row.sample_id: float(pred)
                    for row, pred in zip(train_rows, shape_tree_plus_ensemble_train_pred, strict=True)
                },
                test_predictions={
                    row.sample_id: float(pred)
                    for row, pred in zip(test_rows, shape_tree_plus_ensemble_test_pred, strict=True)
                },
                payload=shape_tree_plus_ensemble.to_json(),
            ),
        ]

        for result in model_results:
            metrics_rows.append(
                {
                    "dataset": dataset,
                    "dataset_name": DISPLAY_NAMES[dataset],
                    "model_name": result.model_name,
                    **result.metrics,
                }
            )

        length_bins = summarize_prompt_length_bins(
            test_rows,
            shape_scores=shape_tree_test_pred,
            ensemble_scores=ensemble_test_pred,
            last_layer_scores=last_layer_test_pred,
            bins=4,
        )
        for row in length_bins:
            bin_rows.append({"dataset": dataset, "dataset_name": DISPLAY_NAMES[dataset], **row})

        for idx, row in enumerate(test_rows):
            sample_rows.append(
                {
                    "dataset": dataset,
                    "dataset_name": DISPLAY_NAMES[dataset],
                    "sample_id": row.sample_id,
                    "prompt_token_count": row.prompt_token_count,
                    "char_length": row.char_length,
                    "newline_count": row.newline_count,
                    "digit_count": row.digit_count,
                    "dollar_count": row.dollar_count,
                    "choice_count": row.choice_count,
                    "mean_relative_length": row.mean_relative_length,
                    "p_loop": row.p_loop,
                    "p_cap": row.p_cap,
                    "prompt_length_prediction": float(length_test_pred[idx]),
                    "shape_linear_prediction": float(shape_linear_test_pred[idx]),
                    "shape_tree_prediction": float(shape_tree_test_pred[idx]),
                    "last_layer_prediction": float(last_layer_test_pred[idx]),
                    "ensemble_prediction": float(ensemble_test_pred[idx]),
                    "shape_tree_plus_ensemble_prediction": float(shape_tree_plus_ensemble_test_pred[idx]),
                    "length_residual": float(row.mean_relative_length - length_test_pred[idx]),
                    "shape_tree_residual": float(row.mean_relative_length - shape_tree_test_pred[idx]),
                    "ensemble_residual_against_shape_tree": float(ensemble_test_pred[idx] - shape_tree_test_pred[idx]),
                    "prompt_excerpt": row.prompt.replace("\n", "\\n")[:220],
                }
            )

        summary["datasets"][dataset] = {
            "display_name": DISPLAY_NAMES[dataset],
            "data_dir": str(data_dir),
            "train_count": len(train_rows),
            "test_count": len(test_rows),
            "prompt_length_test_correlation": {
                "pearson": safe_pearson(
                    np.asarray([row.prompt_token_count for row in test_rows], dtype=np.float64),
                    y_test,
                ),
                "spearman": safe_spearman(
                    np.asarray([row.prompt_token_count for row in test_rows], dtype=np.float64),
                    y_test,
                ),
            },
            "prompt_style_counts": {
                str(key): int(value)
                for key, value in sorted(
                    {
                        (row.prompt_style or "unknown"): sum(
                            1
                            for candidate in archive_rows
                            if (candidate.prompt_style or "unknown") == (row.prompt_style or "unknown")
                        )
                        for row in archive_rows
                    }.items()
                )
            },
            "models": {
                result.model_name: model_summary_payload(result)
                for result in model_results
            },
            "shape_tree_residual_correlation": {
                "last_layer": safe_spearman(
                    y_test - shape_tree_test_pred,
                    last_layer_test_pred,
                ),
                "ensemble": safe_spearman(
                    y_test - shape_tree_test_pred,
                    ensemble_test_pred,
                ),
            },
            "prompt_length_bins": length_bins,
        }

    mean_rows = []
    for model_name in MODEL_ORDER:
        matching = [row for row in metrics_rows if row["model_name"] == model_name]
        if not matching:
            continue
        mean_rows.append(
            {
                "model_name": model_name,
                "dataset_count": len(matching),
                "mean_top_20p_capture": float(np.mean([float(row["top_20p_capture"]) for row in matching])),
                "mean_rmse": float(np.mean([float(row["rmse"]) for row in matching])),
                "mean_spearman": float(np.mean([float(row["spearman"]) for row in matching])),
            }
        )
    summary["cross_dataset_means"] = mean_rows

    write_json(out_dir / "mechanism_summary.json", summary)
    write_csv(
        out_dir / "regression_model_metrics.csv",
        [
            "dataset",
            "dataset_name",
            "model_name",
            "mse",
            "mae",
            "rmse",
            "target_mean",
            "pred_mean",
            "pearson",
            "spearman",
            "top_10p_capture",
            "top_20p_capture",
        ],
        metrics_rows,
    )
    write_csv(
        out_dir / "cross_dataset_means.csv",
        ["model_name", "dataset_count", "mean_top_20p_capture", "mean_rmse", "mean_spearman"],
        mean_rows,
    )
    write_csv(
        out_dir / "prompt_length_bins.csv",
        [
            "dataset",
            "dataset_name",
            "bucket",
            "num_rows",
            "prompt_token_min",
            "prompt_token_max",
            "prompt_token_mean",
            "mean_relative_length_mean",
            "length_vs_target_spearman",
            "shape_vs_target_spearman",
            "ensemble_vs_target_spearman",
            "last_layer_vs_target_spearman",
            "mean_newline_count",
            "mean_dollar_count",
            "mean_choice_count",
        ],
        bin_rows,
    )
    write_csv(
        out_dir / "test_sample_predictions.csv",
        [
            "dataset",
            "dataset_name",
            "sample_id",
            "prompt_token_count",
            "char_length",
            "newline_count",
            "digit_count",
            "dollar_count",
            "choice_count",
            "mean_relative_length",
            "p_loop",
            "p_cap",
            "prompt_length_prediction",
            "shape_linear_prediction",
            "shape_tree_prediction",
            "last_layer_prediction",
            "ensemble_prediction",
            "shape_tree_plus_ensemble_prediction",
            "length_residual",
            "shape_tree_residual",
            "ensemble_residual_against_shape_tree",
            "prompt_excerpt",
        ],
        sample_rows,
    )
    print(f"Wrote prompt-length mechanism audit to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
