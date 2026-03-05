#!/usr/bin/env python3
"""Aggregate probe metrics across multiple seed runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics


SUMMARY_METRICS = (
    "accuracy",
    "macro_f1",
    "roc_auc",
    "pr_auc",
    "positive_precision",
    "positive_recall",
    "positive_f1",
    "prevalence",
)
DEFAULT_SELECTION = ("roc_auc", "macro_f1")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "pr_auc", "macro_f1", "positive_f1", "accuracy"),
        default="roc_auc",
        help="Metric used to select best checkpoint row per run.",
    )
    parser.add_argument(
        "--tie-breaker",
        choices=("roc_auc", "pr_auc", "macro_f1", "positive_f1", "accuracy"),
        default="macro_f1",
        help="Secondary metric used when selection metric ties.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional JSON output path for per-run best rows and aggregate stats.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional CSV output path for aggregate mean/std.",
    )
    return parser.parse_args()


def _rank_value(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(v):
        return float("-inf")
    return v


def _as_float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _infer_selection_pair(row: dict[str, object]) -> tuple[str, str] | None:
    selection_metric = row.get("selection_metric")
    tie_breaker = row.get("tie_breaker")
    if isinstance(selection_metric, str) and isinstance(tie_breaker, str):
        return selection_metric, tie_breaker

    selection_rule = row.get("selection_rule")
    if not isinstance(selection_rule, str):
        return None

    parts = re.findall(r"max\(([^)]+)\)", selection_rule)
    if len(parts) >= 2:
        return parts[0], parts[1]
    return None


def _best_row_from_jsonl(
    path: str,
    *,
    selection_metric: str,
    tie_breaker: str,
) -> dict[str, object]:
    best_row: dict[str, object] | None = None
    best_key = (float("-inf"), float("-inf"))

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_num}") from exc

            rank_key = (
                _rank_value(row.get(selection_metric)),
                _rank_value(row.get(tie_breaker)),
            )
            if rank_key > best_key:
                best_key = rank_key
                best_row = row

    if best_row is None:
        raise SystemExit(f"No metric rows found in {path}")
    return best_row


def _infer_seed(run_dir: str, row: dict[str, object]) -> int | None:
    if "seed" in row:
        try:
            return int(row["seed"])
        except Exception:
            pass

    base = os.path.basename(os.path.normpath(run_dir))
    if base.startswith("seed_"):
        suffix = base[len("seed_") :]
        if suffix.lstrip("-").isdigit():
            return int(suffix)
    return None


def _format_row(run_dir: str, row: dict[str, object]) -> dict[str, object]:
    return {
        "run_dir": run_dir,
        "seed": _infer_seed(run_dir, row),
        "epoch": row.get("epoch"),
        "step": row.get("step"),
        "accuracy": _as_float_or_nan(row.get("accuracy")),
        "macro_f1": _as_float_or_nan(row.get("macro_f1")),
        "roc_auc": _as_float_or_nan(row.get("roc_auc")),
        "pr_auc": _as_float_or_nan(row.get("pr_auc")),
        "positive_precision": _as_float_or_nan(row.get("positive_precision")),
        "positive_recall": _as_float_or_nan(row.get("positive_recall")),
        "positive_f1": _as_float_or_nan(row.get("positive_f1")),
        "prevalence": _as_float_or_nan(row.get("prevalence")),
    }


def _load_best_row(
    run_dir: str,
    *,
    selection_metric: str,
    tie_breaker: str,
) -> dict[str, object]:
    best_metrics_path = os.path.join(run_dir, "best_metrics.json")
    metrics_jsonl = os.path.join(run_dir, "metrics.jsonl")
    best_row: dict[str, object] | None = None
    best_selection: tuple[str, str] | None = None

    if os.path.exists(best_metrics_path):
        with open(best_metrics_path, "r", encoding="utf-8") as f:
            best_row = json.load(f)
        best_selection = _infer_selection_pair(best_row)

    if os.path.exists(metrics_jsonl):
        if best_row is None or best_selection != (selection_metric, tie_breaker):
            row = _best_row_from_jsonl(
                metrics_jsonl,
                selection_metric=selection_metric,
                tie_breaker=tie_breaker,
            )
            return _format_row(run_dir, row)
        return _format_row(run_dir, best_row)

    if best_row is not None:
        return _format_row(run_dir, best_row)

    raise SystemExit(f"Missing both best_metrics.json and metrics.jsonl under {run_dir}")


def _aggregate(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    vals = [
        float(row[metric])
        for row in rows
        if not math.isnan(float(row[metric]))
    ]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(vals)}


def _sanitize_json(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json(v) for v in value]
    return value


def _write_summary_csv(path: str, summary: dict[str, dict[str, object]]) -> None:
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
                    "n": stats["n"],
                }
            )


def _format_metric(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def main() -> None:
    args = _parse_args()

    rows = [
        _load_best_row(
            run_dir,
            selection_metric=args.selection_metric,
            tie_breaker=args.tie_breaker,
        )
        for run_dir in args.run_dirs
    ]

    summary = {
        metric: _aggregate(rows, metric)
        for metric in SUMMARY_METRICS
    }
    payload = {
        "selection_metric": args.selection_metric,
        "tie_breaker": args.tie_breaker,
        "num_runs": len(rows),
        "runs": rows,
        "aggregate": summary,
    }

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(_sanitize_json(payload), f, indent=2, sort_keys=True)
            f.write("\n")

    if args.out_csv:
        _write_summary_csv(args.out_csv, summary)

    print(
        f"Aggregated {len(rows)} run(s) using selection={args.selection_metric} "
        f"tie_breaker={args.tie_breaker}",
        flush=True,
    )
    for metric in SUMMARY_METRICS:
        stats = summary[metric]
        mean_val = float(stats["mean"])
        std_val = float(stats["std"])
        n = int(stats["n"])
        print(
            f"{metric}: mean={_format_metric(mean_val)} std={_format_metric(std_val)} n={n}",
            flush=True,
        )


if __name__ == "__main__":
    main()
