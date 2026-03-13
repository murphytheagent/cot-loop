import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def evaluate_binary_metrics(y_true: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )
    except Exception as exc:
        raise SystemExit(
            "scikit-learn is required to compute evaluation metrics. "
            "Install dependencies with `uv sync`."
        ) from exc

    y_true_np = y_true.detach().cpu().numpy().astype(int)
    probs_np = torch.sigmoid(logits).detach().cpu().numpy()
    pred_np = (probs_np >= 0.5).astype(int)
    prevalence = float(np.mean(y_true_np == 1))

    acc = float(accuracy_score(y_true_np, pred_np))
    f1 = float(f1_score(y_true_np, pred_np, average="macro", zero_division=0))
    pos_precision, pos_recall, pos_f1, _ = precision_recall_fscore_support(
        y_true_np,
        pred_np,
        labels=[1],
        average=None,
        zero_division=0,
    )

    unique_labels = np.unique(y_true_np)
    if unique_labels.size < 2:
        auc = float("nan")
        pr_auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true_np, probs_np))
        pr_auc = float(average_precision_score(y_true_np, probs_np))

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "positive_precision": float(pos_precision[0]),
        "positive_recall": float(pos_recall[0]),
        "positive_f1": float(pos_f1[0]),
        "prevalence": prevalence,
        "roc_auc": auc,
        "pr_auc": pr_auc,
    }
