from __future__ import annotations

from collections.abc import Iterable

LABEL_TARGET_CHOICES = ("eventual_loop", "loop_by_horizon")


def first_ngram_loop_prefix_length(
    token_ids: Iterable[int],
    *,
    n: int = 30,
    k: int = 20,
) -> int | None:
    token_ids = list(token_ids)
    if len(token_ids) < n:
        return None

    base = 1000003
    mod = 1 << 64
    mask = mod - 1

    pow_n = pow(base, n, mod)
    h = 0
    for t in token_ids[:n]:
        h = (h * base + (t + 1)) & mask

    counts = {h: 1}
    for i in range(n, len(token_ids)):
        out_t = token_ids[i - n] + 1
        in_t = token_ids[i] + 1
        h = (h * base + in_t - (out_t * pow_n)) & mask
        c = counts.get(h, 0) + 1
        counts[h] = c
        if c >= k:
            return i + 1

    return None


def has_ngram_loop(token_ids: Iterable[int], n: int = 30, k: int = 20) -> bool:
    return first_ngram_loop_prefix_length(token_ids, n=n, k=k) is not None


def label_from_rollout(
    token_ids: Iterable[int],
    *,
    loop_n: int,
    loop_k: int,
    label_target: str = "eventual_loop",
    label_horizon: int | None = None,
) -> int:
    if label_target not in LABEL_TARGET_CHOICES:
        raise ValueError(
            f"Unknown label_target '{label_target}'. Valid: {LABEL_TARGET_CHOICES}"
        )

    first_loop_prefix = first_ngram_loop_prefix_length(
        token_ids,
        n=loop_n,
        k=loop_k,
    )
    if label_target == "eventual_loop":
        return int(first_loop_prefix is not None)

    if label_horizon is None or label_horizon < 1:
        raise ValueError(
            "label_horizon must be a positive integer when "
            "label_target='loop_by_horizon'."
        )
    return int(first_loop_prefix is not None and first_loop_prefix <= label_horizon)


def labels_from_rollouts(
    rollout_token_ids: list[list[int]],
    *,
    loop_n: int,
    loop_k: int,
    label_target: str = "eventual_loop",
    label_horizon: int | None = None,
) -> list[int]:
    return [
        label_from_rollout(
            token_ids,
            loop_n=loop_n,
            loop_k=loop_k,
            label_target=label_target,
            label_horizon=label_horizon,
        )
        for token_ids in rollout_token_ids
    ]
