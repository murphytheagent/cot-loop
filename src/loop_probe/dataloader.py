import json
import os

import torch
from torch.utils.data import DataLoader, Dataset


def _available_feature_keys(manifest: dict[str, object]) -> list[str]:
    feature_views = manifest.get("feature_views")
    if not isinstance(feature_views, dict):
        return []
    return sorted(str(k) for k in feature_views.keys())


def resolve_feature_key(
    manifest: dict[str, object],
    feature_key: str | None,
) -> str | None:
    if feature_key is not None and feature_key != "":
        return feature_key
    default_key = manifest.get("default_feature_key")
    if isinstance(default_key, str) and default_key:
        return default_key
    return None


def resolve_split_info(
    manifest: dict[str, object],
    *,
    split: str,
    feature_key: str | None,
) -> tuple[dict[str, object], str | None]:
    resolved_feature_key = resolve_feature_key(manifest, feature_key)
    if resolved_feature_key is None:
        if split not in manifest:
            raise SystemExit(f"Split '{split}' not found in manifest.")
        split_info = manifest[split]
        if not isinstance(split_info, dict):
            raise SystemExit(f"Invalid split metadata for split '{split}'.")
        return split_info, None

    feature_views = manifest.get("feature_views")
    if not isinstance(feature_views, dict):
        if feature_key is not None and feature_key != "":
            raise SystemExit(
                "Requested --feature-key but manifest has no feature_views section."
            )
        if split not in manifest:
            raise SystemExit(f"Split '{split}' not found in manifest.")
        split_info = manifest[split]
        if not isinstance(split_info, dict):
            raise SystemExit(f"Invalid split metadata for split '{split}'.")
        return split_info, None

    view_info = feature_views.get(resolved_feature_key)
    if not isinstance(view_info, dict):
        available = ", ".join(_available_feature_keys(manifest))
        raise SystemExit(
            f"Feature key '{resolved_feature_key}' not found in manifest. "
            f"Available keys: [{available}]"
        )

    split_info = view_info.get(split)
    if not isinstance(split_info, dict):
        raise SystemExit(
            f"Split '{split}' not found for feature key '{resolved_feature_key}'."
        )
    return split_info, resolved_feature_key


def resolve_input_dim(manifest: dict[str, object], feature_key: str | None) -> int:
    resolved_feature_key = resolve_feature_key(manifest, feature_key)
    if resolved_feature_key is not None:
        feature_views = manifest.get("feature_views")
        if isinstance(feature_views, dict):
            view_info = feature_views.get(resolved_feature_key)
            if not isinstance(view_info, dict):
                available = ", ".join(_available_feature_keys(manifest))
                raise SystemExit(
                    f"Feature key '{resolved_feature_key}' not found in manifest. "
                    f"Available keys: [{available}]"
                )
            input_dim = view_info.get("input_dim")
            if input_dim is None:
                raise SystemExit(
                    f"Feature key '{resolved_feature_key}' is missing input_dim."
                )
            return int(input_dim)

    input_dim = manifest.get("input_dim")
    if input_dim is None:
        raise SystemExit("Manifest missing required 'input_dim'.")
    return int(input_dim)


class ActivationDataset(Dataset):
    def __init__(self, data_dir: str, split: str, feature_key: str | None = None):
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        split_info, resolved_feature_key = resolve_split_info(
            manifest,
            split=split,
            feature_key=feature_key,
        )
        shard_paths = split_info.get("shards", [])
        if not shard_paths:
            raise SystemExit(f"No shard files listed for split '{split}'.")

        xs = []
        ys = []
        sample_ids = []
        for rel_path in shard_paths:
            full_path = os.path.join(data_dir, rel_path)
            shard = torch.load(full_path, map_location="cpu")
            xs.append(shard["x"].to(dtype=torch.float32))
            ys.append(shard["y"].to(dtype=torch.float32))
            sample_ids.append(shard["sample_ids"].to(dtype=torch.int64))

        self.x = torch.cat(xs, dim=0)
        self.y = torch.cat(ys, dim=0)
        self.sample_ids = torch.cat(sample_ids, dim=0)
        self.feature_key = resolved_feature_key
        if self.x.size(0) != self.y.size(0):
            raise SystemExit(
                f"Split '{split}' has mismatched x/y lengths: "
                f"{self.x.size(0)} vs {self.y.size(0)}"
            )

    def __len__(self) -> int:
        return int(self.x.size(0))

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def make_dataloader(
    data_dir: str,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    feature_key: str | None = None,
) -> DataLoader:
    ds = ActivationDataset(data_dir=data_dir, split=split, feature_key=feature_key)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def read_manifest(data_dir: str) -> dict[str, object]:
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
