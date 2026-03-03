import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .types import SampleRecord

FEATURE_POOLING_CHOICES = ("last_token", "mean_pool")


def select_prefill_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _sdp_kernel_context():
    if torch.cuda.is_available():
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        )
    return contextlib.nullcontext()


def load_prefill_model_and_tokenizer(model_id: str, trust_remote_code: bool):
    dtype = select_prefill_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            **kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            **kwargs,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def _resolve_feature_layer(
    *,
    num_hidden_layers: int,
    feature_layer: int,
) -> int:
    if num_hidden_layers < 1:
        raise RuntimeError("Model returned no transformer hidden layers.")
    if feature_layer >= 0:
        resolved = feature_layer
    else:
        resolved = num_hidden_layers + feature_layer
    if resolved < 0 or resolved >= num_hidden_layers:
        raise SystemExit(
            "--feature-layer is out of range for this model. "
            f"got {feature_layer}, valid=[-{num_hidden_layers}, {num_hidden_layers - 1}]"
        )
    return resolved


def _last_token_idx(
    *,
    attention_mask: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.full(
            (batch_size,),
            seq_len - 1,
            device=device,
            dtype=torch.long,
        )

    token_positions = torch.arange(seq_len, device=device)
    token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)
    masked_positions = token_positions.masked_fill(attention_mask == 0, -1)
    last_token_idx = masked_positions.max(dim=1).values
    if torch.any(last_token_idx < 0):
        raise RuntimeError("Found an empty prompt after tokenization.")
    return last_token_idx


def _pool_hidden_states(
    hidden: torch.Tensor,
    *,
    pooling: str,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    if pooling == "last_token":
        last_token_idx = _last_token_idx(
            attention_mask=attention_mask,
            batch_size=hidden.size(0),
            seq_len=hidden.size(1),
            device=hidden.device,
        )
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_token_idx].float().cpu()

    if pooling == "mean_pool":
        if attention_mask is None:
            return hidden.mean(dim=1).float().cpu()
        mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
        token_count = mask.sum(dim=1).clamp(min=1.0)
        return ((hidden * mask).sum(dim=1) / token_count).float().cpu()

    raise SystemExit(
        f"Unknown feature pooling '{pooling}'. "
        f"Valid: {FEATURE_POOLING_CHOICES}"
    )


def extract_prefill_features_multi(
    model,
    tokenizer,
    device: torch.device,
    records: list[SampleRecord],
    *,
    feature_views: dict[str, tuple[str, int]],
    log_prefix: str,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    total = len(records)
    if total == 0:
        raise SystemExit(f"No records found for split '{log_prefix}'.")
    if batch_size < 1:
        raise SystemExit("--prefill-batch-size must be >= 1.")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise SystemExit(
                "Tokenizer has no pad_token/eos_token; cannot batch prefill prompts."
            )
        tokenizer.pad_token = tokenizer.eos_token
    if not feature_views:
        raise SystemExit("At least one feature view must be configured.")

    normalized_views: dict[str, tuple[str, int]] = {}
    for key, value in feature_views.items():
        pooling, feature_layer = value
        if pooling not in FEATURE_POOLING_CHOICES:
            raise SystemExit(
                f"Unknown feature pooling '{pooling}' for view '{key}'. "
                f"Valid: {FEATURE_POOLING_CHOICES}"
            )
        normalized_views[key] = (pooling, int(feature_layer))

    features_by_key: dict[str, list[torch.Tensor]] = {
        key: [] for key in normalized_views
    }
    resolved_view_specs: dict[str, tuple[str, int]] | None = None

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_records = records[start:end]
            encoded = tokenizer(
                [rec.prompt for rec in batch_records],
                return_tensors="pt",
                padding=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with _sdp_kernel_context():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            if out.hidden_states is None:
                raise RuntimeError("Model did not return hidden states during prefill.")

            if resolved_view_specs is None:
                num_hidden_layers = len(out.hidden_states) - 1
                resolved_view_specs = {}
                for key, (pooling, feature_layer) in normalized_views.items():
                    resolved_view_specs[key] = (
                        pooling,
                        _resolve_feature_layer(
                            num_hidden_layers=num_hidden_layers,
                            feature_layer=feature_layer,
                        ),
                    )

            for key, (pooling, resolved_layer) in resolved_view_specs.items():
                hidden = out.hidden_states[resolved_layer + 1]
                batch_vecs = _pool_hidden_states(
                    hidden,
                    pooling=pooling,
                    attention_mask=attention_mask,
                )
                features_by_key[key].extend(batch_vecs.unbind(dim=0))

            if end == total or start == 0 or end % 50 == 0:
                print(f"[{log_prefix}] prefill {end}/{total}", flush=True)

    return {
        key: torch.stack(view_features, dim=0)
        for key, view_features in features_by_key.items()
    }


def extract_prefill_features(
    model,
    tokenizer,
    device: torch.device,
    records: list[SampleRecord],
    *,
    log_prefix: str,
    batch_size: int = 1,
    feature_pooling: str = "last_token",
    feature_layer: int = -1,
) -> torch.Tensor:
    single_key = "__single_view__"
    features_by_key = extract_prefill_features_multi(
        model,
        tokenizer,
        device,
        records,
        feature_views={single_key: (feature_pooling, feature_layer)},
        log_prefix=log_prefix,
        batch_size=batch_size,
    )
    return features_by_key[single_key]
