from dataclasses import asdict, dataclass, replace

from torch import nn

@dataclass(frozen=True)
class RolloutConfig:
    model_id: str
    temperature: float = 0.0
    max_tokens: int = 30000
    tp: int = 1
    dp: int = 1
    dtype: str = "bfloat16"
    max_model_len: int = 32768
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    trust_remote_code: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeConfig:
    probe_type: str = "linear"
    hidden_dim: int = 1024
    dropout: float = 0.0
    depth: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODEL_ROLLOUT_DEFAULTS: dict[str, RolloutConfig] = {
    "qwq_32b": RolloutConfig(
        model_id="Qwen/QwQ-32B",
        tp=8,
        dp=1,
        max_num_seqs=8,
    ),
    "openthinker3_7b": RolloutConfig(
        model_id="open-thoughts/OpenThinker3-7B",
        tp=1,
        dp=8,
        max_num_seqs=16,
    ),
    "openthinker3_1p5b": RolloutConfig(
        model_id="open-thoughts/OpenThinker3-1.5B",
        tp=1,
        dp=8,
        max_num_seqs=16,
    ),
}

PROBE_DEFAULTS: dict[str, ProbeConfig] = {
    "linear": ProbeConfig(
        probe_type="linear",
    ),
    "mlp": ProbeConfig(
        probe_type="mlp",
        hidden_dim=128,
        dropout=0.1,
        depth=1,
    ),
}

def preset_choices() -> list[str]:
    return sorted(MODEL_ROLLOUT_DEFAULTS.keys())


def probe_preset_choices() -> list[str]:
    return sorted(PROBE_DEFAULTS.keys())


def get_rollout_config(
    preset: str | None,
    *,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    tp: int | None = None,
    dp: int | None = None,
    dtype: str | None = None,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    trust_remote_code: bool | None = None,
) -> RolloutConfig:
    if preset:
        if preset not in MODEL_ROLLOUT_DEFAULTS:
            raise ValueError(
                f"Unknown model preset '{preset}'. Valid presets: {preset_choices()}"
            )
        cfg = MODEL_ROLLOUT_DEFAULTS[preset]
    else:
        if not model_id:
            raise ValueError("Either --model-preset or --model-id must be provided.")
        cfg = RolloutConfig(model_id=model_id)

    if model_id is not None:
        cfg = replace(cfg, model_id=model_id)
    if temperature is not None:
        cfg = replace(cfg, temperature=temperature)
    if max_tokens is not None:
        cfg = replace(cfg, max_tokens=max_tokens)
    if tp is not None:
        cfg = replace(cfg, tp=tp)
    if dp is not None:
        cfg = replace(cfg, dp=dp)
    if dtype is not None:
        cfg = replace(cfg, dtype=dtype)
    if max_model_len is not None:
        cfg = replace(cfg, max_model_len=max_model_len)
    if max_num_seqs is not None:
        cfg = replace(cfg, max_num_seqs=max_num_seqs)
    if max_num_batched_tokens is not None:
        cfg = replace(cfg, max_num_batched_tokens=max_num_batched_tokens)
    if trust_remote_code is not None:
        cfg = replace(cfg, trust_remote_code=trust_remote_code)

    return cfg


def get_probe_config(
    preset: str | None,
    *,
    hidden_dim: int | None = None,
    dropout: float | None = None,
    depth: int | None = None,
) -> ProbeConfig:
    key = preset or "linear"
    if key not in PROBE_DEFAULTS:
        raise ValueError(
            f"Unknown probe preset '{key}'. Valid presets: {probe_preset_choices()}"
        )
    cfg = PROBE_DEFAULTS[key]
    if hidden_dim is not None:
        cfg = replace(cfg, hidden_dim=hidden_dim)
    if dropout is not None:
        cfg = replace(cfg, dropout=dropout)
    if depth is not None:
        cfg = replace(cfg, depth=depth)
    return cfg


def build_probe_model(input_dim: int, probe_cfg: ProbeConfig) -> nn.Module:
    if input_dim < 1:
        raise ValueError("input_dim must be >= 1")

    if probe_cfg.probe_type == "linear":
        from .probes.linear_probe import LinearProbe

        return LinearProbe(input_dim=input_dim)
    if probe_cfg.probe_type == "mlp":
        if probe_cfg.hidden_dim < 1:
            raise ValueError("MLP hidden_dim must be >= 1")
        if probe_cfg.depth < 1:
            raise ValueError("MLP depth must be >= 1")
        if not 0.0 <= probe_cfg.dropout < 1.0:
            raise ValueError("MLP dropout must be in [0, 1)")

        from .probes.mlp_probe import MLPProbe

        return MLPProbe(
            input_dim=input_dim,
            hidden_dim=probe_cfg.hidden_dim,
            dropout=probe_cfg.dropout,
            depth=probe_cfg.depth,
        )

    raise ValueError(f"Unsupported probe_type '{probe_cfg.probe_type}'")
