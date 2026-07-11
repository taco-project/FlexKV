from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy
import os

import yaml

from .models import LayerGroup, ModelPreset, get_preset, model_bytes_per_block, update_from_hf_config


@dataclass
class RunConfig:
    warmup_rounds: int = 1
    rounds: int = 100
    duration_seconds: float = 0
    stop_when: str = "either"
    seed: int = 42
    log_every_rounds: int = 1


@dataclass
class FeatureConfig:
    cpu_stub: bool = False
    layerwise: bool = False
    swa: bool = False
    ssd: bool = False
    concurrency: bool = False
    async_api_probe_interval_rounds: int = 10


@dataclass
class ConversationConfig:
    conversations_per_round: int = 10
    turns_min: int = 2
    turns_max: int = 5
    system_prompt_blocks: int = 8
    first_input_blocks: Any = field(default_factory=lambda: [2, 8])
    added_input_blocks: Any = field(default_factory=lambda: [1, 4])
    output_blocks: Any = field(default_factory=lambda: [1, 2])
    partial_block_tokens: int = 0
    shared_system_prompt: bool = True
    read_after_put: bool = True


@dataclass
class ConcurrencyConfig:
    max_inflight_per_dp: int = 16
    batch_size: int = 4
    target_qps: float = 0
    request_timeout_seconds: float = 30


@dataclass
class ValidationConfig:
    sample_rate: float = 0.001
    min_samples_per_round: int = 1
    sampled_layers_per_group: int = 2
    sampled_blocks_per_request: int = 2
    bytes_per_sample: int = 256
    hit_tolerance_blocks: int = 0
    stop_on_mismatch: bool = True
    minimum_success_rate: float = 0.999


@dataclass
class CacheBenchConfig:
    cpu_cache_gb: float = 8
    ssd_cache_gb: float = 0
    ssd_cache_dir: list[str] = field(default_factory=lambda: ["./flexkv_stress_ssd"])
    num_cpu_blocks: int = 0
    num_ssd_blocks: int = 0
    enable_gds: bool = False
    eviction_policy: str = "lru"
    swa_num_slots: int = 1024
    swa_ssd_slots: int = 0
    force_ssd_reload_interval_rounds: int = 10


@dataclass
class ModelBenchConfig:
    preset: str = "glm5"
    hf_config_path: str | None = None
    tp_size: int = 1
    dp_size: int = 1
    cp_size: int = 1
    gpu_blocks_per_rank: int = 1024
    tokens_per_block: int | None = None
    dtype: str | None = None
    layer_groups: list[dict[str, Any]] | None = None


@dataclass
class OutputConfig:
    directory: str = "./benchmark_results/flexkv_stress"
    resource_interval_seconds: float = 10


@dataclass
class StressConfig:
    run: RunConfig
    model: ModelBenchConfig
    cache: CacheBenchConfig
    features: FeatureConfig
    conversation: ConversationConfig
    concurrency: ConcurrencyConfig
    validation: ValidationConfig
    output: OutputConfig
    preset: ModelPreset
    source_path: Path

    @property
    def tokens_per_block(self) -> int:
        return self.model.tokens_per_block or self.preset.tokens_per_block

    @property
    def required_gpus(self) -> int:
        return self.model.tp_size * self.model.dp_size * self.model.cp_size

    @property
    def bytes_per_block(self) -> int:
        # FlexKV's heterogeneous host block stores all TP slices.
        return self.bytes_per_gpu_block * self.model.tp_size

    @property
    def bytes_per_gpu_block(self) -> int:
        return model_bytes_per_block(self.preset, self.tokens_per_block)

    def validate(self) -> None:
        if self.run.rounds <= 0 and self.run.duration_seconds <= 0:
            raise ValueError("Set run.rounds or run.duration_seconds")
        if self.run.stop_when not in {"either", "both"}:
            raise ValueError("run.stop_when must be 'either' or 'both'")
        if min(self.model.tp_size, self.model.dp_size, self.model.cp_size) < 1:
            raise ValueError("TP, DP and CP sizes must be >= 1")
        if self.model.gpu_blocks_per_rank < 1:
            raise ValueError("model.gpu_blocks_per_rank must be >= 1")
        if not 0 <= self.validation.sample_rate <= 1:
            raise ValueError("validation.sample_rate must be in [0, 1]")
        if self.conversation.turns_min < 1 or self.conversation.turns_max < self.conversation.turns_min:
            raise ValueError("Invalid conversation turn range")
        if self.features.swa and not self.preset.swa_enabled:
            raise ValueError(f"Preset {self.preset.name} does not define an SWA pool")
        if self.features.ssd and self.cache.ssd_cache_gb <= 0 and self.cache.num_ssd_blocks <= 0:
            raise ValueError("SSD is enabled but neither ssd_cache_gb nor num_ssd_blocks is set")
        if self.features.cpu_stub and self.features.ssd:
            raise ValueError("CPU stub mode does not emulate the FlexKV SSD tier")
        for group in self.preset.groups:
            if any(layer < 0 or layer >= self.preset.num_layers for layer in group.layer_indices):
                raise ValueError(f"Layer group {group.name} contains an invalid layer index")
            if self.tokens_per_block % group.compress_ratio:
                raise ValueError(
                    f"tokens_per_block={self.tokens_per_block} must be divisible by "
                    f"{group.name}.compress_ratio={group.compress_ratio}"
                )


def _section(cls, raw: dict[str, Any], name: str):
    return cls(**(raw.get(name) or {}))


def _parse_layers(value: Any, num_layers: int, metadata: dict[str, Any]) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(num_layers))
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, dict) and "compress_ratio" in value:
        ratios = metadata.get("compress_ratios") or []
        return tuple(i for i, ratio in enumerate(ratios) if ratio == int(value["compress_ratio"]))
    if isinstance(value, dict) and "indexer_type" in value:
        types = metadata.get("indexer_types") or []
        return tuple(i for i, kind in enumerate(types) if kind == value["indexer_type"])
    raise ValueError(f"Unsupported layer selector: {value!r}")


def _override_groups(preset: ModelPreset, groups: list[dict[str, Any]]) -> ModelPreset:
    parsed = []
    metadata = preset.metadata or {}
    for raw in groups:
        parsed.append(LayerGroup(
            name=str(raw["name"]),
            layer_indices=_parse_layers(raw.get("layers", "all"), preset.num_layers, metadata),
            num_kv_heads=int(raw.get("num_kv_heads", 1)),
            head_size=int(raw["head_size"]),
            dtype=str(raw.get("dtype", preset.dtype)),
            compress_ratio=int(raw.get("compress_ratio", 1)),
            sliding_window=raw.get("sliding_window"),
        ))
    from dataclasses import replace
    return replace(preset, groups=tuple(parsed))


def load_config(path: str | os.PathLike[str]) -> StressConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text()) or {}
    model = _section(ModelBenchConfig, raw, "model")
    preset = get_preset(model.preset)
    if model.hf_config_path:
        hf_path = Path(model.hf_config_path)
        if not hf_path.is_absolute():
            hf_path = source.parent / hf_path
        preset = update_from_hf_config(preset, hf_path)
    if model.dtype:
        from dataclasses import replace
        groups = tuple(
            replace(group, dtype=model.dtype) if group.dtype == preset.dtype else group
            for group in preset.groups
        )
        preset = replace(preset, dtype=model.dtype, groups=groups)
    if model.layer_groups:
        preset = _override_groups(preset, model.layer_groups)
    config = StressConfig(
        run=_section(RunConfig, raw, "run"),
        model=model,
        cache=_section(CacheBenchConfig, raw, "cache"),
        features=_section(FeatureConfig, raw, "features"),
        conversation=_section(ConversationConfig, raw, "conversation"),
        concurrency=_section(ConcurrencyConfig, raw, "concurrency"),
        validation=_section(ValidationConfig, raw, "validation"),
        output=_section(OutputConfig, raw, "output"),
        preset=preset,
        source_path=source,
    )
    if isinstance(config.cache.ssd_cache_dir, str):
        config.cache.ssd_cache_dir = [
            value.strip() for value in config.cache.ssd_cache_dir.split(";") if value.strip()
        ]
    config.validate()
    return config


def config_as_dict(config: StressConfig) -> dict[str, Any]:
    from dataclasses import asdict
    result = asdict(config)
    result["source_path"] = str(config.source_path)
    return copy.deepcopy(result)
