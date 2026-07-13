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
    mode: str = "latency_hit"
    warmup_rounds: int = 1
    rounds: int = 100
    duration_seconds: float = 0
    stop_when: str = "either"
    seed: int = 42
    log_every_rounds: int = 1


@dataclass
class BandwidthConfig:
    paths: list[str] = field(default_factory=lambda: [
        "gpu_to_cpu_save",
        "cpu_to_gpu_load",
        "gpu_to_ssd_save_e2e",
        "ssd_to_gpu_reload_e2e",
    ])
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    target_payload_gb: float = 0
    min_duration_seconds: float = 30
    min_operations: int = 100
    window_seconds: float = 5
    validation_interval_operations: int = 100


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
    bandwidth: BandwidthConfig
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
        return self.workers_per_dp * self.model.dp_size

    @property
    def kv_tp_size(self) -> int:
        """Attention/KV TP width after SGLang carves CP out of composite TP."""
        return self.model.tp_size // self.model.cp_size

    @property
    def workers_per_dp(self) -> int:
        """Physical SGLang ranks per DP shard (TP and CP ranks overlap)."""
        return self.model.tp_size

    def worker_ranks(self, worker_rank: int) -> tuple[int, int]:
        """Return ``(kv_tp_rank, cp_rank)`` for a flattened physical rank."""
        return worker_rank % self.kv_tp_size, worker_rank // self.kv_tp_size

    @property
    def bytes_per_block(self) -> int:
        # CP is q-split and duplicates KV. Host blocks store only unique KV-TP slices.
        return self.bytes_per_gpu_block * self.kv_tp_size

    @property
    def bytes_per_gpu_block(self) -> int:
        return model_bytes_per_block(self.preset, self.tokens_per_block)

    @property
    def swa_bytes_per_block(self) -> int:
        if not self.features.swa:
            return 0
        return (
            self.tokens_per_block
            * self.preset.swa_num_layers
            * self.preset.swa_head_size
        )

    def validate(self) -> None:
        if self.run.mode not in {"bandwidth", "latency_hit"}:
            raise ValueError("run.mode must be 'bandwidth' or 'latency_hit'")
        if self.run.rounds <= 0 and self.run.duration_seconds <= 0:
            raise ValueError("Set run.rounds or run.duration_seconds")
        if self.run.warmup_rounds < 0:
            raise ValueError("run.warmup_rounds must be >= 0")
        if self.run.log_every_rounds < 1:
            raise ValueError("run.log_every_rounds must be >= 1")
        if self.run.stop_when not in {"either", "both"}:
            raise ValueError("run.stop_when must be 'either' or 'both'")
        if min(self.model.tp_size, self.model.dp_size, self.model.cp_size) < 1:
            raise ValueError("TP, DP and CP sizes must be >= 1")
        if self.model.tp_size % self.model.cp_size:
            raise ValueError(
                "model.tp_size is the composite SGLang TP world size per DP and "
                "must be divisible by model.cp_size"
            )
        if self.model.gpu_blocks_per_rank < 1:
            raise ValueError("model.gpu_blocks_per_rank must be >= 1")
        if self.tokens_per_block < 1:
            raise ValueError("tokens_per_block must be >= 1")
        if not 0 <= self.validation.sample_rate <= 1:
            raise ValueError("validation.sample_rate must be in [0, 1]")
        if not 0 <= self.validation.minimum_success_rate <= 1:
            raise ValueError("validation.minimum_success_rate must be in [0, 1]")
        if min(
            self.validation.min_samples_per_round,
            self.validation.sampled_layers_per_group,
            self.validation.sampled_blocks_per_request,
            self.validation.bytes_per_sample,
        ) < 0:
            raise ValueError("validation sample counts and bytes must be >= 0")
        if min(self.concurrency.max_inflight_per_dp, self.concurrency.batch_size) < 1:
            raise ValueError("concurrency batch_size and max_inflight_per_dp must be >= 1")
        if self.concurrency.request_timeout_seconds <= 0:
            raise ValueError("concurrency.request_timeout_seconds must be > 0")
        if self.output.resource_interval_seconds < 0:
            raise ValueError("output.resource_interval_seconds must be >= 0")
        valid_paths = {
            "gpu_to_cpu_save", "cpu_to_gpu_load",
            "gpu_to_ssd_save_e2e", "ssd_to_gpu_reload_e2e",
        }
        unknown_paths = set(self.bandwidth.paths) - valid_paths
        if unknown_paths:
            raise ValueError(f"Unknown bandwidth paths: {sorted(unknown_paths)}")
        if not self.bandwidth.concurrency_levels or any(
            value < 1 for value in self.bandwidth.concurrency_levels
        ):
            raise ValueError("bandwidth.concurrency_levels must contain positive integers")
        if min(
            self.bandwidth.target_payload_gb,
            self.bandwidth.min_duration_seconds,
            self.bandwidth.min_operations,
            self.bandwidth.window_seconds,
            self.bandwidth.validation_interval_operations,
        ) < 0:
            raise ValueError("bandwidth targets and window settings must be >= 0")
        if self.conversation.turns_min < 1 or self.conversation.turns_max < self.conversation.turns_min:
            raise ValueError("Invalid conversation turn range")
        if self.features.swa and not self.preset.swa_enabled:
            raise ValueError(f"Preset {self.preset.name} does not define an SWA pool")
        if self.features.ssd and self.cache.ssd_cache_gb <= 0 and self.cache.num_ssd_blocks <= 0:
            raise ValueError("SSD is enabled but neither ssd_cache_gb nor num_ssd_blocks is set")
        if self.features.ssd and not self.cache.ssd_cache_dir:
            raise ValueError("SSD is enabled but cache.ssd_cache_dir is empty")
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
        bandwidth=_section(BandwidthConfig, raw, "bandwidth"),
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
    result["derived_parallelism"] = {
        "kv_tp_size": config.kv_tp_size,
        "workers_per_dp": config.workers_per_dp,
        "required_gpus": config.required_gpus,
    }
    return copy.deepcopy(result)
