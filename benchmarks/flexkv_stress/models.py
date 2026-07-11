from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import json


DSV4_PRO_RATIOS = [
    128, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 0,
]

DSV4_FLASH_RATIOS = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0,
]

GLM52_INDEXER_TYPES = [
    "full", "full", "full",
    *[kind for _ in range(18) for kind in ("shared", "shared", "shared", "full")],
    "shared", "shared", "shared",
]


@dataclass(frozen=True)
class LayerGroup:
    name: str
    layer_indices: tuple[int, ...]
    num_kv_heads: int
    head_size: int
    dtype: str
    compress_ratio: int = 1
    sliding_window: int | None = None

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)


@dataclass(frozen=True)
class ModelPreset:
    name: str
    architecture: str
    num_layers: int
    num_kv_heads: int
    head_size: int
    dtype: str
    use_mla: bool
    tokens_per_block: int
    groups: tuple[LayerGroup, ...]
    swa_enabled: bool = False
    swa_head_size: int = 0
    swa_num_layers: int = 0
    metadata: dict[str, Any] | None = None


def _dsv4(name: str, ratios: list[int], num_layers: int | None = None) -> ModelPreset:
    layer_count = num_layers or len(ratios)
    c4 = tuple(i for i, ratio in enumerate(ratios[:layer_count]) if ratio == 4)
    c128 = tuple(i for i, ratio in enumerate(ratios[:layer_count]) if ratio == 128)
    groups = (
        LayerGroup("c4", c4, 1, 585, "uint8", compress_ratio=4),
        LayerGroup("c128", c128, 1, 864, "uint8", compress_ratio=128),
        LayerGroup("c4_indexer", c4, 1, 132, "uint8", compress_ratio=4),
    )
    return ModelPreset(
        name=name,
        architecture="DeepseekV4ForCausalLM",
        num_layers=layer_count,
        num_kv_heads=1,
        head_size=585,
        dtype="uint8",
        use_mla=True,
        tokens_per_block=256,
        groups=groups,
        swa_enabled=True,
        swa_head_size=585,
        swa_num_layers=layer_count,
        metadata={"compress_ratios": list(ratios), "logical_swa_window": 128},
    )


def _glm(name: str, indexer_types: list[str] | None = None) -> ModelPreset:
    num_layers = 78
    groups: list[LayerGroup] = [
        LayerGroup("main", tuple(range(num_layers)), 1, 576, "bfloat16"),
    ]
    if indexer_types is None:
        groups.append(LayerGroup("indexer", tuple(range(num_layers)), 1, 132, "uint8"))
    else:
        if len(indexer_types) != num_layers:
            raise ValueError(f"{name}: expected {num_layers} indexer types, got {len(indexer_types)}")
        for kind in ("full", "shared"):
            layers = tuple(i for i, value in enumerate(indexer_types) if value == kind)
            groups.append(LayerGroup(f"indexer_{kind}", layers, 1, 132, "uint8"))
    return ModelPreset(
        name=name,
        architecture="GlmMoeDsaForCausalLM",
        num_layers=num_layers,
        num_kv_heads=1,
        head_size=576,
        dtype="bfloat16",
        use_mla=True,
        tokens_per_block=64,
        groups=tuple(groups),
        metadata={
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "index_head_dim": 128,
            "indexer_types": indexer_types,
        },
    )


PRESETS = {
    "dsv4": _dsv4("dsv4_pro", DSV4_PRO_RATIOS),
    "dsv4_pro": _dsv4("dsv4_pro", DSV4_PRO_RATIOS),
    "dsv4_flash": _dsv4("dsv4_flash", DSV4_FLASH_RATIOS, num_layers=43),
    "glm5": _glm("glm5"),
    "glm5_2": _glm("glm5_2", GLM52_INDEXER_TYPES),
    "glm5.2": _glm("glm5_2", GLM52_INDEXER_TYPES),
}


def get_preset(name: str) -> ModelPreset:
    try:
        return PRESETS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model preset {name!r}; choose from {sorted(PRESETS)}") from exc


def _groups_from_hf(data: dict[str, Any], fallback: ModelPreset) -> tuple[LayerGroup, ...]:
    architecture = (data.get("architectures") or [fallback.architecture])[0]
    if architecture == "DeepseekV4ForCausalLM":
        ratios = data.get("compress_ratios")
        if not ratios:
            raise ValueError("DeepSeek-V4 config.json is missing compress_ratios")
        return _dsv4(fallback.name, [int(v) for v in ratios]).groups
    if architecture == "GlmMoeDsaForCausalLM":
        types = data.get("indexer_types")
        return _glm(fallback.name, list(types) if types else None).groups
    return fallback.groups


def update_from_hf_config(preset: ModelPreset, config_path: str | Path) -> ModelPreset:
    data = json.loads(Path(config_path).read_text())
    architecture = (data.get("architectures") or [preset.architecture])[0]
    groups = _groups_from_hf(data, preset)
    num_layers = int(data.get("num_hidden_layers", preset.num_layers))
    return replace(
        preset,
        architecture=architecture,
        num_layers=num_layers,
        groups=groups,
        metadata={**(preset.metadata or {}), **data},
    )


def group_bytes_per_block(group: LayerGroup, tokens_per_block: int, kv_dim: int) -> int:
    if tokens_per_block % group.compress_ratio:
        raise ValueError(
            f"tokens_per_block={tokens_per_block} is not divisible by "
            f"{group.name}.compress_ratio={group.compress_ratio}"
        )
    dtype_bytes = {"uint8": 1, "int8": 1, "float16": 2, "bfloat16": 2, "float32": 4}
    try:
        itemsize = dtype_bytes[group.dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported preset dtype {group.dtype!r}") from exc
    return (
        group.num_layers
        * kv_dim
        * (tokens_per_block // group.compress_ratio)
        * group.num_kv_heads
        * group.head_size
        * itemsize
    )


def model_bytes_per_block(preset: ModelPreset, tokens_per_block: int | None = None) -> int:
    tokens = tokens_per_block or preset.tokens_per_block
    kv_dim = 1 if preset.use_mla else 2
    return sum(group_bytes_per_block(group, tokens, kv_dim) for group in preset.groups)
