"""Tests for SWA StorageEngine backing layout geometry."""

import pytest
import torch

from flexkv.common.config import CacheConfig, ModelConfig, SWAPoolConfig
from flexkv.common.transfer import DeviceType
from flexkv.storage.storage_engine import StorageEngine

pytestmark = pytest.mark.unit


def _model_config():
    return ModelConfig(
        num_layers=2,
        num_kv_heads=1,
        head_size=8,
        use_mla=True,
        dtype=torch.uint8,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        cp_size=1,
    )


def _capture_allocations(monkeypatch):
    allocations = []

    def fake_allocate(self, device_type, layout, dtype, device_id=0, raw_data=None, **kwargs):
        allocations.append({
            "device_type": device_type,
            "layout": layout,
            "dtype": dtype,
            "is_swa": bool(kwargs.get("is_swa", False)),
        })
        return True

    monkeypatch.setattr(StorageEngine, "allocate", fake_allocate)
    return allocations


def _swa_layout_by_device(allocations):
    return {
        item["device_type"]: item["layout"]
        for item in allocations
        if item["is_swa"]
    }


def test_swa_storage_layout_uses_tier_specific_slot_counts(monkeypatch, tmp_path):
    allocations = _capture_allocations(monkeypatch)
    cache_config = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=True,
        enable_3rd_remote=True,
        num_cpu_blocks=64,
        num_ssd_blocks=64,
        num_remote_blocks=64,
        ssd_cache_dir=str(tmp_path / "ssd"),
        remote_cache_path=str(tmp_path / "remote"),
        remote_config_custom={"test": True},
        swa=SWAPoolConfig(
            enabled=True,
            num_slots=3,
            num_ssd_slots=5,
            num_remote_slots=7,
            num_swa_layers=2,
            bytes_per_token_per_layer=8,
            pin_memory=False,
        ),
    )

    StorageEngine(_model_config(), cache_config, num_layers_per_pp_stage=2)

    layouts = _swa_layout_by_device(allocations)
    assert layouts[DeviceType.CPU].num_block == 3
    assert layouts[DeviceType.SSD].num_block == 5
    assert layouts[DeviceType.REMOTE].num_block == 7
    assert layouts[DeviceType.SSD].tokens_per_block == cache_config.tokens_per_block
    assert layouts[DeviceType.REMOTE].tokens_per_block == cache_config.tokens_per_block


def test_swa_storage_layout_skips_empty_optional_tiers(monkeypatch, tmp_path):
    allocations = _capture_allocations(monkeypatch)
    cache_config = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=True,
        enable_3rd_remote=True,
        num_cpu_blocks=64,
        num_ssd_blocks=64,
        num_remote_blocks=64,
        ssd_cache_dir=str(tmp_path / "ssd"),
        remote_cache_path=str(tmp_path / "remote"),
        remote_config_custom={"test": True},
        swa=SWAPoolConfig(
            enabled=True,
            num_slots=3,
            num_ssd_slots=0,
            num_remote_slots=0,
            num_swa_layers=2,
            bytes_per_token_per_layer=8,
            pin_memory=False,
        ),
    )

    StorageEngine(_model_config(), cache_config, num_layers_per_pp_stage=2)

    layouts = _swa_layout_by_device(allocations)
    assert DeviceType.CPU in layouts
    assert DeviceType.SSD not in layouts
    assert DeviceType.REMOTE not in layouts
