from __future__ import annotations

from flexkv.common.config import (
    CacheConfig,
    ModelConfig,
    UserConfig,
    load_user_config_from_env,
    update_default_config_from_user_config,
)


def test_load_user_config_from_env_reads_hugepage_flags(monkeypatch) -> None:
    monkeypatch.setenv("FLEXKV_USE_HUGEPAGE_CPU_BUFFER", "1")
    monkeypatch.setenv("FLEXKV_USE_HUGEPAGE_TMP_BUFFER", "1")
    monkeypatch.setenv("FLEXKV_HUGEPAGE_SIZE_BYTES", str(1 << 30))

    user_config = load_user_config_from_env()

    assert user_config.use_hugepage_cpu_buffer is True
    assert user_config.use_hugepage_tmp_buffer is True
    assert user_config.hugepage_size_bytes == 1 << 30


def test_update_default_config_from_user_config_applies_hugepage_flags() -> None:
    model_config = ModelConfig(
        num_layers=1,
        num_kv_heads=1,
        head_size=128,
        use_mla=False,
    )
    cache_config = CacheConfig()
    user_config = UserConfig(
        cpu_cache_gb=16,
        ssd_cache_gb=0,
        use_hugepage_cpu_buffer=True,
        use_hugepage_tmp_buffer=True,
        hugepage_size_bytes=1 << 30,
    )

    update_default_config_from_user_config(model_config, cache_config, user_config)

    assert cache_config.use_hugepage_cpu_buffer is True
    assert cache_config.use_hugepage_tmp_buffer is True
    assert cache_config.hugepage_size_bytes == 1 << 30
