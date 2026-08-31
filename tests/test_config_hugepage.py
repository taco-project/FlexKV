from __future__ import annotations

from flexkv.common.config import (
    CacheConfig,
    ModelConfig,
    RankInfo,
    UserConfig,
    load_user_config_from_file,
    load_user_config_from_env,
    update_default_config_from_user_config,
)


def test_load_user_config_from_env_reads_hugepage_flags(monkeypatch) -> None:
    monkeypatch.setenv("FLEXKV_USE_HUGEPAGE_CPU_BUFFER", "1")
    monkeypatch.setenv("FLEXKV_USE_HUGEPAGE_TMP_BUFFER", "1")
    monkeypatch.setenv("FLEXKV_HUGEPAGE_SIZE_BYTES", str(1 << 30))
    monkeypatch.setenv("FLEXKV_MOONCAKE_MAX_MR_SIZE_BYTES", str(2 << 30))
    monkeypatch.setenv("FLEXKV_MOONCAKE_ALLOW_BLOCK_SPANNING_MRS", "0")
    monkeypatch.setenv("FLEXKV_MOONCAKE_ALLOW_UNALIGNED_BLOCK_MRS", "1")

    user_config = load_user_config_from_env()

    assert user_config.use_hugepage_cpu_buffer is True
    assert user_config.use_hugepage_tmp_buffer is True
    assert user_config.hugepage_size_bytes == 1 << 30
    assert user_config.mooncake_max_mr_size_bytes == 2 << 30
    assert user_config.mooncake_allow_block_spanning_mrs is False
    assert user_config.mooncake_allow_unaligned_block_mrs is True


def test_update_default_config_from_user_config_applies_hugepage_flags() -> None:
    model_config = ModelConfig(
        num_layers=1,
        num_kv_heads=1,
        head_size=128,
        kv_dim=2,
    )
    cache_config = CacheConfig()
    user_config = UserConfig(
        cpu_cache_gb=16,
        ssd_cache_gb=0,
        use_hugepage_cpu_buffer=True,
        use_hugepage_tmp_buffer=True,
        hugepage_size_bytes=1 << 30,
        mooncake_max_mr_size_bytes=2 << 30,
        mooncake_allow_block_spanning_mrs=False,
        mooncake_allow_unaligned_block_mrs=True,
    )

    update_default_config_from_user_config(RankInfo(model_config=model_config), cache_config, user_config)

    assert cache_config.use_hugepage_cpu_buffer is True
    assert cache_config.use_hugepage_tmp_buffer is True
    assert cache_config.hugepage_size_bytes == 1 << 30
    assert cache_config.mooncake_max_mr_size_bytes == 2 << 30
    assert cache_config.mooncake_allow_block_spanning_mrs is False
    assert cache_config.mooncake_allow_unaligned_block_mrs is True


def test_mooncake_max_mr_size_rejects_non_positive_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="must be positive"):
        UserConfig(mooncake_max_mr_size_bytes=0)


def test_mooncake_mr_split_modes_are_mutually_exclusive() -> None:
    import pytest

    with pytest.raises(ValueError, match="mutually exclusive"):
        UserConfig(
            mooncake_allow_block_spanning_mrs=True,
            mooncake_allow_unaligned_block_mrs=True,
        )


def test_load_user_config_from_yaml_reads_mooncake_max_mr_size(tmp_path) -> None:
    config_path = tmp_path / "flexkv.yaml"
    config_path.write_text(
        "cpu_cache_gb: 16\n"
        "mooncake_max_mr_size_bytes: 2147483648\n"
        "mooncake_allow_block_spanning_mrs: false\n"
        "mooncake_allow_unaligned_block_mrs: true\n",
        encoding="utf-8",
    )

    user_config = load_user_config_from_file(str(config_path))

    assert user_config.mooncake_max_mr_size_bytes == 2 << 30
    assert user_config.mooncake_allow_block_spanning_mrs is False
    assert user_config.mooncake_allow_unaligned_block_mrs is True
