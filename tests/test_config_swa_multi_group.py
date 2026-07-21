from __future__ import annotations

import pytest

from flexkv.common.config import UserConfig, load_user_config_from_env


def test_swa_multi_group_defaults_to_auto_enabled(monkeypatch) -> None:
    monkeypatch.delenv("FLEXKV_SWA_MULTI_GROUP", raising=False)

    config = load_user_config_from_env()

    assert config.swa_multi_group is None
    assert config.swa_multi_group is not False


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("0", False), ("1", True)],
)
def test_swa_multi_group_env_override(
    monkeypatch, raw_value: str, expected: bool
) -> None:
    monkeypatch.setenv("FLEXKV_SWA_MULTI_GROUP", raw_value)

    config = load_user_config_from_env()

    assert config.swa_multi_group is expected


def test_swa_multi_group_rejects_non_boolean_config_value() -> None:
    with pytest.raises(ValueError, match="swa_multi_group must be a boolean"):
        UserConfig(swa_multi_group="false")


def test_swa_multi_layer_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("FLEXKV_SWA_MULTI_LAYER", raising=False)

    config = load_user_config_from_env()

    assert config.swa_multi_layer is True


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("0", False), ("1", True)],
)
def test_swa_multi_layer_env_override(
    monkeypatch, raw_value: str, expected: bool
) -> None:
    monkeypatch.setenv("FLEXKV_SWA_MULTI_LAYER", raw_value)

    config = load_user_config_from_env()

    assert config.swa_multi_layer is expected


def test_swa_multi_layer_rejects_non_boolean_config_value() -> None:
    with pytest.raises(ValueError, match="swa_multi_layer must be a boolean"):
        UserConfig(swa_multi_layer="false")
