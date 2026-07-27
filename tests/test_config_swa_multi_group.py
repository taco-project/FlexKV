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
