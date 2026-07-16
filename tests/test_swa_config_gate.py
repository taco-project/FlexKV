import ast
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit
CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "flexkv/integration/config.py"
)


def _resolver():
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_dsv4_swa_transfer_enabled_from_env"
    )
    namespace = {"os": os}
    module = ast.Module([function], type_ignores=[])
    exec(compile(module, CONFIG_PATH, "exec"), namespace)
    return namespace["_dsv4_swa_transfer_enabled_from_env"]


def test_dsv4_swa_transfer_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FLEXKV_ENABLE_SWA_TRANSFER", raising=False)
    assert _resolver()() is True


def test_dsv4_swa_transfer_explicit_zero_stays_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEXKV_ENABLE_SWA_TRANSFER", "0")
    assert _resolver()() is False


def test_dsv4_swa_transfer_explicit_one_stays_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FLEXKV_ENABLE_SWA_TRANSFER", "1")
    assert _resolver()() is True


def test_dsv4_config_does_not_force_the_data_plane_back_on() -> None:
    source = CONFIG_PATH.read_text(encoding="utf-8")

    assert "_dsv4_swa_transfer_enabled_from_env()" in source
    assert "or self.cache_config.swa.enabled" not in source
