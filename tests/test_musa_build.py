"""
Tests for MUSA build configuration.
When FLEXKV_USE_MUSA=1, the build should include flexkv.c_ext_musa.
"""
import os
import pytest

# Import from flexkv.build_config (used by setup.py and tests)
from flexkv.build_config import get_cpp_extension_names


def test_default_build_includes_only_c_ext():
    """Without FLEXKV_USE_MUSA, only flexkv.c_ext is in the extension list."""
    env_orig = os.environ.get("FLEXKV_USE_MUSA")
    try:
        if "FLEXKV_USE_MUSA" in os.environ:
            del os.environ["FLEXKV_USE_MUSA"]
        names = get_cpp_extension_names()
        assert names == ["flexkv.c_ext"]
    finally:
        if env_orig is not None:
            os.environ["FLEXKV_USE_MUSA"] = env_orig
        elif "FLEXKV_USE_MUSA" in os.environ:
            del os.environ["FLEXKV_USE_MUSA"]


@pytest.mark.parametrize("value", ["1"])
def test_musa_build_includes_c_ext_musa(value):
    """With FLEXKV_USE_MUSA=1, flexkv.c_ext_musa is in the extension list."""
    env_orig = os.environ.get("FLEXKV_USE_MUSA")
    try:
        os.environ["FLEXKV_USE_MUSA"] = value
        names = get_cpp_extension_names()
        assert "flexkv.c_ext_musa" in names
    finally:
        if env_orig is not None:
            os.environ["FLEXKV_USE_MUSA"] = env_orig
        elif "FLEXKV_USE_MUSA" in os.environ:
            del os.environ["FLEXKV_USE_MUSA"]


def test_musa_disabled_omits_c_ext_musa():
    """With FLEXKV_USE_MUSA=0 or unset, c_ext_musa is not in the list."""
    for value in ("0", "", "false"):
        env_orig = os.environ.get("FLEXKV_USE_MUSA")
        try:
            if value:
                os.environ["FLEXKV_USE_MUSA"] = value
            elif "FLEXKV_USE_MUSA" in os.environ:
                del os.environ["FLEXKV_USE_MUSA"]
            names = get_cpp_extension_names()
            assert "flexkv.c_ext_musa" not in names
            assert "flexkv.c_ext" in names
        finally:
            if env_orig is not None:
                os.environ["FLEXKV_USE_MUSA"] = env_orig
            elif "FLEXKV_USE_MUSA" in os.environ:
                del os.environ["FLEXKV_USE_MUSA"]
