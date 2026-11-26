import pytest

from flexkv.common import platform
from flexkv.transfer import host_buffer


@pytest.fixture(autouse=True)
def _clean_platform_env(monkeypatch):
    monkeypatch.delenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", raising=False)
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: True)


def test_cudahost_register_flags_default_keeps_nvidia_portable_mapped():
    assert host_buffer._get_cudahost_register_flags() == 3


def test_cudahost_register_flags_default_is_zero_in_host_only_build(monkeypatch):
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: False)

    assert host_buffer._get_cudahost_register_flags() == 0


def test_cudahost_register_flags_accept_hex(monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x3")

    assert host_buffer._get_cudahost_register_flags() == 3


def test_cudahost_register_flags_rejected_in_host_only_build(monkeypatch):
    monkeypatch.setattr(platform, "kernel_transfer_built", lambda: False)
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x3")

    with pytest.raises(ValueError, match="require FLEXKV_CUDAHOST_REGISTER_FLAGS=0"):
        host_buffer._get_cudahost_register_flags()


def test_cudahost_register_flags_reject_invalid_value(monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "mapped")

    with pytest.raises(ValueError, match="must be an int"):
        host_buffer._get_cudahost_register_flags()
