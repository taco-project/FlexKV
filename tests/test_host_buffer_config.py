import pytest

from flexkv.transfer import host_buffer


@pytest.fixture(autouse=True)
def _clean_platform_env(monkeypatch):
    monkeypatch.delenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", raising=False)
    monkeypatch.delenv("FLEXKV_ENABLE_CE_ONLY", raising=False)


def test_cudahost_register_flags_default_keeps_nvidia_portable_mapped():
    assert host_buffer._get_cudahost_register_flags() == 3


def test_cudahost_register_flags_default_is_zero_in_ce_only(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")

    assert host_buffer._get_cudahost_register_flags() == 0


def test_cudahost_register_flags_accept_hex(monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x3")

    assert host_buffer._get_cudahost_register_flags() == 3


def test_cudahost_register_flags_rejected_in_ce_only(monkeypatch):
    monkeypatch.setenv("FLEXKV_ENABLE_CE_ONLY", "1")
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "0x3")

    with pytest.raises(ValueError, match="requires FLEXKV_CUDAHOST_REGISTER_FLAGS=0"):
        host_buffer._get_cudahost_register_flags()


def test_cudahost_register_flags_reject_invalid_value(monkeypatch):
    monkeypatch.setenv("FLEXKV_CUDAHOST_REGISTER_FLAGS", "mapped")

    with pytest.raises(ValueError, match="must be an int"):
        host_buffer._get_cudahost_register_flags()
