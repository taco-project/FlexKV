"""
tests/test_gpu_backend.py — Unit tests for flexkv.gpu_backend abstraction layer.

Covers:
  - GpuBackend interface completeness
  - is_available / device_name classmethods
  - current_backend auto-detection
  - ENV var override
  - transfer_kv_blocks (GenericBackend, no GPU required)
  - layout_transform (GenericBackend)
  - IPC not-implemented guard
  - GDS not-implemented guard
  - stream lifecycle (create / destroy / context)
  - register/unregister host tensor (GenericBackend)
"""

import os
import importlib
import sys
import pytest
import torch


# ─── Helpers ─────────────────────────────────────────────────────────────────

def reload_gpu_backend(env_val: str = ""):
    """Force-reload flexkv.gpu_backend with a given FLEXKV_GPU_BACKEND value."""
    mods_to_remove = [k for k in sys.modules if k.startswith("flexkv.gpu_backend")]
    for m in mods_to_remove:
        del sys.modules[m]

    os.environ["FLEXKV_GPU_BACKEND"] = env_val
    import flexkv.gpu_backend as mod
    importlib.reload(mod)
    return mod


# ─── Interface completeness ───────────────────────────────────────────────────

class TestGpuBackendInterface:
    """Verify GpuBackend ABC defines all required methods."""

    REQUIRED_ABSTRACT = [
        "is_available", "device_name",
        "set_device", "current_device", "device_count", "synchronize",
        "alloc_pinned", "free_pinned",
        "register_host_tensor", "unregister_host_tensor",
        "create_stream", "destroy_stream",
        "transfer_kv_blocks", "layout_transform",
    ]

    def test_abstract_methods_exist(self):
        from flexkv.gpu_backend.interface import GpuBackend
        for name in self.REQUIRED_ABSTRACT:
            assert hasattr(GpuBackend, name), f"GpuBackend missing method: {name}"

    def test_cannot_instantiate_abc(self):
        from flexkv.gpu_backend.interface import GpuBackend
        with pytest.raises(TypeError):
            GpuBackend()


# ─── NvidiaBackend ────────────────────────────────────────────────────────────

class TestNvidiaBackend:
    @pytest.fixture(autouse=True)
    def backend(self):
        from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
        self.cls = NvidiaBackend

    def test_is_classmethod(self):
        assert callable(self.cls.is_available)
        assert callable(self.cls.device_name)

    def test_is_available_returns_bool(self):
        result = self.cls.is_available()
        assert isinstance(result, bool)

    def test_device_name_returns_string(self):
        name = self.cls.device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_isinstance_gpu_backend(self):
        from flexkv.gpu_backend.interface import GpuBackend
        assert issubclass(self.cls, GpuBackend)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_set_get_device(self):
        inst = self.cls()
        inst.set_device(0)
        assert inst.current_device() == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_destroy_stream(self):
        inst = self.cls()
        s = inst.create_stream()
        assert s is not None
        inst.destroy_stream(s)  # should not raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_register_unregister_host_tensor(self):
        inst = self.cls()
        t = torch.zeros(1024, dtype=torch.float32)
        inst.register_host_tensor(t)
        inst.unregister_host_tensor(t)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_transfer_kv_blocks_gpu(self):
        inst = self.cls()
        src = [torch.zeros(256, dtype=torch.float16, device="cuda") for _ in range(2)]
        dst = [torch.ones(256, dtype=torch.float16, device="cuda") for _ in range(2)]
        inst.transfer_kv_blocks(src, dst, [0, 1])
        torch.cuda.synchronize()
        assert torch.allclose(dst[0], src[0])

    def test_ipc_not_implemented_without_gpu(self):
        if torch.cuda.is_available():
            pytest.skip("IPC handle test requires no GPU")
        from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
        inst = NvidiaBackend()
        # On CPU env, get_ipc_handle should either work or raise RuntimeError (not AttributeError)
        t = torch.zeros(4)
        try:
            inst.get_ipc_handle(t)
        except (RuntimeError, OSError):
            pass  # expected without GPU/cudart


# ─── MusaBackend ─────────────────────────────────────────────────────────────

class TestMusaBackend:
    @pytest.fixture(autouse=True)
    def backend(self):
        from flexkv.gpu_backend.musa.backend import MusaBackend
        self.cls = MusaBackend

    def test_isinstance_gpu_backend(self):
        from flexkv.gpu_backend.interface import GpuBackend
        assert issubclass(self.cls, GpuBackend)

    def test_is_available_without_musa(self):
        # In environments without torch_musa, should return False (not raise)
        result = self.cls.is_available()
        assert isinstance(result, bool)

    def test_device_name_without_musa(self):
        name = self.cls.device_name()
        assert isinstance(name, str)


# ─── GenericBackend ───────────────────────────────────────────────────────────

class TestGenericBackend:
    @pytest.fixture(autouse=True)
    def inst(self):
        from flexkv.gpu_backend.generic.backend import GenericBackend
        self.b = GenericBackend()

    def test_always_available(self):
        from flexkv.gpu_backend.generic.backend import GenericBackend
        assert GenericBackend.is_available() is True

    def test_device_name(self):
        from flexkv.gpu_backend.generic.backend import GenericBackend
        assert "Generic" in GenericBackend.device_name()

    def test_transfer_kv_blocks_cpu(self):
        src = [torch.rand(64, dtype=torch.float32) for _ in range(3)]
        dst = [torch.zeros(64, dtype=torch.float32) for _ in range(3)]
        self.b.transfer_kv_blocks(src, dst, [0, 1, 2])
        for s, d in zip(src, dst):
            assert torch.allclose(s, d), "transfer_kv_blocks should copy src→dst"

    def test_layout_transform(self):
        src = torch.rand(128, dtype=torch.float32)
        dst = torch.zeros(128, dtype=torch.float32)
        self.b.layout_transform(src, dst)
        assert torch.allclose(src, dst)

    def test_stream_lifecycle(self):
        s = self.b.create_stream()
        # GenericBackend returns None stream — should not raise
        self.b.destroy_stream(s)

    def test_register_unregister_noop(self):
        t = torch.zeros(64)
        self.b.register_host_tensor(t)
        self.b.unregister_host_tensor(t)

    def test_synchronize_noop(self):
        self.b.synchronize()  # should not raise

    def test_gds_not_supported(self):
        assert self.b.supports_direct_storage() is False
        with pytest.raises(NotImplementedError):
            self.b.read_direct(0, torch.zeros(4), 0)
        with pytest.raises(NotImplementedError):
            self.b.write_direct(0, torch.zeros(4), 0)

    def test_ipc_not_supported(self):
        with pytest.raises(NotImplementedError):
            self.b.get_ipc_handle(torch.zeros(4))

    def test_is_gpu_tensor_cpu(self):
        t = torch.zeros(4)
        assert self.b.is_gpu_tensor(t) is False


# ─── Auto-detection & ENV override ───────────────────────────────────────────

class TestAutoDetection:

    def test_default_is_nvidia_or_generic(self):
        """Without GPU, should fall through to NvidiaBackend (default) or detectable."""
        from flexkv.gpu_backend import current_backend
        from flexkv.gpu_backend.interface import GpuBackend
        assert isinstance(current_backend, GpuBackend)

    def test_env_override_generic(self, monkeypatch):
        monkeypatch.setenv("FLEXKV_GPU_BACKEND", "generic")
        mods = [k for k in sys.modules if k.startswith("flexkv.gpu_backend")]
        for m in mods:
            del sys.modules[m]
        import flexkv.gpu_backend as mod
        from flexkv.gpu_backend.generic.backend import GenericBackend
        assert isinstance(mod.current_backend, GenericBackend)

    def test_env_cuda_maps_to_nvidia(self, monkeypatch):
        monkeypatch.setenv("FLEXKV_GPU_BACKEND", "cuda")
        mods = [k for k in sys.modules if k.startswith("flexkv.gpu_backend")]
        for m in mods:
            del sys.modules[m]
        import flexkv.gpu_backend as mod
        from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
        assert isinstance(mod.current_backend, NvidiaBackend)

    def test_env_kunlun_maps_to_nvidia(self, monkeypatch):
        monkeypatch.setenv("FLEXKV_GPU_BACKEND", "kunlun")
        mods = [k for k in sys.modules if k.startswith("flexkv.gpu_backend")]
        for m in mods:
            del sys.modules[m]
        import flexkv.gpu_backend as mod
        from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
        assert isinstance(mod.current_backend, NvidiaBackend)

    def test_invalid_backend_name_raises(self, monkeypatch):
        """Unknown backend names should fall through to default (not crash)."""
        monkeypatch.setenv("FLEXKV_GPU_BACKEND", "nonexistent_vendor_xyz")
        mods = [k for k in sys.modules if k.startswith("flexkv.gpu_backend")]
        for m in mods:
            del sys.modules[m]
        # Should not raise — unknown names simply don't match _ENV_MAP
        import flexkv.gpu_backend as mod
        from flexkv.gpu_backend.interface import GpuBackend
        assert isinstance(mod.current_backend, GpuBackend)


# ─── make_device ─────────────────────────────────────────────────────────────

class TestMakeDevice:
    def test_generic_cpu_device(self):
        from flexkv.gpu_backend.generic.backend import GenericBackend
        b = GenericBackend()
        d = b.make_device(0)
        assert isinstance(d, torch.device)

    def test_nvidia_cuda_device(self):
        from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
        b = NvidiaBackend()
        d = b.make_device(0)
        assert str(d) in ("cuda:0", "cpu")  # cpu if no GPU


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
