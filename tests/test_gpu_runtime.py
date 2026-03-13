"""
Tests for flexkv.common.gpu_runtime — backend-agnostic GPU runtime abstraction.
"""
import os
import pytest
from unittest.mock import patch, MagicMock


class TestGpuRuntimeBackendDetection:
    """Verify gpu_runtime loads the correct shared library based on backend."""

    def test_import(self):
        from flexkv.common import gpu_runtime
        assert hasattr(gpu_runtime, "host_register")
        assert hasattr(gpu_runtime, "host_unregister")
        assert hasattr(gpu_runtime, "get_current_stream")
        assert hasattr(gpu_runtime, "set_device")
        assert hasattr(gpu_runtime, "synchronize")
        assert hasattr(gpu_runtime, "device_count")
        assert hasattr(gpu_runtime, "current_device")
        assert hasattr(gpu_runtime, "get_device_string")
        assert hasattr(gpu_runtime, "create_stream")
        assert hasattr(gpu_runtime, "stream_context")
        assert hasattr(gpu_runtime, "empty_cache")
        assert hasattr(gpu_runtime, "is_gpu_tensor")
        assert hasattr(gpu_runtime, "ipc_get_mem_handle")
        assert hasattr(gpu_runtime, "ipc_open_mem_handle")


class TestGpuRuntimeMusaPath:
    """Test that MUSA backend path is selected when FLEXKV_GPU_BACKEND=musa."""

    def setup_method(self):
        import flexkv.common.gpu_runtime as mod
        mod._runtime_lib = None
        mod._backend = ""

    @patch.dict(os.environ, {"FLEXKV_GPU_BACKEND": "musa"})
    @patch("ctypes.CDLL")
    def test_loads_musart(self, mock_cdll):
        import flexkv.common.gpu_runtime as mod
        mod._runtime_lib = None
        mod._backend = ""
        mock_lib = MagicMock()
        mock_cdll.return_value = mock_lib
        mod._load_runtime()
        mock_cdll.assert_called_with("libmusart.so")
        assert mod._backend == "musa"

    @patch.dict(os.environ, {}, clear=True)
    @patch("ctypes.CDLL")
    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    def test_loads_cudart(self, mock_backend, mock_cdll):
        import flexkv.common.gpu_runtime as mod
        mod._runtime_lib = None
        mod._backend = ""
        mock_lib = MagicMock()
        mock_cdll.return_value = mock_lib
        mod._load_runtime()
        mock_cdll.assert_called_with("libcudart.so")
        assert mod._backend == "cuda"


class TestGpuRuntimeStreamDevice:
    """Test stream/device helpers dispatch correctly."""

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    @patch("torch.cuda.current_stream")
    def test_get_current_stream_cuda(self, mock_stream, mock_backend):
        from flexkv.common.gpu_runtime import get_current_stream
        sentinel = object()
        mock_stream.return_value = sentinel
        result = get_current_stream()
        assert result is sentinel

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    @patch("torch.cuda.set_device")
    def test_set_device_cuda(self, mock_set, mock_backend):
        from flexkv.common.gpu_runtime import set_device
        set_device(3)
        mock_set.assert_called_once_with(3)

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    @patch("torch.cuda.device_count", return_value=4)
    def test_device_count_cuda(self, mock_count, mock_backend):
        from flexkv.common.gpu_runtime import device_count
        assert device_count() == 4

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    @patch("torch.cuda.current_device", return_value=2)
    def test_current_device_cuda(self, mock_cur, mock_backend):
        from flexkv.common.gpu_runtime import current_device
        assert current_device() == 2

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    @patch("torch.cuda.empty_cache")
    def test_empty_cache_cuda(self, mock_ec, mock_backend):
        from flexkv.common.gpu_runtime import empty_cache
        empty_cache()
        mock_ec.assert_called_once()

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="cuda")
    def test_get_device_string_cuda(self, mock_backend):
        from flexkv.common.gpu_runtime import get_device_string
        assert get_device_string(0) == "cuda:0"
        assert get_device_string(3) == "cuda:3"

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_get_device_string_musa(self, mock_backend):
        from flexkv.common.gpu_runtime import get_device_string
        assert get_device_string(0) == "musa:0"
        assert get_device_string(1) == "musa:1"


class TestMusaUnavailableRaisesError:
    """When backend is 'musa' but torch.musa is absent, functions must raise."""

    def _remove_torch_musa(self):
        """Patch torch so that torch.musa does not exist."""
        import torch as _torch
        return patch.object(_torch, "musa", new=None, create=True)

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_get_current_stream_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import get_current_stream
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                get_current_stream()

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_set_device_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import set_device
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                set_device(0)

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_synchronize_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import synchronize
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                synchronize()

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_device_count_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import device_count
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                device_count()

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_current_device_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import current_device
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                current_device()

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_empty_cache_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import empty_cache
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                empty_cache()

    @patch("flexkv.common.gpu_runtime.get_gpu_backend", return_value="musa")
    def test_create_stream_raises(self, _mock_backend):
        with self._remove_torch_musa():
            from flexkv.common.gpu_runtime import create_stream
            with pytest.raises(RuntimeError, match="torch.musa is not available"):
                create_stream()

