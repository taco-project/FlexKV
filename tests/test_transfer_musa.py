"""
Tests for MUSA transfer path — Phase 2.
Run with: pytest tests/test_transfer_musa.py -v -m musa
Skip when c_ext_musa is not built or MUSA device is not available.
"""
import os
import pytest

try:
    import flexkv.c_ext_musa as musa_ext
    MUSA_EXT_AVAILABLE = hasattr(musa_ext, "transfer_kv_blocks")
except ImportError:
    musa_ext = None
    MUSA_EXT_AVAILABLE = False

HAS_MUSA_SDK = getattr(musa_ext, "HAS_MUSA_SDK", False) if musa_ext else False


def _musa_device_available():
    try:
        import torch
        musa = getattr(torch, "musa", None)
        return musa is not None and musa.is_available()
    except Exception:
        return False


MUSA_DEVICE_AVAILABLE = _musa_device_available()


# ---------------------------------------------------------------------------
# 1. Import / build verification
# ---------------------------------------------------------------------------

@pytest.mark.musa
class TestMusaExtImport:
    """Verify the c_ext_musa extension can be imported and has correct attributes."""

    def test_extension_importable(self):
        if not MUSA_EXT_AVAILABLE:
            pytest.skip("flexkv.c_ext_musa not built (build with FLEXKV_USE_MUSA=1)")
        assert hasattr(musa_ext, "transfer_kv_blocks")

    def test_has_musa_sdk_attribute(self):
        if not MUSA_EXT_AVAILABLE:
            pytest.skip("flexkv.c_ext_musa not built")
        assert hasattr(musa_ext, "HAS_MUSA_SDK")

    @pytest.mark.skipif(not MUSA_EXT_AVAILABLE, reason="extension not built")
    def test_sdk_flag_matches_env(self):
        """When MUSA_HOME points to a real SDK, HAS_MUSA_SDK should be True."""
        musa_home = os.environ.get("MUSA_HOME", "")
        if musa_home and os.path.isdir(musa_home):
            assert musa_ext.HAS_MUSA_SDK is True
        else:
            assert musa_ext.HAS_MUSA_SDK is False


# ---------------------------------------------------------------------------
# 2. Stub transfer (no device required)
# ---------------------------------------------------------------------------

@pytest.mark.musa
@pytest.mark.skipif(
    HAS_MUSA_SDK,
    reason="Stub tests require extension built without MUSA SDK (no-op path); with SDK, real kernel runs and needs valid GPU pointers",
)
class TestMusaTransferStub:
    """Call transfer_kv_blocks in stub mode — must not crash."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_ext(self):
        if not MUSA_EXT_AVAILABLE:
            pytest.skip("flexkv.c_ext_musa not built")

    def _make_dummy_args(self, num_blocks=2, num_layers=1, chunk_size=256):
        import torch
        gpu_block_ids = torch.zeros(num_blocks, dtype=torch.int64)
        cpu_block_ids = torch.zeros(num_blocks, dtype=torch.int64)
        cpu_tensor = torch.zeros(
            chunk_size * num_blocks * 2 * num_layers, dtype=torch.int64
        )
        gpu_tensor_ptrs = torch.zeros(1, dtype=torch.int64)
        return dict(
            gpu_block_id_tensor=gpu_block_ids,
            gpu_tensor_ptrs_tensor=gpu_tensor_ptrs,
            gpu_kv_stride_in_bytes=0,
            gpu_block_stride_in_bytes=0,
            gpu_layer_stride_in_bytes=0,
            cpu_block_id_tensor=cpu_block_ids,
            cpu_tensor=cpu_tensor,
            cpu_kv_stride_in_bytes=chunk_size * 8,
            cpu_layer_stride_in_bytes=chunk_size * 8,
            cpu_block_stride_in_bytes=chunk_size * 8,
            chunk_size_in_bytes=chunk_size * 8,
            start_layer_id=0,
            num_layers=num_layers,
        )

    def test_host_to_device_no_crash(self):
        args = self._make_dummy_args()
        musa_ext.transfer_kv_blocks(**args, is_host_to_device=True)

    def test_device_to_host_no_crash(self):
        args = self._make_dummy_args()
        musa_ext.transfer_kv_blocks(**args, is_host_to_device=False)

    def test_ce_transfer_no_crash(self):
        args = self._make_dummy_args()
        musa_ext.transfer_kv_blocks(**args, use_ce_transfer=True)

    def test_mla_mode_no_crash(self):
        args = self._make_dummy_args()
        musa_ext.transfer_kv_blocks(**args, is_mla=True)

    @pytest.mark.parametrize("block_type", [0, 1, 2])
    def test_all_backend_types(self, block_type):
        args = self._make_dummy_args()
        musa_ext.transfer_kv_blocks(**args, gpu_block_type=block_type)

    def test_multi_layer(self):
        args = self._make_dummy_args(num_blocks=4, num_layers=3, chunk_size=128)
        musa_ext.transfer_kv_blocks(**args)


# ---------------------------------------------------------------------------
# 3. Real MUSA device transfer (requires MUSA SDK + device)
# ---------------------------------------------------------------------------

@pytest.mark.musa
@pytest.mark.skipif(
    not (HAS_MUSA_SDK and MUSA_DEVICE_AVAILABLE),
    reason="Requires MUSA SDK build and MUSA device",
)
class TestMusaTransferDevice:
    """Correctness tests that run on an actual MUSA device."""

    def test_h2d_d2h_roundtrip_vllm(self):
        """Write data H→D then read D→H and verify parity (VLLM backend)."""
        import torch

        num_blocks = 2
        num_layers = 1
        num_kv = 2  # K and V
        block_size = 64
        chunk_bytes = block_size * 8

        # CPU layout: [layer][kv][block]; strides in bytes
        cpu_block_stride = chunk_bytes
        cpu_kv_stride = num_blocks * cpu_block_stride
        cpu_layer_stride = num_kv * cpu_kv_stride

        cpu_src = torch.arange(
            num_blocks * num_layers * num_kv * block_size, dtype=torch.int64
        ).pin_memory()
        cpu_dst = torch.zeros_like(cpu_src).pin_memory()

        # VLLM layout: [K_block0, K_block1, V_block0, V_block1] per layer
        gpu_block_stride = chunk_bytes
        gpu_kv_stride = num_blocks * chunk_bytes
        gpu_tensors = [
            torch.zeros(num_kv * num_blocks * block_size, dtype=torch.int64, device="musa")
            for _ in range(num_layers)
        ]
        gpu_ptrs = torch.tensor(
            [t.data_ptr() for t in gpu_tensors], dtype=torch.int64
        )

        block_ids = torch.arange(num_blocks, dtype=torch.int64)

        # Use CE transfer (memcpy) — kernel path uses __ldg which cannot read host memory
        musa_ext.transfer_kv_blocks(
            block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, 0,
            block_ids, cpu_src,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_bytes, 0, num_layers,
            is_host_to_device=True, use_ce_transfer=True, gpu_block_type=0,
        )

        musa_ext.transfer_kv_blocks(
            block_ids, gpu_ptrs,
            gpu_kv_stride, gpu_block_stride, 0,
            block_ids, cpu_dst,
            cpu_kv_stride, cpu_layer_stride, cpu_block_stride,
            chunk_bytes, 0, num_layers,
            is_host_to_device=False, use_ce_transfer=True, gpu_block_type=0,
        )

        assert torch.equal(cpu_src, cpu_dst), "H→D→H roundtrip data mismatch"


# ---------------------------------------------------------------------------
# 4. Signature parity with CUDA extension
# ---------------------------------------------------------------------------

@pytest.mark.musa
class TestMusaCudaSignatureParity:
    """Ensure c_ext_musa.transfer_kv_blocks accepts the same kwargs as c_ext."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_ext(self):
        if not MUSA_EXT_AVAILABLE:
            pytest.skip("flexkv.c_ext_musa not built")

    def test_same_parameter_names(self):
        import inspect
        try:
            import flexkv.c_ext as cuda_ext
        except ImportError:
            pytest.skip("flexkv.c_ext (CUDA) not available for comparison")
        cuda_sig = inspect.signature(cuda_ext.transfer_kv_blocks)
        musa_sig = inspect.signature(musa_ext.transfer_kv_blocks)
        assert list(cuda_sig.parameters.keys()) == list(musa_sig.parameters.keys())

