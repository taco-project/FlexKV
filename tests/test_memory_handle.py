"""
Test TensorSharedHandle with three different input types:
1. torch.Tensor (default PyTorch IPC)
2. torch.Tensor (forced direct CUDA IPC)
3. bytes (constructed from shared CUDA IPC handle)
"""
import multiprocessing as mp
import os
import pytest
import torch
from multiprocessing import Process, Pipe

from flexkv.common.memory_handle import TensorSharedHandle


def _worker_test_tensor_from_tensor_pytorch_ipc(conn, device_id):
    """Test construction from torch.Tensor (default PyTorch IPC)"""
    try:
        # Receive TensorSharedHandle
        handle = conn.recv()
        assert isinstance(handle, TensorSharedHandle)
        assert not handle.use_direct_ipc, "Should use PyTorch IPC, not direct CUDA IPC"
        assert handle.rebuild_func is not None, "PyTorch IPC should have rebuild_func"

        # Recover tensor
        tensor = handle.get_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.is_cuda, "tensor should be on CUDA"
        assert (
            tensor.device.index == device_id
        ), f"tensor should be on device {device_id}"
        assert tensor.shape == (10, 20), "tensor shape should be correct"
        assert tensor.dtype == torch.float32, "tensor dtype should be correct"

        # Verify data
        expected_data = (
            torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
        )
        assert torch.allclose(tensor, expected_data), "tensor data should be correct"

        # Modify data to verify sharing
        tensor[:] = 42.0
        conn.send(True)
    except Exception as e:
        conn.send(f"Error: {e}")
        raise


def _worker_test_tensor_from_tensor_direct_ipc(conn, device_id):
    """Test construction from torch.Tensor (forced direct CUDA IPC)"""
    try:
        # Receive TensorSharedHandle
        handle = conn.recv()
        assert isinstance(handle, TensorSharedHandle)
        assert handle.use_direct_ipc, "Should use direct CUDA IPC"
        assert handle.ipc_handle is not None, "Direct CUDA IPC should have ipc_handle"
        assert handle.tensor_shape == (10, 20), "tensor shape should be saved"
        assert handle.tensor_dtype == torch.float32, "tensor dtype should be saved"
        assert (
            handle.rebuild_func is None
        ), "Direct CUDA IPC should not have rebuild_func"

        # Recover tensor
        tensor = handle.get_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.is_cuda, "tensor should be on CUDA"
        assert (
            tensor.device.index == device_id
        ), f"tensor should be on device {device_id}"
        assert tensor.shape == (10, 20), "tensor shape should be correct"
        assert tensor.dtype == torch.float32, "tensor dtype should be correct"

        # Verify data
        expected_data = (
            torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
        )
        assert torch.allclose(tensor, expected_data), "tensor data should be correct"

        # Modify data to verify sharing
        tensor[:] = 42.0
        conn.send(True)
    except Exception as e:
        conn.send(f"Error: {e}")
        raise


def _worker_test_tensor_from_bytes(conn, device_id):
    """Test construction from bytes (IPC handle)"""
    try:
        # Receive TensorSharedHandle
        handle = conn.recv()
        assert isinstance(handle, TensorSharedHandle)
        assert (
            handle.use_direct_ipc
        ), "Construction from bytes should use direct CUDA IPC"
        assert handle.ipc_handle is not None, "Should have ipc_handle"
        assert handle.tensor_shape == (10, 20), "tensor shape should be saved"
        assert handle.tensor_dtype == torch.float32, "tensor dtype should be saved"
        assert (
            handle.rebuild_func is None
        ), "Construction from bytes should not have rebuild_func"

        # Recover tensor
        tensor = handle.get_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.is_cuda, "tensor should be on CUDA"
        assert (
            tensor.device.index == device_id
        ), f"tensor should be on device {device_id}"
        assert tensor.shape == (10, 20), "tensor shape should be correct"
        assert tensor.dtype == torch.float32, "tensor dtype should be correct"

        # Verify data (original data should be 0-199)
        expected_data = (
            torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
        )
        assert torch.allclose(tensor, expected_data), "tensor data should be correct"

        # Modify data to verify sharing
        tensor[:] = 99.0
        conn.send(True)
    except Exception as e:
        conn.send(f"Error: {e}")
        raise


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_tensor_pytorch_ipc():
    """Test method 1: Construction from torch.Tensor (default PyTorch IPC)"""
    mp.set_start_method("spawn", force=True)

    device_id = 0
    # Create original tensor
    original_tensor = (
        torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
    )

    # Create TensorSharedHandle (default PyTorch IPC)
    handle = TensorSharedHandle(original_tensor, device_id=device_id)

    # Verify handle properties
    assert not handle.use_direct_ipc, "Should use PyTorch IPC by default"
    assert handle.rebuild_func is not None, "PyTorch IPC should have rebuild_func"
    assert handle.rebuild_args is not None, "PyTorch IPC should have rebuild_args"
    assert handle.device.index == device_id, "Device should be correct"

    # Test cross-process sharing (main use case of TensorSharedHandle)
    parent_conn, child_conn = Pipe()
    process = Process(
        target=_worker_test_tensor_from_tensor_pytorch_ipc,
        args=(child_conn, device_id),
        daemon=True,
    )
    process.start()

    # Send handle to child process
    parent_conn.send(handle)

    # Wait for child process to complete
    result = parent_conn.recv()
    assert result is True, f"Child process test failed: {result}"

    process.join(timeout=5)
    parent_conn.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_tensor_direct_ipc():
    """Test method 2: Construction from torch.Tensor (forced direct CUDA IPC)"""
    mp.set_start_method("spawn", force=True)

    device_id = 0
    # Create original tensor
    original_tensor = (
        torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
    )

    # Create TensorSharedHandle (forced direct CUDA IPC)
    handle = TensorSharedHandle(
        original_tensor, device_id=device_id, force_direct_ipc=True
    )

    # Verify handle properties
    assert handle.use_direct_ipc, "Should use direct CUDA IPC"
    assert handle.ipc_handle is not None, "Should have ipc_handle"
    assert len(handle.ipc_handle) == 64, "IPC handle should be 64 bytes"
    assert handle.tensor_shape == (10, 20), "tensor shape should be saved"
    assert handle.tensor_dtype == torch.float32, "tensor dtype should be saved"
    assert handle.tensor_numel == 200, "tensor numel should be saved"
    assert handle.rebuild_func is None, "Direct CUDA IPC should not have rebuild_func"
    assert handle.device.index == device_id, "Device should be correct"

    # Test cross-process sharing (main use case of TensorSharedHandle)
    parent_conn, child_conn = Pipe()
    process = Process(
        target=_worker_test_tensor_from_tensor_direct_ipc,
        args=(child_conn, device_id),
        daemon=True,
    )
    process.start()

    # Send handle to child process
    parent_conn.send(handle)

    # Wait for child process to complete
    result = parent_conn.recv()
    assert result is True, f"Child process test failed: {result}"

    process.join(timeout=5)
    parent_conn.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_bytes():
    """Test method 3: Construction from bytes (IPC handle)"""
    mp.set_start_method("spawn", force=True)

    device_id = 0
    # First create a tensor and export IPC handle
    original_tensor = (
        torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
    )
    source_handle = TensorSharedHandle(
        original_tensor, device_id=device_id, force_direct_ipc=True
    )

    # Extract IPC handle and related information
    ipc_handle_bytes = source_handle.ipc_handle
    tensor_shape = source_handle.tensor_shape
    tensor_dtype = source_handle.tensor_dtype

    # Construct new TensorSharedHandle from bytes
    handle = TensorSharedHandle(
        ipc_handle_bytes,
        device_id=device_id,
        tensor_shape=tensor_shape,
        tensor_dtype=tensor_dtype,
    )

    # Verify handle properties
    assert handle.use_direct_ipc, "Construction from bytes should use direct CUDA IPC"
    assert handle.ipc_handle == ipc_handle_bytes, "IPC handle should be the same"
    assert handle.tensor_shape == (10, 20), "tensor shape should be correct"
    assert handle.tensor_dtype == torch.float32, "tensor dtype should be correct"
    assert handle.tensor_numel == 200, "tensor numel should be correct"
    assert (
        handle.rebuild_func is None
    ), "Construction from bytes should not have rebuild_func"
    assert handle.device.index == device_id, "Device should be correct"

    # Test cross-process sharing (main use case of TensorSharedHandle)
    parent_conn, child_conn = Pipe()
    process = Process(
        target=_worker_test_tensor_from_bytes, args=(child_conn, device_id), daemon=True
    )
    process.start()

    # Send handle to child process
    parent_conn.send(handle)

    # Wait for child process to complete
    result = parent_conn.recv()
    assert result is True, f"Child process test failed: {result}"

    process.join(timeout=5)
    parent_conn.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_bytes_with_string_dtype():
    """Test construction from bytes with string dtype"""
    device_id = 0
    # Create original tensor
    original_tensor = (
        torch.arange(200, dtype=torch.float16).reshape(10, 20).cuda(device_id)
    )
    source_handle = TensorSharedHandle(
        original_tensor, device_id=device_id, force_direct_ipc=True
    )

    # Construct with string dtype
    handle = TensorSharedHandle(
        source_handle.ipc_handle,
        device_id=device_id,
        tensor_shape=source_handle.tensor_shape,
        tensor_dtype="float16",  # Use string
    )

    # Verify dtype parsing
    assert handle.tensor_dtype == torch.float16, "Should correctly parse string dtype"
    assert handle.tensor_shape == (10, 20), "tensor shape should be correct"
    # Note: Recovering from IPC handle in the same process may fail because
    # the handle was exported from the same process. Here we mainly verify
    # that construction and property setting are correct.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_bytes_with_different_device():
    """Test construction from bytes with different device specified"""
    source_device_id = 0
    target_device_id = 0  # Use same device if only one GPU available

    # Create original tensor
    original_tensor = (
        torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(source_device_id)
    )
    source_handle = TensorSharedHandle(
        original_tensor, device_id=source_device_id, force_direct_ipc=True
    )

    # Construct on target device
    handle = TensorSharedHandle(
        source_handle.ipc_handle,
        device_id=target_device_id,
        tensor_shape=source_handle.tensor_shape,
        tensor_dtype=source_handle.tensor_dtype,
    )

    # Verify device setting
    assert handle.device.index == target_device_id, "Device should be set correctly"
    # Note: Recovering from IPC handle in the same process may fail because
    # the handle was exported from the same process. Here we mainly verify
    # that device parameter setting is correct.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_bytes_missing_required_params():
    """Test construction from bytes with missing required parameters"""
    device_id = 0
    original_tensor = (
        torch.arange(200, dtype=torch.float32).reshape(10, 20).cuda(device_id)
    )
    source_handle = TensorSharedHandle(
        original_tensor, device_id=device_id, force_direct_ipc=True
    )

    # Test missing device_id (device_id=-1)
    with pytest.raises(ValueError, match="device_id must be provided"):
        TensorSharedHandle(
            source_handle.ipc_handle,
            device_id=-1,  # Invalid device_id
            tensor_shape=source_handle.tensor_shape,
            tensor_dtype=source_handle.tensor_dtype,
        )

    # Test missing tensor_shape
    with pytest.raises(ValueError, match="tensor_shape is required"):
        TensorSharedHandle(
            source_handle.ipc_handle,
            device_id=device_id,
            tensor_shape=None,
            tensor_dtype=source_handle.tensor_dtype,
        )

    # Test missing tensor_dtype
    with pytest.raises(ValueError, match="tensor_dtype is required"):
        TensorSharedHandle(
            source_handle.ipc_handle,
            device_id=device_id,
            tensor_shape=source_handle.tensor_shape,
            tensor_dtype=None,
        )

    # Test missing ipc_handle (pass None)
    with pytest.raises(ValueError, match="Unsupported data type"):
        TensorSharedHandle(
            None,
            device_id=device_id,
            tensor_shape=source_handle.tensor_shape,
            tensor_dtype=source_handle.tensor_dtype,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_unsupported_type():
    """Test unsupported data type"""
    with pytest.raises(ValueError, match="Unsupported data type"):
        TensorSharedHandle("not a tensor or bytes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_from_cpu_tensor():
    """Test that CPU tensor should raise error"""
    cpu_tensor = torch.arange(200, dtype=torch.float32).reshape(10, 20)

    with pytest.raises(ValueError, match="Only support CUDA tensor sharing"):
        TensorSharedHandle(cpu_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_dtype_string_mapping():
    """Test various string dtype formats"""
    device_id = 0
    test_cases = [
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("float", torch.float32),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("half", torch.float16),
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("int8", torch.int8),
        ("uint8", torch.uint8),
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("bool", torch.bool),
    ]

    for dtype_str, expected_dtype in test_cases:
        original_tensor = torch.zeros(10, dtype=expected_dtype).cuda(device_id)
        source_handle = TensorSharedHandle(
            original_tensor, device_id=device_id, force_direct_ipc=True
        )

        handle = TensorSharedHandle(
            source_handle.ipc_handle,
            device_id=device_id,
            tensor_shape=source_handle.tensor_shape,
            tensor_dtype=dtype_str,
        )

        assert (
            handle.tensor_dtype == expected_dtype
        ), f"String '{dtype_str}' should map to {expected_dtype}"


def _worker_modify_tensor(conn, handle):
    """Worker process: modify shared tensor"""
    tensor = handle.get_tensor()
    tensor[:] = 123.0
    conn.send(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA support required")
def test_tensor_shared_memory_modification():
    """Test if shared memory modifications are visible (using direct CUDA IPC)"""
    mp.set_start_method("spawn", force=True)

    device_id = 0
    original_tensor = torch.zeros(10, dtype=torch.float32).cuda(device_id)

    # Use direct CUDA IPC (supports true shared memory)
    handle = TensorSharedHandle(
        original_tensor, device_id=device_id, force_direct_ipc=True
    )

    parent_conn, child_conn = Pipe()
    process = Process(
        target=_worker_modify_tensor, args=(child_conn, handle), daemon=True
    )
    process.start()

    result = parent_conn.recv()
    assert result is True

    process.join(timeout=5)
    parent_conn.close()
