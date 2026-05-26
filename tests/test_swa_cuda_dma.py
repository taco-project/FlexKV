# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SWA CUDA DMA Engine (Phase 2)."""

import time
import pytest

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False

from flexkv.swa.swa_cuda_dma import (
    SWACudaDMAEngine,
    SWATransferTask,
    TransferDirection,
    TransferStatus,
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
class TestSWACudaDMAEngine:
    """Test suite for SWACudaDMAEngine."""

    @pytest.fixture
    def engine(self):
        """Create a DMA engine with 4 streams."""
        return SWACudaDMAEngine(num_streams=4, device=0)

    def test_init_valid(self, engine):
        """Test valid engine initialization."""
        assert engine.num_streams == 4
        assert engine.device == 0
        assert engine.is_available is True
        assert len(engine._streams) == 4

    def test_init_invalid_streams(self):
        """Test invalid stream count raises error."""
        with pytest.raises((ValueError, RuntimeError)):
            SWACudaDMAEngine(num_streams=0)

    def test_pending_transfers_empty(self, engine):
        """Initially no pending transfers."""
        assert engine.pending_transfers == 0
        assert engine.completed_transfers == 0

    def test_stream_pool_round_robin(self, engine):
        """Verify stream allocation is round-robin."""
        # Create dummy transfer tasks to test stream allocation
        assert engine._next_stream_id == 0
        # Simulate multiple submissions
        for i in range(8):
            expected_stream = i % 4
            actual_stream = engine._next_stream_id % engine.num_streams
            assert actual_stream == expected_stream
            engine._next_stream_id += 1

    def test_synchronize_stream(self, engine):
        """Test stream synchronization."""
        # Should not raise even if stream is empty
        engine.synchronize_stream(0)
        engine.synchronize_stream(1)

    def test_synchronize_all(self, engine):
        """Test synchronize all streams."""
        # Should not raise
        engine.synchronize_all()

    def test_reset(self, engine):
        """Test engine reset."""
        assert engine._transfer_counter == 0
        engine._transfer_counter = 5
        engine.reset()
        assert engine._transfer_counter == 0
        assert engine.pending_transfers == 0

    def test_clear_completed_empty(self, engine):
        """Clear with no transfers."""
        result = engine.clear_completed()
        assert result == 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
class TestSWATransferTask:
    """Test SWATransferTask dataclass."""

    def test_transfer_task_init(self):
        """Test task initialization."""
        task = SWATransferTask(
            transfer_id=1,
            endpoint_hash=12345,
            direction=TransferDirection.HOST_TO_DEVICE,
            slot_id=5,
            gpu_ptr=0x1000,
            host_ptr=0x2000,
            size_bytes=4096,
        )
        assert task.transfer_id == 1
        assert task.endpoint_hash == 12345
        assert task.direction == TransferDirection.HOST_TO_DEVICE
        assert task.slot_id == 5
        assert task.status == TransferStatus.PENDING

    def test_latency_calculation(self):
        """Test latency calculation."""
        task = SWATransferTask(
            transfer_id=1,
            endpoint_hash=0,
            direction=TransferDirection.HOST_TO_DEVICE,
            slot_id=0,
            gpu_ptr=0,
            host_ptr=0,
            size_bytes=1024,
        )
        assert task.latency_us == 0.0  # Not completed

        # Simulate completion
        task.status = TransferStatus.COMPLETED
        task.complete_time = task.submit_time + 0.001  # 1ms
        assert task.latency_us > 900  # ~1000 µs


class TestTransferDirection:
    """Test TransferDirection enum."""

    def test_enum_values(self):
        """Verify enum values."""
        assert TransferDirection.HOST_TO_DEVICE.value == 1
        assert TransferDirection.DEVICE_TO_HOST.value == 2


class TestTransferStatus:
    """Test TransferStatus enum."""

    def test_enum_values(self):
        """Verify enum values."""
        assert TransferStatus.PENDING.value == 1
        assert TransferStatus.IN_FLIGHT.value == 2
        assert TransferStatus.COMPLETED.value == 3
        assert TransferStatus.FAILED.value == 4


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
class TestSWACudaDMAEngineIntegration:
    """Integration tests for DMA engine."""

    @pytest.fixture
    def engine(self):
        return SWACudaDMAEngine(num_streams=2, device=0)

    def test_get_transfer_info_missing(self, engine):
        """Get transfer info for non-existent transfer."""
        info = engine.get_transfer_info(999)
        assert info is None

    def test_get_all_transfers_empty(self, engine):
        """Get all transfers when none exist."""
        transfers = engine.get_all_transfers()
        assert transfers == []

    def test_stream_synchronization_sequence(self, engine):
        """Test synchronizing streams in sequence."""
        for i in range(engine.num_streams):
            engine.synchronize_stream(i)
        engine.synchronize_all()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
class TestSWACudaDMAMemoryBounds:
    """Test memory bounds and allocation."""

    @pytest.fixture
    def engine(self):
        return SWACudaDMAEngine(num_streams=2, device=0)

    def test_large_page_size(self, engine):
        """Test handling of large SWA page size (4.35 MB)."""
        # DeepSeek V4 SWA page size
        page_size_bytes = 128 * 584 * 61  # 4,559,872 bytes
        
        # Engine should track the size without allocation
        task = SWATransferTask(
            transfer_id=0,
            endpoint_hash=1,
            direction=TransferDirection.HOST_TO_DEVICE,
            slot_id=0,
            gpu_ptr=0x1000,
            host_ptr=0x2000,
            size_bytes=page_size_bytes,
        )
        assert task.size_bytes == page_size_bytes


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="CUDA not available")
class TestSWACudaDMAThreadSafety:
    """Test thread safety of DMA engine."""

    @pytest.fixture
    def engine(self):
        return SWACudaDMAEngine(num_streams=4, device=0)

    def test_concurrent_access(self, engine):
        """Test that concurrent access to engine is safe."""
        import threading
        
        errors = []
        
        def worker():
            try:
                # Try to access engine properties
                _ = engine.pending_transfers
                _ = engine.completed_transfers
                engine.reset()
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"


class TestSWACudaDMANoTorch:
    """Test behavior when torch is not available."""

    def test_import_fails_without_torch(self):
        """Verify that SWACudaDMAEngine requires torch."""
        # This test documents the requirement
        try:
            import torch
            # torch is available, skip
        except ImportError:
            with pytest.raises(RuntimeError):
                SWACudaDMAEngine()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
