"""Tests for SWA Host Pool — CPU pinned memory slot allocation."""
import numpy as np
import pytest

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_host_pool import SWAHostPool


@pytest.fixture
def small_config():
    return SWAPoolConfig(
        enabled=True,
        num_slots=8,
        window_size=4,
        num_swa_layers=2,
        bytes_per_token_per_layer=8,
        pin_memory=False,  # CPU-only tests
    )


@pytest.fixture
def pool(small_config):
    return SWAHostPool(small_config)


class TestSWAHostPoolAllocation:
    def test_initial_state(self, pool):
        assert pool.num_free == 8
        assert pool.num_used == 0
        assert pool.num_slots == 8

    def test_allocate_one(self, pool):
        slot = pool.allocate()
        assert slot is not None
        assert 0 <= slot < 8
        assert pool.num_free == 7
        assert pool.num_used == 1

    def test_allocate_all(self, pool):
        slots = []
        for _ in range(8):
            s = pool.allocate()
            assert s is not None
            slots.append(s)
        assert pool.num_free == 0
        assert pool.allocate() is None  # Pool full

    def test_free(self, pool):
        slot = pool.allocate()
        pool.free(slot)
        assert pool.num_free == 8

    def test_allocate_after_free(self, pool):
        slots = [pool.allocate() for _ in range(8)]
        assert pool.allocate() is None
        pool.free(slots[0])
        new_slot = pool.allocate()
        assert new_slot is not None


class TestSWAHostPoolIO:
    def test_write_read_numpy(self, pool, small_config):
        slot = pool.allocate()
        data = np.arange(small_config.slot_size_bytes, dtype=np.uint8)
        pool.write(slot, data)
        result = pool.read(slot)
        np.testing.assert_array_equal(np.asarray(result)[:len(data)], data)

    def test_write_read_bytes(self, pool, small_config):
        slot = pool.allocate()
        data = bytes(range(min(256, small_config.slot_size_bytes)))
        pool.write(slot, data)
        result = np.asarray(pool.read(slot))
        np.testing.assert_array_equal(result[:len(data)], np.frombuffer(data, dtype=np.uint8))

    def test_slot_size(self, pool, small_config):
        expected = 4 * 2 * 8  # window_size * layers * bytes
        assert pool.slot_size_bytes == expected

    def test_read_copy_independence(self, pool, small_config):
        slot = pool.allocate()
        data = np.ones(small_config.slot_size_bytes, dtype=np.uint8) * 42
        pool.write(slot, data)
        copy1 = pool.read_copy(slot)
        copy2 = pool.read_copy(slot)
        # Modify copy1, copy2 should be unaffected
        np.asarray(copy1)[0] = 99
        assert np.asarray(copy2)[0] == 42
