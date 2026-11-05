"""Tests for SWA Host Pool — CPU-side SWA slot-id allocation (free-list)."""
import pytest

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_host_pool import SWAHostPool

pytestmark = pytest.mark.unit


@pytest.fixture
def small_config():
    return SWAPoolConfig(
        enabled=True,
        num_slots=8,
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

    def test_duplicate_free_is_idempotent(self, pool):
        slot = pool.allocate()
        assert pool.num_free == 7
        pool.free(slot)
        pool.free(slot)
        assert pool.num_free == 8

    def test_allocate_after_free(self, pool):
        slots = [pool.allocate() for _ in range(8)]
        assert pool.allocate() is None
        pool.free(slots[0])
        new_slot = pool.allocate()
        assert new_slot is not None

    def test_reset_rearms_all_slots(self, pool):
        for _ in range(5):
            pool.allocate()
        assert pool.num_used == 5
        pool.reset()
        assert pool.num_free == 8
        assert pool.num_used == 0

    def test_config_rejects_legacy_window_size(self):
        with pytest.raises(TypeError):
            SWAPoolConfig(
                enabled=True,
                num_slots=8,
                window_size=4,
                num_swa_layers=2,
                bytes_per_token_per_layer=8,
            )
