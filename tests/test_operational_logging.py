import logging

from flexkv.common.debug import EvictionLogAggregator


class _FakeLogger:
    def __init__(self):
        self.records = []

    def debug(self, message, *args):
        self.records.append(("debug", message % args))

    def info(self, message, *args):
        self.records.append(("info", message % args))

    def warning(self, message, *args):
        self.records.append(("warning", message % args))

    def is_enabled_for(self, level):
        return level >= logging.DEBUG


def _record(aggregator, *, target_met=True, evicted_blocks=3):
    aggregator.record(
        tier="cpu",
        scope="full",
        reason="capacity",
        requested_blocks=2,
        required_blocks=3,
        evicted_blocks=evicted_blocks,
        free_blocks_before=0,
        free_blocks_after=evicted_blocks,
        total_blocks=100,
        duration_ms=1.25,
        sample_block_hashes=["0x0000000000000001"],
        target_met=target_met,
    )


def test_eviction_logging_keeps_batches_at_debug_and_aggregates_info():
    now = [0.0]
    logger = _FakeLogger()
    aggregator = EvictionLogAggregator(
        interval_s=10.0, logger=logger, clock=lambda: now[0]
    )

    _record(aggregator)
    assert [level for level, _ in logger.records] == ["debug"]

    now[0] = 10.0
    _record(aggregator)

    levels = [level for level, _ in logger.records]
    assert levels == ["debug", "debug", "info"]
    summary = logger.records[-1][1]
    assert "operation=eviction action=summary status=success" in summary
    assert "schema_version" not in summary
    assert "batches=2" in summary
    assert "evicted_blocks=6" in summary
    assert "eviction_time=0.0025s" in summary


def test_eviction_target_miss_warns_immediately():
    logger = _FakeLogger()
    aggregator = EvictionLogAggregator(
        interval_s=10.0, logger=logger, clock=lambda: 0.0
    )

    _record(aggregator, target_met=False, evicted_blocks=0)

    assert [level for level, _ in logger.records] == ["warning"]
    warning = logger.records[-1][1]
    assert "operation=eviction action=batch status=target_miss" in warning
    assert "target_met=false" in warning
    assert "eviction_time=0.0013s" in warning
