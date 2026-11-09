"""Async send ownership and error propagation, without a GPU or live peer."""
import pytest

from flexkv.integration.sglang.comm import FlexKVComm


class Work:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def wait(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        # A timeout is a transport operation, not a harmless readiness probe.
        assert not args and not kwargs, 'reaper must use the process-group timeout'
        if self.error is not None:
            raise self.error
        return True

    def is_completed(self):
        raise AssertionError('Gloo completion must be driven by wait()')


def comm(works, high=4, drain=2):
    obj = FlexKVComm.__new__(FlexKVComm)
    obj.world_rank = 0
    obj._async_works = list(works)
    obj._reap_high = high
    obj._REAP_MAX_DRAIN = drain
    obj._reap_calls = 0
    obj._reap_drained_total = 0
    obj._reap_stuck_total = 0
    return obj


def test_below_watermark_retains_ownership_without_waiting():
    works = [Work() for _ in range(4)]
    obj = comm(works)
    obj._reap_completed_async_works()
    assert obj._async_works == works
    assert all(not item.calls for item in works)


def test_backpressure_waits_without_short_timeout_and_drains_fifo():
    works = [Work() for _ in range(7)]
    obj = comm(works)
    obj._reap_completed_async_works()
    assert obj._async_works == works[2:]
    assert [len(item.calls) for item in works] == [1, 1, 0, 0, 0, 0, 0]
    assert obj._reap_high == 4
    assert obj._reap_drained_total == 2


def test_real_communication_error_propagates_and_retains_failed_work():
    error = RuntimeError('peer disconnected')
    works = [Work(), Work(error), Work(), Work(), Work()]
    obj = comm(works, drain=5)
    with pytest.raises(RuntimeError, match='peer disconnected') as caught:
        obj._reap_completed_async_works()
    assert caught.value is error
    assert obj._async_works == works[1:]
    assert all(not item.calls for item in works[2:])


def test_empty_queue_is_a_noop():
    obj = comm([])
    obj._reap_completed_async_works()
    assert obj._reap_calls == obj._reap_drained_total == 0
