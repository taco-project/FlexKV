# SPDX-License-Identifier: Apache-2.0

"""Request/physical completion boundaries; no CUDA or radixshmem dependency."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("waiter", Path(__file__).parents[2] /
                                            "flexkv/common/radix_prefetch.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Job:
    job_id, local_hit, planned_hit = 1, 2, 8
    resources_released, complete, cancelled = False, False, False

    def done(self):
        return self.complete

    def cancel(self):
        self.cancelled = True

    def wait(self, timeout):
        assert self.complete
        return SimpleNamespace(common_hit=8, remote_blocks=6)


def test_deadline_returns_miss_while_resources_continue_draining():
    job = Job()
    waiter = m.PrefetchWaiter(job, 0)
    assert waiter.done() and job.cancelled
    assert not waiter.resources_released
    result = waiter.wait(0)
    assert result.common_hit == 2 and result.remote_blocks == 0
    job.resources_released = True
    assert waiter.resources_released


def test_independent_waiter_cancellation_does_not_report_physical_completion():
    job = Job()
    waiter = m.PrefetchWaiter(job, 1000)
    waiter.cancel()
    assert waiter.done() and waiter.cancelled
    assert not waiter.resources_released
    assert waiter.wait(0).remote_bytes == 0


def test_completed_pull_returns_verified_result_before_deadline():
    job = Job()
    job.complete = job.resources_released = True
    waiter = m.PrefetchWaiter(job, 1000)
    assert waiter.done() and not job.cancelled
    assert waiter.wait(0).common_hit == 8
