# SPDX-License-Identifier: Apache-2.0

import time
from types import SimpleNamespace


class PrefetchWaiter:
    """A request deadline is independent of the physical pull's ownership.

    RadixClient owns the transfer until release/ACK, including after this
    waiter returns a miss. Each caller has its own waiter; node-wide exact-key
    singleflight is implemented by the RadixServer admission protocol.
    """

    def __init__(self, job, timeout_ms):
        self._job = job
        self._deadline = time.monotonic() + timeout_ms / 1000
        self._expired = self._cancelled = False
        self.job_id = job.job_id
        self.local_hit, self.planned_hit = job.local_hit, job.planned_hit

    @property
    def resources_released(self):
        return bool(getattr(self._job, "resources_released", self._job.done()))

    @property
    def cancelled(self):
        return self._cancelled

    def done(self):
        if not self._job.done() and time.monotonic() >= self._deadline and not self._expired:
            self._expired = True
            self._job.cancel()
        return self._expired or self._cancelled or self._job.done()

    def wait(self, timeout=None):
        self.done()
        if self._expired or self._cancelled:
            return SimpleNamespace(common_hit=self.local_hit, remote_blocks=0, remote_bytes=0,
                                   source_rank=-1, finalize=lambda: None, fallback_reason="deadline")
        return self._job.wait(timeout)

    def cancel(self):
        self._cancelled = True
        self._job.cancel()
