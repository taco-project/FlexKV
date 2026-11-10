# SGLang connector readiness polling deadline

The SGLang connector uses `FLEXKV_READY_TIMEOUT_S` to stop repeatedly polling
a manager that keeps returning `False` from `is_ready()`. This is the reviewed
readiness-loop subset of upstream FlexKV #306
(`04a88abc4f4e55ae26e7853c7862b325ee3dd059`).

- Default: **360 seconds**; override with a finite positive number.
- The timer starts when `_wait_kv_manager_ready()` is entered. Earlier GPU
  registration and eventfd setup are not included in this budget.
- The default poll interval remains 10 seconds. Each sleep is capped by the
  remaining budget; a returned `False` after expiry raises `TimeoutError`.
- `is_ready()` exceptions still propagate. Invalid timeout values raise
  `ValueError` before this method's first readiness query.

This is **not** an end-to-end RPC deadline. A call blocked inside `is_ready()`
cannot be interrupted by this outer loop. A slow call that eventually returns
`True` can also exceed the polling budget. KVServer request/response deadlines,
late-response isolation and startup-failure resource cleanup are separate
contracts; this patch does not implement them.

For some special machine types and scenarios deployments, choose the budget using actual Host-pool/worker startup
measurements. A legitimate startup that keeps returning `False` beyond the
budget will now fail instead of waiting indefinitely. CPU clock/endpoint
tests do not establish a suitable production timeout or test real CUDA/RDMA
startup and teardown.
