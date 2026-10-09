"""Pure, startup-registered stop policies; resource cleanup is never a policy."""

import threading
from typing import Callable, Dict, Optional

Policy = Callable[[str, float, Optional[float]], Optional[str]]
_policies: Dict[str, Policy] = {}
_lock = threading.Lock()
_frozen = False


def register_prefetch_policy(name: str, policy: Policy) -> None:
    with _lock:
        if _frozen:
            raise RuntimeError("prefetch policy registry is frozen")
        if not name or name in _policies or not callable(policy):
            raise ValueError(f"invalid or duplicate prefetch policy: {name!r}")
        _policies[name] = policy


def freeze_policies() -> None:
    global _frozen
    with _lock:
        _frozen = True


def get_policy(name: str) -> Policy:
    try:
        return _policies[name]
    except KeyError as exc:
        raise ValueError(f"unknown prefetch policy: {name!r}") from exc


def _wait_complete(event, now, deadline):
    return None


def _timeout(event, now, deadline):
    return "deadline" if deadline is not None and now >= deadline else None


def _best_effort(event, now, deadline):
    return "demand" if event == "demand" else None


register_prefetch_policy("wait_complete", _wait_complete)
register_prefetch_policy("timeout", _timeout)
register_prefetch_policy("best_effort", _best_effort)
