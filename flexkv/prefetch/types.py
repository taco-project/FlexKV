"""Pickle-safe prefetch protocol. All positions are absolute token offsets."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


class PrefetchCapacityExhausted(RuntimeError):
    """Expected admission stop when existing prefixes exhaust the pin budget."""


@dataclass(frozen=True)
class PrefetchHandle:
    epoch: str
    session_id: int


@dataclass(frozen=True)
class PrefetchOptions:
    policy: str = "wait_complete"
    chunk_max_blocks: int = 128
    max_inflight_chunks: int = 2
    timeout_base_s: float = 2.0
    timeout_per_ki_token_s: float = 0.1
    timeout_max_s: float = 30.0
    timeout_budget_s: Optional[float] = None
    candidate_start_token: int = 0
    swa_aware: bool = False

    def validate(self) -> None:
        from .policy import get_policy

        get_policy(self.policy)
        for name in ("chunk_max_blocks", "max_inflight_chunks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.candidate_start_token, bool)
            or not isinstance(self.candidate_start_token, int)
            or self.candidate_start_token < 0
        ):
            raise ValueError("candidate_start_token must be a nonnegative integer")
        for name in (
            "timeout_base_s",
            "timeout_per_ki_token_s",
            "timeout_max_s",
            "timeout_budget_s",
        ):
            value = getattr(self, name)
            if value is None and name == "timeout_budget_s":
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")

    def budget(self, candidate_tokens: int) -> float:
        if self.timeout_budget_s is not None:
            return self.timeout_budget_s
        return min(
            self.timeout_max_s,
            self.timeout_base_s + self.timeout_per_ki_token_s * candidate_tokens / 1024,
        )


@dataclass(frozen=True)
class PrefetchSnapshot:
    handle: PrefetchHandle
    version: int
    state: str
    terminal: bool
    outcome: Optional[str]
    stop_reason: Optional[str]
    planned_end_token: Optional[int]
    reusable_prefix_end_token: int
    l3_loaded_spans: Tuple[Tuple[int, int], ...]
    inflight_chunks: int
    inflight_bytes: int
    submitted_chunks: int
    sealed_submit_seq: Optional[int]
    lease_valid: bool
    error: Optional[str] = None

    @property
    def loaded_tokens(self) -> int:
        return sum(end - begin for begin, end in self.l3_loaded_spans)


@dataclass(frozen=True)
class PrefetchCapabilities:
    protocol_version: int = 1
    policies: Tuple[str, ...] = ("wait_complete", "timeout", "best_effort")
    stop_and_drain: bool = True
    partial_dense: bool = True
    partial_swa_checkpoints: bool = False
