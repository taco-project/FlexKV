"""Centralized Redis key layout for the sharing-domain era.

Every Redis key produced by FlexKV's distributed metadata layer is required
to flow through this module.  Keeping the formatting in one place lets us
audit the namespace ``sd:<sd_key>:*`` invariant (design doc §4.7) and avoid
the kind of prefix drift the legacy ``CPUB:`` / ``SSDB:`` / ``PCFSB:`` key
families suffered from.

Two layers of keys live here:

1. **Per-SD keys** — anything tied to a single sharing domain
   (``node:`` / ``meta:`` / ``buffer:`` / ``block:`` / ``aggregate:``).
2. **Cross-SD instance keys** — discovery & failure detection
   (``flexkv:instance:<id>:session`` and ``flexkv:instance:<id>:sd_nodes``).
   These are not parameterized by an SD because they describe a whole
   FlexKV instance, which spans every SD it owns.

Phase 0 task 0-B in ``docs/dist_reuse/plan.md``.
"""

from __future__ import annotations

import re
from typing import Final

from .sharing_domain import SharingDomainKey


__all__ = [
    "SharingDomainNamespace",
    "INSTANCE_KEY_PREFIX",
    "SD_KEY_PREFIX",
]


SD_KEY_PREFIX: Final[str] = "sd"
INSTANCE_KEY_PREFIX: Final[str] = "flexkv:instance"

# Hash digests are non-negative; keep an explicit type guard for callers
# that may pass arbitrary Python ints (which can be negative).
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class SharingDomainNamespace:
    """Builds Redis keys for one sharing domain.

    A namespace is **immutable after construction** — the SD key it wraps is
    a frozen dataclass and we never mutate the cached prefix.
    """

    __slots__ = ("_sd_key", "_serialized", "_prefix")

    def __init__(self, sd_key: SharingDomainKey) -> None:
        if not isinstance(sd_key, SharingDomainKey):
            raise TypeError(
                f"SharingDomainNamespace expects a SharingDomainKey, got {type(sd_key).__name__}"
            )
        self._sd_key: SharingDomainKey = sd_key
        self._serialized: str = sd_key.serialize()
        # Cache the full ``sd:<sd_key>`` prefix to avoid string concat on the
        # hot path (every block insert / publish hits a key builder).
        self._prefix: str = f"{SD_KEY_PREFIX}:{self._serialized}"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    @property
    def sd_key(self) -> SharingDomainKey:
        return self._sd_key

    @property
    def serialized_sd(self) -> str:
        """Return the bare ``sd_key`` string without the ``sd:`` prefix."""
        return self._serialized

    @property
    def prefix(self) -> str:
        """``sd:<sd_key>`` — the common prefix of every per-SD key."""
        return self._prefix

    # ------------------------------------------------------------------
    # Per-SD keys
    # ------------------------------------------------------------------
    def node_key(self, node_id: int) -> str:
        return f"{self._prefix}:node:{int(node_id)}"

    def meta_key(self, node_id: int) -> str:
        return f"{self._prefix}:meta:{int(node_id)}"

    def buffer_key(self, node_id: int, buffer_ptr: int) -> str:
        return f"{self._prefix}:buffer:{int(node_id)}:{int(buffer_ptr)}"

    def block_key(self, node_id: int, block_hash: int) -> str:
        """Per-block metadata key.

        ``block_hash`` is rendered as **lowercase hex without 0x prefix**
        because the C++ ``RedisMetaChannel::make_block_key`` will be
        retrofitted (Phase 0 task 0-D) to format it the same way.  We accept
        both signed and unsigned 64-bit hashes by masking to 64 bits first.
        """
        h = int(block_hash) & 0xFFFFFFFFFFFFFFFF
        return f"{self._prefix}:block:{int(node_id)}:{h:x}"

    def aggregate_key(self, request_prefix_hash: int) -> str:
        """Aggregate-radix marker (design doc §4.7) for tracking
        fully-ready prefixes across SDs in this instance."""
        h = int(request_prefix_hash) & 0xFFFFFFFFFFFFFFFF
        return f"{self._prefix}:aggregate:{h:x}"

    # ------------------------------------------------------------------
    # SCAN-friendly patterns
    # ------------------------------------------------------------------
    def node_key_pattern(self) -> str:
        return f"{self._prefix}:node:*"

    def meta_key_pattern(self) -> str:
        return f"{self._prefix}:meta:*"

    def buffer_key_pattern(self) -> str:
        return f"{self._prefix}:buffer:*"

    def block_key_pattern(self) -> str:
        """Match every block in the SD regardless of node_id.  Used by the
        global-SCAN optimization in design doc §4.7.1.2."""
        return f"{self._prefix}:block:*"

    def block_key_pattern_for_node(self, node_id: int) -> str:
        """Per-node block SCAN pattern (legacy path; the global pattern
        above is preferred)."""
        return f"{self._prefix}:block:{int(node_id)}:*"

    # ------------------------------------------------------------------
    # Cross-SD (instance-level) keys — static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def instance_session_key(instance_id: str) -> str:
        """Layer-1 failure-detector heartbeat key (design doc §4.3.2)."""
        SharingDomainNamespace._validate_instance_id(instance_id)
        return f"{INSTANCE_KEY_PREFIX}:{instance_id}:session"

    @staticmethod
    def instance_sd_nodes_key(instance_id: str) -> str:
        """``sd_key → node_id`` mapping written once per instance startup
        (design doc §4.7.1.6)."""
        SharingDomainNamespace._validate_instance_id(instance_id)
        return f"{INSTANCE_KEY_PREFIX}:{instance_id}:sd_nodes"

    @staticmethod
    def instance_session_key_pattern() -> str:
        return f"{INSTANCE_KEY_PREFIX}:*:session"

    @staticmethod
    def parse_instance_session_key(key: str) -> str:
        """Extract ``instance_id`` from a session key.  Raises ``ValueError``
        if the key does not match the expected layout."""
        prefix = f"{INSTANCE_KEY_PREFIX}:"
        suffix = ":session"
        if not key.startswith(prefix) or not key.endswith(suffix):
            raise ValueError(f"Not a flexkv instance session key: {key!r}")
        instance_id = key[len(prefix):-len(suffix)]
        SharingDomainNamespace._validate_instance_id(instance_id)
        return instance_id

    @staticmethod
    def _validate_instance_id(instance_id: str) -> None:
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"instance_id must be a non-empty str, got {instance_id!r}")
        if not _INSTANCE_ID_RE.match(instance_id):
            raise ValueError(
                f"instance_id {instance_id!r} contains forbidden characters; allowed [A-Za-z0-9_.-]"
            )

    # ------------------------------------------------------------------
    # Equality / hashing — useful for caching namespaces in dicts
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SharingDomainNamespace):
            return NotImplemented
        return self._sd_key == other._sd_key

    def __hash__(self) -> int:
        return hash(self._sd_key)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SharingDomainNamespace({self._serialized!r})"
