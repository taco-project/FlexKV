"""Sharing Domain abstraction for distributed KV cache reuse.

A *sharing domain* (SD) groups together FlexKV instances that hold the **same
KV slice** along two orthogonal dimensions:

- ``pp_node_idx`` (which physical node within the PP pipeline) — which
  layer range
- ``tp_node_idx`` (cross-node TP node index) — which KV head shard

**CP is not part of the SD key.**  Based on the code-fact review (see
``docs/dist_reuse/dist_reuse_with_cp_pp_multinode_tp.md`` §1.3 / §2.3 / §4.5):
CP attention does an all-gather before writing back to the local
``token_to_kv_pool``, so every ``cp_rank`` in the same CP group holds
bit-wise identical main KV and (for NSA) indexer K.  Cross-instance reuse is
therefore legal at the CP-group granularity — the SD layer does not need
``cp_rank``/``cp_size`` isolation.  The CP dimension is handled purely as a
sync_leader control-plane + scatter-based data-plane concern in the
connector layer.

**Why node-granularity for PP, not rank-granularity?**  The KV cache
sharing boundary is the *physical CPU pool* of a node — every PP rank
co-located on the same machine writes into the *same* CPU pool.  Keying
the SD by PP rank would (a) inflate the SD count to ``pp_size`` even on
single-node PP=2/4 deployments where no remote-peer SD actually exists,
and (b) make the master coordinator believe two PP ranks on the same
machine are remote peers.  Keying by ``pp_node_idx`` collapses
co-located PP ranks into the same SD, so SD count exactly equals the
number of physical nodes participating in this instance's KV sharing.

  ``pp_node_count = max(min(pp_size, nnodes), 1)``
  ``pp_per_node   = max(pp_size // pp_node_count, 1)``
  ``pp_node_idx   = pp_rank // pp_per_node``

Only instances that share the same ``(pp_node_idx, tp_node_idx)`` pair
(plus the same ``model_id`` and ``is_nsa`` layout flag) may participate in
P2P KVCache reuse with each other.  See design doc §3.1 / §4.1 / §4.5.

The serialized form is used as a Redis key namespace prefix:

    sd:<model_id>:ppn<pp_node_idx>/<pp_node_count>:tpn<tp_node_idx>/<tp_node_count>:nsa<0|1>:<...>

``is_nsa`` distinguishes NSA model layouts (extra indexer K cache buffer)
from non-NSA models; it is independent of whether CP is enabled.

**IMPORTANT — sd_key equality is NOT physical-pool equality.**
Two SDs being equal only means "same SD namespace, eligible to negotiate
KV reuse".  It does *not* imply that the underlying CPU pools store the
same physical layer range.  In particular, on a single-node PP=2/4
deployment all PP ranks collapse to ``pp_node_idx=0``, but each rank
physically owns a disjoint ``[pp_start_layer, pp_end_layer)`` slice of
the CPU pool — so cross-rank KV reads inside the same node would return
the wrong layer's bytes.  This is why
:func:`SharingDomainKey.from_model_config` raises on
``pp_size > 1 and nnodes == 1`` until the CPU pool is reworked to store
full-layer KV.

This module is **pure Python** and has no runtime dependency on RedisMeta /
Mooncake / CacheEngine, so it is safe to import from any layer (config,
transfer manager, tests).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, List, Optional


__all__ = [
    "SharingDomainKey",
    "DEFAULT_MODEL_ID",
    "derive_model_id",
]


# Sentinel used by :meth:`SharingDomainKey.default` to mark the degenerate
# single-SD fallback (when sharing-domain support is disabled).  Anything that
# uses ``DEFAULT_MODEL_ID`` is opting out of any cross-instance reuse.
DEFAULT_MODEL_ID = "__default__"


# A model_id must be safe to embed in a Redis key.  We restrict it to
# ``[A-Za-z0-9_.-]`` so the resulting ``sd:`` prefix never accidentally
# contains the ``:`` separator we use ourselves.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def derive_model_id(
    *,
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    dtype: Any,
    use_mla: bool,
) -> str:
    """Build a stable, process-independent ``model_id`` from architecture knobs.

    Two FlexKV instances that produce **physically interchangeable** CPU blocks
    (same layer count, same KV head count, same head size, same dtype, same
    MLA flag) will derive the same ``model_id`` regardless of process / host.

    The dtype is normalized to its ``str`` form (e.g. ``"torch.bfloat16"``)
    so that Python ``hash()`` randomization does not leak in.  The result is
    a 16-char hex digest, short enough to keep Redis keys compact and long
    enough to make collisions a non-issue across realistic deployments.
    """
    payload = f"{int(num_layers)}|{int(num_kv_heads)}|{int(head_size)}|{dtype!s}|{int(bool(use_mla))}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def _compute_pp_node_dims(pp_size: int, nnodes: int) -> tuple[int, int]:
    """Derive ``(pp_node_count, pp_per_node)`` from a model topology.

    Rules:

    * Single-node deployment (``nnodes == 1``): all PP ranks live on
      one machine → ``pp_node_count = 1``, ``pp_per_node = pp_size``.
      (For ``pp_size > 1`` the caller is responsible for refusing
      dist_reuse — see :meth:`SharingDomainKey.from_model_config`.)
    * PP-pipelining stretches across nodes (``pp_size >= nnodes``):
      ``pp_node_count = nnodes``, ``pp_per_node = pp_size // nnodes``.
    * Cross-node TP only (``pp_size < nnodes``, e.g. ``PP=1 × nnodes=2``):
      PP does not span multiple nodes → ``pp_node_count = 1``,
      ``pp_per_node = pp_size`` (which equals 1 here in practice).
    """
    pp_size = max(int(pp_size), 1)
    nnodes = max(int(nnodes), 1)
    pp_node_count = max(min(pp_size, nnodes), 1)
    pp_per_node = max(pp_size // pp_node_count, 1)
    return pp_node_count, pp_per_node


@dataclass(frozen=True)
class SharingDomainKey:
    """Immutable identifier of a single sharing domain.

    Attributes are validated lazily on construction (see ``__post_init__``).
    Comparison / hashing is by value, so two ``SharingDomainKey`` instances
    with identical fields are interchangeable as ``dict`` keys.

    Fields:
        model_id: topology-derived (or user-set) identifier of the model
            architecture.  Two instances with the same ``model_id`` produce
            physically interchangeable CPU blocks.
        pp_node_idx / pp_node_count: index of the physical node this SD
            sits on within the pipeline-parallel axis, and the total
            number of PP-spanning nodes for this instance.  Always
            ``0/1`` on a single-node deployment regardless of ``pp_size``.
        tp_node_idx / tp_node_count: cross-node TP shard index and count
            (always 1 / 1 when TP fits within a single node).
        is_nsa: NSA-model layout flag.  ``True`` means the model has an
            extra indexer K cache buffer (NSA / DeepSeek-V3-sparse-attn);
            ``False`` means plain MLA / MHA.  Independent of CP.
    """

    model_id: str
    pp_node_idx: int
    pp_node_count: int
    tp_node_idx: int
    tp_node_count: int
    is_nsa: bool

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError(f"SharingDomainKey.model_id must be a non-empty str, got {self.model_id!r}")
        if not _MODEL_ID_RE.match(self.model_id):
            raise ValueError(
                f"SharingDomainKey.model_id {self.model_id!r} contains forbidden characters; "
                f"allowed: [A-Za-z0-9_.-]"
            )

        for name, val in (
            ("pp_node_count", self.pp_node_count),
            ("tp_node_count", self.tp_node_count),
        ):
            if not isinstance(val, int) or val < 1:
                raise ValueError(f"SharingDomainKey.{name} must be int>=1, got {val!r}")

        for name, idx, count in (
            ("pp_node_idx", self.pp_node_idx, self.pp_node_count),
            ("tp_node_idx", self.tp_node_idx, self.tp_node_count),
        ):
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"SharingDomainKey.{name} must be int>=0, got {idx!r}")
            if idx >= count:
                raise ValueError(
                    f"SharingDomainKey.{name}={idx} out of range for "
                    f"{name.replace('idx', 'count')}={count}"
                )

        if not isinstance(self.is_nsa, bool):
            raise ValueError(f"SharingDomainKey.is_nsa must be bool, got {self.is_nsa!r}")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def serialize(self) -> str:
        """Stable string form, usable as a Redis namespace prefix.

        Format::

            <model_id>:ppn<pp_node_idx>/<pp_node_count>:tpn<tp_node_idx>/<tp_node_count>:nsa<0|1>

        The ``sd:`` prefix is **not** included here — it is added by
        :class:`SharingDomainNamespace` when building actual Redis keys.

        Note: CP is intentionally absent from this key.  See module docstring
        and design doc §4.5 for the rationale (CP all-gather makes all
        ``cp_rank`` pool contents bit-wise identical within a CP group).
        """
        nsa = 1 if self.is_nsa else 0
        return (
            f"{self.model_id}"
            f":ppn{self.pp_node_idx}/{self.pp_node_count}"
            f":tpn{self.tp_node_idx}/{self.tp_node_count}"
            f":nsa{nsa}"
        )

    @classmethod
    def deserialize(cls, s: str) -> "SharingDomainKey":
        """Parse the canonical form produced by :meth:`serialize`.

        Raises ``ValueError`` on malformed input.  Round-trip with
        :meth:`serialize` is guaranteed.
        """
        # Split on ':' but the model_id itself is forbidden from containing
        # ':' (see _MODEL_ID_RE), so the leading segment is unambiguous.
        parts = s.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"SharingDomainKey.deserialize: expected 4 ':'-separated segments, got {len(parts)} in {s!r}"
            )
        model_id, ppn_part, tpn_part, nsa_part = parts

        pp_node_idx, pp_node_count = cls._parse_idx_count(ppn_part, "ppn")
        tp_node_idx, tp_node_count = cls._parse_idx_count(tpn_part, "tpn")

        if not nsa_part.startswith("nsa") or nsa_part[3:] not in ("0", "1"):
            raise ValueError(f"SharingDomainKey.deserialize: bad nsa segment {nsa_part!r} in {s!r}")
        is_nsa = nsa_part[3:] == "1"

        return cls(
            model_id=model_id,
            pp_node_idx=pp_node_idx,
            pp_node_count=pp_node_count,
            tp_node_idx=tp_node_idx,
            tp_node_count=tp_node_count,
            is_nsa=is_nsa,
        )

    @staticmethod
    def _parse_idx_count(seg: str, prefix: str) -> tuple[int, int]:
        if not seg.startswith(prefix):
            raise ValueError(f"SharingDomainKey: segment {seg!r} does not start with {prefix!r}")
        body = seg[len(prefix):]
        if "/" not in body:
            raise ValueError(f"SharingDomainKey: segment {seg!r} missing '/' between idx and count")
        idx_s, count_s = body.split("/", 1)
        try:
            return int(idx_s), int(count_s)
        except ValueError as e:
            raise ValueError(f"SharingDomainKey: cannot parse idx/count from {seg!r}: {e}") from e

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------
    @classmethod
    def default(cls) -> "SharingDomainKey":
        """The degenerate single-SD fallback used when
        ``CacheConfig.enable_sharing_domain`` is False.

        All dimensions collapse to count 1 / idx 0, ``is_nsa=False`` and
        ``model_id=DEFAULT_MODEL_ID``.  Two instances opting into the default
        SD always reuse with each other regardless of model topology (matches
        the legacy single-instance dist_reuse semantics).
        """
        return cls(
            model_id=DEFAULT_MODEL_ID,
            pp_node_idx=0, pp_node_count=1,
            tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        *,
        rank_info: Any = None,
        overrides: Optional[dict] = None,
    ) -> "SharingDomainKey":
        """Derive the SD key for *this node* from a ``ModelConfig``.

        Per-rank physical fields (``pp_rank``, ``tp_node_idx``)
        historically lived on ``ModelConfig`` but were moved into
        ``RankInfo`` by the RankInfo refactor (PR #165).  Callers that
        have a ``rank_info`` should pass it as the keyword argument; in
        that case ``pp_rank`` / ``tp_node_idx`` come from ``rank_info``
        and the per-cluster fields (``pp_size``, ``nnodes``,
        ``tp_node_count``, ``model_id``, ``is_nsa``) from
        ``model_config``.  When ``rank_info`` is ``None`` we fall back
        to reading the per-rank fields from ``model_config`` itself
        (backwards-compatible for legacy stubs in unit tests and the
        older ModelConfig layout).

        The PP-axis is collapsed from rank-granularity to
        node-granularity::

            pp_node_count = max(min(pp_size, nnodes), 1)
            pp_per_node   = max(pp_size // pp_node_count, 1)
            pp_node_idx   = pp_rank // pp_per_node

        which means PP ranks co-located on the same physical node share
        the same SD key (same Redis namespace).  See module docstring
        for why.

        ``overrides`` lets a Master node craft an SD key for a Remote
        node on-the-fly (e.g. setting ``pp_node_idx=1`` while the Master
        itself is on ``pp_node_idx=0``).  Only the six dataclass fields are
        valid keys.

        ``is_nsa`` is read from ``model_config.is_nsa``.

        Raises:
            ValueError: if ``pp_size > 1 and nnodes == 1`` — single-node
                PP>1 deployments cannot participate in dist_reuse until
                the CPU pool is reworked to store full-layer KV (see
                module docstring).  This guard is **only** active when
                the caller is opting into a sharing domain (i.e. went
                through this factory); the
                :meth:`SharingDomainKey.default` fallback path is
                untouched.
        """
        pp_size = max(int(getattr(model_config, "pp_size", 1)), 1)
        nnodes = max(int(getattr(model_config, "nnodes", 1)), 1)

        if pp_size > 1 and nnodes == 1:
            raise ValueError(
                "SharingDomainKey.from_model_config: dist_reuse is not supported "
                f"on a single-node PP>1 deployment (pp_size={pp_size}, nnodes=1). "
                "Each PP rank physically owns only a slice of the CPU pool layers, "
                "so a single sd_key would alias incompatible layer shards.  "
                "Either disable enable_sharing_domain, scale to nnodes>=pp_size, "
                "or wait for the full-layer CPU pool refactor."
            )

        pp_node_count, pp_per_node = _compute_pp_node_dims(pp_size, nnodes)

        # Per-rank fields: prefer ``rank_info`` (post-refactor source of
        # truth); fall back to ``model_config`` for legacy callers.
        if rank_info is not None:
            _pp_rank = int(getattr(rank_info, "pp_rank", 0))
            _tp_node_idx = int(
                getattr(rank_info, "tp_node_idx",
                        getattr(model_config, "tp_node_idx", 0))
            )
        else:
            # Legacy path: PR #165 moved ``pp_rank`` / ``tp_node_idx``
            # off ``ModelConfig`` onto :class:`RankInfo`.  These
            # ``getattr(..., 0)`` reads therefore now return ``0``
            # for any post-#165 ``ModelConfig`` instance — i.e. the
            # caller silently gets the master-position SD even if it
            # is actually on (pp_rank>0, tp_node_idx>0).  This is
            # only safe for unit-test fakes that explicitly set
            # ``pp_rank`` / ``tp_node_idx`` on the stub ModelConfig
            # (see ``tests/test_sharing_domain_key.py``); production
            # callers should pass ``rank_info=`` explicitly.  We log
            # a one-time warning when the heuristic is exercised on
            # a multi-rank topology so the error surfaces during
            # bring-up instead of silently corrupting Redis keys.
            _pp_rank = int(getattr(model_config, "pp_rank", 0))
            _tp_node_idx = int(getattr(model_config, "tp_node_idx", 0))
            _has_pp_rank = hasattr(model_config, "pp_rank")
            _has_tp_node_idx = hasattr(model_config, "tp_node_idx")
            if (
                pp_size > 1 or int(getattr(model_config, "tp_node_count", 1)) > 1
            ) and not (_has_pp_rank and _has_tp_node_idx):
                # Local import to avoid pulling logger into module
                # import-time graph (sharing_domain.py is imported
                # very early by config.py via ``derive_model_id``).
                from flexkv.common.debug import flexkv_logger
                flexkv_logger.warning(
                    "SharingDomainKey.from_model_config: called without "
                    "rank_info on a multi-rank topology (pp_size=%d, "
                    "tp_node_count=%s); per-rank fields default to 0, "
                    "which only matches the master SD.  Pass rank_info="
                    "<RankInfo> so the per-rank position is honoured.",
                    pp_size,
                    getattr(model_config, "tp_node_count", 1),
                )

        _pp_node_idx = _pp_rank // pp_per_node

        kwargs: dict = {
            "model_id": _resolve_model_id(model_config),
            "pp_node_idx": _pp_node_idx,
            "pp_node_count": pp_node_count,
            "tp_node_idx": _tp_node_idx,
            "tp_node_count": int(getattr(model_config, "tp_node_count", 1)),
            "is_nsa": bool(_resolve_is_nsa(model_config)),
        }
        if overrides:
            unknown = set(overrides) - set(kwargs)
            if unknown:
                raise ValueError(
                    f"SharingDomainKey.from_model_config: unknown override keys {sorted(unknown)}"
                )
            kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def is_master(self) -> bool:
        """The Master role of an instance is the SD with both node-idx dims = 0.

        Note: CP is not part of the SD key, so whether a physical node is
        the *sync_leader* (cp_rank=0 within a CP group) is determined at the
        connector layer, not here.  This method only tells you whether this
        SD is the (ppn=0, tpn=0) SD of its instance.
        """
        return self.pp_node_idx == 0 and self.tp_node_idx == 0

    def total_sd_count(self) -> int:
        """Total number of SDs per instance (= ``pp_node_count × tp_node_count``).

        This equals the number of *physical nodes* spanned by the
        instance's KV-sharing topology, which is at most ``nnodes`` for
        any deployment.  CP does not multiply this.
        """
        return self.pp_node_count * self.tp_node_count

    def enumerate_peers(self) -> List["SharingDomainKey"]:
        """Enumerate every SD belonging to the *same instance* as ``self``.

        Order is deterministic: outer loop ``pp_node_idx``, inner loop
        ``tp_node_idx``.  Always returns ``self.total_sd_count()`` items.

        CP dimension is not enumerated here — see module docstring for why.
        """
        out: List["SharingDomainKey"] = []
        for ppn in range(self.pp_node_count):
            for tpn in range(self.tp_node_count):
                out.append(replace(self, pp_node_idx=ppn, tp_node_idx=tpn))
        return out

    def __str__(self) -> str:  # pragma: no cover — purely cosmetic
        return self.serialize()


def _resolve_model_id(model_config: Any) -> str:
    """Pick up an explicit ``model_id`` if the user set one, otherwise derive."""
    explicit = getattr(model_config, "model_id", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    # Fall back to topology-only digest.  Important: do *not* depend on
    # tp_size/pp_size/cp_size here — those are dimensions of the SD key
    # itself, not of the underlying model.
    return derive_model_id(
        num_layers=int(getattr(model_config, "num_layers", 1)),
        num_kv_heads=int(getattr(model_config, "num_kv_heads", 1)),
        head_size=int(getattr(model_config, "head_size", 1)),
        dtype=getattr(model_config, "dtype", "unknown"),
        use_mla=bool(getattr(model_config, "use_mla", False)),
    )


def _resolve_is_nsa(model_config: Any) -> bool:
    """Read the NSA layout flag from the model config.

    Only the ``is_nsa`` attribute is consulted.  Defaults to ``False`` when
    the attribute is absent so that non-NSA configs (older vllm / sglang
    branches that never knew about NSA) continue to produce valid SD keys.
    """
    return bool(getattr(model_config, "is_nsa", False))
