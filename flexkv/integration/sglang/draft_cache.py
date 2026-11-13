"""Physical draft members of one atomic FlexKV cache entry.

Compatible rows join an existing group as tail layers (packed). Other rows
remain native heterogeneous groups (sidecars), in the same registration and
transfer task. No independent cache index or publication is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json


def _signature(group):
    sample = group["buffers"][0]
    return (
        group["ratio"],
        group["sub_page_size"],
        tuple(sample.shape),
        tuple(sample.stride()),
        str(sample.dtype),
        str(sample.device),
    )


def _group(name, buffers, layers, page, ratio=1):
    if not buffers or all(b.numel() == 0 for b in buffers):
        return None
    sample = buffers[0]
    if sample.ndim != 2 or not sample.is_contiguous():
        raise ValueError(f"{name}: expected contiguous physical page rows")
    if sample.shape[1] % page:
        raise ValueError(f"{name}: row width does not tile its physical page")
    if len(buffers) != len(layers) or any(
        b.shape != sample.shape or b.dtype != sample.dtype or b.device != sample.device or not b.is_contiguous()
        for b in buffers
    ):
        raise ValueError(f"{name}: inconsistent layer buffers")
    return dict(
        name=name,
        buffers=list(buffers),
        layer_ids=list(layers),
        sub_page_size=page,
        ratio=ratio,
        bytes_per_token=sample.shape[1] // page * sample.element_size(),
        dtype=sample.dtype,
    )


def _paged_rows(buffers, page, name):
    rows = []
    for buffer in buffers:
        if buffer.ndim != 3 or not buffer.is_contiguous() or buffer.shape[0] % page:
            raise ValueError(f"{name}: MLA token rows must tile full pages")
        rows.append(buffer.view(buffer.shape[0] // page, -1))
    return rows


def _append(groups, candidate):
    if candidate is None:
        return "empty"
    for existing in groups:
        if _signature(existing) == _signature(candidate):
            existing["buffers"].extend(candidate["buffers"])
            existing["layer_ids"].extend(candidate["layer_ids"])
            return "packed"
    groups.append(candidate)
    return "sidecar"


@dataclass
class DraftCachePlan:
    full_groups: list
    swa_groups: list
    target_layers: int
    transfer_layers: int
    modes: list
    fingerprint: str


def build_draft_cache_plan(*, target_pool, draft_pools, target_layers, page_size, target_groups, is_dsv4, identity):
    """Build a PP1 plan for MLA and DeepSeek split pools with shared slot maps.

    Request rings and unified/fused arenas have different lifetimes. They are
    rejected here until their owning reconstruction/alias contract is passed
    explicitly, instead of serializing request-scoped state as prefix pages.
    """
    full = [{**g, "buffers": list(g["buffers"]), "layer_ids": list(g["layer_ids"])} for g in target_groups]
    swa = []
    target_swa = getattr(target_pool, "swa_kv_pool", None)
    if target_swa is not None:
        buffers = list(target_swa.kv_buffer)
        ids = list(range(len(buffers)))
        swa.append(_group("swa", buffers, ids, int(target_swa.page_size)))
    modes = []
    tail = target_layers
    for index, pool in enumerate(draft_pools):
        count = int(pool.layer_num)
        layers = list(range(tail, tail + count))
        if getattr(pool, "_unified_kv", False) or getattr(pool, "swa_req_ring_size", None) is not None:
            raise NotImplementedError("FlexKV draft request rings/unified arenas require reconstruction support")
        if hasattr(pool, "compression_ratios"):
            start = int(getattr(pool, "_stage_start", 0))
            end = int(getattr(pool, "_stage_end", start + count))
            if any(pool.compression_ratios[start:end]):
                raise NotImplementedError("FlexKV compressed full draft pools are not supported yet")
            draft_swa = getattr(pool, "swa_kv_pool", None)
            if target_swa is None or draft_swa is None:
                raise NotImplementedError("Paged DeepSeek draft SWA requires the target SWA mapping")
            # SGLang binds the draft to the target allocator. Both paged SWA
            # pools must cover the same physical slot space; a full-capacity
            # identity-mapped draft must not follow the target SWA map.
            if int(getattr(pool, "swa_size", -1)) != int(getattr(target_pool, "swa_size", -2)):
                raise ValueError("Draft/target SWA slot capacities differ")
            source_map = getattr(target_pool, "full_to_swa_index_mapping", None)
            draft_map = getattr(pool, "full_to_swa_index_mapping", None)
            if (
                source_map is not None
                and draft_map is not source_map
                and (draft_map is None or draft_map.data_ptr() != source_map.data_ptr())
            ):
                raise ValueError("Draft SWA must share the target full-to-SWA mapping")
            candidate = _group(f"draft{index}_swa", list(draft_swa.kv_buffer), layers, int(draft_swa.page_size))
            modes.append(_append(swa, candidate))
        else:
            buffers = getattr(pool, "kv_buffer", None)
            if buffers is None:
                raise NotImplementedError("FlexKV draft support currently requires MLA or DeepSeek split pools")
            rows = _paged_rows(buffers, page_size, "draft KV")
            required_pages = full[0]["buffers"][0].shape[0]
            if rows[0].shape[0] < required_pages:
                raise ValueError("Draft KV does not cover the target allocator's physical page space")
            modes.append(_append(full, _group(f"draft{index}_kv", rows, layers, page_size)))
            indexers = list(getattr(pool, "index_k_with_scale_buffer", None) or [])
            if indexers:
                if any(buffer.numel() and buffer.shape[0] < required_pages for buffer in indexers):
                    raise ValueError("Draft indexer does not cover the target allocator's physical page space")
                modes.append(_append(full, _group(f"draft{index}_indexer", indexers, layers, 1, page_size)))
        tail += count
    target_device = full[0]["buffers"][0].device
    if any(b.device != target_device for group in full + swa for b in group["buffers"]):
        raise ValueError("Draft and target buffers must belong to the same rank device")
    description = {
        "version": 1,
        "identity": identity,
        "target_layers": target_layers,
        "transfer_layers": tail,
        "full": [],
        "swa": [],
    }
    description["physical_formats"] = [
        dict(
            type=type(p).__name__,
            layout=str(getattr(p, "kv_layout", None)),
            start_layer=getattr(p, "start_layer", getattr(p, "_stage_start", None)),
            end_layer=getattr(p, "end_layer", getattr(p, "_stage_end", None)),
        )
        for p in (target_pool, *draft_pools)
    ]
    for label, groups in [("full", full), ("swa", swa)]:
        for group in groups:
            sample = group["buffers"][0]
            # Capacity may differ across otherwise compatible instances. Cache
            # identity describes the bytes of a page, not the allocation size.
            description[label].append(
                dict(
                    layers=group["layer_ids"],
                    ratio=group["ratio"],
                    page=group["sub_page_size"],
                    width=sample.shape[1],
                    dtype=str(sample.dtype),
                )
            )
    fingerprint = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
    return DraftCachePlan(full, swa, target_layers, tail, modes, fingerprint)


def standard_target_groups(pool, page_size, target_layers, indexer_group=None):
    buffers = getattr(pool, "kv_buffer", None)
    if buffers is None:
        raise NotImplementedError("FlexKV draft target must use MLA or DeepSeek split pools")
    groups = [_group("kv", _paged_rows(buffers, page_size, "target KV"), range(target_layers), page_size)]
    indexers = (
        list(indexer_group.buffers)
        if indexer_group is not None
        else list(getattr(pool, "index_k_with_scale_buffer", None) or [])
    )
    if indexers:
        layer_ids = indexer_group.layer_indices if indexer_group is not None else range(target_layers)
        indexer = _group("indexer", indexers, layer_ids, 1, page_size)
        if indexer is not None:
            groups.append(indexer)
    return groups


class TargetDraftTransferCounter:
    """Preserve target overlap, and drain draft tail layers at its last layer."""

    def __init__(self, counter, target_layers, transfer_layers):
        self.counter = counter
        self.target_layers = target_layers
        self.transfer_layers = transfer_layers

    def wait_until(self, layer):
        self.counter.wait_until(layer)
        if layer == self.target_layers - 1:
            for pending in range(self.transfer_layers):
                self.counter.wait_until(pending)


class DraftTransferCounter:
    def __init__(self, counter, transfer_layers):
        self.counter = counter
        self.transfer_layers = transfer_layers

    def wait_until(self, layer):
        # Some draft backends can read before the last target attention hook.
        # Fence every outstanding member before the counter's terminal layer
        # retires the consumer; do not assume completion order across workers.
        for pending in range(self.transfer_layers):
            self.counter.wait_until(pending)
