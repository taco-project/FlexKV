"""Single-node PP: one CPU block packs every local PP stage.

GLM-5.3-shaped fixture: 78 layers split 18/20/20/20 (uneven), each stage with
a main-KV group plus a page-packed DSA indexer group. Covers the stage
layout on ModelConfig, block sizing, per-stage host regions, server-side
registration checks and the per-worker kwargs the TransferEngine builds.
"""
import multiprocessing as mp
import os
import socket
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch as mock_patch

import pytest
import torch

from flexkv.common.config import (
    CacheConfig,
    LayerGroupSpec,
    ModelConfig,
    block_size_in_bytes_for_cache,
)
from flexkv.common.pool import PoolId
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import WorkerKey
from flexkv.server.request import RegisterTPClientRequest
from flexkv.transfer.template import (
    compile_host_regions,
    compile_stage_host_regions,
)
from flexkv.transfer_manager import TransferManager

pytestmark = pytest.mark.unit

PARTITION = (18, 20, 20, 20)
TPB = 64
KV_HEAD = 656
INDEXER_ROW = TPB * (128 + 4)
KV_DTYPE = torch.float8_e4m3fn


def _stage_groups(num_layers, indexer_indices=None):
    if indexer_indices is None:
        indexer_indices = list(range(num_layers))
    return [
        LayerGroupSpec(
            num_layers=num_layers, num_kv_heads=1, head_size=KV_HEAD,
            layer_indices=list(range(num_layers)), dtype=KV_DTYPE,
        ),
        LayerGroupSpec(
            num_layers=len(indexer_indices), num_kv_heads=1,
            head_size=INDEXER_ROW, layer_indices=list(indexer_indices),
            dtype=torch.uint8, compress_ratio=TPB,
        ),
    ]


def _all_stage_groups(partition=PARTITION):
    # Stage 2 dedups two aliased indexer layers, as FLEXKV_DEDUP_INDEXER_GROUP
    # does: the stages differ in group shape, not only in layer count.
    out = []
    for pp, n in enumerate(partition):
        idx = [i for i in range(n) if not (pp == 2 and i in (3, 7))]
        out.append(_stage_groups(n, idx))
    return out


def _model_config(partition=PARTITION, tp_size=1, freeze=True):
    stages = _all_stage_groups(partition)
    mc = ModelConfig(
        num_layers=sum(partition), num_kv_heads=1, head_size=KV_HEAD, kv_dim=1,
        dtype=KV_DTYPE, tp_size=tp_size, pp_size=len(partition), dp_size=2,
        enable_dp_attention=True,
        layer_groups=[g for s in stages for g in s],
        pp_stage_group_counts=tuple(len(s) for s in stages),
        pp_stage_num_layers=tuple(partition),
    )
    if freeze:
        mc.freeze()
    return mc, stages


def _group_bytes(g, tp_size=1):
    return (tp_size * g.num_layers * (TPB // g.compress_ratio)
            * g.num_kv_heads * g.head_size * g.dtype.itemsize)


def _cpu_layout(mc, num_block=4):
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST, num_layer=mc.num_layers,
        num_block=num_block, tokens_per_block=TPB, num_head=1,
        head_size=KV_HEAD, kv_dim=1, num_kv_heads=1,
        layer_groups=mc.layer_groups, tp_size=mc.tp_size,
    )


# ---------------------------------------------------------------------------
# ModelConfig stage layout
# ---------------------------------------------------------------------------

def test_stage_slices_follow_pp_rank_order():
    mc, stages = _model_config()
    assert mc.has_pp_stage_layout
    for pp, groups in enumerate(stages):
        assert mc.pp_stage_layer_groups(pp) == groups
        assert mc.pp_stage_num_layers_of(pp) == PARTITION[pp]
    assert mc.pp_stage_group_slice(3) == slice(6, 8)
    assert "pp_stage_layers=[18, 20, 20, 20]" in str(mc)


def test_single_stage_config_keeps_legacy_behaviour():
    groups = _stage_groups(78)
    mc = ModelConfig(num_layers=78, kv_dim=1, dtype=KV_DTYPE,
                     layer_groups=groups)
    mc.freeze()
    assert not mc.has_pp_stage_layout
    assert mc.pp_stage_layer_groups(0) == groups


@pytest.mark.parametrize("kwargs, match", [
    (dict(pp_stage_num_layers=(18, 20, 20, 21)), "sum to num_layers"),
    (dict(pp_stage_group_counts=(2, 2, 2, 1)), "partition"),
    (dict(pp_stage_group_counts=(2, 2, 4)), "describe pp_size"),
    (dict(nnodes=4), "single-node only"),
    (dict(pp_stage_group_counts=None), "set together"),
])
def test_invalid_stage_layout_fails_fast(kwargs, match):
    mc, _ = _model_config(freeze=False)
    for k, v in kwargs.items():
        object.__setattr__(mc, k, v)
    with pytest.raises(ValueError, match=match):
        mc.freeze()


def test_stage_layer_index_must_stay_stage_local():
    mc, stages = _model_config(freeze=False)
    # pp0 has 18 layers; index 19 belongs to no layer of that stage.
    stages[0][1].layer_indices[-1] = 19
    with pytest.raises(ValueError, match="outside its 18 local layers"):
        mc.freeze()


# ---------------------------------------------------------------------------
# Block sizing: one block = every local stage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp_size", [1, 2])
def test_block_size_sums_every_stage(tp_size):
    mc, stages = _model_config(tp_size=tp_size)
    cache = CacheConfig(tokens_per_block=TPB)
    expected = sum(_group_bytes(g, tp_size) for s in stages for g in s)
    assert block_size_in_bytes_for_cache(mc, cache) == expected
    assert _cpu_layout(mc).get_block_stride() == expected
    # Far below the 128 MiB Mooncake per-call limit of the launch script.
    assert expected < 128 * 1024 * 1024


# ---------------------------------------------------------------------------
# Host regions: stages are contiguous, disjoint and fill the block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tp_size", [1, 2])
def test_stage_regions_tile_the_block(tp_size):
    mc, stages = _model_config(tp_size=tp_size)
    layout = _cpu_layout(mc)
    cursor = 0
    for pp, groups in enumerate(stages):
        regions = compile_stage_host_regions(
            mc.layer_groups, mc.pp_stage_group_slice(pp), layout, 1, KV_DTYPE)
        assert [r.group_index for r in regions] == list(range(len(groups)))
        assert [r.num_layers for r in regions] == [g.num_layers for g in groups]
        assert regions[0].base_offset == cursor, f"stage {pp} not contiguous"
        for a, b in zip(regions, regions[1:]):
            assert b.base_offset == a.base_offset + a.span_bytes
        assert all(r.block_stride == layout.get_block_stride() for r in regions)
        cursor = regions[-1].base_offset + regions[-1].span_bytes
    # Offsets live inside one TP rank's slice; the slices fill the block.
    assert cursor * tp_size == layout.get_block_stride()


def test_stage_offsets_match_uneven_partition():
    mc, stages = _model_config()
    layout = _cpu_layout(mc)
    offsets = [
        compile_stage_host_regions(
            mc.layer_groups, mc.pp_stage_group_slice(pp), layout, 1, KV_DTYPE,
        )[0].base_offset
        for pp in range(4)
    ]
    stage_bytes = [sum(_group_bytes(g) for g in s) for s in stages]
    assert offsets == [0, stage_bytes[0], sum(stage_bytes[:2]), sum(stage_bytes[:3])]
    assert stage_bytes[0] < stage_bytes[1]  # pp0 is the 18-layer stage


def test_single_stage_regions_equal_legacy_regions():
    groups = _stage_groups(78)
    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST, num_layer=78, num_block=4,
        tokens_per_block=TPB, num_head=1, head_size=KV_HEAD, kv_dim=1,
        num_kv_heads=1, layer_groups=groups,
    )
    legacy = compile_host_regions(groups, layout, 1, KV_DTYPE)
    staged = compile_stage_host_regions(groups, slice(0, 2), layout, 1, KV_DTYPE)
    assert staged == legacy


def test_stage_regions_reject_non_blockfirst():
    layout = MagicMock(type=KVCacheLayoutType.LAYERFIRST)
    with pytest.raises(ValueError, match="BLOCKFIRST"):
        compile_stage_host_regions([], slice(0, 1), layout, 1, KV_DTYPE)


# ---------------------------------------------------------------------------
# Server-side registration
# ---------------------------------------------------------------------------

def _manager(mc):
    manager = TransferManager.__new__(TransferManager)
    for name in ("all_gpu_layouts", "all_gpu_blocks", "gpu_worker_key_mapping",
                 "gpu_device_id_mapping", "all_gpu_layouts_per_group",
                 "all_gpu_blocks_per_group", "all_swa_gpu_blocks",
                 "all_swa_gpu_layouts", "all_swa_gpu_layouts_per_group",
                 "all_swa_gpu_blocks_per_group"):
        setattr(manager, name, {})
    manager.swa_layer_groups = None
    manager.model_config = mc
    return manager


def _req(dp, pp, groups, num_layer):
    return RegisterTPClientRequest(
        dp_client_id=dp, pp_rank=pp, intra_client_id=pp, device_id=dp + 2 * pp,
        handles=[], gpu_layout=SimpleNamespace(num_layer=num_layer),
        layer_groups=groups, gpu_layouts=[None] * len(groups),
        handles_per_group=[[] for _ in groups],
    )


def test_uneven_stages_register_against_their_own_slice():
    mc, stages = _model_config()
    manager = _manager(mc)
    for dp in range(2):
        for pp, groups in enumerate(stages):
            manager._handle_gpu_blocks_registration(
                _req(dp, pp, groups, PARTITION[pp]))
    assert len(manager.all_gpu_blocks) == 8
    assert {k.pp_rank for k in manager.gpu_worker_key_mapping.values()} == {0, 1, 2, 3}
    # The whole-block list is untouched by registration.
    assert mc.layer_groups == [g for s in stages for g in s]


def test_registration_with_another_stage_groups_fails():
    mc, stages = _model_config()
    manager = _manager(mc)
    with pytest.raises(ValueError, match="differ from the PP stage layout"):
        manager._handle_gpu_blocks_registration(_req(0, 1, stages[0], 18))
    assert not manager.all_gpu_blocks


def test_legacy_registration_mismatch_still_fails():
    groups = _stage_groups(20)
    mc = ModelConfig(num_layers=20, kv_dim=1, dtype=KV_DTYPE,
                     layer_groups=groups)
    manager = _manager(mc)
    with pytest.raises(ValueError, match="layer groups differ"):
        manager._handle_gpu_blocks_registration(_req(0, 0, _stage_groups(18), 18))


# ---------------------------------------------------------------------------
# TransferEngine per-worker kwargs
# ---------------------------------------------------------------------------

def _engine(mc):
    from flexkv.transfer.transfer_engine import TransferEngine

    engine = object.__new__(TransferEngine)
    engine.model_config = mc
    engine._cpu_handle = SimpleNamespace(kv_layout=_cpu_layout(mc))
    engine._swa_layer_groups = None
    engine._swa_gpu_blocks_per_group = None
    engine._swa_gpu_layouts_per_group = None
    blocks, layouts = {}, {}
    for dp in range(2):
        for pp in range(4):
            n = len(mc.pp_stage_layer_groups(pp))
            wk = WorkerKey(dp_client_id=dp, pp_rank=pp)
            blocks[wk] = [[f"h{dp}{pp}g{g}" for g in range(n)]]
            layouts[wk] = [[f"l{dp}{pp}g{g}" for g in range(n)]]
    engine._gpu_blocks_per_group = blocks
    engine._gpu_layouts_per_group = layouts
    return engine


def test_worker_kwargs_carry_their_stage():
    mc, stages = _model_config()
    engine = _engine(mc)
    for pp in range(4):
        wk = WorkerKey(dp_client_id=1, pp_rank=pp)
        mg = engine._get_multi_group_kwargs(wk)
        assert mg["layer_groups"] == stages[pp]
        assert mg["gpu_blocks_per_group"] == [[f"h1{pp}g{g}"] for g in range(2)]
        st = engine._get_pp_stage_kwargs(wk)
        sl = mc.pp_stage_group_slice(pp)
        assert st["host_group_range"] == (sl.start, sl.stop)
        assert st["num_original_layers"] == PARTITION[pp]
        assert st["host_layer_groups"][sl] == stages[pp]


def test_worker_kwargs_reject_group_count_mismatch():
    mc, _ = _model_config()
    engine = _engine(mc)
    wk = WorkerKey(dp_client_id=0, pp_rank=1)
    engine._gpu_blocks_per_group[wk] = [["only-one-group"]]
    with pytest.raises(ValueError, match="PP stage layout expects 2"):
        engine._get_multi_group_kwargs(wk)


def test_legacy_engine_has_no_stage_kwargs():
    from flexkv.transfer.transfer_engine import TransferEngine

    engine = object.__new__(TransferEngine)
    engine.model_config = ModelConfig(num_layers=2)
    assert engine._get_pp_stage_kwargs(WorkerKey(0, 0)) == {}


# ---------------------------------------------------------------------------
# FlexKVComm.all_gather_pp under TP2 PP4 DP2 (attn_tp=1), gloo
# ---------------------------------------------------------------------------

def _gather_worker(rank, world, port, tp, pp, q):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        import torch.distributed as dist

        from flexkv.integration.sglang.comm import FlexKVComm

        dist.init_process_group("gloo", rank=rank, world_size=world,
                                timeout=timedelta(seconds=30))
        my_pp, tp_rank = divmod(rank, tp)
        my_group = None
        for t in range(tp):
            g = dist.new_group([s * tp + t for s in range(pp)])
            if t == tp_rank:
                my_group = g
        rank_info = SimpleNamespace(
            model_config=SimpleNamespace(
                pp_size=pp, attn_tp_size=1, attn_cp_size=1, attn_dp_size=tp),
            pp_rank=my_pp, attn_tp_rank=0, attn_cp_rank=0, pp_size_per_node=pp,
        )
        comm = FlexKVComm(rank_info=rank_info, world_rank=rank,
                          pp_group=my_group, world_group=dist.group.WORLD)
        got = comm.all_gather_pp({"pp": my_pp, "dp": tp_rank})
        q.put((rank, got))
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        q.put((rank, f"ERROR {type(e).__name__}: {e}"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_all_gather_pp_stays_inside_dp_replica():
    tp, pp = 2, 4
    world = tp * pp
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_gather_worker, args=(r, world, port, tp, pp, q))
             for r in range(world)]
    for p in procs:
        p.start()
    results = dict(q.get(timeout=120) for _ in range(world))
    for p in procs:
        p.join(timeout=30)
    for rank, got in results.items():
        dp = rank % tp
        assert got == [{"pp": s, "dp": dp} for s in range(pp)], (rank, got)


def test_all_gather_pp_without_pp_returns_own_item():
    from flexkv.integration.sglang.comm import FlexKVComm

    comm = FlexKVComm.__new__(FlexKVComm)
    comm.pp_size = 1
    assert comm.all_gather_pp("x") == ["x"]


# ---------------------------------------------------------------------------
# PP fan-out: one op, every stage worker, same block ids
# ---------------------------------------------------------------------------

def test_d2h_fans_out_to_every_stage_with_the_same_blocks():
    """Stages share one slot_mapping; each writes its own byte range.

    So the op count must not change with the stage layout: one parent op,
    one replica per stage, all carrying the same block ids.
    """
    import numpy as np

    from flexkv.common.transfer import TransferOp, TransferType

    mc, _ = _model_config()
    engine = _engine(mc)
    engine.op_id_to_op = {}
    engine.op_id_to_nvtx_range = {}
    engine.pin_buffer = MagicMock()
    engine._child_id_to_child = {}
    engine._child_to_parent_op_id = {}
    workers = {
        WorkerKey(dp_client_id=1, pp_rank=pp): MagicMock(name=f"pp{pp}")
        for pp in range(4)
    }
    engine._register_worker(PoolId.FULL_KV, TransferType.D2H, workers)

    op = TransferOp(
        graph_id=0, transfer_type=TransferType.D2H,
        src_block_ids=np.array([7, 8], dtype=np.int64),
        dst_block_ids=np.array([11, 12], dtype=np.int64),
        dp_client_id=1,
    )
    with mock_patch('flexkv.transfer.transfer_engine.register_op_to_buffer'), \
         mock_patch('nvtx.start_range', return_value=MagicMock()):
        engine._assign_op_to_worker(op)

    assert op.pending_count == 4
    for pp, worker in workers.items():
        worker.submit_transfer.assert_called_once()
        replica = worker.submit_transfer.call_args[0][0]
        assert list(replica.src_block_ids) == [7, 8], pp
        assert list(replica.dst_block_ids) == [11, 12], pp


# ---------------------------------------------------------------------------
# Guards that only matter once a block carries every stage
# ---------------------------------------------------------------------------

def test_mooncake_object_must_fit_one_sdk_call():
    from flexkv.transfer.backends import _assert_block_fits_batch_limits

    limits = SimpleNamespace(max_put_batch_bytes=1 << 27,
                             max_get_batch_bytes=1 << 27)
    mc, _ = _model_config()
    whole_block = _cpu_layout(mc).get_block_stride()
    _assert_block_fits_batch_limits(limits, whole_block)  # ~3.9 MB, fits

    with pytest.raises(ValueError, match="max_put_batch_bytes"):
        _assert_block_fits_batch_limits(limits, (1 << 27) + 1)
    # 0 = unset, so an unconfigured limit must not reject anything.
    _assert_block_fits_batch_limits(
        SimpleNamespace(max_put_batch_bytes=0, max_get_batch_bytes=0), 1 << 40)


def test_global_layer_member_map_is_refused_under_stage_layout():
    """Stage-local layer ids would all collapse onto global layer 0."""
    mc, _ = _model_config()
    with pytest.raises(ValueError, match="not defined under a single-node PP"):
        _ = mc.layer_member_map

    groups = _stage_groups(78)
    single = ModelConfig(num_layers=78, kv_dim=1, dtype=KV_DTYPE,
                         layer_groups=groups)
    single.freeze()
    assert single.layer_member_map.num_original_layers == 78


def test_layerwise_handshake_rejects_another_stages_layer_count(tmp_path):
    """pp0's worker waits on 18 fds; a 20-layer consumer must be refused.

    Accepting it would shape the fd table to the wrong stage, so the layer a
    forward waits on would be signalled by another stage's copy.
    """
    import struct
    import threading

    from flexkv.transfer.layer_eventfd import receive_layer_eventfds

    sock_path = str(tmp_path / "lw.sock")
    out = {}

    def serve():
        out["fds"] = receive_layer_eventfds(
            sock_path, tp_group_size=1, num_layers=18,
            max_retries=2, retry_interval=0.5)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    for _ in range(100):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(sock_path)
    # (effective_tp_rank, tp_size, num_layers, num_counters) -- 20 != 18.
    client.sendall(struct.pack("iiii", 0, 1, 20, 2))
    # Not ACKed: the loop drops the connection (the ``with conn`` has already
    # closed it by the time the handler tries to NACK, so the consumer sees
    # EOF) and keeps waiting rather than binding the wrong stage's fds.
    assert client.recv(1) != b"\x01", "a mismatched layer count must not be ACKed"
    client.close()

    server.join(timeout=30)
    assert out["fds"].numel() == 0, "no fd table may be built from a refusal"


def test_layerwise_handshake_accepts_its_own_stage_layer_count(tmp_path):
    import struct
    import threading

    from flexkv.integration.sglang.comm import eventfd, send_fds
    from flexkv.transfer.layer_eventfd import receive_layer_eventfds

    sock_path = str(tmp_path / "lw_ok.sock")
    out = {}

    def serve():
        out["fds"] = receive_layer_eventfds(
            sock_path, tp_group_size=1, num_layers=18,
            max_retries=4, retry_interval=0.5)

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    for _ in range(100):
        if os.path.exists(sock_path):
            break
        time.sleep(0.05)

    fds = [eventfd(0, 0) for _ in range(18)]
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(10)
    client.connect(sock_path)
    client.sendall(struct.pack("iiii", 0, 1, 18, 1))
    send_fds(client, fds, struct.pack("i", 0))
    assert client.recv(1) == b"\x01"
    client.close()

    server.join(timeout=30)
    assert out["fds"].numel() == 18
    for fd in fds:
        os.close(fd)
