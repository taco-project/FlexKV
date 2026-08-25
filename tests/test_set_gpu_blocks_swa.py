"""set_gpu_blocks must not overwrite SWA ops with main-KV slot_mapping slices."""
import numpy as np

from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType


def test_set_gpu_blocks_skips_swa_h2d():
    graph = TransferOpGraph()
    main_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([10, 11], dtype=np.int64),
        dst_block_ids=np.array([0, 0], dtype=np.int64),
    )
    swa_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([200], dtype=np.int64),
        dst_block_ids=np.array([42], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(main_h2d)
    graph.add_transfer_op(swa_h2d)

    # Main KV slots 100..104; only first two used for main H2D.
    graph.set_gpu_blocks(np.array([100, 101, 102, 103, 104], dtype=np.int64))

    assert np.array_equal(main_h2d.dst_block_ids, np.array([100, 101], dtype=np.int64))
    # SWA dst must stay as built in the graph, not gpu_blocks[:1] == 100.
    assert np.array_equal(swa_h2d.dst_block_ids, np.array([42], dtype=np.int64))


def test_clear_gpu_blocks_skips_swa_d2h():
    graph = TransferOpGraph()
    main_d2h = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([50, 51], dtype=np.int64),
        dst_block_ids=np.array([10, 11], dtype=np.int64),
    )
    swa_d2h = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([77], dtype=np.int64),
        dst_block_ids=np.array([200], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(main_d2h)
    graph.add_transfer_op(swa_d2h)

    graph.clear_gpu_blocks()

    assert main_d2h.src_block_ids.size == 0
    assert main_d2h.dst_block_ids.size == 0
    assert np.array_equal(swa_d2h.src_block_ids, np.array([77], dtype=np.int64))
    assert np.array_equal(swa_d2h.dst_block_ids, np.array([200], dtype=np.int64))


def test_set_gpu_blocks_can_fill_swa_when_explicit_mapping_provided():
    graph = TransferOpGraph()
    main_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([10, 11], dtype=np.int64),
        dst_block_ids=np.array([0, 0], dtype=np.int64),
    )
    swa_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([200, 201], dtype=np.int64),
        dst_block_ids=np.array([42, 43], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(main_h2d)
    graph.add_transfer_op(swa_h2d)

    graph.set_gpu_blocks(
        np.array([100, 101, 102, 103], dtype=np.int64),
        np.array([900, 901, 902], dtype=np.int64),
    )

    assert np.array_equal(main_h2d.dst_block_ids, np.array([100, 101], dtype=np.int64))
    assert np.array_equal(swa_h2d.dst_block_ids, np.array([900, 901], dtype=np.int64))


def test_set_gpu_blocks_explicit_swa_mapping_consumes_per_op():
    graph = TransferOpGraph()
    first = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([10], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
        is_swa=True,
    )
    second = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([11], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(first)
    graph.add_transfer_op(second)

    graph.set_gpu_blocks(
        np.array([], dtype=np.int64),
        np.array([900, 901], dtype=np.int64),
    )

    assert np.array_equal(first.dst_block_ids, np.array([900], dtype=np.int64))
    assert np.array_equal(second.dst_block_ids, np.array([901], dtype=np.int64))


def test_set_swa_gpu_blocks_consumes_per_op():
    graph = TransferOpGraph()
    first = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([10], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
        is_swa=True,
    )
    second = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([0], dtype=np.int64),
        dst_block_ids=np.array([11], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(first)
    graph.add_transfer_op(second)

    graph.set_swa_gpu_blocks(np.array([900, 901], dtype=np.int64))

    assert np.array_equal(first.dst_block_ids, np.array([900], dtype=np.int64))
    assert np.array_equal(second.src_block_ids, np.array([901], dtype=np.int64))
