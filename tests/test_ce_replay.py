"""Replay bounds/geometry checks, independent of a GPU or native extension."""
import copy
import json

import pytest

from flexkv.transfer.ce_replay import CETraceEntry, load_trace, replay


def record(kind=0, direction="H2D"):
    return {"schema_version": 1, "event": "transfer_launch", "trace_id": 1,
            "device": 0, "dropped_before": 0, "tensor_kind": kind,
            "transfer_backend": "copy_engine", "direction": direction,
            "num_blocks": 2, "start_layer_id": 1, "num_layers": 2,
            "total_num_layers": 4, "kv_dim": 2, "chunk_size_in_bytes": 16,
            "transfer_num_cta": 4,
            "strides": {"gpu_kv": 512, "gpu_block": 64, "gpu_layer": 2048,
                        "cpu_kv": 1024, "cpu_block": 64, "cpu_layer": 4096},
            "offsets": {"gpu": 8, "cpu": 16},
            "ce_config": {"segment_threshold": 8, "path_opt_enabled": False,
                          "force_path": -1, "enable_memcpy2d": False,
                          "is_blockfirst": False, "num_kv_heads": 1,
                          "gather_threads": 4, "gather_nt": True},
            "ce_path_id": -1, "gpu_block_ids": [1, 3], "cpu_block_ids": [2, 5],
            "block_ids_truncated": False}


@pytest.mark.parametrize("kind", [0, 1, 2])
@pytest.mark.parametrize("direction", ["H2D", "D2H"])
def test_partial_layers_and_rank_offsets_have_sufficient_backing(kind, direction):
    entry = CETraceEntry.from_json(json.dumps(record(kind, direction)))
    layout = entry.layout()
    assert layout.cpu_bytes == 2 * 4096 + 1024 + 5 * 64 + 16 + 16
    expected_gpu = 3 * 64 + 8 + 16
    if kind in (0, 1):
        expected_gpu += 512
    if kind == 1:
        expected_gpu += 2 * 2048
    assert layout.gpu_buffer_bytes == expected_gpu
    assert layout.gpu_buffer_count == (4, 1, 8)[kind]
    for pointer, gpu_offset, cpu_offset in entry.copies():
        assert 0 <= pointer < layout.gpu_buffer_count
        assert gpu_offset + 16 <= layout.gpu_buffer_bytes
        assert cpu_offset + 16 <= layout.cpu_bytes
    assert len(list(entry.copies())) == 8
    assert entry.transfer_bytes == 128


def test_sglang_uses_total_layers_for_kv_pointer_split():
    entry = CETraceEntry(record(2))
    assert {pointer for pointer, _, _ in entry.copies()} == {1, 2, 5, 6}


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(block_ids_truncated=True), "truncated"),
    (lambda d: d.update(gpu_block_ids=[1]), "incomplete"),
    (lambda d: d.update(cpu_block_ids=[-1, 2]), "integer"),
    (lambda d: d.update(tensor_kind=3), "tensor_kind"),
    (lambda d: d.update(num_layers=5), "layer range"),
    (lambda d: d["offsets"].update(gpu=1), "aligned"),
    (lambda d: d.update(chunk_size_in_bytes=12), "aligned"),
    (lambda d: d.update(gpu_block_ids=[1, 1]), "overlapping"),
    (lambda d: d["strides"].update(gpu_block=0), "overlapping"),
    (lambda d: d.update(gpu_block_ids=[1, 1 << 30]), "budget"),
    (lambda d: d["strides"].update(cpu_layer=1 << 62), "budget"),
    (lambda d: d.update(direction="unknown"), "direction"),
])
def test_reject_before_cuda_or_allocation(mutate, match):
    data = record()
    mutate(data)
    with pytest.raises(ValueError, match=match):
        replay(CETraceEntry(data))


def test_d2h_rejects_duplicate_cpu_destinations():
    data = record(direction="D2H")
    data["cpu_block_ids"] = [2, 2]
    with pytest.raises(ValueError, match="overlapping"):
        CETraceEntry(data).layout()


def test_sglang_distinct_pointer_spaces_may_share_byte_offsets():
    data = record(2)
    data["strides"]["gpu_kv"] = 0
    data["strides"]["gpu_layer"] = 0
    CETraceEntry(data).layout()


def test_incomplete_trace_remains_inspectable_but_not_replayable():
    data = record()
    data.update(gpu_block_ids=[1], cpu_block_ids=[2], block_ids_truncated=True)
    entry = CETraceEntry.from_json(json.dumps(data))
    assert entry.transfer_bytes == 128
    with pytest.raises(ValueError, match="truncated"):
        entry.layout()


def test_trace_reader_reports_line_and_rejects_unknown_version(tmp_path):
    path = tmp_path / "capture.jsonl"
    invalid = copy.deepcopy(record())
    invalid["schema_version"] = 2
    path.write_text(json.dumps(record()) + "\n" + json.dumps(invalid))
    with pytest.raises(ValueError, match=r"capture.jsonl:2:"):
        load_trace(path)


def test_budget_includes_oracle_and_readback():
    entry = CETraceEntry(record())
    layout = entry.layout()
    with pytest.raises(ValueError, match="budget"):
        entry.layout(layout.peak_bytes - 1)
    assert entry.layout(layout.peak_bytes) == layout


def test_forced_path_is_inspectable_but_rejected_before_cuda():
    data = record()
    data["ce_config"]["force_path"] = 0
    entry = CETraceEntry.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="forced benchmark"):
        replay(entry)


def test_invalid_cli_entry_keeps_original_error(tmp_path):
    import subprocess
    import sys
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    result = subprocess.run([sys.executable, "-m", "flexkv.transfer.ce_replay", str(path), "--replay", "0"],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "replay entry is outside the trace" in result.stderr
    assert "Traceback" not in result.stderr
