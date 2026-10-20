"""Inspect native transfer traces and replay one launch on synthetic buffers.

No model/cache data or original pointers are read. A replay preserves recorded
strides, offsets, IDs and layer numbering, then checks all destination bytes,
including untouched regions. It is not a replay of distributed scheduling.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

PATH_NAMES = {-1: "PER_BLOCK", 0: "CONTIG_DIRECT", 1: "SEGMENT_DIRECT",
              2: "SEGMENT_SCATTER", 3: "GATHER_SCATTER", 4: "GATHER_DIRECT"}
MAX_INT64 = (1 << 63) - 1


def _integer(value, name, minimum=0, maximum=MAX_INT64):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class ReplayLayout:
    cpu_bytes: int
    gpu_buffer_bytes: int
    gpu_buffer_count: int
    peak_bytes: int


@dataclass
class CETraceEntry:
    record: Dict[str, Any]

    @classmethod
    def from_json(cls, line):
        entry = cls(json.loads(line))
        entry.validate()
        return entry

    @property
    def transfer_bytes(self):
        d = self.record
        return d["num_blocks"] * d["num_layers"] * d["kv_dim"] * d["chunk_size_in_bytes"]

    def validate(self):
        d = self.record
        if not isinstance(d, dict) or d.get("schema_version") != 1 or d.get("event") != "transfer_launch":
            raise ValueError("expected a version 1 transfer_launch record")
        if d.get("direction") not in ("H2D", "D2H"):
            raise ValueError("unknown transfer direction")
        if d.get("transfer_backend") not in ("copy_engine", "sm_kernel"):
            raise ValueError("unknown transfer backend")
        for name in ("trace_id", "device", "dropped_before"):
            _integer(d[name], name)
        _integer(d["tensor_kind"], "tensor_kind", maximum=2)
        for name in ("num_blocks", "num_layers", "total_num_layers", "transfer_num_cta"):
            _integer(d[name], name, 1, (1 << 31) - 1)
        _integer(d["start_layer_id"], "start_layer_id", maximum=(1 << 31) - 1)
        if d["start_layer_id"] + d["num_layers"] > d["total_num_layers"]:
            raise ValueError("layer range exceeds total_num_layers")
        _integer(d["kv_dim"], "kv_dim", 1, 2)
        _integer(d["chunk_size_in_bytes"], "chunk_size_in_bytes", 8)
        for name in ("gpu_kv", "gpu_block", "gpu_layer", "cpu_kv", "cpu_block", "cpu_layer"):
            value = _integer(d["strides"][name], name)
            if value % 8:
                raise ValueError("strides must be aligned to 8 bytes")
        for name in ("gpu", "cpu"):
            if _integer(d["offsets"][name], name) % 8:
                raise ValueError("offsets must be aligned to 8 bytes")
        if d["chunk_size_in_bytes"] % 8:
            raise ValueError("chunk size must be aligned to 8 bytes")
        if type(d["block_ids_truncated"]) is not bool:
            raise ValueError("block_ids_truncated must be boolean")
        for name in ("gpu_block_ids", "cpu_block_ids"):
            values = d[name]
            if not isinstance(values, list) or len(values) > d["num_blocks"]:
                raise ValueError(f"invalid {name} length")
            for value in values:
                # Native kernels use 32-bit block indexes.
                _integer(value, name, maximum=(1 << 31) - 1)
            if not d["block_ids_truncated"] and len(values) != d["num_blocks"]:
                raise ValueError(f"incomplete {name}")
        if len(d["gpu_block_ids"]) != len(d["cpu_block_ids"]):
            raise ValueError("block ID lengths disagree")
        config = d["ce_config"]
        for name in ("path_opt_enabled", "enable_memcpy2d", "is_blockfirst", "gather_nt"):
            if type(config[name]) is not bool:
                raise ValueError(f"{name} must be boolean")
        _integer(config["segment_threshold"], "segment_threshold", 1)
        _integer(config["gather_threads"], "gather_threads", 0, 1024)
        _integer(config["num_kv_heads"], "num_kv_heads", 1, (1 << 31) - 1)
        _integer(config["force_path"], "force_path", -1, 4)
        if d["transfer_backend"] == "copy_engine":
            _integer(d["ce_path_id"], "ce_path_id", -1, 4)
        elif d.get("ce_path_id") is not None:
            raise ValueError("SM kernel records have no CE path")

    def _address(self, layer, kv, gpu_block, cpu_block):
        d, s, o = self.record, self.record["strides"], self.record["offsets"]
        cpu = layer * s["cpu_layer"] + kv * s["cpu_kv"] + cpu_block * s["cpu_block"] + o["cpu"]
        gpu = gpu_block * s["gpu_block"] + o["gpu"]
        if d["tensor_kind"] == 0:  # vLLM: one pointer per layer
            return layer, gpu + kv * s["gpu_kv"], cpu
        if d["tensor_kind"] == 1:  # TRT-LLM: one pointer for all layers
            return 0, gpu + layer * s["gpu_layer"] + kv * s["gpu_kv"], cpu
        return kv * d["total_num_layers"] + layer, gpu, cpu  # SGLang K/V arrays

    def copies(self):
        d = self.record
        for layer in range(d["start_layer_id"], d["start_layer_id"] + d["num_layers"]):
            for kv in range(d["kv_dim"]):
                for gpu_block, cpu_block in zip(d["gpu_block_ids"], d["cpu_block_ids"], strict=True):
                    yield self._address(layer, kv, gpu_block, cpu_block)

    def layout(self, max_bytes=256 * 1024 * 1024):
        """Bound allocation and reject ambiguous destination writes before CUDA."""
        self.validate()
        d = self.record
        if d["block_ids_truncated"]:
            raise ValueError("truncated block IDs cannot be replayed; capture with FLEXKV_CE_TRACE_MAX_BLOCKS=0")
        if d["ce_config"]["force_path"] != -1:
            raise ValueError("forced benchmark paths are inspectable but cannot be replayed safely")
        if d["num_blocks"] * d["num_layers"] * d["kv_dim"] > 1_000_000:
            raise ValueError("too many copy spans for a diagnostic replay")
        _integer(max_bytes, "max_bytes", 1)
        layer = d["start_layer_id"] + d["num_layers"] - 1
        _, gpu_end, cpu_end = self._address(layer, d["kv_dim"] - 1,
                                           max(d["gpu_block_ids"]), max(d["cpu_block_ids"]))
        width = d["chunk_size_in_bytes"]
        gpu_bytes, cpu_bytes = gpu_end + width, cpu_end + width
        count = (d["total_num_layers"], 1, d["total_num_layers"] * d["kv_dim"])[d["tensor_kind"]]
        # Actual tensors + initial/oracle/readback copies; also bound ID tables.
        peak = 4 * (cpu_bytes + count * gpu_bytes) + 32 * (2 * d["num_blocks"] + count)
        if max(cpu_bytes, gpu_bytes, peak) > MAX_INT64 or peak > max_bytes:
            raise ValueError(f"replay needs up to {peak} bytes, above budget {max_bytes}")
        destinations = {}
        for pointer, gpu, cpu in self.copies():
            key, offset = (pointer, gpu) if d["direction"] == "H2D" else (0, cpu)
            destinations.setdefault(key, []).append(offset)
        for offsets in destinations.values():
            offsets.sort()
            if any(a + width > b for a, b in zip(offsets, offsets[1:])):
                raise ValueError("overlapping destination spans cannot be replayed")
        return ReplayLayout(cpu_bytes, gpu_bytes, count, peak)


def load_trace(path):
    entries = []
    with Path(path).open() as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                entries.append(CETraceEntry.from_json(line))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    return entries


def replay(entry, *, device=0, max_bytes=256 * 1024 * 1024, per_block=False):
    """Replay synchronously; tensor owners survive the complete native drain."""
    layout = entry.layout(max_bytes)
    import torch
    from flexkv import c_ext

    d, s, config = entry.record, entry.record["strides"], entry.record["ce_config"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for replay (inspection is CPU-only)")
    if per_block and d["transfer_backend"] != "copy_engine":
        raise ValueError("--per-block requires a copy-engine record")
    with torch.cuda.device(device):
        generator = torch.Generator().manual_seed(1729)
        cpu = torch.randint(0, 256, (layout.cpu_bytes,), dtype=torch.uint8, generator=generator).pin_memory()
        seeds = [torch.randint(0, 256, (layout.gpu_buffer_bytes,), dtype=torch.uint8, generator=generator)
                 for _ in range(layout.gpu_buffer_count)]
        gpu_buffers = [seed.to(device=f"cuda:{device}") for seed in seeds]
        expected_cpu = cpu.clone()
        expected_gpu = [seed.clone() for seed in seeds]
        width = d["chunk_size_in_bytes"]
        for pointer, gpu_offset, cpu_offset in entry.copies():
            if d["direction"] == "H2D":
                expected_gpu[pointer][gpu_offset:gpu_offset + width] = cpu[cpu_offset:cpu_offset + width]
            else:
                expected_cpu[cpu_offset:cpu_offset + width] = seeds[pointer][gpu_offset:gpu_offset + width]
        gpu_ids = torch.tensor(d["gpu_block_ids"], dtype=torch.int64, pin_memory=True)
        cpu_ids = torch.tensor(d["cpu_block_ids"], dtype=torch.int64, pin_memory=True)
        pointers = torch.tensor([tensor.data_ptr() for tensor in gpu_buffers], dtype=torch.int64, pin_memory=True)
        # Never reproduce a service's live memory, stream order, or completion
        # protocol: one recorded, already-resolved rank/region launch only.
        torch.cuda.synchronize(device)
        started = time.perf_counter_ns()
        try:
            c_ext.transfer_kv_blocks(
                gpu_block_id_tensor=gpu_ids, gpu_tensor_ptrs_tensor=pointers,
                gpu_kv_stride_in_bytes=s["gpu_kv"], gpu_block_stride_in_bytes=s["gpu_block"],
                gpu_layer_stride_in_bytes=s["gpu_layer"], cpu_block_id_tensor=cpu_ids,
                cpu_tensor=cpu, cpu_kv_stride_in_bytes=s["cpu_kv"],
                cpu_layer_stride_in_bytes=s["cpu_layer"], cpu_block_stride_in_bytes=s["cpu_block"],
                chunk_size_in_bytes=width, start_layer_id=d["start_layer_id"], num_layers=d["num_layers"],
                transfer_num_cta=d["transfer_num_cta"], is_host_to_device=d["direction"] == "H2D",
                use_ce_transfer=d["transfer_backend"] == "copy_engine", kv_dim=d["kv_dim"],
                num_kv_heads=config["num_kv_heads"], gpu_block_type=d["tensor_kind"], sync=True,
                ce_path_opt=False if per_block else config["path_opt_enabled"],
                ce_segment_threshold=config["segment_threshold"], ce_force_path=config["force_path"],
                ce_enable_memcpy2d=config["enable_memcpy2d"], is_blockfirst=config["is_blockfirst"],
                ce_gather_threads=config["gather_threads"], ce_gather_nt=config["gather_nt"],
                gpu_startoff_inside_chunks=d["offsets"]["gpu"], cpu_startoff_inside_chunks=d["offsets"]["cpu"],
                total_num_layers=d["total_num_layers"],
            )
        finally:
            # Includes the error path: a failed launch can have earlier copies
            # queued. Do not release raw-pointer backing tensors before drain.
            torch.cuda.synchronize(device)
        elapsed_ns = time.perf_counter_ns() - started
        if not torch.equal(cpu, expected_cpu):
            raise RuntimeError("CPU bytes differ from replay oracle (including untouched bytes)")
        for tensor, expected in zip(gpu_buffers, expected_gpu, strict=True):
            if not torch.equal(tensor.cpu(), expected):
                raise RuntimeError("GPU bytes differ from replay oracle (including untouched bytes)")
        return {"trace_id": d["trace_id"], "bytes": entry.transfer_bytes,
                "elapsed_us": elapsed_ns / 1000, "bandwidth_GB_s": entry.transfer_bytes / elapsed_ns,
                "byte_check": "passed", "policy": "per_block" if per_block else "recorded"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_file")
    parser.add_argument("--replay", type=int, metavar="ENTRY", help="replay one zero-based entry on synthetic buffers")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-mib", type=int, default=256, help="upper bound for replay buffers and oracle copies")
    parser.add_argument("--per-block", action="store_true", help="compare with baseline CE per-block copies")
    args = parser.parse_args()
    extension = None
    try:
        entries = load_trace(args.trace_file)
        if args.replay is None:
            for index, entry in enumerate(entries):
                d = entry.record
                print(json.dumps({"entry": index, "trace_id": d["trace_id"], "device": d["device"],
                                  "direction": d["direction"], "backend": d["transfer_backend"],
                                  "ce_path": PATH_NAMES.get(d["ce_path_id"]), "bytes": entry.transfer_bytes,
                                  "truncated": d["block_ids_truncated"], "dropped_before": d["dropped_before"]}))
        else:
            if not 0 <= args.replay < len(entries):
                raise ValueError("replay entry is outside the trace")
            from flexkv import c_ext as extension
            print(json.dumps(replay(entries[args.replay], device=args.device,
                                    max_bytes=args.max_mib * 1024 * 1024, per_block=args.per_block)))
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"ce_replay: {exc}\n")
    finally:
        if extension is not None:
            extension.ce_trace_shutdown()


if __name__ == "__main__":
    main()
