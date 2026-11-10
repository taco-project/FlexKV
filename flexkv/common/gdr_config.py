"""Validated, process-independent GDR configuration; no CUDA initialization."""

import copy
import json
import os

_DIRECTION = dict(path="auto", selector="avgbytes", on_init_failure="original",
                  gdr_when_ce_avg_bytes_below=524288,
                  gdr_max_block_fragment_bytes=131072, segment_threshold=8)
_DEFAULT = dict(enabled=False, qp_count=4, use_dedicated_gdr_thread=True,
                nic_names=[], h2d=dict(_DIRECTION, batch_requests=4, max_batch_blocks=8192),
                d2h=dict(_DIRECTION, auto_layer_parallel=True, read_depth=16,
                         batch_requests=1, max_batch_blocks=8192),
                diagnostics=dict(summary=False, timing=False, probe=False))


def parse_gdr_config(value=None):
    def merge(default, given, path):
        if not isinstance(given, dict):
            raise ValueError(f"{path} must be a mapping")
        unknown = given.keys() - default.keys()
        if unknown:
            raise ValueError(f"Unknown {path} keys: {sorted(unknown)}")
        result = copy.deepcopy(default)
        for key, val in given.items():
            if isinstance(default[key], dict):
                result[key] = merge(default[key], val, f"{path}.{key}")
            elif type(val) is not type(default[key]):
                raise ValueError(f"{path}.{key} must be {type(default[key]).__name__}")
            else:
                result[key] = copy.deepcopy(val)
        return result

    config = merge(_DEFAULT, {} if value is None else value, "gdr")
    if not 1 <= config["qp_count"] <= 16:
        raise ValueError("gdr.qp_count must be 1..16")
    if any(not isinstance(name, str) or not name.strip() for name in config["nic_names"]):
        raise ValueError("gdr.nic_names must contain nonempty NIC names")
    for direction in ("h2d", "d2h"):
        entry = config[direction]
        if entry["path"] not in {"original", "ce", "kernel", "gdr", "auto"}:
            raise ValueError(f"gdr.{direction}.path is invalid")
        if entry["selector"] not in {"avgbytes", "segments"}:
            raise ValueError(f"gdr.{direction}.selector is invalid")
        if entry["on_init_failure"] not in {"original", "error"}:
            raise ValueError(f"gdr.{direction}.on_init_failure is invalid")
        for key in ("gdr_when_ce_avg_bytes_below", "gdr_max_block_fragment_bytes",
                    "segment_threshold", "max_batch_blocks"):
            if not 0 < entry[key] <= 2**31 - 1:
                raise ValueError(f"gdr.{direction}.{key} must be 1..2^31-1")
        if not 1 <= entry["batch_requests"] <= 64:
            raise ValueError(f"gdr.{direction}.batch_requests must be 1..64")
    depth = config["d2h"]["read_depth"]
    if depth < 0 or depth > 128 or (depth and depth & (depth - 1)):
        raise ValueError("gdr.d2h.read_depth must be 0 or a power of two <=128")
    from flexkv.common.platform import ce_only_mode
    if config["enabled"] and ce_only_mode():
        for direction in ("h2d", "d2h"):
            if config[direction]["path"] == "kernel":
                raise ValueError(f"gdr.{direction}.path=kernel conflicts with CE-only mode")
    return config


def worker_gdr_config():
    config = parse_gdr_config(json.loads(os.environ.get("FLEXKV_GDR_CONFIG", "{}")))
    if os.environ.get("FLEXKV_KV_SHARED_ACROSS_RANKS_MODE"):
        config["d2h"]["auto_layer_parallel"] = False
    return config


def publish_gdr_config(value):
    config = parse_gdr_config(value)
    os.environ["FLEXKV_GDR_CONFIG"] = json.dumps(config)
    return config
