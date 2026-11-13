"""Opt-in loopback control for a dedicated, quiescent FlexKV benchmark.

Call start(self) after KVTaskEngine starts. The benchmark must own the only
request producer and flush GPU ownership before L2 reset. Mount this file only
in the benchmark container. FLEXKV_BENCHMARK_CONTROL_PORT enables the hook.
The reset_all action starts a separate cold experiment; clear_l2 preserves L3.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE_DIR = Path(os.environ.get("FLEXKV_BENCHMARK_STATE_DIR", "/tmp/flexkv-draft-replay"))


def start(engine):
    port = int(os.environ.get("FLEXKV_BENCHMARK_CONTROL_PORT", "0"))
    if not port:
        return
    runtime = getattr(engine, "_runtime", None)

    # Chunked prefetch commits outside KVTaskManager's legacy transfer
    # counters. Observe the original commit result without changing its work.
    # These are successfully published main-KV block bytes. SWA snapshots
    # and NIC wire overhead are separate; native transfer logs include SWA.
    backend = getattr(engine, "_prefetch_backend", None)
    original_commit = backend.commit if backend is not None else None
    completed = dict(tokens=0, bytes=0)

    def observed_commit(context, chunk):
        end = original_commit(context, chunk)
        tokens = max(0, end - chunk.begin)
        completed["tokens"] += tokens
        completed["bytes"] += tokens // backend.block_size * backend.block_bytes
        return end

    if backend is not None:
        backend.commit = observed_commit
    elif engine.cache_config.use_mooncake_store_backend:
        from flexkv.common.config import block_size_in_bytes_for_cache, RankInfo

        original_finalize = engine._finalize_prefetch_return_mask
        block = engine.cache_config.tokens_per_block
        block_bytes = block_size_in_bytes_for_cache(
            engine.model_config, engine.cache_config, RankInfo(engine.model_config)
        )

        def observed_finalize(task):
            result = original_finalize(task)
            tokens = int(task.return_mask.sum())
            completed["tokens"] += tokens
            completed["bytes"] += tokens // block * block_bytes
            return result

        engine._finalize_prefetch_return_mask = observed_finalize

    def call(function, *arguments):
        if runtime is not None:
            return runtime.call(function, *arguments)
        return function(*arguments)

    def state():
        from prometheus_client import REGISTRY

        transfers = {}
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name.startswith("flexkv_py_transfer_") and sample.name.endswith("_total"):
                    key = sample.name + json.dumps(sample.labels, sort_keys=True)
                    transfers[key] = sample.value
        transfers["flexkv_benchmark_REMOTE2H_committed_bytes_total"] = completed["bytes"]
        transfers["flexkv_benchmark_REMOTE2H_committed_tokens_total"] = completed["tokens"]
        cpu = engine.cache_engine.cpu_cache_engine
        active = [t.task_id for t in list(engine.tasks.values()) if not t.is_completed()]
        coordinator = getattr(engine, "_prefetch", None)
        return dict(
            valid=True,
            active_tasks=active,
            sessions=len(coordinator.sessions) if coordinator is not None else 0,
            l2_total_blocks=int(cpu.mempool.num_total_blocks),
            l2_free_blocks=int(cpu.mempool.num_free_blocks),
            control_owner="task_runtime" if runtime is not None else "quiescent_in_process",
            chunked_runtime=runtime is not None,
            transfers=transfers,
            remote_byte_counter_scope="main KV block payload; excludes SWA snapshots and transport overhead",
            time=time.time(),
        )

    def clear(prefixes):
        import numpy as np
        from flexkv.common.block import SequenceMeta

        before = state()
        if before["active_tasks"] or before["sessions"]:
            raise RuntimeError("L2 reset requires all transfers and prefetch sessions drained")
        cache = engine.cache_engine
        block = cache.cpu_cache_engine.tokens_per_block
        if not prefixes:
            raise ValueError("L3 verification requires nonempty block-aligned prefixes")
        remote = cache.remote_cache_engine
        if remote is None:
            raise RuntimeError("Mooncake remote engine is absent")
        matches = []
        layout = json.loads((STATE_DIR / "layout.json").read_text())
        for item in prefixes:
            prefix = item["input_ids"]
            count = int(item["cached_prefix_tokens"])
            if count <= 0 or count % block:
                raise ValueError("Expected replay prefix must be block aligned")
            if layout["bigram"]:
                prefix = [(a << 32) | b for a, b in zip(prefix, prefix[1:])][:count]
            else:
                prefix = prefix[:count]
            namespace = ([layout["format"]] if layout.get("format") else []) + [
                json.dumps(["sglang-cache-v1", None, item["cache_salt"]], separators=(",", ":"))
            ]
            seq = SequenceMeta(np.asarray(prefix, dtype=np.int64), block, namespace=namespace)
            match = remote.match(seq)
            matched = getattr(match, "num_matched_blocks", None)
            if matched is None:
                matched = getattr(match, "num_ready_matched_blocks", None)
            if matched is None:
                raise RuntimeError("Remote match result does not report a prefix length")
            length = int(matched) * block
            if length != len(prefix):
                raise RuntimeError(f"L3 prefix incomplete: expected {len(prefix)}, got {length}")
            matches.append(length)
        with cache._cache_tree_lock:
            # In-process wait_complete has no task runtime. The sole benchmark
            # producer is quiescent after GPU flush; recheck under the radix
            # lock before recycling any CPU destinations.
            locked = state()
            if locked["active_tasks"] or locked["sessions"]:
                raise RuntimeError("Transfers became active before L2 reset")
            cache.cpu_cache_engine.reset()
        after = state()
        if after["l2_free_blocks"] != after["l2_total_blocks"]:
            raise RuntimeError("L2 allocator did not become empty")
        return dict(
            valid=True,
            before=before,
            after=after,
            l3_verified_prefix_tokens=matches,
            l3_preserved=True,
            reset_scope="cpu_cache_engine only",
        )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or "{}")
                if self.path not in ("/snapshot", "/drain", "/clear_l2", "/reset_all"):
                    raise ValueError("Unknown benchmark action")
                if self.path != "/snapshot":
                    deadline = time.monotonic() + 120
                    quiet_since = None
                    while True:
                        value = call(state)
                        now = time.monotonic()
                        if not value["active_tasks"] and not value["sessions"]:
                            if quiet_since is None:
                                quiet_since = now
                            elif now - quiet_since >= 0.1:
                                value["quiescent_observed_s"] = now - quiet_since
                                break
                        else:
                            quiet_since = None
                        if now >= deadline:
                            raise RuntimeError(f"Transfer drain deadline: {value}")
                        time.sleep(0.025)
                if self.path == "/clear_l2":
                    value = call(clear, body.get("prefixes", []))
                elif self.path == "/reset_all":
                    call(engine.reset_cache)
                    value = call(state)
                elif self.path == "/snapshot":
                    value = call(state)
                status = 200
            except Exception as exc:
                value, status = dict(valid=False, error=repr(exc)), 409
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def install():
    """Copy this module as sitecustomize.py in the dedicated server's PYTHONPATH."""
    if not int(os.environ.get("FLEXKV_BENCHMARK_CONTROL_PORT", "0")):
        return
    from flexkv.kvtask import KVTaskEngine
    from flexkv.integration.sglang.connector import FlexKVConnector

    original_connector_init = FlexKVConnector.__init__

    def connector_init(connector, *args, **kwargs):
        # This benchmark switch is outside production source. Both arms share
        # the correct bigram codec and all other runtime changes.
        root = STATE_DIR
        root.mkdir(parents=True, exist_ok=True)
        pools = (args[0] if args else kwargs.get("kvcache"), *kwargs.get("draft_kv_pools", ()))
        geometry = []
        target_map = getattr(pools[0], "full_to_swa_index_mapping", None)
        for pool in pools:
            mapping = getattr(pool, "full_to_swa_index_mapping", None)
            geometry.append(
                dict(
                    type=type(pool).__name__,
                    layers=getattr(pool, "layer_num", None),
                    swa_size=getattr(pool, "swa_size", None),
                    full_size=getattr(pool, "full_size", None),
                    swa_uses_full_indices=getattr(pool, "swa_uses_full_indices", None),
                    shared_swa_map=mapping is target_map
                    or (mapping is not None and target_map is not None and mapping.data_ptr() == target_map.data_ptr()),
                    swa_shapes=[
                        list(buffer.shape) for buffer in getattr(getattr(pool, "swa_kv_pool", None), "kv_buffer", [])
                    ],
                )
            )
        (root / f"pool-geometry-{os.getpid()}.json").write_text(json.dumps(geometry, indent=2))
        if os.environ.get("FLEXKV_BENCHMARK_DRAFT_CACHE", "1") == "0":
            kwargs.pop("draft_kv_pools", None)
        original_connector_init(connector, *args, **kwargs)
        if connector._sync_ctx.is_sync_leader:
            algorithm = getattr(args[1] if len(args) > 1 else kwargs.get("server_args"), "speculative_algorithm", "")
            algorithm = str(getattr(algorithm, "value", algorithm)).upper()
            plan = getattr(connector, "_draft_cache_plan", None)
            data = dict(
                format=getattr(connector, "cache_format_namespace", None),
                bigram=algorithm in ("EAGLE", "NEXTN", "EAGLE3"),
                modes=plan.modes if plan else [],
                draft_cache=os.environ.get("FLEXKV_BENCHMARK_DRAFT_CACHE", "1"),
            )
            (root / "layout.json").write_text(json.dumps(data, indent=2))

    FlexKVConnector.__init__ = connector_init
    original_start = KVTaskEngine.start

    def start_with_control(engine):
        original_start(engine)
        start(engine)

    KVTaskEngine.start = start_with_control


install()
