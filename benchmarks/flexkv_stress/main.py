from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import sys
import time

import yaml

from .config import config_as_dict, load_config
from .metrics import Reporter
from .models import model_bytes_per_block
from .workload import WorkloadGenerator


LOG = logging.getLogger("flexkv_stress")


def _directory_bytes(paths: list[str]) -> int:
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            total += sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total


def _describe(config) -> None:
    LOG.info(
        "model=%s architecture=%s layers=%d TP=%d DP=%d CP=%d page=%d required_gpus=%d",
        config.preset.name, config.preset.architecture, config.preset.num_layers,
        config.model.tp_size, config.model.dp_size, config.model.cp_size,
        config.tokens_per_block, config.required_gpus,
    )
    for group in config.preset.groups:
        LOG.info(
            "group=%s layers=%d heads=%d head_size=%d dtype=%s compress=%d",
            group.name, group.num_layers, group.num_kv_heads, group.head_size,
            group.dtype, group.compress_ratio,
        )
    gpu_bytes = config.bytes_per_gpu_block * config.model.gpu_blocks_per_rank
    LOG.info(
        "main_kv_bytes_per_block=%d estimated_main_gpu_bytes_per_rank=%d features=%s",
        config.bytes_per_block, gpu_bytes, config.features,
    )
    if config.features.swa:
        swa_bytes = (
            config.cache.swa_num_slots * config.tokens_per_block
            * config.preset.swa_num_layers * config.preset.swa_head_size
        )
        LOG.info("estimated_swa_gpu_bytes_per_rank=%d", swa_bytes)


def _should_stop(config, completed_rounds: int, elapsed: float) -> bool:
    round_done = config.run.rounds > 0 and completed_rounds >= config.run.rounds
    time_done = config.run.duration_seconds > 0 and elapsed >= config.run.duration_seconds
    enabled = []
    if config.run.rounds > 0:
        enabled.append(round_done)
    if config.run.duration_seconds > 0:
        enabled.append(time_done)
    return any(enabled) if config.run.stop_when == "either" else all(enabled)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FlexKV long-running correctness and stress benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, help="Override run.rounds")
    parser.add_argument("--duration", type=float, help="Override run.duration_seconds")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print memory geometry")
    parser.add_argument(
        "--cpu-stub",
        action="store_true",
        help="Run with in-memory CPU PyTorch workers instead of FlexKV/CUDA",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.rounds is not None:
        config.run.rounds = args.rounds
    if args.duration is not None:
        config.run.duration_seconds = args.duration
    if args.cpu_stub:
        config.features.cpu_stub = True
    config.validate()
    _describe(config)
    if args.dry_run:
        print(yaml.safe_dump(config_as_dict(config), sort_keys=False))
        return 0

    # These variables must exist before FlexKV modules read GLOBAL_CONFIG_FROM_ENV.
    os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = "1" if config.features.layerwise else "0"
    os.environ.setdefault("FLEXKV_SERVER_RECV_PORT", f"ipc:///tmp/flexkv_stress_server_{os.getpid()}")

    from .runner import FlexKVRuntime, StressRunner

    reporter = Reporter(config.output.directory)
    (reporter.directory / "effective_config.yaml").write_text(
        yaml.safe_dump(config_as_dict(config), sort_keys=False)
    )
    runtime = FlexKVRuntime(config)
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        LOG.warning("received signal %s; finishing the current batch", signum)
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    failed = False
    started = time.monotonic()
    try:
        runtime.start()
        runner = StressRunner(config, runtime)
        workload = WorkloadGenerator(config.conversation, config.tokens_per_block, config.run.seed)
        for warmup in range(config.run.warmup_rounds):
            LOG.info("warmup round %d/%d", warmup + 1, config.run.warmup_rounds)
            warmup_results, warmup_operations = runner.run_round(
                workload.generate_round(1_000_000_000 + warmup), -1
            )
            if any(not result.success for result in warmup_results) or any(
                not operation.success for operation in warmup_operations
            ):
                raise RuntimeError("warmup failed; inspect FlexKV logs before starting the soak run")

        completed = 0
        while not stop_requested and not _should_stop(config, completed, time.monotonic() - started):
            round_started = time.perf_counter()
            conversations = workload.generate_round(completed)
            results, operations = runner.run_round(conversations, completed)
            row = reporter.write_round(completed, round_started, results, operations)
            for result in results:
                if not result.success:
                    reporter.write_error(
                        completed,
                        "turn",
                        f"hit_delta={result.hit_delta_tokens}, validation_ok={result.validation_ok}",
                        result.conversation_id,
                        result.turn_id,
                    )
            for operation in operations:
                if not operation.success:
                    reporter.write_error(completed, operation.operation, operation.error or "operation failed")
            ssd_bytes = _directory_bytes(config.cache.ssd_cache_dir) if config.features.ssd else 0
            for worker in runtime.workers:
                memory = worker.request({"operation": "memory"})
                reporter.write_resource(completed, worker.device_id, {
                    "allocated_bytes": memory.get("allocated_bytes", 0),
                    "reserved_bytes": memory.get("reserved_bytes", 0),
                    "max_allocated_bytes": memory.get("max_allocated_bytes", 0),
                    "rss_bytes": memory.get("rss_bytes", 0),
                    "ssd_bytes": ssd_bytes,
                })
            if completed % config.run.log_every_rounds == 0:
                LOG.info(
                    "round=%d turns=%d success=%.5f qps=%.2f hit_ratio=%.5f "
                    "hit_delta=%d p99_ms=%.2f samples=%d sample_failures=%d",
                    completed, row["turns"], row["success_rate"], row["qps"], row["hit_ratio"],
                    row["hit_delta_tokens"], row["latency_p99_ms"], row["validation_samples"],
                    row["validation_failures"],
                )
            if row["success_rate"] < config.validation.minimum_success_rate:
                failed = True
            if any(not operation.success for operation in operations):
                failed = True
            if row["validation_failures"]:
                failed = True
                if config.validation.stop_on_mismatch:
                    break
            completed += 1
    except Exception as exc:
        failed = True
        reporter.write_error(-1, "runtime", f"{type(exc).__name__}: {exc}")
        LOG.exception("benchmark failed")
    finally:
        runtime.close()
        reporter.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        LOG.info("results=%s", reporter.directory)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
