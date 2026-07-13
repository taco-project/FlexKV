from __future__ import annotations

from datetime import datetime
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
from .workload import WorkloadGenerator


LOG = logging.getLogger("flexkv_stress")


def _directory_bytes(paths: list[str]) -> int:
    total = 0
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            for file in path.rglob("*"):
                try:
                    if file.is_file():
                        total += file.stat().st_size
                except OSError:
                    continue
    return total


def _describe(config) -> None:
    LOG.info(
        "mode=%s model=%s architecture=%s composite_tp=%d kv_tp=%d DP=%d CP=%d "
        "page=%d required_gpus=%d",
        config.run.mode, config.preset.name, config.preset.architecture,
        config.model.tp_size, config.kv_tp_size, config.model.dp_size,
        config.model.cp_size, config.tokens_per_block, config.required_gpus,
    )
    LOG.info(
        "main_page_gb=%.9f CPU_GB=%.3f SSD_GB=%.3f features=%s",
        config.bytes_per_block / 1e9, config.cache.cpu_cache_gb,
        config.cache.ssd_cache_gb, config.features,
    )


def _should_stop(config, completed_rounds: int, elapsed: float) -> bool:
    round_done = config.run.rounds > 0 and completed_rounds >= config.run.rounds
    time_done = config.run.duration_seconds > 0 and elapsed >= config.run.duration_seconds
    enabled = []
    if config.run.rounds > 0:
        enabled.append(round_done)
    if config.run.duration_seconds > 0:
        enabled.append(time_done)
    return any(enabled) if config.run.stop_when == "either" else all(enabled)


def _sample_resources(config, runtime) -> list[dict[str, int]]:
    ssd_bytes = _directory_bytes(config.cache.ssd_cache_dir) if config.features.ssd else 0
    resources = []
    for worker in runtime.workers:
        memory = worker.request({"operation": "memory"})
        resources.append({
            "device_id": worker.device_id,
            "allocated_bytes": memory.get("allocated_bytes", 0),
            "reserved_bytes": memory.get("reserved_bytes", 0),
            "max_allocated_bytes": memory.get("max_allocated_bytes", 0),
            "rss_bytes": memory.get("rss_bytes", 0),
            "ssd_bytes": ssd_bytes,
        })
    return resources


def _record_failures(reporter, scenario, window_id, results, operations) -> bool:
    failed = False
    for result in results:
        if not result.success:
            failed = True
            reporter.write_error(
                window_id, "request",
                f"hit_delta={result.hit_delta_tokens}, validation_ok={result.validation_ok}",
                scenario=scenario,
            )
    for operation in operations:
        if not operation.success:
            failed = True
            reporter.write_error(
                window_id, operation.operation, operation.error or "operation failed",
                scenario=scenario,
            )
    return failed


def _warmup(config, runner, workload, reporter) -> bool:
    """Run warmup rounds. Failures are recorded to errors.csv and folded into
    the exit code, but never abort the run: a diagnostic run should surface
    every failure across the full workload rather than stop at a warmup miss.
    Returns True if any warmup round had failures."""
    failed = False
    for warmup in range(config.run.warmup_rounds):
        LOG.info("warmup round %d/%d", warmup + 1, config.run.warmup_rounds)
        conversations = workload.generate_round(1_000_000_000 + warmup)
        results, operations = runner.run_round(conversations, -1)
        if _record_failures(reporter, "warmup", -1, results, operations):
            LOG.warning("warmup round %d had failures; see errors.csv (continuing)", warmup + 1)
            failed = True
    return failed


def _run_latency_hit(config, runtime, reporter, stop_requested) -> bool:
    from .runner import StressRunner

    runner = StressRunner(config, runtime)
    workload = WorkloadGenerator(config.conversation, config.tokens_per_block, config.run.seed)
    # Warm the shared prefix so its boundary carries an SWA snapshot before any
    # conversation matches it (cross-conversation SWA reuse). No-op without a
    # shared system prompt.
    runner.warm_shared_prefix(workload.shared_system)
    failed = _warmup(config, runner, workload, reporter)
    original = (
        config.features.concurrency,
        config.concurrency.batch_size,
        config.concurrency.max_inflight_per_dp,
    )
    global_window = 0
    round_offset = 0
    for scenario in ("unloaded", "loaded"):
        if scenario == "unloaded":
            config.features.concurrency = False
            config.concurrency.batch_size = 1
            config.concurrency.max_inflight_per_dp = 1
        else:
            (config.features.concurrency, config.concurrency.batch_size,
             config.concurrency.max_inflight_per_dp) = original
        completed = 0
        scenario_started = time.monotonic()
        while not stop_requested() and not _should_stop(
            config, completed, time.monotonic() - scenario_started
        ):
            conversations = workload.generate_round(round_offset + completed)
            window_started = datetime.now().astimezone()
            started = time.perf_counter()
            results, operations = runner.run_round(conversations, round_offset + completed)
            duration = time.perf_counter() - started
            resources = _sample_resources(config, runtime)
            row = reporter.write_latency_window(
                scenario, global_window, window_started, duration, results, resources,
                batch_size=config.concurrency.batch_size,
                concurrency=config.concurrency.max_inflight_per_dp,
            )
            failed |= _record_failures(reporter, scenario, global_window, results, operations)
            if row["request_success_rate"] < config.validation.minimum_success_rate:
                failed = True
            if row["byte_validation_accuracy"] < 1:
                # Record the mismatch and keep going. A diagnostic run must expose
                # every failure across the full workload, not stop at the first one.
                failed = True
            LOG.info(
                "scenario=%s window=%d requests=%d qps=%.2f hit=%.5f "
                "match/load/save_p99_ms=%.3f/%.3f/%.3f",
                scenario, global_window, row["requests"], row["qps"], row["hit_ratio"],
                row["match_p99_ms"], row["load_p99_ms"], row["save_p99_ms"],
            )
            completed += 1
            global_window += 1
        round_offset += max(1, completed)
    (config.features.concurrency, config.concurrency.batch_size,
     config.concurrency.max_inflight_per_dp) = original
    return failed


_PATH_OPERATIONS = {
    "gpu_to_cpu_save": {"launch_put"},
    "cpu_to_gpu_load": {"launch_get"},
    # FlexKV exposes completion only for the end-to-end GPU/SSD paths.
    "gpu_to_ssd_save_e2e": {"launch_put"},
    "ssd_to_gpu_reload_e2e": {"ssd_reload_get"},
}


def _bandwidth_done(config, operations: int, active_s: float, payload_bytes: int) -> bool:
    target_bytes = config.bandwidth.target_payload_gb * 1e9
    return (
        operations >= config.bandwidth.min_operations
        and active_s >= config.bandwidth.min_duration_seconds
        and (target_bytes <= 0 or payload_bytes >= target_bytes)
    )


def _run_bandwidth(config, runtime, reporter, stop_requested) -> bool:
    from .runner import StressRunner

    runner = StressRunner(config, runtime)
    workload = WorkloadGenerator(config.conversation, config.tokens_per_block, config.run.seed)
    # Warm the shared prefix so its boundary carries an SWA snapshot before any
    # conversation matches it (cross-conversation SWA reuse). No-op without a
    # shared system prompt.
    runner.warm_shared_prefix(workload.shared_system)
    failed = _warmup(config, runner, workload, reporter)
    window_id = 0
    round_id = 0
    original_ssd_interval = config.cache.force_ssd_reload_interval_rounds
    levels = sorted({
        value for value in config.bandwidth.concurrency_levels
        if value <= config.concurrency.max_inflight_per_dp
    })
    if not levels:
        levels = [config.concurrency.max_inflight_per_dp]
    paths = [
        path for path in config.bandwidth.paths
        if config.features.ssd or "ssd" not in path
    ]
    for path in paths:
        config.cache.force_ssd_reload_interval_rounds = (
            1 if path == "ssd_to_gpu_reload_e2e" else original_ssd_interval
        )
        for concurrency in levels:
            config.features.concurrency = concurrency > 1
            config.concurrency.batch_size = concurrency
            cumulative_ops = 0
            cumulative_seconds = 0.0
            cumulative_bytes = 0
            window_operations = []
            window_validation_samples = 0
            window_validation_passes = 0
            window_seconds = 0.0
            window_started_at = None
            window_resources = []
            empty_rounds = 0
            first = True
            while first or not _bandwidth_done(
                config, cumulative_ops, cumulative_seconds, cumulative_bytes
            ):
                first = False
                if stop_requested():
                    return failed
                conversations = workload.generate_round(round_id)
                started_at = datetime.now().astimezone()
                if window_started_at is None:
                    window_started_at = started_at
                results, operations = runner.run_round(conversations, round_id)
                selected = [
                    item for item in operations if item.operation in _PATH_OPERATIONS[path]
                ]
                active_s = sum(item.latency_ms for item in selected) / 1000
                validation_samples = sum(result.validation_sampled for result in results)
                validation_passes = sum(
                    result.validation_sampled and result.validation_ok for result in results
                )
                resources = _sample_resources(config, runtime)
                failed |= _record_failures(reporter, path, window_id, results, operations)
                cumulative_ops += len(selected)
                cumulative_seconds += active_s
                cumulative_bytes += sum(item.transfer_bytes for item in selected)
                window_operations.extend(selected)
                window_seconds += active_s
                window_validation_samples += validation_samples
                window_validation_passes += validation_passes
                window_resources = resources
                round_id += 1
                empty_rounds = 0 if selected else empty_rounds + 1
                done = _bandwidth_done(
                    config, cumulative_ops, cumulative_seconds, cumulative_bytes
                )
                flush = bool(window_operations) and (
                    config.bandwidth.window_seconds == 0
                    or window_seconds >= config.bandwidth.window_seconds
                    or done
                )
                if flush:
                    reporter.write_bandwidth_window(
                        path, concurrency, window_id, window_started_at, window_seconds,
                        window_operations, window_validation_samples,
                        window_validation_passes, window_resources,
                    )
                    window_id += 1
                    window_operations = []
                    window_validation_samples = 0
                    window_validation_passes = 0
                    window_seconds = 0.0
                    window_started_at = None
                    window_resources = []
                if empty_rounds >= 3:
                    reporter.write_error(window_id, path, "No timed transfer operation was produced",
                                         scenario=path)
                    failed = True
                    break
            LOG.info(
                "path=%s concurrency=%d operations=%d payload_gb=%.3f active_s=%.3f",
                path, concurrency, cumulative_ops, cumulative_bytes / 1e9, cumulative_seconds,
            )
    config.cache.force_ssd_reload_interval_rounds = original_ssd_interval
    return failed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="FlexKV bandwidth and latency/hit stress benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("bandwidth", "latency_hit"))
    parser.add_argument("--rounds", type=int, help="Override latency_hit run.rounds")
    parser.add_argument("--duration", type=float, help="Override latency_hit run.duration_seconds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu-stub", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def run(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.mode:
        config.run.mode = args.mode
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

    os.environ["FLEXKV_ENABLE_LAYERWISE_TRANSFER"] = "1" if config.features.layerwise else "0"
    os.environ.setdefault("FLEXKV_SERVER_RECV_PORT", f"ipc:///tmp/flexkv_stress_server_{os.getpid()}")

    from .runner import FlexKVRuntime

    reporter = Reporter(config)
    (reporter.directory / "effective_config.yaml").write_text(
        yaml.safe_dump(config_as_dict(config), sort_keys=False), encoding="utf-8"
    )
    runtime = None
    stopping = False

    def request_stop(signum, _frame):
        nonlocal stopping
        LOG.warning("received signal %s; finishing the current operation", signum)
        stopping = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    failed = False
    try:
        runtime = FlexKVRuntime(config)
        runtime.start()
        stop_requested = lambda: stopping
        if config.run.mode == "latency_hit":
            failed = _run_latency_hit(config, runtime, reporter, stop_requested)
        else:
            failed = _run_bandwidth(config, runtime, reporter, stop_requested)
    except Exception as exc:
        failed = True
        reporter.write_error(-1, "runtime", f"{type(exc).__name__}: {exc}")
        LOG.exception("benchmark failed")
    finally:
        if runtime is not None:
            runtime.close()
        reporter.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        LOG.info("results=%s", reporter.directory)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
