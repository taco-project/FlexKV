from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape
import csv
import json
import math
import os
import time


SCHEMA_VERSION = "1.0"
DECIMAL_GB = 1_000_000_000


@dataclass
class OperationResult:
    operation: str
    success: bool
    latency_ms: float
    tokens: int = 0
    hit_tokens: int = 0
    transfer_bytes: int = 0
    error: str = ""


@dataclass
class TurnResult:
    round_id: int
    conversation_id: int
    turn_id: int
    added_input_tokens: int
    output_tokens: int
    total_tokens: int
    expected_hit_tokens: int
    actual_hit_tokens: int
    hit_delta_tokens: int
    put_unmatched_tokens: int
    match_ms: float
    get_ms: float
    put_ms: float
    success: bool
    validation_sampled: bool = False
    validation_ok: bool = True
    query_tokens: int = 0


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile retained for callers and small unit tests."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def decimal_gb(byte_count: int | float) -> float:
    return float(byte_count) / DECIMAL_GB


def throughput_gb_s(byte_count: int | float, seconds: float) -> float:
    return decimal_gb(byte_count) / seconds if byte_count > 0 and seconds > 0 else 0.0


class LogHistogram:
    """Fixed-size, mergeable logarithmic histogram for long-running latency data."""

    def __init__(self, lowest_ms: float = 0.001, ratio: float = 1.01, bins: int = 4096):
        self.lowest_ms = lowest_ms
        self.ratio = ratio
        self.bins = [0] * bins
        self.count = 0

    def record(self, value_ms: float) -> None:
        value = max(0.0, float(value_ms))
        if value <= self.lowest_ms:
            index = 0
        else:
            index = int(math.log(value / self.lowest_ms, self.ratio)) + 1
            index = min(len(self.bins) - 1, max(0, index))
        self.bins[index] += 1
        self.count += 1

    def merge(self, other: "LogHistogram") -> None:
        if (self.lowest_ms, self.ratio, len(self.bins)) != (
            other.lowest_ms, other.ratio, len(other.bins)
        ):
            raise ValueError("Cannot merge histograms with different geometries")
        for index, count in enumerate(other.bins):
            self.bins[index] += count
        self.count += other.count

    def percentile(self, fraction: float) -> float:
        if not self.count:
            return 0.0
        target = max(1, math.ceil(self.count * fraction))
        seen = 0
        for index, count in enumerate(self.bins):
            seen += count
            if seen >= target:
                if index == 0:
                    return self.lowest_ms
                return self.lowest_ms * self.ratio ** (index - 1)
        return self.lowest_ms * self.ratio ** (len(self.bins) - 2)


class CsvFile:
    def __init__(self, path: Path, fields: list[str]):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({
            key: str(value).lower() if isinstance(value, bool) else value
            for key, value in row.items()
        })
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


LATENCY_FIELDS = [
    "run_id", "mode", "scenario", "backend", "performance_valid", "model",
    "architecture", "composite_tp", "kv_tp", "cp", "dp", "gpu_count",
    "page_tokens", "main_page_gb", "cpu_cache_gb", "ssd_cache_gb", "layerwise",
    "ssd_enabled", "swa", "conversations_per_round", "turns_min", "turns_max",
    "system_prompt_blocks", "first_input_blocks", "added_input_blocks", "output_blocks",
    "partial_block_tokens", "shared_system_prompt", "read_after_put",
    "batch_size", "concurrency", "requests", "qps",
    "match_p50_ms", "match_p95_ms", "match_p99_ms",
    "load_p50_ms", "load_p95_ms", "load_p99_ms",
    "save_p50_ms", "save_p95_ms", "save_p99_ms",
    "hit_ratio", "hit_exact_rate", "hit_token_accuracy", "request_success_rate",
    "byte_validation_accuracy", "gpu_memory_gb", "rss_gb", "ssd_used_gb",
]

BANDWIDTH_FIELDS = [
    "run_id", "mode", "scenario", "backend", "performance_valid", "model",
    "architecture", "composite_tp", "kv_tp", "cp", "dp", "gpu_count",
    "page_tokens", "main_page_gb", "cpu_cache_gb", "ssd_cache_gb", "layerwise",
    "ssd_enabled", "swa", "conversations_per_round", "turns_min", "turns_max",
    "system_prompt_blocks", "first_input_blocks", "added_input_blocks", "output_blocks",
    "partial_block_tokens", "shared_system_prompt", "read_after_put",
    "batch_size", "concurrency",
    "payload_gb", "operations", "duration_s", "throughput_gb_s",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "operation_success_rate", "byte_validation_accuracy", "gpu_memory_gb",
    "rss_gb", "ssd_used_gb",
]

WINDOW_FIELDS = ["window_id", "window_started_at", "window_duration_s"]
ERROR_FIELDS = ["time", "scenario", "window_id", "operation", "error"]


def _rounded(value: float) -> float:
    return round(float(value), 9)


def _resource_values(resources: list[dict[str, int]], cpu_stub: bool) -> tuple[float, float, float]:
    if not resources:
        return 0.0, 0.0, 0.0
    gpu = sum(item.get("allocated_bytes", 0) for item in resources)
    # CPU stub workers share one process; accelerator workers are separate processes.
    rss_values = [item.get("rss_bytes", 0) for item in resources]
    rss = max(rss_values, default=0) if cpu_stub else sum(rss_values)
    ssd = max((item.get("ssd_bytes", 0) for item in resources), default=0)
    return _rounded(decimal_gb(gpu)), _rounded(decimal_gb(rss)), _rounded(decimal_gb(ssd))


class _LatencyAggregate:
    def __init__(self, batch_size: int, concurrency: int):
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.requests = self.successes = 0
        self.query_tokens = self.actual = self.absolute_delta = self.exact = 0
        self.validation_samples = self.validation_passes = 0
        self.duration_s = 0.0
        self.latency = {name: LogHistogram() for name in ("match", "load", "save")}
        self.gpu = self.rss = self.ssd = 0.0

    def add(self, results: list[TurnResult], duration_s: float, resources: tuple[float, float, float],
            tolerance_tokens: int) -> None:
        self.duration_s += duration_s
        self.requests += len(results)
        self.successes += sum(result.success for result in results)
        for result in results:
            query = result.query_tokens or result.total_tokens
            self.query_tokens += query
            self.actual += result.actual_hit_tokens
            self.absolute_delta += abs(result.actual_hit_tokens - result.expected_hit_tokens)
            self.exact += abs(result.hit_delta_tokens) <= tolerance_tokens
            self.validation_samples += int(result.validation_sampled)
            self.validation_passes += int(result.validation_sampled and result.validation_ok)
            self.latency["match"].record(result.match_ms)
            self.latency["load"].record(result.get_ms)
            self.latency["save"].record(result.put_ms)
        self.gpu = max(self.gpu, resources[0])
        self.rss = max(self.rss, resources[1])
        self.ssd = max(self.ssd, resources[2])


class _BandwidthAggregate:
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.payload_bytes = self.operations = self.successes = 0
        self.validation_samples = self.validation_passes = 0
        self.duration_s = 0.0
        self.latency = LogHistogram()
        self.gpu = self.rss = self.ssd = 0.0

    def add(self, operations: list[OperationResult], duration_s: float,
            validation_samples: int, validation_passes: int,
            resources: tuple[float, float, float]) -> None:
        self.payload_bytes += sum(item.transfer_bytes for item in operations)
        self.operations += len(operations)
        self.successes += sum(item.success for item in operations)
        self.validation_samples += validation_samples
        self.validation_passes += validation_passes
        self.duration_s += duration_s
        for item in operations:
            self.latency.record(item.latency_ms)
        self.gpu = max(self.gpu, resources[0])
        self.rss = max(self.rss, resources[1])
        self.ssd = max(self.ssd, resources[2])


class Reporter:
    """Mode-aware stable CSV/JSON/SVG reporter with bounded-memory histograms."""

    def __init__(self, config, output_dir: str | Path | None = None):
        self.config = config
        self.started_at = datetime.now().astimezone()
        self._started_monotonic = time.monotonic()
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{timestamp}_{os.getpid()}"
        base = Path(output_dir or config.output.directory).resolve()
        self.directory = base / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.mode = config.run.mode
        self.backend = self._backend_name()
        self.performance_valid = not config.features.cpu_stub
        self.windows: list[dict[str, Any]] = []
        self.latency_aggregates: dict[str, _LatencyAggregate] = {}
        self.bandwidth_aggregates: dict[tuple[str, int], _BandwidthAggregate] = {}
        self.errors: CsvFile | None = None
        self.error_count = 0

    def _backend_name(self) -> str:
        if self.config.features.cpu_stub:
            return "cpu_stub"
        try:
            import torch
            return "rocm" if getattr(torch.version, "hip", None) else "cuda"
        except Exception:
            return "cuda"

    def write_error(self, window_id: int, operation: str, error: str,
                    conversation_id: int = -1, turn_id: int = -1,
                    scenario: str = "") -> None:
        del conversation_id, turn_id
        if self.errors is None:
            self.errors = CsvFile(self.directory / "errors.csv", ERROR_FIELDS)
        self.error_count += 1
        self.errors.write({
            "time": datetime.now().astimezone().isoformat(), "scenario": scenario,
            "window_id": window_id, "operation": operation, "error": error,
        })

    def _cache_gb(self, tier: str) -> float:
        count = getattr(self.config.cache, f"num_{tier}_blocks")
        configured = getattr(self.config.cache, f"{tier}_cache_gb")
        return _rounded(decimal_gb(count * self.config.bytes_per_block) if count else configured)

    def _common(self, scenario: str) -> dict[str, Any]:
        config = self.config
        def csv_value(value: Any) -> Any:
            return json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value

        return {
            "run_id": self.run_id, "mode": self.mode, "scenario": scenario,
            "backend": self.backend, "performance_valid": self.performance_valid,
            "model": config.preset.name, "architecture": config.preset.architecture,
            "composite_tp": config.model.tp_size, "kv_tp": config.kv_tp_size,
            "cp": config.model.cp_size, "dp": config.model.dp_size,
            "gpu_count": config.required_gpus, "page_tokens": config.tokens_per_block,
            "main_page_gb": _rounded(decimal_gb(config.bytes_per_block)),
            "cpu_cache_gb": self._cache_gb("cpu"), "ssd_cache_gb": self._cache_gb("ssd"),
            "layerwise": config.features.layerwise, "ssd_enabled": config.features.ssd,
            "swa": config.features.swa,
            "conversations_per_round": config.conversation.conversations_per_round,
            "turns_min": config.conversation.turns_min,
            "turns_max": config.conversation.turns_max,
            "system_prompt_blocks": config.conversation.system_prompt_blocks,
            "first_input_blocks": csv_value(config.conversation.first_input_blocks),
            "added_input_blocks": csv_value(config.conversation.added_input_blocks),
            "output_blocks": csv_value(config.conversation.output_blocks),
            "partial_block_tokens": config.conversation.partial_block_tokens,
            "shared_system_prompt": config.conversation.shared_system_prompt,
            "read_after_put": config.conversation.read_after_put,
        }

    @staticmethod
    def _latency_values(histograms: dict[str, LogHistogram]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, histogram in histograms.items():
            for label, fraction in (("p50", .50), ("p95", .95), ("p99", .99)):
                result[f"{name}_{label}_ms"] = _rounded(histogram.percentile(fraction))
        return result

    def _latency_row(self, scenario: str, aggregate: _LatencyAggregate) -> dict[str, Any]:
        requests = aggregate.requests
        query_tokens = aggregate.query_tokens
        row = {
            **self._common(scenario), "batch_size": aggregate.batch_size,
            "concurrency": aggregate.concurrency, "requests": requests,
            "qps": _rounded(requests / aggregate.duration_s if aggregate.duration_s else 0),
            **self._latency_values(aggregate.latency),
            "hit_ratio": _rounded(aggregate.actual / query_tokens if query_tokens else 0),
            "hit_exact_rate": _rounded(aggregate.exact / requests if requests else 1),
            "hit_token_accuracy": _rounded(
                max(0.0, 1 - aggregate.absolute_delta / query_tokens) if query_tokens else 1
            ),
            "request_success_rate": _rounded(aggregate.successes / requests if requests else 1),
            "byte_validation_accuracy": _rounded(
                aggregate.validation_passes / aggregate.validation_samples
                if aggregate.validation_samples else 1
            ),
            "gpu_memory_gb": aggregate.gpu, "rss_gb": aggregate.rss,
            "ssd_used_gb": aggregate.ssd,
        }
        return row

    def write_latency_window(self, scenario: str, window_id: int, started_at: datetime,
                             duration_s: float, results: list[TurnResult],
                             resources: list[dict[str, int]] | None = None,
                             batch_size: int | None = None,
                             concurrency: int | None = None) -> dict[str, Any]:
        batch = batch_size or (1 if scenario == "unloaded" else self.config.concurrency.batch_size)
        inflight = concurrency or (1 if scenario == "unloaded" else self.config.concurrency.max_inflight_per_dp)
        aggregate = self.latency_aggregates.setdefault(scenario, _LatencyAggregate(batch, inflight))
        window = _LatencyAggregate(batch, inflight)
        values = _resource_values(resources or [], self.config.features.cpu_stub)
        tolerance = self.config.validation.hit_tolerance_blocks * self.config.tokens_per_block
        window.add(results, max(duration_s, 1e-9), values, tolerance)
        aggregate.add(results, max(duration_s, 1e-9), values, tolerance)
        row = {
            **self._latency_row(scenario, window), "window_id": window_id,
            "window_started_at": started_at.isoformat(),
            "window_duration_s": _rounded(duration_s),
        }
        self.windows.append(row)
        return row

    def _bandwidth_row(self, scenario: str, aggregate: _BandwidthAggregate) -> dict[str, Any]:
        common = self._common(scenario)
        row = {key: common[key] for key in BANDWIDTH_FIELDS if key in common}
        row.update({
            "batch_size": aggregate.concurrency, "concurrency": aggregate.concurrency,
            "payload_gb": _rounded(decimal_gb(aggregate.payload_bytes)),
            "operations": aggregate.operations, "duration_s": _rounded(aggregate.duration_s),
            "throughput_gb_s": _rounded(throughput_gb_s(
                aggregate.payload_bytes, aggregate.duration_s
            )),
            "latency_p50_ms": _rounded(aggregate.latency.percentile(.50)),
            "latency_p95_ms": _rounded(aggregate.latency.percentile(.95)),
            "latency_p99_ms": _rounded(aggregate.latency.percentile(.99)),
            "operation_success_rate": _rounded(
                aggregate.successes / aggregate.operations if aggregate.operations else 1
            ),
            "byte_validation_accuracy": _rounded(
                aggregate.validation_passes / aggregate.validation_samples
                if aggregate.validation_samples else 1
            ),
            "gpu_memory_gb": aggregate.gpu, "rss_gb": aggregate.rss,
            "ssd_used_gb": aggregate.ssd,
        })
        return row

    def write_bandwidth_window(self, scenario: str, concurrency: int, window_id: int,
                               started_at: datetime, duration_s: float,
                               operations: list[OperationResult], validation_samples: int,
                               validation_passes: int,
                               resources: list[dict[str, int]] | None = None) -> dict[str, Any]:
        key = (scenario, concurrency)
        aggregate = self.bandwidth_aggregates.setdefault(key, _BandwidthAggregate(concurrency))
        window = _BandwidthAggregate(concurrency)
        values = _resource_values(resources or [], self.config.features.cpu_stub)
        active_s = max(duration_s, 1e-9)
        window.add(operations, active_s, validation_samples, validation_passes, values)
        aggregate.add(operations, active_s, validation_samples, validation_passes, values)
        row = {
            **self._bandwidth_row(scenario, window), "window_id": window_id,
            "window_started_at": started_at.isoformat(),
            "window_duration_s": _rounded(duration_s),
        }
        self.windows.append(row)
        return row

    def _summary_rows(self) -> list[dict[str, Any]]:
        if self.mode == "latency_hit":
            order = {"unloaded": 0, "loaded": 1}
            return [
                self._latency_row(name, aggregate)
                for name, aggregate in sorted(
                    self.latency_aggregates.items(), key=lambda item: order.get(item[0], 99)
                )
            ]
        return [
            self._bandwidth_row(name, aggregate)
            for (name, _), aggregate in sorted(
                self.bandwidth_aggregates.items(),
                key=lambda item: (
                    self.config.bandwidth.paths.index(item[0][0]), item[0][1]
                ),
            )
        ]

    def _json(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        config = self.config
        run = {
            "run_id": self.run_id, "mode": self.mode, "backend": self.backend,
            "performance_valid": self.performance_valid,
            "started_at": self.started_at.isoformat(),
            "duration_s": _rounded(time.monotonic() - self._started_monotonic),
        }
        model = {
            "preset": config.preset.name, "architecture": config.preset.architecture,
            "page_tokens": config.tokens_per_block,
            "main_page_gb": _rounded(decimal_gb(config.bytes_per_block)),
            "swa_page_gb": _rounded(decimal_gb(config.swa_bytes_per_block)),
        }
        topology = {
            "composite_tp": config.model.tp_size, "kv_tp": config.kv_tp_size,
            "cp": config.model.cp_size, "dp": config.model.dp_size,
            "gpu_count": config.required_gpus,
        }
        cache = {
            "cpu_gb": self._cache_gb("cpu"), "ssd_gb": self._cache_gb("ssd"),
            "ssd_enabled": config.features.ssd, "layerwise": config.features.layerwise,
            "swa_enabled": config.features.swa,
        }
        workload = {
            "conversations_per_round": config.conversation.conversations_per_round,
            "turns_min": config.conversation.turns_min,
            "turns_max": config.conversation.turns_max,
            "system_prompt_blocks": config.conversation.system_prompt_blocks,
            "first_input_blocks": config.conversation.first_input_blocks,
            "input_blocks": config.conversation.added_input_blocks,
            "output_blocks": config.conversation.output_blocks,
            "partial_block_tokens": config.conversation.partial_block_tokens,
            "shared_system_prompt": config.conversation.shared_system_prompt,
            "read_after_put": config.conversation.read_after_put,
        }
        scenarios = []
        if self.mode == "latency_hit":
            for row in rows:
                scenarios.append({
                    "name": row["scenario"], "batch_size": row["batch_size"],
                    "concurrency": row["concurrency"], "requests": row["requests"],
                    "qps": row["qps"],
                    "latency_ms": {
                        name: {label: row[f"{name}_{label}_ms"] for label in ("p50", "p95", "p99")}
                        for name in ("match", "load", "save")
                    },
                    "hit": {"ratio": row["hit_ratio"], "exact_rate": row["hit_exact_rate"],
                            "token_accuracy": row["hit_token_accuracy"]},
                    "correctness": {
                        "request_success_rate": row["request_success_rate"],
                        "byte_validation_accuracy": row["byte_validation_accuracy"],
                        "validation_samples": self.latency_aggregates[row["scenario"]].validation_samples,
                    },
                    "resources_gb": {"gpu_memory": row["gpu_memory_gb"], "rss": row["rss_gb"],
                                     "ssd_used": row["ssd_used_gb"]},
                })
        else:
            for row in rows:
                scenarios.append({
                    "name": row["scenario"], "concurrency": row["concurrency"],
                    "payload_gb": row["payload_gb"], "operations": row["operations"],
                    "duration_s": row["duration_s"], "throughput_gb_s": row["throughput_gb_s"],
                    "latency_ms": {label: row[f"latency_{label}_ms"] for label in ("p50", "p95", "p99")},
                    "correctness": {
                        "operation_success_rate": row["operation_success_rate"],
                        "byte_validation_accuracy": row["byte_validation_accuracy"],
                    },
                    "resources_gb": {"gpu_memory": row["gpu_memory_gb"], "rss": row["rss_gb"],
                                     "ssd_used": row["ssd_used_gb"]},
                })
        return {"schema_version": SCHEMA_VERSION, "run": run, "model": model,
                "topology": topology, "cache": cache, "workload": workload,
                "scenarios": scenarios}

    def _write_svg(self, rows: list[dict[str, Any]]) -> None:
        title = (
            f"FlexKV Stress — {self.config.preset.name} — {self.mode} — {self.backend} — "
            f"TP/CP/DP {self.config.model.tp_size}/{self.config.model.cp_size}/{self.config.model.dp_size}"
        )
        if self.mode == "bandwidth":
            panels = ["Throughput (GB/s)", "Operation latency p50/p95/p99 (ms)",
                      "Success and byte validation accuracy", "GPU / RSS / SSD usage (GB)"]
        else:
            panels = ["Match / load / save latency p50/p95/p99 (ms)", "QPS",
                      "Hit and byte accuracy", "GPU / RSS / SSD usage (GB)"]
        parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">',
            '<rect width="1200" height="820" fill="#0b1220"/>',
            f'<text x="40" y="38" fill="#f8fafc" font-size="22">{escape(title)}</text>',
        ]
        if not self.performance_valid:
            parts.append('<text id="cpu-stub-watermark" x="600" y="70" text-anchor="middle" '
                         'fill="#ff5a5f" font-size="20" font-weight="bold">'
                         'CPU STUB — PERFORMANCE NUMBERS ARE NOT VALID</text>')
        colors = ["#38bdf8", "#fb7185", "#a3e635", "#fbbf24", "#c084fc"]
        stable_peaks: set[str] = set()
        if self.mode == "bandwidth":
            by_path: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                if row["operation_success_rate"] == 1 and row["byte_validation_accuracy"] == 1:
                    by_path.setdefault(str(row["scenario"]), []).append(row)
            for path, candidates in by_path.items():
                best = max(candidates, key=lambda item: item["throughput_gb_s"])
                stable_peaks.add(f'{path}@{best["concurrency"]}:throughput_gb_s')
        for panel_id, panel_title in enumerate(panels):
            x = 35 + (panel_id % 2) * 580
            y = 90 + (panel_id // 2) * 350
            parts.extend([
                f'<g id="panel-{panel_id + 1}" data-panel="{escape(panel_title)}">',
                f'<rect x="{x}" y="{y}" width="550" height="320" rx="8" fill="#111c2e" stroke="#334155"/>',
                f'<text x="{x + 18}" y="{y + 28}" fill="#e2e8f0" font-size="16">{escape(panel_title)}</text>',
                f'<line class="x-axis" x1="{x + 55}" y1="{y + 275}" x2="{x + 525}" y2="{y + 275}" stroke="#64748b"/>',
                f'<line class="y-axis" x1="{x + 55}" y1="{y + 55}" x2="{x + 55}" y2="{y + 275}" stroke="#64748b"/>',
            ])
            values: list[tuple[str, float]] = []
            if self.mode == "bandwidth":
                fields = [
                    ("throughput_gb_s",),
                    ("latency_p50_ms", "latency_p95_ms", "latency_p99_ms"),
                    ("operation_success_rate", "byte_validation_accuracy"),
                    ("gpu_memory_gb", "rss_gb", "ssd_used_gb"),
                ][panel_id]
            else:
                fields = [
                    tuple(f"{operation}_{quantile}_ms" for operation in ("match", "load", "save")
                          for quantile in ("p50", "p95", "p99")),
                    ("qps",),
                    ("hit_ratio", "hit_exact_rate", "hit_token_accuracy", "byte_validation_accuracy"),
                    ("gpu_memory_gb", "rss_gb", "ssd_used_gb"),
                ][panel_id]
            plot_rows = self.windows if panel_id == 3 and self.windows else rows
            for row in plot_rows:
                base = (f'{row["scenario"]}@{row["concurrency"]}'
                        if self.mode == "bandwidth" else str(row["scenario"]))
                if panel_id == 3 and "window_id" in row:
                    base += f'#window-{row["window_id"]}'
                values.extend((f"{base}:{field}", float(row[field])) for field in fields)
            positive = [(label, value) for label, value in values if value > 0]
            maximum = max((value for _, value in positive), default=1.0)
            points = []
            point_records: list[tuple[str, str]] = []
            for index, (label, value) in enumerate(positive):
                px = x + 75 + index * (430 / max(1, len(positive) - 1))
                py = y + 270 - 200 * value / maximum
                points.append(f"{px:.1f},{py:.1f}")
                point_records.append((label, points[-1]))
                peak = label in stable_peaks
                bar_panel = (
                    (self.mode == "bandwidth" and panel_id in (1, 2))
                    or (self.mode == "latency_hit" and panel_id in (0, 1, 2))
                )
                if bar_panel:
                    parts.append(f'<rect class="grouped-bar" data-series="{escape(label)}" '
                                 f'x="{px - 3:.1f}" y="{py:.1f}" width="6" '
                                 f'height="{max(0.0, y + 275 - py):.1f}" '
                                 f'fill="{colors[index % len(colors)]}" opacity="0.55"/>')
                parts.append(f'<circle data-series="{escape(label)}" '
                             f'class="{"highest-stable-bandwidth" if peak else "data-point"}" '
                             f'cx="{px:.1f}" cy="{py:.1f}" r="{8 if peak else 5}" '
                             f'fill="{colors[index % len(colors)]}"/>')
                parts.append(f'<text x="{px:.1f}" y="{py - 9:.1f}" text-anchor="middle" fill="#cbd5e1" '
                             f'font-size="10">{value:.3g}</text>')
            if points and self.mode == "bandwidth" and panel_id == 0:
                for path in self.config.bandwidth.paths:
                    path_points = [point for label, point in point_records if label.startswith(path + "@")]
                    if path_points:
                        parts.append(f'<polyline data-series="{escape(path)}" '
                                     f'points="{" ".join(path_points)}" fill="none" '
                                     'stroke="#38bdf8" stroke-width="2"/>')
            elif points:
                parts.append(f'<polyline data-series="{panel_id + 1}" points="{" ".join(points)}" '
                             'fill="none" stroke="#38bdf8" stroke-width="2"/>')
            parts.append('</g>')
        parts.append('</svg>')
        (self.directory / "charts.svg").write_text("\n".join(parts) + "\n", encoding="utf-8")

    def close(self) -> None:
        rows = self._summary_rows()
        fields = LATENCY_FIELDS if self.mode == "latency_hit" else BANDWIDTH_FIELDS
        summary = CsvFile(self.directory / "summary.csv", fields)
        for row in rows:
            summary.write(row)
        summary.close()
        metrics = CsvFile(self.directory / "metrics.csv", WINDOW_FIELDS + fields)
        for row in self.windows:
            metrics.write(row)
        metrics.close()
        (self.directory / "summary.json").write_text(
            json.dumps(self._json(rows), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._write_svg(rows)
        if self.errors is not None:
            self.errors.close()
