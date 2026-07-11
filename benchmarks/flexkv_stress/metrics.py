from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import csv
import math
import time


@dataclass
class OperationResult:
    operation: str
    success: bool
    latency_ms: float
    tokens: int = 0
    hit_tokens: int = 0
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


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


class CsvFile:
    def __init__(self, path: Path, fields: list[str]):
        self.handle = path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=fields, extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class Reporter:
    TURN_FIELDS = list(TurnResult.__dataclass_fields__)
    ROUND_FIELDS = [
        "round_id", "elapsed_seconds", "turns", "successes", "failures", "success_rate",
        "qps", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "expected_hit_tokens",
        "actual_hit_tokens", "hit_ratio", "hit_delta_tokens", "validation_samples",
        "validation_failures",
    ]
    ERROR_FIELDS = ["time", "round_id", "conversation_id", "turn_id", "operation", "error"]
    OP_FIELDS = ["round_id", "operation", "calls", "successes", "failures", "p50_ms", "p95_ms", "p99_ms"]
    RESOURCE_FIELDS = [
        "time", "round_id", "device_id", "allocated_bytes", "reserved_bytes",
        "max_allocated_bytes", "rss_bytes", "ssd_bytes",
    ]
    VALIDATION_FIELDS = ["round_id", "conversation_id", "turn_id", "ok"]

    def __init__(self, output_dir: str | Path):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base = Path(output_dir).resolve()
        self.directory = base / timestamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.turns = CsvFile(self.directory / "turns.csv", self.TURN_FIELDS)
        self.rounds = CsvFile(self.directory / "rounds.csv", self.ROUND_FIELDS)
        self.windows = CsvFile(self.directory / "windows.csv", self.ROUND_FIELDS)
        self.errors = CsvFile(self.directory / "errors.csv", self.ERROR_FIELDS)
        self.operations = CsvFile(self.directory / "operations.csv", self.OP_FIELDS)
        self.resources = CsvFile(self.directory / "resources.csv", self.RESOURCE_FIELDS)
        self.validation = CsvFile(self.directory / "validation_samples.csv", self.VALIDATION_FIELDS)
        self.all_rounds: list[dict[str, Any]] = []

    def write_error(self, round_id: int, operation: str, error: str,
                    conversation_id: int = -1, turn_id: int = -1) -> None:
        self.errors.write({
            "time": time.time(), "round_id": round_id, "conversation_id": conversation_id,
            "turn_id": turn_id, "operation": operation, "error": error,
        })

    def write_round(self, round_id: int, started_at: float, results: list[TurnResult],
                    operations: list[OperationResult]) -> dict[str, Any]:
        for result in results:
            self.turns.write(asdict(result))
            if result.validation_sampled:
                self.validation.write({
                    "round_id": result.round_id,
                    "conversation_id": result.conversation_id,
                    "turn_id": result.turn_id,
                    "ok": result.validation_ok,
                })
        elapsed = max(time.perf_counter() - started_at, 1e-9)
        successes = sum(result.success for result in results)
        expected = sum(result.expected_hit_tokens for result in results)
        actual = sum(result.actual_hit_tokens for result in results)
        latencies = [op.latency_ms for op in operations]
        validation_samples = sum(result.validation_sampled for result in results)
        validation_failures = sum(result.validation_sampled and not result.validation_ok for result in results)
        row = {
            "round_id": round_id,
            "elapsed_seconds": elapsed,
            "turns": len(results),
            "successes": successes,
            "failures": len(results) - successes,
            "success_rate": successes / len(results) if results else 1.0,
            "qps": len(results) / elapsed,
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_p99_ms": percentile(latencies, 0.99),
            "expected_hit_tokens": expected,
            "actual_hit_tokens": actual,
            "hit_ratio": actual / sum(result.total_tokens for result in results) if results else 0.0,
            "hit_delta_tokens": actual - expected,
            "validation_samples": validation_samples,
            "validation_failures": validation_failures,
        }
        self.rounds.write(row)
        self.windows.write(row)
        self.all_rounds.append(row)
        grouped: dict[str, list[OperationResult]] = defaultdict(list)
        for op in operations:
            grouped[op.operation].append(op)
        for name, items in grouped.items():
            values = [item.latency_ms for item in items]
            ok = sum(item.success for item in items)
            self.operations.write({
                "round_id": round_id, "operation": name, "calls": len(items), "successes": ok,
                "failures": len(items) - ok, "p50_ms": percentile(values, 0.50),
                "p95_ms": percentile(values, 0.95), "p99_ms": percentile(values, 0.99),
            })
        return row

    def write_resource(self, round_id: int, device_id: int, values: dict[str, int]) -> None:
        self.resources.write({"time": time.time(), "round_id": round_id, "device_id": device_id, **values})

    def close(self) -> None:
        summary_fields = ["rounds", "turns", "successes", "failures", "success_rate", "validation_failures"]
        total_turns = sum(int(row["turns"]) for row in self.all_rounds)
        successes = sum(int(row["successes"]) for row in self.all_rounds)
        validation_failures = sum(int(row["validation_failures"]) for row in self.all_rounds)
        summary = CsvFile(self.directory / "summary.csv", summary_fields)
        summary.write({
            "rounds": len(self.all_rounds), "turns": total_turns, "successes": successes,
            "failures": total_turns - successes,
            "success_rate": successes / total_turns if total_turns else 1.0,
            "validation_failures": validation_failures,
        })
        summary.close()
        for output in (
            self.turns, self.rounds, self.windows, self.errors, self.operations,
            self.resources, self.validation,
        ):
            output.close()
