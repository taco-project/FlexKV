# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark utilities for SWA CUDA DMA performance measurement."""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import statistics

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    operation: str
    num_trials: int
    total_bytes: int
    latencies: List[float]  # in microseconds

    @property
    def mean_latency_us(self) -> float:
        """Mean latency in microseconds."""
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def median_latency_us(self) -> float:
        """Median latency in microseconds."""
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p99_latency_us(self) -> float:
        """99th percentile latency."""
        if not self.latencies or len(self.latencies) < 100:
            return max(self.latencies) if self.latencies else 0.0
        sorted_lats = sorted(self.latencies)
        idx = int(len(sorted_lats) * 0.99)
        return sorted_lats[idx]

    @property
    def throughput_gbps(self) -> float:
        """Throughput in GB/s."""
        if self.mean_latency_us <= 0:
            return 0.0
        bytes_per_sec = self.total_bytes / (self.mean_latency_us / 1_000_000)
        return bytes_per_sec / 1e9

    def __str__(self) -> str:
        return (
            f"{self.operation}:\n"
            f"  Trials: {self.num_trials}\n"
            f"  Mean: {self.mean_latency_us:.2f} µs\n"
            f"  Median: {self.median_latency_us:.2f} µs\n"
            f"  P99: {self.p99_latency_us:.2f} µs\n"
            f"  Throughput: {self.throughput_gbps:.1f} GB/s"
        )


class SWABenchmark:
    """Benchmarking suite for SWA CUDA DMA."""

    def __init__(self, device: int = 0):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch with CUDA required")
        self.device = device

    def benchmark_stream_creation(self, num_streams: int = 4, num_trials: int = 100) -> BenchmarkResult:
        """Benchmark CUDA stream creation overhead.

        Args:
            num_streams: Number of streams to create per trial.
            num_trials: Number of trials.

        Returns:
            BenchmarkResult with latencies.
        """
        latencies = []
        with torch.cuda.device(self.device):
            for _ in range(num_trials):
                start = time.perf_counter()
                streams = [torch.cuda.Stream() for _ in range(num_streams)]
                end = time.perf_counter()
                latencies.append((end - start) * 1_000_000)

        return BenchmarkResult(
            operation=f"Stream creation ({num_streams} streams)",
            num_trials=num_trials,
            total_bytes=0,
            latencies=latencies
        )

    def benchmark_event_creation(self, num_trials: int = 100) -> BenchmarkResult:
        """Benchmark CUDA event creation overhead.

        Args:
            num_trials: Number of trials.

        Returns:
            BenchmarkResult with latencies.
        """
        latencies = []
        with torch.cuda.device(self.device):
            for _ in range(num_trials):
                start = time.perf_counter()
                event = torch.cuda.Event(blocking=False, enable_timing=True)
                end = time.perf_counter()
                latencies.append((end - start) * 1_000_000)

        return BenchmarkResult(
            operation="Event creation",
            num_trials=num_trials,
            total_bytes=0,
            latencies=latencies
        )

    def benchmark_event_query(self, num_trials: int = 1000) -> BenchmarkResult:
        """Benchmark CUDA event query overhead.

        Args:
            num_trials: Number of queries.

        Returns:
            BenchmarkResult with latencies.
        """
        latencies = []
        with torch.cuda.device(self.device):
            event = torch.cuda.Event(blocking=False)
            for _ in range(num_trials):
                start = time.perf_counter()
                _ = event.query()
                end = time.perf_counter()
                latencies.append((end - start) * 1_000_000)

        return BenchmarkResult(
            operation="Event query",
            num_trials=num_trials,
            total_bytes=0,
            latencies=latencies
        )

    def benchmark_memcpy(self,
                        size_bytes: int = 4_559_872,  # 4.35 MB (SWA page)
                        num_trials: int = 100,
                        direction: str = "h2d") -> BenchmarkResult:
        """Benchmark async memcpy latency.

        Args:
            size_bytes: Size of each transfer.
            num_trials: Number of trials.
            direction: "h2d" for Host-to-Device, "d2h" for Device-to-Host.

        Returns:
            BenchmarkResult with latencies.
        """
        latencies = []
        with torch.cuda.device(self.device):
            # Allocate buffers
            if direction == "h2d":
                host_buf = torch.zeros(size_bytes, dtype=torch.uint8, pin_memory=True)
                device_buf = torch.zeros(size_bytes, dtype=torch.uint8, device=f"cuda:{self.device}")
            else:
                device_buf = torch.zeros(size_bytes, dtype=torch.uint8, device=f"cuda:{self.device}")
                host_buf = torch.zeros(size_bytes, dtype=torch.uint8, pin_memory=True)

            stream = torch.cuda.Stream()
            for _ in range(num_trials):
                event = torch.cuda.Event(blocking=False, enable_timing=True)

                with torch.cuda.stream(stream):
                    start = time.perf_counter()
                    if direction == "h2d":
                        device_buf.copy_(host_buf, non_blocking=True)
                    else:
                        host_buf.copy_(device_buf, non_blocking=True)
                    event.record()
                    end = time.perf_counter()

                stream.synchronize()
                latencies.append((end - start) * 1_000_000)

        return BenchmarkResult(
            operation=f"Memcpy {direction.upper()} ({size_bytes / 1e6:.1f} MB)",
            num_trials=num_trials,
            total_bytes=size_bytes,
            latencies=latencies
        )

    def benchmark_concurrent_memcpy(self,
                                    size_bytes: int = 4_559_872,
                                    num_concurrent: int = 4,
                                    num_trials: int = 50) -> BenchmarkResult:
        """Benchmark concurrent async memcpy.

        Args:
            size_bytes: Size of each transfer.
            num_concurrent: Number of concurrent transfers.
            num_trials: Number of trials.

        Returns:
            BenchmarkResult with total latencies.
        """
        latencies = []
        with torch.cuda.device(self.device):
            # Allocate buffers
            host_bufs = [
                torch.zeros(size_bytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(num_concurrent)
            ]
            device_bufs = [
                torch.zeros(size_bytes, dtype=torch.uint8, device=f"cuda:{self.device}")
                for _ in range(num_concurrent)
            ]
            streams = [torch.cuda.Stream() for _ in range(num_concurrent)]

            for _ in range(num_trials):
                start = time.perf_counter()
                events = []
                for i in range(num_concurrent):
                    with torch.cuda.stream(streams[i]):
                        device_bufs[i].copy_(host_bufs[i], non_blocking=True)
                        event = torch.cuda.Event(blocking=False)
                        event.record()
                        events.append(event)

                # Wait for all
                for event in events:
                    event.wait()
                end = time.perf_counter()
                latencies.append((end - start) * 1_000_000)

        return BenchmarkResult(
            operation=f"Concurrent memcpy ({num_concurrent} parallel, {size_bytes / 1e6:.1f} MB each)",
            num_trials=num_trials,
            total_bytes=size_bytes * num_concurrent,
            latencies=latencies
        )

    def run_full_suite(self) -> Dict[str, BenchmarkResult]:
        """Run complete benchmark suite.

        Returns:
            Dict of operation name → BenchmarkResult.
        """
        results = {}

        print("Running SWA CUDA benchmark suite...")
        print("=" * 60)

        # Stream creation
        print("Benchmarking stream creation...")
        results["stream_creation"] = self.benchmark_stream_creation()
        print(results["stream_creation"])
        print()

        # Event creation
        print("Benchmarking event creation...")
        results["event_creation"] = self.benchmark_event_creation()
        print(results["event_creation"])
        print()

        # Event query
        print("Benchmarking event query...")
        results["event_query"] = self.benchmark_event_query()
        print(results["event_query"])
        print()

        # Single memcpy (H2D)
        print("Benchmarking single H2D memcpy...")
        results["memcpy_h2d"] = self.benchmark_memcpy(direction="h2d")
        print(results["memcpy_h2d"])
        print()

        # Single memcpy (D2H)
        print("Benchmarking single D2H memcpy...")
        results["memcpy_d2h"] = self.benchmark_memcpy(direction="d2h")
        print(results["memcpy_d2h"])
        print()

        # Concurrent memcpy
        for num_concurrent in [2, 4]:
            print(f"Benchmarking {num_concurrent} concurrent memcpy...")
            key = f"concurrent_memcpy_{num_concurrent}"
            results[key] = self.benchmark_concurrent_memcpy(num_concurrent=num_concurrent)
            print(results[key])
            print()

        print("=" * 60)
        print("Benchmark suite complete!")
        return results


if __name__ == "__main__":
    bench = SWABenchmark(device=0)
    results = bench.run_full_suite()

    print("\nSummary:")
    print("=" * 60)
    for name, result in results.items():
        print(f"{name}: mean={result.mean_latency_us:.2f} µs, "
              f"p99={result.p99_latency_us:.2f} µs, "
              f"throughput={result.throughput_gbps:.1f} GB/s")
