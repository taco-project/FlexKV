#!/usr/bin/env python3
"""Summarize isolated/cumulative PUT/GET diagnostic runs and worker traces."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values) if values else None


def transfer_bandwidth(row, operation):
    key = f"{operation}_transfer_GB_s"
    if key in row:
        return float(row[key])
    transfer_ns = float(row[f"{operation}_transfer_ms"]) * 1e6
    return float(row["payload_bytes"]) / max(transfer_ns, 1.0)


def correlation(xs, ys):
    if len(xs) < 2:
        return 0.0
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    numerator = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs)
        * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def slope(xs, ys):
    if len(xs) < 2:
        return 0.0
    x_mean, y_mean = statistics.mean(xs), statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return (
        sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        / denominator
        if denominator else 0.0
    )


def milliseconds(end, begin):
    return (end - begin) / 1e6


def load_trace(path):
    by_task = defaultdict(dict)
    event_counts = defaultdict(int)
    failures = []
    if not path.exists():
        return by_task, event_counts, failures
    with path.open() as handle:
        for line in handle:
            event = json.loads(line)
            event_counts[event["event"]] += 1
            if "task" in event:
                by_task[event["task"]][event["event"]] = event
            if event.get("status") not in (None, "ok", "completed"):
                failures.append(event)
    return by_task, event_counts, failures


def task_stage(by_task, task_id):
    events = by_task.get(task_id, {})
    required = ("request_created", "planned", "worker_submit",
                "worker_done", "task_end")
    if any(name not in events for name in required):
        return None
    done = events["worker_done"]
    points = dict(done.get("points", []))
    result = {
        "plan_ms": milliseconds(
            events["planned"]["ns"], events["request_created"]["ns"]),
        "worker_ipc_queue_ms": milliseconds(
            done["received_ns"], done["submitted_ns"]),
        "worker_body_ms": milliseconds(
            points["worker_end"], points["worker_start"]),
        "completion_return_ms": milliseconds(
            events["task_end"]["ns"], done["ns"]),
    }
    release = events.get("gpu_release_eligible")
    if release is not None:
        result["request_to_gpu_release_ms"] = milliseconds(
            release["ns"], events["request_created"]["ns"])
        launch = events.get("graph_launch")
        if launch is not None:
            result["launch_to_gpu_release_ms"] = milliseconds(
                release["ns"], launch["ns"])
    if "producer_fence_begin" in points:
        result["producer_fence_ms"] = milliseconds(
            points["producer_fence_end"], points["producer_fence_begin"])
    if "d2h_backend_begin" in points:
        result["d2h_backend_ms"] = milliseconds(
            points["d2h_host_ready"], points["d2h_backend_begin"])
    return result


def parse_native_probes(log_path):
    marker = "[GDR-D2H-PROBE]"
    probes = []
    if not log_path.exists():
        return probes
    with log_path.open(errors="replace") as handle:
        for line in handle:
            offset = line.find(marker)
            if offset < 0:
                continue
            payload = line.find("{", offset + len(marker))
            if payload < 0:
                continue
            try:
                probes.append(json.JSONDecoder().raw_decode(line[payload:])[0])
            except json.JSONDecodeError:
                continue
    return probes


def summarize_run(run_dir):
    rows = [
        json.loads(line)
        for line in (run_dir / "raw.jsonl").read_text().splitlines()
        if line.strip()
    ]
    measured = [row for row in rows if not row["warmup"]]
    if not measured:
        return [], {}
    by_task, counts, failures = load_trace(run_dir / "request-trace.jsonl")
    grouped = defaultdict(list)
    for row in measured:
        grouped[(row["blocks_per_request"], row["concurrency"])].append(row)
    summaries = []
    for (blocks, concurrency), samples in grouped.items():
        samples.sort(key=lambda row: row["iteration"])
        active = [
            row.get("active_roundtrip_ms", (
                row["put_e2e_ms"] + row["prefetch_e2e_ms"]
                + row["get_e2e_ms"]))
            for row in samples
        ]
        xs = list(range(len(samples)))
        first = samples[0]
        stages = {"put": [], "get": []}
        for sample in samples:
            for operation, key in (
                    ("put", "put_task_ids"), ("get", "get_task_ids")):
                for task_id in sample[key]:
                    stage = task_stage(by_task, task_id)
                    if stage:
                        stages[operation].append(stage)
        summary = {
            "run": run_dir.name,
            "run_label": first.get("run_label", run_dir.name),
            "arm": first["arm"],
            "blocks": blocks,
            "concurrency": concurrency,
            "logical_payload_mib": first["payload_bytes"] / 2**20,
            "rounds": len(samples),
            "qps": first.get("qps"),
            "read_depth": first.get("read_depth"),
            "dedicated": first.get("dedicated"),
            "h2d_batch_requests": first.get("h2d_batch_requests"),
            "d2h_batch_requests": first.get("d2h_batch_requests"),
            "gdr_diagnostics": first.get("gdr_diagnostics"),
            "kv_shared_mode": first.get("kv_shared_mode"),
            "owner_sync_after_put": first.get("owner_sync_after_put"),
            "round_gap_ms": first.get("round_gap_ms"),
            "put_median_ms": median([row["put_e2e_ms"] for row in samples]),
            "get_median_ms": median([row["get_e2e_ms"] for row in samples]),
            "put_transfer_median_GB_s": median([
                transfer_bandwidth(row, "put") for row in samples]),
            "get_transfer_median_GB_s": median([
                transfer_bandwidth(row, "get") for row in samples]),
            "active_roundtrip_median_ms": median(active),
            "active_roundtrip_first_ms": active[0],
            "active_roundtrip_last_ms": active[-1],
            "drift_percent": (
                (active[-1] / active[0] - 1) * 100 if active[0] else 0),
            "slope_ms_per_round": slope(xs, active),
            "round_correlation": correlation(xs, active),
            "verified": all(row["verified"] for row in samples),
            "trace_failures": len(failures),
        }
        for operation in ("put", "get"):
            for stage_name in (
                    "plan_ms", "worker_ipc_queue_ms", "worker_body_ms",
                    "producer_fence_ms", "d2h_backend_ms",
                    "completion_return_ms", "request_to_gpu_release_ms",
                    "launch_to_gpu_release_ms"):
                values = [
                    stage[stage_name] for stage in stages[operation]
                    if stage_name in stage
                ]
                summary[f"{operation}_{stage_name}"] = median(values)
        summaries.append(summary)
    trace_meta = {
        "event_counts": dict(counts),
        "trace_failures": len(failures),
        "native_probes": parse_native_probes(run_dir / "run.log"),
    }
    return summaries, trace_meta


def write_report(output, rows, metadata):
    def ms(value):
        return "-" if value is None else f"{value:.2f}"

    fields = list(rows[0])
    with (output / "diagnostic-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "diagnostic-summary.json").write_text(
        json.dumps({"runs": rows, "trace": metadata}, indent=2) + "\n")
    lines = [
        "# PUT/GET cumulative-state diagnostics",
        "",
        "Active roundtrip is PUT + PREFETCH + GET and excludes optional owner sync "
        "and round cooldown. Positive slope/correlation indicates within-run accumulation.",
        "",
        "| Run | Arm | Blocks×C | QPs | Depth | Dedicated | Shared | Gap ms | PUT transfer GB/s | GET transfer GB/s | PUT request→unlock ms | PUT launch→unlock ms | GET request→unlock ms | GET launch→unlock ms | Median ms | First→last ms | Drift | Slope ms/round | r |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_label']} | {row['arm']} | "
            f"{row['blocks']}×{row['concurrency']} | {row['qps']} | "
            f"{row['read_depth']} | {row['dedicated']} | "
            f"{row['kv_shared_mode']} | {row['round_gap_ms']} | "
            f"{row['put_transfer_median_GB_s']:.3f} | "
            f"{row['get_transfer_median_GB_s']:.3f} | "
            f"{ms(row.get('put_request_to_gpu_release_ms'))} | "
            f"{ms(row.get('put_launch_to_gpu_release_ms'))} | "
            f"{ms(row.get('get_request_to_gpu_release_ms'))} | "
            f"{ms(row.get('get_launch_to_gpu_release_ms'))} | "
            f"{row['active_roundtrip_median_ms']:.2f} | "
            f"{row['active_roundtrip_first_ms']:.2f}→"
            f"{row['active_roundtrip_last_ms']:.2f} | "
            f"{row['drift_percent']:+.1f}% | "
            f"{row['slope_ms_per_round']:.2f} | "
            f"{row['round_correlation']:.3f} |"
        )
    suspicious = [
        row for row in rows
        if row["round_correlation"] > 0.8 and row["drift_percent"] > 20
    ]
    lines.extend([
        "",
        f"Suspicious cumulative runs: {len(suspicious)} / {len(rows)} "
        "(r > 0.8 and drift > 20%).",
        "",
        "Native probe payloads and full stage medians are in diagnostic-summary.json/csv.",
    ])
    (output / "diagnostic-report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    all_rows, metadata = [], {}
    for raw_path in sorted(output.glob("*/raw.jsonl")):
        rows, trace_meta = summarize_run(raw_path.parent)
        all_rows.extend(rows)
        metadata[raw_path.parent.name] = trace_meta
    if not all_rows:
        raise SystemExit(f"no completed diagnostic runs under {output}")
    write_report(output, all_rows, metadata)
    print(
        f"analyzed {len(all_rows)} run/case rows -> "
        f"{output / 'diagnostic-report.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
