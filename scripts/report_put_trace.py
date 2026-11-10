#!/usr/bin/env python3
"""Read FlexKV request trace JSONL; print GET/PUT/PREFETCH timelines and queue occupancy.

No FlexKV/torch dependency. Times are monotonic ns, compared only within a host.
Missing completion logs remain UNKNOWN, never assumed to be queued or successful.
"""
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import json
import sys

MARKER = "[PUT-TRACE] "
PAIRS = (
    ("prepare", "d2h_prepare_begin", "d2h_backend_begin"),
    ("owner_fence", "producer_fence_begin", "producer_fence_end"),
    ("backend", "d2h_backend_begin", "d2h_host_ready"),
    ("mc_prepare", "mooncake_prepare_begin", "mooncake_prepare_end"),
    ("mc_exists", "mooncake_exists_begin", "mooncake_exists_end"),
    ("mc_write", "mooncake_write_begin", "mooncake_write_end"),
    ("mc_read", "mooncake_read_begin", "mooncake_read_end"),
)


def read_records(paths):
    records, bad = [], 0
    for path in paths:
        source = sys.stdin if path == "-" else open(path)
        try:
            for line in source:
                if MARKER in line:
                    payload = line.split(MARKER, 1)[1]
                elif line.lstrip().startswith("{"):
                    payload = line
                else:
                    continue
                try:
                    row = json.loads(payload)
                    if row.get("v") != 1:
                        raise ValueError("unsupported trace schema")
                    records.append(row)
                except (ValueError, TypeError):
                    bad += 1
        finally:
            if source is not sys.stdin:
                source.close()
    return records, bad


def worker_intervals(records):
    jobs = {}
    for row in records:
        if row.get("event") not in ("worker_submit", "worker_done") or not row.get("ticket"):
            continue
        # Record type keys make repeated/tail-overlapping input files idempotent.
        jobs.setdefault(row["ticket"], {})[row["event"]] = row
    result = []
    for ticket, entries in jobs.items():
        submit = entries.get("worker_submit")
        done = entries.get("worker_done")
        base = dict(submit or done)
        points = dict(done.get("points", [])) if done else {}
        base.update(start=points.get("worker_start"), end=points.get("worker_end"),
                    points=points, complete=bool(done), has_submit=bool(submit),
                    status=done.get("status", "incomplete") if done else "incomplete",
                    physical=done.get("physical", ticket) if done else ticket,
                    volume=done.get("volume", {}) if done else {},
                    logical_blocks=done.get("logical_blocks") if done else None)
        result.append(base)
    return result


def occupancy(job, jobs):
    """Other already-submitted logical ops at this worker's arrival boundary.

    End is exclusive; simultaneous submissions aren't 'existing'. Unknown means
    no complete interval, not zero. Counts include reads sharing the worker.
    """
    t = job["submitted_ns"]
    counts = dict(queued=0, running=0, unknown=0)
    requests = set()
    for other in jobs:
        if (other["queue"] != job["queue"] or other["ticket"] == job["ticket"]
                or other["submitted_ns"] >= t or
                (other["end"] is not None and other["end"] <= t)):
            continue
        if other["start"] is None or other["end"] is None:
            counts["unknown"] += 1
        elif other["start"] > t:
            counts["queued"] += 1
        else:
            counts["running"] += 1
        requests.add(other.get("request", other["ticket"]))
    counts["requests"] = len(requests)
    return counts


def queue_snapshots(jobs):
    """O(n log n) sweep; snapshots exclude all simultaneous new submissions."""
    queues = defaultdict(list)
    for job in jobs:
        queues[job['queue']].append(job)
    snapshots = {}
    for members in queues.values():
        events = defaultdict(lambda: dict(arrive=[], start=[], end=[]))
        for job in members:
            events[job['submitted_ns']]['arrive'].append(job)
            if job['start'] is not None and job['end'] is not None:
                events[job['start']]['start'].append(job)
                events[job['end']]['end'].append(job)
        states, request_counts = {}, defaultdict(int)
        counts = dict(queued=0, running=0, unknown=0)
        for stamp, group in sorted(events.items()):
            for job in group['end']:
                previous = states.pop(job['ticket'], None)
                if previous is not None:
                    counts[previous] -= 1
                    key = job.get('request', job['ticket'])
                    request_counts[key] -= 1
                    if not request_counts[key]:
                        del request_counts[key]
            for job in group['start']:
                if states.get(job['ticket']) == 'queued':
                    states[job['ticket']] = 'running'
                    counts['queued'] -= 1
                    counts['running'] += 1
            for job in group['arrive']:
                snapshots[job['ticket']] = dict(counts, requests=len(request_counts))
            for job in group['arrive']:
                if job['end'] is not None and job['end'] <= stamp:
                    continue
                state = ('unknown' if job['start'] is None or job['end'] is None else
                         'running' if job['start'] <= stamp else 'queued')
                states[job['ticket']] = state
                counts[state] += 1
                request_counts[job.get('request', job['ticket'])] += 1
    return snapshots


def stage_occupancy(job, jobs, begin, finish):
    """Other physical operations active at this stage's start, host-local.

    Same physical coalesced batch is counted once and never as a predecessor of
    itself. Incomplete intervals are reported separately as unknown.
    """
    t = job["points"].get(begin)
    if t is None:
        return None
    active, unknown = set(), set()
    for other in jobs:
        if (other["host"] != job["host"] or other.get("run") != job.get("run")
                or other["physical"] == job["physical"]
                or other.get("direction") != job.get("direction")):
            continue
        b, e = other["points"].get(begin), other["points"].get(finish)
        if b is not None and b < t and e is not None and t < e:
            active.add(other["physical"])
        elif other["submitted_ns"] < t and other["end"] is None:
            unknown.add(other["physical"])
    return len(active), len(unknown)


class StageIndex:
    """Sorted interval endpoints: O(log n) occupancy, not O(n) per output row."""
    def __init__(self, jobs):
        self.starts, self.ends, self.unknown = defaultdict(list), defaultdict(list), defaultdict(list)
        seen, unknown_seen = set(), set()
        for job in jobs:
            scope = (job['host'], job.get('run'), job.get('direction'))
            identity = (*scope, job['physical'])
            if job['end'] is None and identity not in unknown_seen:
                self.unknown[scope].append(job['submitted_ns'])
                unknown_seen.add(identity)
            for _, begin, finish in PAIRS:
                key = (*scope, begin, finish)
                unique = (*key, job['physical'])
                if unique in seen:
                    continue
                b, e = job['points'].get(begin), job['points'].get(finish)
                if b is not None and e is not None and e > b:
                    self.starts[key].append(b)
                    self.ends[key].append(e)
                    seen.add(unique)
        for collection in (self.starts, self.ends, self.unknown):
            for values in collection.values():
                values.sort()

    def at(self, job, begin, finish):
        t = job['points'][begin]
        scope = (job['host'], job.get('run'), job.get('direction'))
        key = (*scope, begin, finish)
        return (bisect_left(self.starts[key], t) - bisect_right(self.ends[key], t),
                bisect_left(self.unknown[scope], t))


def ms(a, b):
    return "?" if a is None or b is None else f"{(b-a)/1e6:.3f}"


def render(records, out=sys.stdout, request_filter=None):
    jobs = worker_intervals(records)
    snapshots = queue_snapshots(jobs)
    jobs_by_request = defaultdict(list)
    stages = StageIndex(jobs)
    for job in jobs:
        jobs_by_request[job.get('request')].append(job)
    by_request = defaultdict(list)
    task_requests = defaultdict(set)
    for row in records:
        request = row.get("request")
        if request:
            by_request[request].append(row)
            if row["event"] == "planned":
                task_requests[(row["run"], row["task"])].add(request)
    ambiguous = 0
    for row in records:
        if row["event"] in ("store_created", "gpu_unlock_notified"):
            candidates = task_requests[(row["run"], row["task"])]
            if len(candidates) == 1:
                by_request[next(iter(candidates))].append(row)
            else:
                ambiguous += 1
    print("Times: monotonic ns; durations ms. Q/R/U = queued/running/unknown logical ops", file=out)
    print("Queue counts include all operations. Stage other_active counts physical executions on this host/direction.", file=out)
    print("Backend includes native preparation + copies + completion; nested spans must not be summed.", file=out)
    for request, rows in sorted(by_request.items()):
        if request_filter and request_filter not in request and not any(request_filter == r.get("rid") for r in rows):
            continue
        plan = next((r for r in rows if r["event"] == "planned"), rows[0])
        rid = next((r.get("rid") for r in rows if r.get("rid")), "-")
        operation = str(plan.get("operation", "put")).upper()
        print(f"\n{operation} {request} rid={rid} task={plan.get('task')} graph={plan.get('graph')}", file=out)
        for row in sorted(rows, key=lambda r: (r["host"], r["ns"])):
            if row["event"] in ("worker_submit", "worker_done"):
                continue
            stamp = datetime.fromtimestamp(row["wall_ns"] / 1e9, timezone.utc).isoformat(timespec="milliseconds")
            extra = ""
            if row["event"] == "planned":
                extra = f" plan_ms={ms(row.get('created_ns'), row['ns'])}"
                if row.get("children"):
                    extra += f" shared_children={row['children']}"
            if row.get("direction"):
                extra += f" type={row['direction']} op={row.get('op')}"
            if row.get("status"):
                extra += f" status={row['status']}"
            print(f"  {row['event']:24} ns={row['ns']} wall={stamp} host={row['host']}{extra}", file=out)
        for admitted in (r for r in rows if r['event'] == 'scheduler_admit'):
            local = [r for r in rows if r['host'] == admitted['host'] and r['pid'] == admitted['pid']]
            enqueued = next((r for r in local if r['event'] == 'scheduler_enqueue'), None)
            if enqueued:
                print(f"  scheduler_queue host={admitted['host']} ms={ms(enqueued['ns'], admitted['ns'])}", file=out)
            for ready in (r for r in local if r['event'] == 'op_ready'):
                dispatch = next((r for r in local if r['event'] == 'op_dispatch' and r['op'] == ready['op']), None)
                if dispatch:
                    print(f"  op={ready['op']} dependency_or_admit_ms={ms(admitted['ns'], ready['ns'])} "
                          f"ready_to_dispatch_ms={ms(ready['ns'], dispatch['ns'])}", file=out)
        for job in sorted(jobs_by_request[request], key=lambda j: (j["host"], j["submitted_ns"])):
            q = snapshots[job['ticket']]
            print(f"  {job['direction']} op={job['op']} parent={job.get('parent_op')} pool={job.get('pool')} "
                  f"queue={job['queue']} status={job['status']} physical={job['physical']}", file=out)
            print(f"    submit={job['submitted_ns']} received={job.get('received_ns', '?')} "
                  f"start={job['start']} end={job['end']} Q/R/U={q['queued']}/{q['running']}/{q['unknown']} "
                  f"prior_request_groups={q['requests']} queue_ms={ms(job['submitted_ns'], job['start'])} "
                  f"execution_ms={ms(job['start'], job['end'])}", file=out)
            if job['volume']:
                print(f"    logical_blocks={job['logical_blocks']} physical_volume="
                      + json.dumps(job['volume'], sort_keys=True), file=out)
            points = job['points']
            if 'batch_prepare_begin' in points:
                print(f"    shared_batch_prepare begin={points['batch_prepare_begin']} "
                      f"end={points['batch_prepare_end']} "
                      f"ms={ms(points['batch_prepare_begin'], points['batch_prepare_end'])} (inside queue_ms)", file=out)
            for label, begin, finish in PAIRS:
                if begin not in points:
                    continue
                end = points.get(finish)
                skipped = False
                active, unknown = stages.at(job, begin, finish)
                print(f"    {label:14} begin={points[begin]} end={end} ms={ms(points[begin], end)} "
                      f"other_active={active} unknown={unknown}" + (" write=skipped_exists" if skipped else ""), file=out)
            if ("mooncake_prepare_begin" in points
                    and "mooncake_results_end" in points
                    and "mooncake_write_begin" not in points):
                print("    mc_write       begin=- end="
                      f"{points['mooncake_results_end']} ms=0.000 write=skipped_exists",
                      file=out)
        created = next((r for r in rows if r["event"] == "request_created"), None)
        terminal = next((r for r in reversed(sorted(rows, key=lambda r: r["ns"]))
                         if r["event"] == "task_end"
                         and (created is None or r["host"] == created["host"])), None)
        if created and terminal:
            print(f"  request_total_ms={ms(created['ns'], terminal['ns'])}", file=out)
        if not any(r["event"] == "task_end" for r in rows):
            merged = [r['request'] for r in records if request in r.get('children', [])]
            print(f"  terminal=shared_with:{','.join(merged)}" if merged else "  terminal=missing_or_still_running", file=out)
    if ambiguous:
        print(f"\nUnlinked connector events: {ambiguous} (missing/ambiguous run+task; not guessed).", file=out)
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="stdout log files, or - for stdin")
    parser.add_argument("--request", help="request key substring or exact SGLang rid")
    args = parser.parse_args()
    rows, bad = read_records(args.logs)
    render(rows, request_filter=args.request)
    if bad:
        print(f"WARNING: {bad} malformed/unsupported trace records ignored; counts may be incomplete.", file=sys.stderr)


if __name__ == "__main__":
    main()
