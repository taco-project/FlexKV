#!/usr/bin/env python3
"""Summarize completion polling and scheduler CPU spans without GPU dependencies."""
import argparse
from collections import defaultdict
import sys

from report_put_trace import read_records


def render(rows, out=sys.stdout, limit=30):
    # Host/run/PID scope is essential: TP ranks may have independent run IDs.
    groups = defaultdict(list)
    for row in rows:
        groups[(row['host'], row['run'], row['pid'])].append(row)
    print('CPU observation spans; nested durations overlap and MUST NOT be summed.', file=out)
    print('Missing ends mean incomplete observation, not proof of a hung function.', file=out)
    for identity, events in sorted(groups.items()):
        events.sort(key=lambda r: r['ns'])
        print(f'\nprocess host/run/pid={identity}', file=out)
        for row in events:
            if row['event'] == 'store_poll_first':
                print(f"  first_poll task={row['task']} rid={row['rid']} rank={row['rank']} ns={row['ns']}", file=out)
            elif row['event'] == 'store_poll_complete':
                print(f"  poll_complete task={row['task']} rid={row['rid']} rank={row['rank']} "
                      f"polls={row['poll_count']} errors={row['error_count']} "
                      f"max_gap_ms={row['max_gap_ns']/1e6:.3f} "
                      f"max_poll_ms={row['max_poll_ns']/1e6:.3f}", file=out)
                for stage, timing in row['stages'].items():
                    print(f"    slowest_{stage} ms={timing['duration_ns']/1e6:.3f} "
                          f"ns={timing['begin_ns']}..{timing['end_ns']} status={timing['status']}", file=out)
            elif row['event'] == 'store_poll_error':
                print(f"  first_poll_error task={row['task']} rid={row['rid']} ns={row['ns']}", file=out)
        summaries = [r for r in events if r['event'] == 'store_poll_summary']
        if summaries:
            poll_count = sum(r['poll_count'] for r in summaries)
            empty_count = sum(r['empty_poll_count'] for r in summaries)
            active_count = sum(r['active_poll_count'] for r in summaries)
            total_poll_ns = sum(r['total_poll_ns'] for r in summaries)
            total_sync_ns = sum(r['total_sync_ns'] for r in summaries)
            total_try_wait_ns = sum(r['total_try_wait_ns'] for r in summaries)
            print(
                f"  poll_summary windows={len(summaries)} polls={poll_count} "
                f"active={active_count} empty={empty_count} "
                f"errors={sum(r['error_count'] for r in summaries)} "
                f"detail_suppressed={sum(r.get('detail_suppressed_poll_count', 0) for r in summaries)} "
                f"total_poll_ms={total_poll_ns/1e6:.3f} "
                f"total_sync_ms={total_sync_ns/1e6:.3f} "
                f"total_try_wait_ms={total_try_wait_ns/1e6:.3f} "
                f"max_poll_ms={max(r['max_poll_ns'] for r in summaries)/1e6:.3f} "
                f"max_gap_ms={max(r['max_gap_ns'] for r in summaries)/1e6:.3f}",
                file=out,
            )
            forward_count = sum(r.get('forward_count', 0) for r in summaries)
            if forward_count:
                slowest = max(summaries, key=lambda r: r.get('max_forward_ns', 0))
                print(
                    f"  forward_summary forwards={forward_count} "
                    f"errors={sum(r.get('forward_error_count', 0) for r in summaries)} "
                    f"total_ms={sum(r.get('total_forward_ns', 0) for r in summaries)/1e6:.3f} "
                    f"max_ms={slowest.get('max_forward_ns', 0)/1e6:.3f} "
                    f"mode={slowest.get('max_forward_mode', '?')} "
                    f"batch_size={slowest.get('max_forward_batch_size', '?')} "
                    f"ns={slowest.get('max_forward_begin_ns', '?')}.."
                    f"{slowest.get('max_forward_end_ns', '?')}",
                    file=out,
                )
        polls = [r for r in events if r['event'] == 'store_poll_begin']
        gaps = sorted(((b['ns']-a['ns'], a, b) for a, b in zip(polls, polls[1:])),
                      key=lambda item: item[0], reverse=True)
        print(f'  observed store_poll calls={len(polls)} (older traces have no poll hooks)', file=out)
        for duration, a, b in gaps[:limit]:
            print(f"  poll_gap_ms={duration/1e6:.3f} ns={a['ns']}..{b['ns']} "
                  f"previous_inflight={a.get('inflight', '?')}", file=out)
        begins = {r['span']: r for r in events if 'span' in r and r['event'].endswith('_begin')}
        ends = {r['span']: r for r in events if 'span' in r and r['event'].endswith('_end')}
        for key, row in begins.items():
            if key not in ends:
                print(f"  missing_end event={row['event']} span={key} ns={row['ns']}", file=out)
        by_stage = defaultdict(list)
        for row in ends.values():
            by_stage[row['event']].append(row)
        for stage, samples in sorted(by_stage.items()):
            slowest = max(samples, key=lambda r: r['duration_ns'])
            print(f"  {stage}: count={len(samples)} max_ms={slowest['duration_ns']/1e6:.3f} "
                  f"ns={slowest['begin_ns']}..{slowest['ns']} status={slowest['status']}", file=out)
        for row in sorted(ends.values(), key=lambda r: r['duration_ns'], reverse=True)[:limit]:
            print(f"  slow_span {row['event']} ms={row['duration_ns']/1e6:.3f} "
                  f"span={row['span']} parent={row.get('parent_span')} "
                  f"ns={row['begin_ns']}..{row['ns']}", file=out)
        for row in events:
            if row['event'] in ('completion_consumed', 'gpu_release_eligible',
                                'gpu_unlock_notified', 'gpu_unlock_begin',
                                'gpu_unlock_done', 'gpu_unlock_untracked'):
                print(f"  {row['event']} ns={row['ns']} task={row.get('task', '-')} "
                      f"op={row.get('op', '-')} rid={row.get('rid', '-')} "
                      f"parent={row.get('parent_span')}", file=out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--limit', type=int, default=30, help='largest gaps/spans per process')
    args = parser.parse_args()
    rows, bad = read_records(args.logs)
    render(rows, limit=max(0, args.limit))
    if bad:
        print(f'WARNING: {bad} malformed/unsupported records ignored', file=sys.stderr)


if __name__ == '__main__':
    main()
