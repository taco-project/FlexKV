"""Low-volume H2D/D2H stage capture using the unchanged compare_modes workload.

Run twice with --probe 0/1. submit_ns excludes statistics collection and file IO;
use probe_calls.jsonl / probe_summary.csv, not the delegated benchmark's elapsed
column (which includes the Python capture wrapper).
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time


def difference(before, after):
    return {k: v-before.get(k, 0) for k, v in after.items() if k.startswith(("probe.gpu.", "h2d_completion.gpu."))}


def summarize(records):
    groups = {}
    for row in records:
        if row["phase"] != "measure" or not row["success"]:
            continue
        key = (row["arm"], row.get("direction", "D2H"), row["blocks"], row["layout"])
        groups.setdefault(key, []).append(row)
    output = []
    for (arm, direction, blocks, layout), rows in groups.items():
        ranks = sorted({int(k.split('.')[2]) for r in rows for k in r['stages']}) or [-1]
        for rank in ranks:
            prefix = f"{'h2d_completion' if direction == 'H2D' else 'probe'}.gpu.{rank}."
            fields = sorted({k[len(prefix):] for r in rows for k in r['stages'] if k.startswith(prefix)})
            item = dict(arm=arm, direction=direction, blocks=blocks, layout=layout, rank=rank, samples=len(rows),
                        submit_us=statistics.median(r['submit_ns'] for r in rows)/1000)
            elapsed = sorted(r['submit_ns'] for r in rows)
            at = .95*(len(elapsed)-1);lo = int(at);hi = min(lo+1,len(elapsed)-1)
            item['submit_p95_us'] = (elapsed[lo]+(elapsed[hi]-elapsed[lo])*(at-lo))/1000
            for field in fields:
                value = statistics.median(r['stages'].get(prefix+field, 0) for r in rows)
                item[field[:-3]+'_us' if field.endswith('_ns') else field] = value/1000 if field.endswith('_ns') else value
            if item.get('wqes'):
                item['bytes_per_wqe'] = item['bytes']/item['wqes']
            if item.get('poll_calls'):
                item['empty_poll_fraction'] = item['empty_polls']/item['poll_calls']
            output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--share-mode', choices=['sharded', 'layer_parallel'], default=None)
    parser.add_argument('--probe', type=int, choices=[0, 1], default=1)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--concurrency', default='1')
    parser.add_argument('--directions', default='D2H')
    parser.add_argument('--diagnostics', type=int, default=0)
    options, _ = parser.parse_known_args()
    if options.concurrency != '1' or options.directions not in ('H2D', 'D2H') or options.diagnostics != 0:
        parser.error('stage capture requires --concurrency 1 --directions H2D or D2H --diagnostics 0')
    command = list(sys.argv)
    import compare_modes as target
    if options.share_mode is not None:
        from tune import set_share_mode
        set_share_mode(target, options.share_mode)
    # Forward the original benchmark options; remove only our probe switch.
    forwarded = sys.argv[1:]
    mini = argparse.ArgumentParser(add_help=False)
    mini.add_argument('--probe', type=int)
    mini.add_argument('--share-mode')
    _, forwarded = mini.parse_known_args(forwarded)
    sys.argv = [sys.argv[0]]+forwarded
    if '--directions' not in forwarded and not any(a.startswith('--directions=') for a in forwarded):
        sys.argv += ['--directions', 'D2H']
    if '--concurrency' not in forwarded and not any(a.startswith('--concurrency=') for a in forwarded):
        sys.argv += ['--concurrency', '1']
    if '--warmup' not in forwarded and not any(a.startswith('--warmup=') for a in forwarded):
        sys.argv += ['--warmup', str(options.warmup)]
    records, case = [], {}
    run_case, build, config_for = target.run_case, target.build_region_batch, target.config_for
    def configured(arm, args, diagnostics):
        config = config_for(arm, args, diagnostics)
        if options.share_mode is not None:
            config['d2h']['auto_layer_parallel'] = False
        config['diagnostics']['probe'] = bool(options.probe)
        config['diagnostics']['timing'] = False
        return config
    def run(args, arm, direction, blocks, concurrency, layout):
        case.clear();case.update(arm=arm, direction=direction, blocks=blocks, layout=layout)
        return run_case(args, arm, direction, blocks, concurrency, layout)
    class Capture:
        def __init__(self, group):
            self.group, self.case = group, dict(case)
            self.generation, self.calls = 0, 0
        def __getattr__(self, name):
            return getattr(self.group, name)
        def configure_gdr(self, config):
            self.generation += 1;self.calls = 0
            return self.group.configure_gdr(config)
        def _begin(self):
            self.before = dict(self.group.gdr_statistics())
            self.started = time.perf_counter_ns()
        def _end(self, success):
            elapsed = time.perf_counter_ns()-self.started
            # A failed launch may still have native lanes running. Do not read
            # unsynchronized counters; the original harness drains in finally.
            after = dict(self.group.gdr_statistics()) if success else {}
            phase = 'verify' if self.generation == 1 else ('warmup' if self.calls < options.warmup else 'measure')
            records.append(dict(self.case, phase=phase, index=self.calls, success=success,
                                submit_ns=elapsed, stages=difference(self.before, after),
                                capabilities={k:v for k,v in after.items() if k.startswith('gpu.')}))
            self.calls += 1
        def submit(self, *args, **kwargs):
            self._begin()
            success = False
            try:
                result = self.group.submit(*args, **kwargs)
                success = True
                return result
            finally:
                self._end(success)
        def submit_layerwise(self, *args, **kwargs):
            self._begin()
            try:
                return self.group.submit_layerwise(*args, **kwargs)
            except BaseException:
                self._end(False)
                raise
        def wait_layer_completion(self, *args, **kwargs):
            success = False
            try:
                result = self.group.wait_layer_completion(*args, **kwargs)
                success = bool(result[0])
                return result
            finally:
                self._end(success)
    target.run_case, target.config_for = run, configured
    target.build_region_batch = lambda *a, **kw: Capture(build(*a, **kw))
    out = Path(options.output);out.mkdir(parents=True, exist_ok=True)
    try:
        target.main()
    finally:
        (out/'probe_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out/'probe_metadata.json').write_text(json.dumps(dict(command=command, probe=options.probe, share_mode=options.share_mode or "sharded",
            clock='host monotonic nanoseconds', scope='RegionBatchGroup submit through native completion; no Python worker/owner RPC',
            note='Nested timers; TP rank times are parallel. Warmup excluded from probe_summary.csv.'),indent=2)+'\n')
        summary = summarize(records)
        if summary:
            fields = list(dict.fromkeys(k for row in summary for k in row))
            with (out/'probe_summary.csv').open('w', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(summary)
        print(f'Transfer stage results: {out}/probe_summary.csv (use submit_us for probe on/off comparison)')


if __name__ == '__main__':
    main()
