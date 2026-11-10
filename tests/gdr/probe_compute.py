"""FlexKV GDR versus production HiCache copy/compute interference.

The default comparison uses HiCache's some special machine types and scenarios native gather/CE/scatter operators.
--comparison flexkv-ce retains the older FlexKV-only CE/GDR control explicitly.

This measures synthetic GEMM, not GLM forward or an SM occupancy percentage.
capacity_estimates.json translates probe performance into retained capacity
and conditional model budgets; these are assumptions, not hardware counters.
Use compute_calls.jsonl/compute_summary.csv, not compare_modes' elapsed columns:
the wrapper inserts a compute-alone baseline before each measured copy.
"""
import argparse
import csv
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import sys
import threading
import time


def validate_special_platform_torch_runtime():
    """Do not benchmark plain/partially patched torch after a swallowed hook error."""
    if 'xpytorch_import_hook' in sys.modules:
        xmlir = sys.modules.get('torch_xmlir')
        if xmlir is None or getattr(xmlir, '_XMLIRC', None) is None:
            raise ImportError(
                'torch imported, but some special machine types and scenarios import hook did not initialize '
                'torch_xmlir._XMLIRC. Check the FIRST import-hook error; '
                'the current environment torch/lib (libc10_cuda.so and '
                'libtorch_python.so) must be discoverable before Python starts.')


def capacity_estimates(records, affected_fractions, exposed_delta_fraction=0):
    """Fixed-work throughput ratios and an explicitly conditional time budget.

    f is the CE critical-path time spent in affected compute segments / total
    time (not just their slowdown penalty, nor all CPU forward spans).
    delta is net additional exposed NON-compute time / CE total time, including
    H2D, synchronization and control changes; overlapped durations do not count.
    Neither f nor delta can be inferred from synthetic copy duration alone.
    """
    if not math.isfinite(exposed_delta_fraction):
        raise ValueError('exposed delta fraction must be finite')
    if not affected_fractions or any(not math.isfinite(f) or not 0 <= f <= 1 for f in affected_fractions):
        raise ValueError('affected fractions must be finite and between 0 and 1')
    groups = {}
    for row in records:
        if row['phase'] != 'measure' or not row['success']:
            continue
        for rank in row['compute']:
            key=(row['direction'],row['blocks'],row['layout'],rank['rank'])
            groups.setdefault(key,{}).setdefault(row['arm'],[]).append(rank)
    estimates=[]
    def valid(value):
        return isinstance(value,(int,float)) and math.isfinite(value) and value > 0
    def samples(rows,clock):
        # A CPU-observed overlap is a necessary proxy, not GPU overlap proof.
        return [(r['baseline'].get(clock),r['overlap'].get(clock)) for r in rows
                if r['overlap']['observed_overlap_ns'] > 0
                and valid(r['baseline'].get(clock)) and valid(r['overlap'].get(clock))]
    for (direction,blocks,layout,rank),arms in groups.items():
        reference = 'HiCache' if 'HiCache' in arms else 'CE'
        if not {reference,'GDR'} <= arms.keys():
            continue
        clock='device_ms'
        ce,gdr=samples(arms[reference],clock),samples(arms['GDR'],clock)
        if not ce or not gdr:
            clock='wall_ns'
            ce,gdr=samples(arms[reference],clock),samples(arms['GDR'],clock)
        if not ce or not gdr:
            continue
        retained_ce=statistics.median(a/b for a,b in ce)
        retained_gdr=statistics.median(a/b for a,b in gdr)
        gain=retained_gdr/retained_ce
        scenarios=[]
        for f in affected_fractions:
            # Non-compute time after the change cannot be negative. Do not
            # manufacture a speedup from an impossible supplied time budget.
            remainder=1-f+exposed_delta_fraction
            denominator=remainder+f/gain
            scenarios.append(dict(affected_fraction=f,
                exposed_delta_fraction=exposed_delta_fraction,
                valid=remainder >= 0 and denominator > 0,
                estimated_tpm_gain_pct=(1/denominator-1)*100 if remainder >= 0 and denominator > 0 else None))
        estimates.append(dict(direction=direction,blocks=blocks,layout=layout,rank=rank,
            clock=clock,reference_arm=reference,reference_samples=len(ce),gdr_samples=len(gdr),
            reference_retained_capacity_pct=retained_ce*100,gdr_retained_capacity_pct=retained_gdr*100,
            **({'ce_samples':len(ce),'ce_retained_capacity_pct':retained_ce*100} if reference == 'CE'
               else {'hicache_samples':len(ce),'hicache_retained_capacity_pct':retained_ce*100}),
            recovered_capacity_percentage_points=(retained_gdr-retained_ce)*100,
            normalized_probe_speedup_pct=(gain-1)*100,
            baseline_gdr_to_reference_ratio=statistics.median(a for a,b in gdr)/statistics.median(a for a,b in ce),
            **({'baseline_gdr_to_ce_ratio':statistics.median(a for a,b in gdr)/statistics.median(a for a,b in ce)}
               if reference == 'CE' else {}),
            conditional_model_scenarios=scenarios))
    return dict(estimates=estimates,
        scope='Same fixed GEMM work per rank; normalized to each arm baseline. Device performance or host wall proxy, never measured SM occupancy.',
        assumptions='Model scenarios assume this probe speedup applies to the affected model work, fixed token counts and request mix, and no other scheduling/cache/concurrency changes.',
        formula='retained=T0/Tcopy; gain=retained_GDR/retained_reference; TPM_ratio=1/((1-f)+f/gain+delta)',
        note='Only observed-overlap samples included. Baseline drift and host observation lag remain confounders. No estimate when an arm lacks usable overlapping samples. Physical SM counts require hardware counters or a validated resource-scaling calibration.')


def summarize(records):
    groups = {}
    for row in records:
        if row['phase'] != 'measure' or not row['success']:
            continue
        for rank in row['compute']:
            groups.setdefault((row['arm'], row['direction'], row['blocks'],
                               row['layout'], rank['rank']), []).append((row, rank))
    result = []
    for (arm, direction, blocks, layout, rank), pairs in groups.items():
        base = [r['baseline']['wall_ns']/1e6 for _, r in pairs]
        overlap = [r['overlap']['wall_ns']/1e6 for _, r in pairs]
        item = dict(arm=arm, direction=direction, blocks=blocks, layout=layout,
                    rank=rank, samples=len(pairs), baseline_wall_ms=statistics.median(base),
                    overlap_wall_ms=statistics.median(overlap),
                    wall_slowdown_pct=statistics.median((b/a-1)*100 for a,b in zip(base,overlap)),
                    transfer_ms=statistics.median(r['transfer_ns']/1e6 for r,_ in pairs),
                    overlap_observed_fraction=sum(r['overlap']['observed_overlap_ns'] > 0
                                                  for _,r in pairs)/len(pairs))
        if 'payload_bytes' in pairs[0][0]:
            item['payload_bytes'] = pairs[0][0]['payload_bytes']
            item['compute_process'] = pairs[0][0]['compute_process']
            item['main_layers'] = sum(len(g['layer_ids']) for g in pairs[0][0]['geometry'] if g['name'] == 'main_kv')
            item['indexer_layers'] = sum(len(g['layer_ids']) for g in pairs[0][0]['geometry'] if g['name'] == 'indexer')
        valid = [(r['baseline']['device_ms'],r['overlap']['device_ms']) for _,r in pairs
                 if r['baseline']['device_ms'] is not None and r['overlap']['device_ms'] is not None]
        item['valid_device_samples'] = len(valid)
        item['device_slowdown_pct'] = statistics.median((b/a-1)*100 for a,b in valid) if valid else None
        result.append(item)
    return result


class LocalCompute:
    def __init__(self, torch, tp, options, only_rank=None, ranks=None):
        self.torch, self.iterations, self.states = torch, options.compute_iters, []
        for rank in (ranks if ranks is not None else (range(tp) if only_rank is None else [only_rank])):
            with torch.cuda.device(rank):
                stream = torch.cuda.Stream()
                # KV and compute allocations never alias; D2H source is immutable.
                with torch.cuda.stream(stream):
                    a = torch.ones((options.compute_m, options.compute_k), device=f'cuda:{rank}', dtype=torch.bfloat16)
                    b = torch.ones((options.compute_k, options.compute_n), device=f'cuda:{rank}', dtype=torch.bfloat16)
                    c = torch.empty((options.compute_m, options.compute_n), device=f'cuda:{rank}', dtype=torch.bfloat16)
                    torch.mm(a,b,out=c)
                stream.synchronize()  # setup/warmup only
                self.states.append((rank,stream,a,b,c))

    def launch(self):
        pending = []
        for rank,stream,a,b,c in self.states:
            row = dict(rank=rank, error=None)
            with self.torch.cuda.device(rank), self.torch.cuda.stream(stream):
                begin, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
                row['begin_ns'] = time.perf_counter_ns()
                begin.record(stream)
                for _ in range(self.iterations):
                    self.torch.mm(a,b,out=c)
                end.record(stream)
                row['enqueue_ns'] = time.perf_counter_ns()-row['begin_ns']
            def wait(row=row, rank=rank, begin=begin, end=end):
                try:
                    with self.torch.cuda.device(rank):
                        end.synchronize()  # this compute stream only, never device-wide
                        row['end_ns'] = time.perf_counter_ns()
                        try:
                            elapsed = float(begin.elapsed_time(end))
                            row['device_ms'] = elapsed if math.isfinite(elapsed) and elapsed > 0 else None
                        except Exception:
                            row['device_ms'] = None
                except BaseException as exc:
                    row['error'] = repr(exc)
            thread = threading.Thread(target=wait)
            thread.start()
            pending.append((thread,row))
        return pending

    def verify(self):
        for rank,stream,a,b,c in self.states:
            with self.torch.cuda.device(rank):
                if not bool(self.torch.all(c == a.shape[1]).item()):
                    raise AssertionError(f'compute result mismatch on rank {rank}')

    def close(self):
        pass

    @staticmethod
    def finish(pending, copy_begin=0, copy_end=0):
        rows = []
        for thread,row in pending:
            thread.join()
        for thread,row in pending:
            if row['error']:
                raise RuntimeError(f"compute completion failed: {row['error']}")
            row['wall_ns'] = row['end_ns']-row['begin_ns']
            row['observed_overlap_ns'] = max(0,min(copy_end,row['end_ns'])-max(copy_begin,row['begin_ns']))
            rows.append(row)
        return rows


def _compute_child(pipe, rank, options):
    """Separate CUDA context, like serving rank versus transfer worker."""
    try:
        import torch
        validate_special_platform_torch_runtime()
        torch.set_num_threads(1)
        compute=LocalCompute(torch,1,options,only_rank=rank)
        pipe.send(('ready',None))
        while True:
            command=pipe.recv()
            if command == 'stop': break
            if command == 'verify':
                compute.verify();pipe.send(('verified',None))
            elif command == 'run':
                pending=compute.launch()
                pipe.send(('enqueued',None))
                pipe.send(('done',compute.finish(pending)))
            else: raise ValueError(command)
    except BaseException as exc:
        try: pipe.send(('error',repr(exc)))
        except (BrokenPipeError,EOFError,OSError): pass
    finally:
        pipe.close()


class Compute:
    def __init__(self, torch, tp, options, ranks=None):
        self.local=None;self.workers=[];self.timeout=options.compute_timeout
        if options.compute_process == 'local':
            self.local=LocalCompute(torch,tp,options,ranks=ranks)
            return
        ctx=multiprocessing.get_context('spawn')
        try:
            for rank in (range(tp) if ranks is None else ranks):
                parent,child=ctx.Pipe()
                process=ctx.Process(target=_compute_child,args=(child,rank,options),daemon=True)
                process.start();child.close()
                self.workers.append((parent,process))
            for pipe,process in self.workers: self._receive(pipe,'ready')
        except BaseException:
            self.close();raise

    def _receive(self, pipe, expected):
        if not pipe.poll(self.timeout):
            raise TimeoutError(f'compute process did not report {expected} in {self.timeout}s')
        kind,value=pipe.recv()
        if kind != expected: raise RuntimeError(f'compute process expected {expected}, received {kind}: {value}')
        return value

    def launch(self):
        if self.local: return self.local.launch()
        for pipe,process in self.workers: pipe.send('run')
        for pipe,process in self.workers: self._receive(pipe,'enqueued')
        return [pipe for pipe,process in self.workers]

    def finish(self, pending, copy_begin=0, copy_end=0):
        if self.local: return self.local.finish(pending,copy_begin,copy_end)
        rows=[]
        for pipe in pending:
            for row in self._receive(pipe,'done'):
                row['observed_overlap_ns']=max(0,min(copy_end,row['end_ns'])-max(copy_begin,row['begin_ns']))
                rows.append(row)
        return rows

    def verify(self):
        if self.local: return self.local.verify()
        for pipe,process in self.workers: pipe.send('verify')
        for pipe,process in self.workers: self._receive(pipe,'verified')

    def close(self):
        for pipe,process in self.workers:
            try: pipe.send('stop')
            except (BrokenPipeError,EOFError,OSError): pass
        for pipe,process in self.workers:
            process.join(timeout=1)
            if process.is_alive(): process.terminate();process.join(timeout=5)
            pipe.close()
        self.workers=[]


def flexkv_ce_main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name,default in [('m',512),('n',2048),('k',2048),('iters',64)]:
        parser.add_argument(f'--compute-{name}', type=int, default=default)
    parser.add_argument('--compute-process',choices=['isolated','local'],default='isolated')
    parser.add_argument('--compute-timeout',type=float,default=120)
    parser.add_argument('--model-affected-fractions',default='0.1,0.3,0.5,0.8',
                        help='Conditional CE critical-path compute fractions; scenarios, not inferred measurements.')
    parser.add_argument('--exposed-transfer-delta-fraction',type=float,default=0,
                        help='Net extra exposed non-compute time / CE total time; default zero is an assumption.')
    options, forwarded = parser.parse_known_args()
    if min(options.compute_m,options.compute_n,options.compute_k,options.compute_iters) <= 0 or not 0 < options.compute_timeout < float('inf'):
        parser.error('compute dimensions and iterations must be positive')
    try:
        fractions=[float(v) for v in options.model_affected_fractions.split(',')]
        capacity_estimates([],fractions,options.exposed_transfer_delta_fraction)
    except ValueError as exc:
        parser.error(str(exc))
    guard = argparse.ArgumentParser(add_help=False)
    guard.add_argument('--output', required=True)
    guard.add_argument('--arms', default='CE,GDR')
    guard.add_argument('--concurrency', default='1')
    guard.add_argument('--warmup', type=int, default=3)
    guard.add_argument('--diagnostics', type=int, default=0)
    args,_ = guard.parse_known_args(forwarded)
    if args.concurrency != '1' or args.diagnostics != 0 or not set(args.arms.split(',')) <= {'CE','GDR'}:
        parser.error('requires --concurrency 1 --diagnostics 0; arms CE,GDR only')
    command = list(sys.argv)
    import compare_modes as target  # defer native/torch imports, allow CPU summary tests
    for name,value in [('arms',args.arms),('concurrency','1'),('warmup',str(args.warmup))]:
        if not any(v == '--'+name or v.startswith('--'+name+'=') for v in forwarded):
            forwarded += ['--'+name,value]
    sys.argv = [sys.argv[0]]+forwarded
    records, case, computations = [], {}, []
    run_case, build = target.run_case, target.build_region_batch
    def run(a,arm,direction,blocks,concurrency,layout):
        case.clear();case.update(arm=arm,direction=direction,blocks=blocks,layout=layout)
        try:
            result = run_case(a,arm,direction,blocks,concurrency,layout)
            # The delegated harness has drained all copies before checking.
            computations[-1].verify()
            result["measurement"] = "Compute-interference wrapper includes inserted baselines; use compute_summary.csv/compute_calls.jsonl, not delegated transfer timing columns."
            return result
        finally:
            for compute in computations: compute.close()
            computations.clear()
    class Capture:
        def __init__(self, group, tp):
            self.group, self.case = group, dict(case)
            self.compute = Compute(target.torch,tp,options)
            self.generation, self.calls, self.pending = 0,0,None
            computations.append(self.compute)
        def __getattr__(self,name):
            return getattr(self.group,name)
        def configure_gdr(self,cfg):
            self.generation += 1;self.calls = 0
            return self.group.configure_gdr(cfg)
        def _begin(self):
            if self.generation == 1:
                return
            self.baseline = self.compute.finish(self.compute.launch())
            self.pending = self.compute.launch()
            self.started = time.perf_counter_ns()
        def _end(self,success):
            if self.pending is None:
                return
            ended = time.perf_counter_ns()
            try:
                overlap = self.compute.finish(self.pending,self.started,ended)
                records.append(dict(self.case,phase='warmup' if self.calls < args.warmup else 'measure',
                                    index=self.calls,success=success,transfer_ns=ended-self.started,
                                    compute=[dict(rank=b['rank'],baseline=b,overlap=o)
                                             for b,o in zip(self.baseline,overlap)]))
            finally:
                self.pending = None
                self.calls += 1
        def submit(self,*a,**kw):
            self._begin();success=False
            try:
                result=self.group.submit(*a,**kw);success=True
                return result
            finally:
                self._end(success)
        def submit_layerwise(self,*a,**kw):
            self._begin()
            try:
                return self.group.submit_layerwise(*a,**kw)
            except BaseException:
                self._end(False)
                raise
        def wait_layer_completion(self,*a,**kw):
            success=False
            try:
                result=self.group.wait_layer_completion(*a,**kw);success=bool(result[0])
                return result
            finally:
                self._end(success)
    target.run_case = run
    target.build_region_batch = lambda *a,**kw:Capture(build(*a,**kw),len(a[1]))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    try:
        target.main()
    finally:
        (out/'compute_calls.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out/'compute_metadata.json').write_text(json.dumps(dict(command=command,compute=vars(options),
            scope='fixed synthetic GEMM; default separate processes/contexts like serving versus worker; not model forward or SM occupancy',
            clocks='device events when positive; host wall includes launch and host scheduling',
            note='Observed overlap is a host-clock proxy, not proof of device kernel overlap. Copy may wait for compute through existing GDR device-wide visibility sync. Zero some special machine types and scenarios event times are unavailable. Compare per-rank results; do not sum parallel rank times.'),indent=2)+'\n')
        if (out/'summary.md').exists():
            path=out/'summary.md'
            path.write_text('COMPUTE CONTROL: these delegated transfer timings include inserted compute baselines. Use compute_summary.csv/compute_calls.jsonl.\n\n'+path.read_text())
        rows=summarize(records)
        (out/'capacity_estimates.json').write_text(json.dumps(
            capacity_estimates(records,fractions,options.exposed_transfer_delta_fraction),indent=2)+'\n')
        if rows:
            with (out/'compute_summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        print(f'Compute interference results: {out}/compute_summary.csv')


def main():
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument('--comparison', choices=['hicache', 'flexkv-ce'], default='hicache')
    selection, remaining = selector.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    if selection.comparison == 'flexkv-ce':
        return flexkv_ce_main()
    from probe_hicache_compute import main as hicache_main
    return hicache_main()


if __name__ == '__main__':
    main()
