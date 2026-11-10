"""Matched per-GPU FlexKV GDR / production HiCache some special machine types and scenarios transfer interference.

Each GPU is tested separately with its CP-held layers; --tp defines ownership,
not a simultaneous multi-GPU bandwidth test. CP broadcasts, scheduler work and
model forward are outside this scope. Native production transfer code is reused.
"""
import argparse
import copy
import csv
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

# Observed GLM-5.2 index_topk_freq=4 configuration (21/78 active layers).
GLM_INDEXER_LAYERS = (0, 1, 2) + tuple(range(6, 78, 4))


def import_runtime_module(name):
    # Use import statements: importlib.import_module bypasses __import__, whose
    # some special machine types and scenarios hook coordinates torch initialization with torch_xmlir registration.
    if name == 'torch':
        import torch
        return torch
    if name == 'flexkv.c_ext':
        from flexkv import c_ext
        return c_ext
    if name == 'sgl_kernel.kvcacheio':
        from sgl_kernel import kvcacheio
        return kvcacheio
    raise ValueError(f'unsupported runtime module: {name}')


def load_runtime(output, rank=None):
    """Keep the original native load order and record fatal import boundaries."""
    environment = {k: os.environ.get(k, '') for k in
                   ('LD_LIBRARY_PATH', 'PYTHONPATH', 'CONDA_PREFIX',
                    'LD_PRELOAD', 'XPU_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES',
                    'PYTHONFAULTHANDLER', 'PYTHONUNBUFFERED')}
    (output / 'runtime-environment.json').write_text(json.dumps(dict(
        executable=sys.executable, environment=environment), indent=2) + '\n')
    journal = output / 'runtime-imports.jsonl'
    def record(module, phase, value=None):
        # C++ registration can abort without raising a Python exception. Flush
        # each boundary so the last attempted import survives that failure.
        row = dict(pid=os.getpid(), module=module, phase=phase,
                   time_ns=time.time_ns(), path=getattr(value, '__file__', None))
        with journal.open('a') as file:
            file.write(json.dumps(row) + '\n')
        print(f'[runtime pid={row["pid"]}] {phase}: {module}', flush=True)
    stage = 'torch'
    try:
        modules = {}
        # torch -> c_ext succeeded in the original some special machine types and scenarios log. Do not eagerly
        # import xtorch_ops: sgl_kernel imports its own dependencies in order.
        # Resolve apiinfer via the launcher environment, not extra dlopen calls.
        for stage in ('torch', 'flexkv.c_ext', 'sgl_kernel.kvcacheio'):
            record(stage, 'begin')
            modules[stage] = import_runtime_module(stage)
            if stage == 'torch':
                from probe_compute import validate_special_platform_torch_runtime
                validate_special_platform_torch_runtime()
                if rank is not None:
                    modules[stage].cuda.set_device(rank)
            record(stage, 'ready', modules[stage])
        return modules['torch'], modules['flexkv.c_ext'], modules['sgl_kernel.kvcacheio']
    except (ImportError, OSError) as exc:
        (output / 'failure.json').write_text(json.dumps(dict(
            stage='runtime_import', module=stage, error=repr(exc),
            executable=sys.executable, environment=environment), indent=2) + '\n')
        raise RuntimeError(
            f'some special machine types and scenarios runtime import failed at {stage}: {exc}. '
            'Set LD_LIBRARY_PATH BEFORE launching Python, using the same '
            'torch/lib and xtorch_ops library directories as serving. '
            f'Details: {output}/failure.json') from exc


def owned_geometry(args, rank):
    choose = lambda layers: [l for l in layers if args.ownership == 'all-layers' or l % args.tp == rank]
    specs = [('main_kv', choose(range(args.layers)), args.page_size, args.main_token_bytes),
             ('indexer', choose(args.indexer_layer_ids), 1, args.indexer_page_bytes)]
    return [dict(name=name, layer_ids=layers, page_size=page, item_bytes=item)
            for name, layers, page, item in specs if layers]


def expand_pages(torch, pages, page_size):
    return (pages[:, None] * page_size + torch.arange(page_size, dtype=torch.int64)[None, :]).reshape(-1)


def check_gdr_route(routes, direction):
    prefix = direction.lower()
    if routes.get(prefix + '.gdr.calls', 0) <= 0 or routes.get(prefix + '.init_fallback.calls', 0):
        raise RuntimeError('GDR route was not verified or fell back; refuse this comparison')


class Buffer:
    def __init__(self, torch, fill_pattern, geometry, capacity, rank, cpu_pages, gpu_pages):
        self.geometry, self.capacity = geometry, capacity
        self.page, self.item = geometry['page_size'], geometry['item_bytes']
        self.layers = geometry['layer_ids']
        count = len(self.layers)
        self.host = torch.empty((capacity, count, self.page, 1, self.item),
                                dtype=torch.uint8, pin_memory=True)
        self.gpu = torch.empty((count, capacity * self.page, 1, self.item),
                               dtype=torch.uint8, device=f'cuda:{rank}')
        self.host_ids = expand_pages(torch, cpu_pages, self.page).pin_memory()
        self.device_ids_cpu = expand_pages(torch, gpu_pages, self.page).pin_memory()
        self.device_ids = self.device_ids_cpu.to(self.gpu.device)
        self.fill_pattern, self.torch = fill_pattern, torch
        self.layer_map = {layer: local for local, layer in enumerate(self.layers)}

    def reset(self, h2d, seed):
        if h2d:
            self.fill_pattern(self.host, 17, seed)
            self.gpu.zero_()
        else:
            self.fill_pattern(self.gpu, 13, seed)
            self.host.zero_()

    def expected(self, h2d, cpu_pages, gpu_pages):
        # Verification is outside timing; compare untouched pages as well.
        if h2d:
            result = self.gpu.cpu().reshape(len(self.layers), self.capacity, -1)
            source = self.host.reshape(self.capacity, len(self.layers), -1)
            result[:, gpu_pages, :] = source[cpu_pages, :, :].transpose(0, 1)
        else:
            result = self.host.clone().reshape(self.capacity, len(self.layers), -1)
            source = self.gpu.cpu().reshape(len(self.layers), self.capacity, -1)
            result[cpu_pages, :, :] = source[:, gpu_pages, :].transpose(0, 1)
        return result

    def verify(self, expected, h2d):
        actual = self.gpu.cpu().reshape(expected.shape) if h2d else self.host.reshape(expected.shape)
        if not self.torch.equal(actual, expected):
            raise AssertionError(f'{self.geometry["name"]}: moved or untouched KV bytes differ')


class HiCacheTransfer:
    def __init__(self, torch, buffers, rank, io=None):
        if io is None:
            from sgl_kernel import kvcacheio as io
        if not io.IS_KUNLUN:
            raise RuntimeError('HiCache comparison requires some special machine types and scenarios/KUNLUN production wrapper')
        self.torch, self.buffers, self.io = torch, buffers, io
        with torch.cuda.device(rank):
            self.stream = torch.cuda.Stream()
        self.rank = rank

    def execute(self, h2d):
        with self.torch.cuda.device(self.rank), self.torch.cuda.stream(self.stream):
            if h2d:
                # Global layer order, as in HiCacheController's per-layer loop.
                for layer in sorted({l for b in self.buffers for l in b.layers}):
                    for b in self.buffers:
                        if layer not in b.layer_map:
                            continue
                        local = b.layer_map[layer]
                        self.io.transfer_kv_per_layer_mla_pf_direct_lf(
                            src=b.host, dst=b.gpu[local], src_indices=b.host_ids,
                            dst_indices=b.device_ids, layer_id=local, page_size=b.page,
                            item_size=b.item, src_indices_cpu=b.host_ids,
                            dst_indices_cpu=b.device_ids_cpu)
            else:
                for b in self.buffers:
                    self.io.transfer_kv_all_layer_mla_lf_pf_direct(
                        src_layers=list(b.gpu.unbind(0)), dst=b.host,
                        src_indices=b.device_ids, dst_indices=b.host_ids,
                        item_size=b.item, page_size=b.page, num_layers=len(b.layers))
        # H2D API return is only enqueue. Include its final XPU scatter in ready.
        # D2H production API already waits through Host scatter; this is a
        # stream-scoped endpoint and never waits on the compute stream.
        self.stream.synchronize()

    def close(self):
        self.stream.synchronize()


class GdrTransfer:
    def __init__(self, torch, buffers, rank, args, cpu_pages, gpu_pages):
        from flexkv.transfer.region_batch import RegionSpec, build_region_batch, make_requests
        from flexkv.common.gdr_config import parse_gdr_config
        self.buffers = buffers
        specs = []
        for b in buffers:
            fragment = b.page * b.item
            count = len(b.layers)
            specs.append(RegionSpec(b.geometry['name'], b.host.data_ptr(), fragment,
                fragment, count * fragment, count * fragment,
                [layer.data_ptr() for layer in b.gpu.unbind(0)], count,
                [b.capacity * fragment], [fragment], [b.capacity * fragment],
                [fragment], count, 1, 1))
        self.group = build_region_batch(specs, [rank], is_blockfirst=True)
        self.config = parse_gdr_config(dict(enabled=True, qp_count=args.qps,
            nic_names=args.nics.split(',') if args.nics else [],
            use_dedicated_gdr_thread=True,
            h2d=dict(path='gdr', on_init_failure='error'),
            d2h=dict(path='gdr', on_init_failure='error', read_depth=args.read_depth),
            diagnostics=dict(summary=True, timing=False)))
        self.configure(True)
        self.cpu_pages, self.gpu_pages = cpu_pages.pin_memory(), gpu_pages.pin_memory()
        self.requests = {}
        h2d = []
        for layer in sorted({l for b in buffers for l in b.layers}):
            for i, b in enumerate(buffers):
                if layer in b.layer_map:
                    h2d += make_requests(len(buffers), self.gpu_pages, self.cpu_pages,
                        True, layer_id=b.layer_map[layer], layer_granularity=1,
                        region_indices=[i])
        self.requests[True] = h2d
        self.requests[False] = make_requests(len(buffers), self.gpu_pages, self.cpu_pages, False)

    def configure(self, diagnostics):
        self.config['diagnostics']['summary'] = diagnostics
        self.group.configure_gdr(self.config)
        for b in self.buffers:
            self.group.gdr_register_local(0, b.gpu)

    def execute(self, h2d):
        # Blocking finish includes native CQ drain and H2D visibility fences.
        self.group.submit(self.requests[h2d], True)

    def close(self):
        self.group.wait_all_streams()
        self.group = None


def run_case(args, arm, direction, blocks, rank, records, routes):
    import torch
    from compare_modes import block_ids, fill_pattern
    from probe_compute import Compute
    h2d = direction == 'H2D'
    geometry = owned_geometry(args, rank)
    if not geometry:
        return
    cpu_pages = torch.from_numpy(block_ids(blocks, args.cpu_run_pages, '', 0, 1))
    gpu_pages = torch.from_numpy(block_ids(blocks, args.gpu_run_pages, '', 0, 0))
    capacity = max(int(cpu_pages.max()), int(gpu_pages.max())) + 2
    with torch.cuda.device(rank):
        buffers = [Buffer(torch, fill_pattern, g, capacity, rank, cpu_pages, gpu_pages) for g in geometry]
        options = copy.copy(args)
        options.compute_process = ('local' if arm == 'HiCache' else 'isolated') \
            if args.compute_process == 'auto' else args.compute_process
        compute = transfer = None
        try:
            compute = Compute(torch, args.tp, options, ranks=[rank])
            transfer = HiCacheTransfer(torch, buffers, rank) if arm == 'HiCache' else \
                GdrTransfer(torch, buffers, rank, args, cpu_pages, gpu_pages)
            for b in buffers:
                b.reset(h2d, 0)
            torch.cuda.synchronize(rank)
            expected = [b.expected(h2d, cpu_pages, gpu_pages) for b in buffers]
            transfer.execute(h2d)
            for b, value in zip(buffers, expected):
                b.verify(value, h2d)
            if arm == 'GDR':
                stats = dict(transfer.group.gdr_statistics())
                check_gdr_route(stats, direction)
                routes.append(dict(arm=arm, direction=direction, blocks=blocks, rank=rank, stats=stats))
                transfer.configure(False)
            for i in range(args.warmup + args.rounds):
                # Reset and rotate the source so a no-op copy cannot reuse old
                # correct destination bytes. All setup/verification stays out
                # of the copy/compute timing interval.
                for b in buffers:
                    b.reset(h2d, i + 1)
                torch.cuda.synchronize(rank)
                expected = [b.expected(h2d, cpu_pages, gpu_pages) for b in buffers]
                baseline = compute.finish(compute.launch())
                pending = compute.launch()
                begin = time.perf_counter_ns()
                success = False
                try:
                    transfer.execute(h2d)
                    success = True
                finally:
                    end = time.perf_counter_ns()
                    overlap = compute.finish(pending, begin, end)
                    records.append(dict(arm=arm, direction=direction, blocks=blocks,
                        layout='PAGE_FIRST_DIRECT_' + args.ownership, rank=rank,
                        phase='warmup' if i < args.warmup else 'measure', index=i,
                        success=success, transfer_ns=end-begin, geometry=geometry,
                        payload_bytes=blocks * sum(len(g['layer_ids']) * g['page_size'] * g['item_bytes'] for g in geometry),
                        compute_process=options.compute_process,
                        compute=[dict(rank=b['rank'], baseline=b, overlap=o) for b, o in zip(baseline, overlap)]))
                try:
                    for b, value in zip(buffers, expected):
                        b.verify(value, h2d)
                    compute.verify()
                except BaseException:
                    records[-1]['success'] = False
                    raise
        finally:
            try:
                if transfer is not None:
                    transfer.close()
            finally:
                if compute is not None:
                    compute.close()


def write_results(output, records, routes, fractions, exposed_delta):
    from probe_compute import capacity_estimates, summarize
    (output / 'compute_calls.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (output / 'route_verification.json').write_text(json.dumps(routes, indent=2) + '\n')
    (output / 'capacity_estimates.json').write_text(json.dumps(
        capacity_estimates(records, fractions, exposed_delta), indent=2) + '\n')
    rows = summarize(records)
    if rows:
        with (output / 'compute_summary.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def stop_rank_worker(process):
    # start_new_session gives this worker and its compute descendants their own
    # process group. Clean only this group, including after a native abort.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def run_rank_workers(args, output, fractions):
    """HiCache's thread-local events are device-bound: never switch GPUs in a worker."""
    records, routes = [], []
    manifest = dict(arguments=vars(args), command=sys.argv,
        comparison='FlexKV direct GDR vs HiCache production some special machine types and scenarios gather/CE/scatter',
        rank_scope='one fresh process per CP rank, GPUs tested sequentially; not TP8 throughput',
        rank_metadata={})
    (output / 'compute_metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        for rank in range(args.tp):
            rank_dir = output / f'rank-{rank}'
            rank_dir.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, '-u', str(Path(__file__).with_name('probe_compute.py')),
                       *sys.argv[1:], '--output', str(rank_dir), '--rank-worker', str(rank)]
            print(f'Isolated CP rank={rank}/{args.tp}: {rank_dir}', flush=True)
            process = subprocess.Popen(command, start_new_session=True)
            try:
                try:
                    status = process.wait(timeout=args.rank_worker_timeout)
                finally:
                    stop_rank_worker(process)
            finally:
                # Preserve completed cases even if this worker aborts or times
                # out. The worker checkpoints between cases, outside timing.
                calls_file, routes_file = rank_dir / 'compute_calls.jsonl', rank_dir / 'route_verification.json'
                if calls_file.is_file():
                    records.extend(json.loads(line) for line in calls_file.read_text().splitlines() if line)
                if routes_file.is_file():
                    routes.extend(json.loads(routes_file.read_text()))
                metadata_file = rank_dir / 'compute_metadata.json'
                if metadata_file.is_file():
                    metadata = json.loads(metadata_file.read_text())
                    manifest['rank_metadata'][str(rank)] = metadata
                    for key in ('compute_context', 'excluded', 'negative_control', 'hicache_operators'):
                        if key in metadata:
                            manifest[key] = metadata[key]
                    (output / 'compute_metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
            if status != 0:
                raise RuntimeError(f'CP rank {rank} worker exited with code {status}; details: {rank_dir}')
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(exc)), indent=2) + '\n')
        raise
    finally:
        write_results(output, records, routes, fractions, args.exposed_transfer_delta_fraction)
    print(f'Results: {output}/compute_summary.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--arms', default='HiCache,GDR')
    parser.add_argument('--directions', default='H2D,D2H')
    parser.add_argument('--blocks', default='32,256')
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--concurrency', type=int, choices=[1], default=1)
    parser.add_argument('--ownership', choices=['cp-layer-split', 'all-layers'], default='cp-layer-split')
    parser.add_argument('--layers', type=int, default=78)
    parser.add_argument('--page-size', type=int, default=64)
    parser.add_argument('--main-token-bytes', type=int, default=656)
    parser.add_argument('--indexer-page-bytes', type=int, default=16384)
    parser.add_argument('--indexer-layer-ids', default=','.join(map(str, GLM_INDEXER_LAYERS)))
    parser.add_argument('--cpu-run-pages', type=int, default=0)
    parser.add_argument('--gpu-run-pages', type=int, default=1)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--qps', type=int, default=4)
    parser.add_argument('--read-depth', type=int, default=16)
    parser.add_argument('--nics', default='')
    parser.add_argument('--sglang-root', default='/data/kelvinlv/sglang-special-platform')
    parser.add_argument('--compute-process', choices=['auto', 'local', 'isolated'], default='auto')
    for name, default in [('m', 512), ('n', 2048), ('k', 2048), ('iters', 64)]:
        parser.add_argument('--compute-' + name, type=int, default=default)
    parser.add_argument('--compute-timeout', type=float, default=120)
    parser.add_argument('--rank-worker-timeout', type=float, default=600,
                        help='Maximum seconds for all cases on one isolated GPU process.')
    parser.add_argument('--rank-worker', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--model-affected-fractions', default='0.1,0.3,0.5,0.8')
    parser.add_argument('--exposed-transfer-delta-fraction', type=float, default=0)
    args = parser.parse_args()
    try:
        args.indexer_layer_ids = [int(v) for v in args.indexer_layer_ids.split(',') if v]
        blocks = [int(v) for v in args.blocks.split(',')]
        fractions = [float(v) for v in args.model_affected_fractions.split(',')]
        arms, directions = args.arms.split(','), args.directions.split(',')
        if set(arms) != {'HiCache', 'GDR'} or len(arms) != 2:
            raise ValueError('--arms must contain HiCache,GDR; CE is the FlexKV backend, not HiCache')
        if not directions or not set(directions) <= {'H2D', 'D2H'}:
            raise ValueError('directions must be H2D,D2H')
        positive = blocks + [args.tp, args.layers, args.page_size, args.main_token_bytes,
                            args.indexer_page_bytes, args.rounds, args.warmup, args.compute_m,
                            args.compute_n, args.compute_k, args.compute_iters, args.qps, args.read_depth]
        if min(positive) <= 0 or min(args.cpu_run_pages, args.gpu_run_pages) < 0:
            raise ValueError('sizes/rounds/warmup must be positive; run lengths nonnegative')
        if args.main_token_bytes % 8 or args.indexer_page_bytes % 8:
            raise ValueError('production XPU gather/scatter needs 8-byte aligned item sizes')
        if args.indexer_layer_ids != sorted(set(args.indexer_layer_ids)) or any(
                not 0 <= l < args.layers for l in args.indexer_layer_ids):
            raise ValueError('indexer layer IDs must be unique, sorted and within main layers')
        from probe_compute import capacity_estimates, summarize
        capacity_estimates([], fractions, args.exposed_transfer_delta_fraction)
        if not 0 < args.compute_timeout < float('inf'):
            raise ValueError('compute timeout must be finite and positive')
        if not 0 < args.rank_worker_timeout < float('inf'):
            raise ValueError('rank worker timeout must be finite and positive')
        if args.rank_worker is not None and not 0 <= args.rank_worker < args.tp:
            raise ValueError('rank worker must be within the CP rank range')
    except ValueError as exc:
        parser.error(str(exc))
    sys.path.insert(0, str(Path(args.sglang_root) / 'python'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    faulthandler.enable(all_threads=True)
    if args.rank_worker is None:
        # The coordinator stays CPU-only; all native caches belong to a fresh
        # process permanently assigned to exactly one GPU, as in serving.
        return run_rank_workers(args, output, fractions)
    torch, c_ext, kvcacheio = load_runtime(output, args.rank_worker)
    if args.tp > torch.cuda.device_count() or not c_ext.gdr_compiled() or not kvcacheio.IS_KUNLUN:
        parser.error('requires enough some special machine types and scenarios GPUs, native GDR and KUNLUN HiCache operators')
    manifest = dict(arguments=vars(args), command=sys.argv,
        comparison='FlexKV direct GDR vs HiCache production some special machine types and scenarios gather/CE/scatter',
        rank_scope='single CP rank in a fresh GPU process; coordinator tests ranks sequentially',
        compute_context='auto: same process/context for HiCache, isolated serving context for GDR',
        excluded='CP broadcasts, scheduler/LPM, inactive indexer layers and actual model forward',
        negative_control=args.compute_process != 'auto',
        hicache_operators=['sgl_kernel.transfer_kv_per_layer_mla_pf_direct_lf_ce',
                          'sgl_kernel.transfer_kv_all_layer_mla_lf_pf_direct_ce'],
        loaded_native_libraries={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in map(Path, torch.ops.loaded_libraries) if p.is_file()},
        source_files={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__), Path(kvcacheio.__file__), Path(c_ext.__file__))},
        environment={k: v for k, v in os.environ.items() if k in
                     ('LD_LIBRARY_PATH', 'PYTHONPATH', 'CONDA_PREFIX') or
                     k.startswith(('XSGL_TRANSFER_', 'BKCL_', 'CUDA_', 'XPU_', 'OMP_', 'MKL_'))})
    (output / 'compute_metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records, routes = [], []
    try:
        for direction in directions:
            for count in blocks:
                for arm in arms:
                    rank = args.rank_worker
                    print(f'{arm} {direction} blocks={count} CP rank={rank}/{args.tp}', flush=True)
                    run_case(args, arm, direction, count, rank, records, routes)
                    write_results(output, records, routes, fractions, args.exposed_transfer_delta_fraction)
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(error=repr(exc)), indent=2) + '\n')
        raise
    finally:
        write_results(output, records, routes, fractions, args.exposed_transfer_delta_fraction)
    print(f'Results: {output}/compute_summary.csv', flush=True)


if __name__ == '__main__':
    main()
