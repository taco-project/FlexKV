"""Run existing GPU checks/benchmarks with GDR tuning overrides.

The existing benchmark and shell scripts are left unchanged. --ipc selects
real cross-process correctness checks; otherwise arguments go to compare_modes.
"""
import argparse
import json
from pathlib import Path
import sys



def set_share_mode(target, name):
    """Override request ownership without changing the reference workload."""
    from flexkv.transfer.region_batch import rank_share_mode
    mode = rank_share_mode(name)
    original = target.make_requests
    def requests(*args, **kwargs):
        kwargs["share_mode"] = mode
        return original(*args, **kwargs)
    target.make_requests = requests


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--ipc", action="store_true")
    p.add_argument("--share-mode", choices=["sharded", "layer_parallel"], default=None)
    p.add_argument("--read-depth", type=int, default=16)
    tuning, remaining = p.parse_known_args()
    if tuning.ipc and tuning.share_mode is not None:
        p.error("--share-mode applies to the comparison benchmark, not --ipc")
    command = list(sys.argv)
    sys.argv = [sys.argv[0]] + remaining
    configs = {}
    if tuning.ipc:
        import test_direct_ipc as target
    else:
        import compare_modes as target
        if tuning.share_mode is not None:
            set_share_mode(target, tuning.share_mode)
    # Both existing entry points validate configuration with this module-level
    # function. Override only tuning fields; their arm/path/verification logic
    # and measurement boundaries stay identical.
    parse = target.parse_gdr_config
    def configured(value=None):
        c = parse(value)
        c["d2h"]["read_depth"] = tuning.read_depth
        if tuning.share_mode is not None:
            c["d2h"]["auto_layer_parallel"] = False
        c = parse(c)
        configs[c["h2d"]["path"]] = c
        return c
    target.parse_gdr_config = configured
    if tuning.ipc:
        target.main()
    else:
        output_parser = argparse.ArgumentParser(add_help=False)
        output_parser.add_argument("--output", required=True)
        output, _ = output_parser.parse_known_args(remaining)
        Path(output.output).mkdir(parents=True, exist_ok=True)
        try:
            target.main()
        finally:
            (Path(output.output)/"tuning.json").write_text(json.dumps(
                dict(command=command, tuning=vars(tuning), configs=configs), indent=2)+"\n")


if __name__ == "__main__":
    main()
