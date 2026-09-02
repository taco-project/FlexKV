"""flexkv collect-env - Collect environment information for troubleshooting."""

import argparse
import os
import platform
import subprocess
import sys

from flexkv.cli.commands.base import BaseCommand, print_table, print_json


def _run(cmd: list, timeout: int = 10) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "N/A"


class CollectEnvCommand(BaseCommand):
    def name(self) -> str:
        return "collect-env"

    def help(self) -> str:
        return "Collect environment information for troubleshooting."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        rows = []
        rows.append(("OS", f"{platform.system()} {platform.release()}"))
        rows.append(("Python", sys.version.split()[0]))

        git_commit = _run(["git", "rev-parse", "--short", "HEAD"])
        rows.append(("Git Commit", git_commit))

        try:
            import torch
            rows.append(("PyTorch", torch.__version__))
            rows.append(("CUDA Available", str(torch.cuda.is_available())))
            if torch.cuda.is_available():
                rows.append(("CUDA Version", torch.version.cuda or "N/A"))
                gpu_names = []
                for i in range(torch.cuda.device_count()):
                    gpu_names.append(torch.cuda.get_device_name(i))
                rows.append(("GPU Count", str(len(gpu_names))))
                rows.append(("GPU Model", "; ".join(gpu_names)))
        except ImportError:
            rows.append(("PyTorch", "not installed"))

        driver = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if driver != "N/A":
            rows.append(("NVIDIA Driver", driver.split("\n")[0]))

        flexkv_vars = {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith("FLEXKV_") or k == "ENABLE_FLEXKV"
        }
        rows.append(("FLEXKV_* Vars", f"{len(flexkv_vars)} set" if flexkv_vars else "none set"))

        pip_list = _run(["pip", "list", "--format=freeze"])
        key_pkgs = []
        for line in (pip_list or "").split("\n"):
            pkg = line.split("==")[0].lower()
            if pkg in ("torch", "transformers", "vllm", "pyzmq", "prometheus-client", "numpy"):
                key_pkgs.append(line)
        if key_pkgs:
            rows.append(("Key Packages", "; ".join(key_pkgs)))

        if getattr(args, "format", "terminal") == "json":
            print_json(dict(rows))
        elif not getattr(args, "quiet", False):
            print_table("FlexKV Environment", rows)
