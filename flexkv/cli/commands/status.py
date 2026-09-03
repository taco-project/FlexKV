"""flexkv status - Discover local FlexKV processes and runtime endpoints."""

import argparse
import os
import socket

from flexkv.cli.commands.base import BaseCommand, print_table, print_json

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, OSError, socket.timeout):
        return False


def _proc_environ(pid: int) -> dict:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
        result = {}
        for pair in raw.split(b"\x00"):
            if b"=" in pair:
                k, _, v = pair.partition(b"=")
                result[k.decode("utf-8", errors="replace")] = v.decode(
                    "utf-8", errors="replace"
                )
        return result
    except (FileNotFoundError, PermissionError):
        return {}


def _scan() -> list:
    processes = []
    if not _HAS_PSUTIL:
        import glob
        for cmdline_path in glob.glob("/proc/[0-9]*/cmdline"):
            pid_str = cmdline_path.split("/")[2]
            try:
                with open(cmdline_path, "rb") as f:
                    cmdline = f.read().replace(b"\x00", b" ").decode(
                        "utf-8", errors="replace"
                    ).strip()
            except (FileNotFoundError, PermissionError):
                continue
            pid = int(pid_str)
            if pid == os.getpid():
                continue
            if "flexkv" in cmdline.lower() or "FLEXKV" in cmdline:
                env = _proc_environ(int(pid_str))
                processes.append({
                    "pid": pid,
                    "cmdline": cmdline[:120],
                    "instance_id": env.get("FLEXKV_INSTANCE_ID", ""),
                    "server_port": env.get("FLEXKV_SERVER_RECV_PORT", ""),
                })
        return processes

    for proc in psutil.process_iter(["pid", "cmdline", "environ"]):
        try:
            if proc.info["pid"] == os.getpid():
                continue
            cmdline = " ".join(proc.info["cmdline"] or [])
            if "flexkv" not in cmdline.lower() and "FLEXKV" not in cmdline:
                continue
            env = proc.info.get("environ") or {}
            processes.append({
                "pid": proc.info["pid"],
                "cmdline": cmdline[:120],
                "instance_id": env.get("FLEXKV_INSTANCE_ID", ""),
                "server_port": env.get("FLEXKV_SERVER_RECV_PORT", ""),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def _check_ipc() -> list:
    import glob
    results = []
    for path in sorted(glob.glob("/tmp/flexkv_server*")):
        results.append((path, "exists" if os.path.exists(path) else "missing"))
    return results


def _check_metrics_ports() -> list:
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    results = []
    for name, port_attr in [("python", "py_metrics_port"), ("cpp", "cpp_metrics_port")]:
        port = getattr(GLOBAL_CONFIG_FROM_ENV, port_attr, None)
        if port is not None:
            status = "listening" if _port_open(port) else "not listening"
            results.append((f"metrics_{name}_port_{port}", status))
    return results


class StatusCommand(BaseCommand):
    def name(self) -> str:
        return "status"

    def help(self) -> str:
        return "Show local FlexKV processes, IPC endpoints, and metrics port status."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        procs = _scan()
        ipc = _check_ipc()
        metrics = _check_metrics_ports()

        if getattr(args, "format", "terminal") == "json":
            import json
            data = {
                "processes": procs,
                "ipc_endpoints": dict(ipc),
                "metrics_ports": dict(metrics),
            }
            print_json(data)
            return

        if getattr(args, "quiet", False):
            return

        rows = []
        if procs:
            for p in procs:
                rows.append((f"PID {p['pid']}", p["cmdline"]))
                if p.get("instance_id"):
                    rows.append((f"  instance_id", p["instance_id"]))
                if p.get("server_port"):
                    rows.append((f"  server_port", p["server_port"]))
        else:
            rows.append(("processes", "no FlexKV processes found"))

        rows.append(("", ""))
        if ipc:
            for path, st in ipc:
                rows.append((f"IPC {path}", st))
        else:
            rows.append(("IPC endpoints", "none found in /tmp"))

        rows.append(("", ""))
        for name, st in metrics:
            rows.append((name, st))

        print_table("FlexKV Status", rows)
