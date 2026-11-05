"""flexkv trace - List and show FlexKV trace files."""

import argparse
import glob
import os
import sys
import time

from flexkv.cli.commands.base import (
    BaseCommand,
    add_output_arguments,
    print_json,
    print_table,
)


def _trace_files() -> list:
    base = os.getenv("FLEXKV_TRACE_FILE_PATH", "./flexkv_trace.log")
    pattern = base + ".*"
    files = sorted(glob.glob(pattern), key=lambda f: os.path.getmtime(f), reverse=True)
    if os.path.exists(base):
        files.insert(0, base)
    return files


class TraceCommand(BaseCommand):
    def name(self) -> str:
        return "trace"

    def help(self) -> str:
        return "List and show FlexKV trace files."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="action")
        p_list = sub.add_parser("list", help="List trace files with size and mtime.")
        p_show = sub.add_parser("show", help="Show trace file content.")
        add_output_arguments(p_list)
        add_output_arguments(p_show)
        p_show.add_argument("-n", "--lines", type=int, default=0,
                            help="Show last N lines (0 = all).")
        p_show.add_argument("-f", "--file", type=str, default=None,
                            help="Specific trace file path.")

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "action", None)
        if action is None:
            action = "list"
        if action == "list":
            self._list(args)
        elif action == "show":
            self._show(args)

    def _list(self, args: argparse.Namespace) -> None:
        files = _trace_files()
        if getattr(args, "format", "terminal") == "json":
            import json
            data = []
            for f in files:
                st = os.stat(f)
                data.append({
                    "path": f,
                    "size_bytes": st.st_size,
                    "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                })
            print_json(data)
            return
        if getattr(args, "quiet", False):
            return
        if not files:
            base = os.getenv("FLEXKV_TRACE_FILE_PATH", "./flexkv_trace.log")
            print(
                f"No trace files found (searched: {base} and rotated files).",
                file=sys.stderr,
            )
            print(
                "Set FLEXKV_TRACE_FILE_PATH to point to your trace file.",
                file=sys.stderr,
            )
            return
        rows = []
        for f in files:
            st = os.stat(f)
            size_str = _fmt_size(st.st_size)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            rows.append((f, f"{size_str}  {mtime}"))
        print_table("FlexKV Trace Files", rows)

    def _show(self, args: argparse.Namespace) -> None:
        path = getattr(args, "file", None) or os.getenv(
            "FLEXKV_TRACE_FILE_PATH", "./flexkv_trace.log"
        )
        if not os.path.exists(path):
            print(f"Trace file not found: {path}", file=sys.stderr)
            sys.exit(1)
        n = getattr(args, "lines", 0) or 0
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            if n > 0:
                lines = f.readlines()
                for line in lines[-n:]:
                    sys.stdout.write(line)
            else:
                shutil_copyfileobj(f, sys.stdout)


def _fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def shutil_copyfileobj(fsrc, fdst):
    import shutil
    shutil.copyfileobj(fsrc, fdst)
