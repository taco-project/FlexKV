"""flexkv storage - Show SSD cache directory usage."""

import argparse
import os
import sys

from flexkv.cli.commands.base import BaseCommand, print_table, print_json


def _dir_stats(path: str) -> dict:
    total_size = 0
    file_count = 0
    if not os.path.isdir(path):
        return {"path": path, "exists": False, "size_bytes": 0, "files": 0}
    for root, _, files in os.walk(path):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                total_size += os.path.getsize(fpath)
                file_count += 1
            except OSError:
                continue
    return {"path": path, "exists": True, "size_bytes": total_size, "files": file_count}


def _fmt_size(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.2f} GB"
    if n >= 1024 ** 2:
        return f"{n / 1024 ** 2:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


class StorageCommand(BaseCommand):
    def name(self) -> str:
        return "storage"

    def help(self) -> str:
        return "Show SSD cache directory usage and file counts."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        from flexkv.common.config import load_user_config_from_env

        user_config = load_user_config_from_env()
        dirs = user_config.ssd_cache_dir
        if isinstance(dirs, str):
            dirs = [dirs]

        results = [_dir_stats(d) for d in dirs]

        if getattr(args, "format", "terminal") == "json":
            print_json(results)
            return
        if getattr(args, "quiet", False):
            return

        rows = []
        for r in results:
            if not r["exists"]:
                rows.append((r["path"], "directory does not exist"))
            else:
                rows.append((
                    r["path"],
                    f"{_fmt_size(r['size_bytes'])}  ({r['files']} files)",
                ))
        if not rows:
            rows.append(("ssd_cache_dir", "not configured"))
        print_table("FlexKV Storage", rows)
