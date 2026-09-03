"""flexkv env - List FLEXKV_* environment variables."""

import argparse
import os

from flexkv.cli.commands.base import BaseCommand, print_table, print_json


class EnvCommand(BaseCommand):
    def name(self) -> str:
        return "env"

    def help(self) -> str:
        return "List all FLEXKV_* environment variables."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> None:
        rows = []
        for key, value in sorted(os.environ.items()):
            if key.startswith("FLEXKV_") or key == "ENABLE_FLEXKV":
                rows.append((key, value))

        if getattr(args, "format", "terminal") == "json":
            print_json(dict(rows))
        elif not getattr(args, "quiet", False):
            print_table("FlexKV Environment Variables", rows)
