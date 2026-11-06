"""Abstract base class and shared output helpers for CLI subcommands."""

import abc
import argparse
import json
import sys


class BaseCommand(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def help(self) -> str:
        ...

    @abc.abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        ...

    @abc.abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        ...

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(self.name(), help=self.help())
        self.add_arguments(parser)
        add_output_arguments(parser)
        parser.set_defaults(func=self.execute)


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        default="terminal",
        choices=["terminal", "json"],
        help="Output format (default: terminal).",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress stdout output (exit code only).",
    )


def print_table(title: str, rows: list, width: int = 48) -> None:
    if not rows:
        print(f"\n{title}\n{'=' * width}\n  (no data)\n")
        return
    label_width = max(len(str(r[0])) for r in rows)
    print(f"\n{title}")
    print("=" * width)
    for label, value in rows:
        print(f"  {label:<{label_width}}  {value}")
    print()


def print_json(data) -> None:
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False, default=str)
    print()
