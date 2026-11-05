"""FlexKV CLI entry point."""

import argparse
import sys

from flexkv.cli.commands import ALL_COMMANDS


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="flexkv",
        description="FlexKV local CLI for configuration and status inspection.",
    )
    subparsers = parser.add_subparsers(dest="command")
    for cmd in ALL_COMMANDS:
        cmd.register(subparsers)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()