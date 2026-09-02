"""flexkv config - Show or validate FlexKV configuration."""

import argparse
import os
import sys
from dataclasses import fields

from flexkv.cli.commands.base import (
    BaseCommand,
    add_output_arguments,
    print_json,
    print_table,
)


def _display_value(name: str, value) -> str:
    lowered = name.lower()
    if any(word in lowered for word in ("password", "secret", "token")):
        return "<redacted>"
    return str(value)


class ConfigCommand(BaseCommand):
    def name(self) -> str:
        return "config"

    def help(self) -> str:
        return "Show or validate FlexKV configuration."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        sub = parser.add_subparsers(dest="action")
        for action_parser in (
            sub.add_parser("show", help="Show resolved configuration."),
            sub.add_parser("validate", help="Validate configuration constraints."),
        ):
            add_output_arguments(action_parser)

    def execute(self, args: argparse.Namespace) -> None:
        action = getattr(args, "action", None)
        if action is None:
            action = "show"
        if action == "show":
            self._show(args)
        elif action == "validate":
            self._validate(args)

    def _load_config(self):
        config_path = os.getenv("FLEXKV_CONFIG_PATH")
        if config_path:
            from flexkv.common.config import load_user_config_from_file
            user_config = load_user_config_from_file(config_path)
            source = f"file: {config_path}"
        else:
            from flexkv.common.config import load_user_config_from_env
            user_config = load_user_config_from_env()
            source = "environment variables"
        return user_config, source

    def _show(self, args: argparse.Namespace) -> None:
        user_config, source = self._load_config()
        rows = [("config_source", source)]
        for f in fields(user_config):
            value = getattr(user_config, f.name)
            rows.append((f.name, _display_value(f.name, value)))

        from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
        for key in [
            "enable_metrics", "py_metrics_port", "cpp_metrics_port",
            "server_recv_port", "instance_num", "instance_id",
            "enable_trace", "trace_file_path",
        ]:
            if hasattr(GLOBAL_CONFIG_FROM_ENV, key):
                value = getattr(GLOBAL_CONFIG_FROM_ENV, key)
                rows.append((f"env.{key}", _display_value(key, value)))

        if getattr(args, "format", "terminal") == "json":
            print_json(dict(rows))
        elif not getattr(args, "quiet", False):
            print_table("FlexKV Configuration", rows)

    def _validate(self, args: argparse.Namespace) -> None:
        try:
            user_config, _ = self._load_config()
        except Exception as e:
            print(f"Config load failed: {e}", file=sys.stderr)
            raise SystemExit(1) from None

        violations = []
        if user_config.cpu_cache_gb <= 0:
            violations.append(f"cpu_cache_gb must be > 0, got {user_config.cpu_cache_gb}")
        if user_config.ssd_cache_gb < 0:
            violations.append(f"ssd_cache_gb must be >= 0, got {user_config.ssd_cache_gb}")
        if user_config.ssd_cache_gb > 0 and user_config.ssd_cache_gb <= user_config.cpu_cache_gb:
            violations.append(
                f"ssd_cache_gb ({user_config.ssd_cache_gb}) must be > "
                f"cpu_cache_gb ({user_config.cpu_cache_gb})"
            )
        if user_config.ssd_cache_gb > 0:
            dirs = user_config.ssd_cache_dir
            if isinstance(dirs, str):
                dirs = [dirs]
            for d in dirs:
                if not os.path.isdir(d):
                    violations.append(f"SSD cache directory does not exist: {d}")

        if violations:
            for v in violations:
                print(f"ERROR: {v}", file=sys.stderr)
            self._show(args)
            raise SystemExit(1)
        else:
            if not getattr(args, "quiet", False):
                print("Configuration OK.")
