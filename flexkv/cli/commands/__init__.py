"""Discover and register FlexKV CLI commands."""

import importlib
import inspect
import pkgutil

from flexkv.cli.commands.base import BaseCommand


def _discover_commands():
    commands = []
    seen = set()
    for _finder, name, ispkg in pkgutil.iter_modules(__path__):
        if ispkg or name == "base":
            continue
        module = importlib.import_module(f"{__name__}.{name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls is BaseCommand or not issubclass(cls, BaseCommand):
                continue
            if inspect.isabstract(cls) or cls.__module__ != module.__name__:
                continue
            command = cls()
            command_name = command.name()
            if command_name in seen:
                raise RuntimeError(f"Duplicate CLI command name: {command_name}")
            seen.add(command_name)
            commands.append(command)
    commands.sort(key=lambda command: command.name())
    return commands


ALL_COMMANDS = _discover_commands()
