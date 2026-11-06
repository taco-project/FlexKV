import json
import os
import sys
import types

import pytest

from flexkv.cli.commands import ALL_COMMANDS
from flexkv.cli.commands import status as status_command
from flexkv.cli.commands.config import ConfigCommand
from flexkv.cli.commands.storage import StorageCommand
from flexkv.cli.main import main


def test_cli_entry_point_help(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["flexkv", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    expected_commands = {
        "collect-env", "config", "env", "metrics", "status", "storage", "trace"
    }
    assert expected_commands.issubset({command.name() for command in ALL_COMMANDS})
    for name in expected_commands:
        assert name in output


def test_config_show_redacts_sensitive_fields(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "flexkv.json"
    config_path.write_text(json.dumps({
        "cpu_cache_gb": 16,
        "ssd_cache_gb": 0,
        "redis_password": "super-secret",
    }))
    monkeypatch.setenv("FLEXKV_CONFIG_PATH", str(config_path))

    ConfigCommand().execute(type("Args", (), {
        "action": "show", "format": "json", "quiet": False
    })())

    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["redis_password"] == "<redacted>"
    assert "super-secret" not in output


def test_config_validate_failure_returns_nonzero(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "flexkv.json"
    config_path.write_text(json.dumps({
        "cpu_cache_gb": 1,
        "ssd_cache_gb": 2,
        "ssd_cache_dir": str(tmp_path / "missing"),
    }))
    monkeypatch.setenv("FLEXKV_CONFIG_PATH", str(config_path))

    with pytest.raises(SystemExit) as excinfo:
        ConfigCommand().execute(type("Args", (), {
            "action": "validate", "format": "terminal", "quiet": True
        })())

    assert excinfo.value.code == 1
    assert "SSD cache directory does not exist" in capsys.readouterr().err


def test_config_validate_load_failure_is_stable(capsys, monkeypatch):
    monkeypatch.setenv("FLEXKV_CPU_CACHE_GB", "not-a-number")

    with pytest.raises(SystemExit) as excinfo:
        ConfigCommand().execute(type("Args", (), {
            "action": "validate", "format": "terminal", "quiet": True
        })())

    assert excinfo.value.code == 1
    assert "Config load failed" in capsys.readouterr().err


def test_config_validate_accepts_common_arguments_after_action(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "flexkv", "config", "validate", "--quiet"
    ])
    monkeypatch.setenv("FLEXKV_CPU_CACHE_GB", "not-a-number")

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1


def test_status_excludes_current_cli_process(monkeypatch):
    class Process:
        info = {
            "pid": os.getpid(),
            "cmdline": ["flexkv", "status"],
            "environ": {},
        }

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    fake_psutil = types.SimpleNamespace(
        process_iter=lambda attrs: [Process()],
        NoSuchProcess=NoSuchProcess,
        AccessDenied=AccessDenied,
    )
    monkeypatch.setattr(status_command, "_HAS_PSUTIL", True)
    monkeypatch.setattr(
        status_command, "psutil", fake_psutil, raising=False
    )

    assert status_command._scan() == []


def test_storage_existing_directory(tmp_path, capsys, monkeypatch):
    cache_path = tmp_path / "cache"
    cache_path.mkdir()
    (cache_path / "data.bin").write_bytes(b"x")
    monkeypatch.setenv("FLEXKV_SSD_CACHE_DIR", str(cache_path))

    StorageCommand().execute(type("Args", (), {
        "format": "terminal", "quiet": False
    })())

    output = capsys.readouterr().out
    assert str(cache_path) in output
    assert "1 B" in output
    assert "1 files" in output
