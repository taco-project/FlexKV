from unittest.mock import MagicMock, patch

import pytest

from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVDPClient
from flexkv.server.request import UnregisterDPClientRequest
from flexkv.server.server import KVServer


@pytest.fixture
def server_client_config(monkeypatch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "server_client_mode", True)
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "server_launch_mode", "embedded")
    return ModelConfig(), CacheConfig()


def test_embedded_mode_preserves_server_owner_behavior(server_client_config):
    model_config, cache_config = server_client_config
    server_handle = MagicMock()
    dp_client = MagicMock()

    with patch("flexkv.kvmanager.KVServer.create_server", return_value=server_handle) as create_server, \
         patch("flexkv.kvmanager.KVDPClient", return_value=dp_client):
        manager = KVManager(model_config, cache_config, dp_client_id=0)
        manager.start()
        manager.shutdown()

    create_server.assert_called_once()
    dp_client.start_server_and_register.assert_called_once()
    dp_client.shutdown.assert_called_once()
    dp_client.unregister.assert_not_called()
    server_handle.shutdown.assert_called_once()


def test_external_mode_never_starts_or_stops_shared_server(
    monkeypatch, server_client_config
):
    model_config, cache_config = server_client_config
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "server_launch_mode", "external")
    dp_client = MagicMock()

    with patch("flexkv.kvmanager.KVServer.create_server") as create_server, \
         patch("flexkv.kvmanager.KVDPClient", return_value=dp_client):
        manager = KVManager(model_config, cache_config, dp_client_id=0)
        manager.start()
        manager.shutdown()

    assert manager.server_handle is None
    assert not manager.owns_mps
    create_server.assert_not_called()
    dp_client.start_server_and_register.assert_called_once()
    dp_client.unregister.assert_called_once()
    dp_client.shutdown.assert_not_called()


def test_external_mode_requires_server_client_mode(monkeypatch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "server_client_mode", False)
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "server_launch_mode", "external")

    with pytest.raises(ValueError, match="requires server-client mode"):
        KVManager(ModelConfig(), CacheConfig(), dp_client_id=0)


def test_unregister_request_does_not_use_global_shutdown():
    client = object.__new__(KVDPClient)
    client.dp_client_id = 7
    client.send_to_server = MagicMock()

    client.unregister()

    request = client.send_to_server.send_pyobj.call_args.args[0]
    assert isinstance(request, UnregisterDPClientRequest)
    assert request.dp_client_id == 7


def test_server_unregisters_only_the_requesting_client():
    server = object.__new__(KVServer)
    server.client_manager = MagicMock()
    server.kv_task_engine = MagicMock()
    server._running = True

    server._handle_unregister_dp_client_request(UnregisterDPClientRequest(3))

    server.client_manager.delete_dp_client.assert_called_once_with(3)
    assert server._running
