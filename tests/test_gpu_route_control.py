"""Route lifecycle control stays scoped to a registered inference client."""
from threading import Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import zmq

from flexkv.common.transfer import WorkerKey
from flexkv.server.client import KVTPClient
from flexkv.transfer_manager import TransferManager

pytestmark = pytest.mark.unit


def manager():
    m = TransferManager.__new__(TransferManager)
    m.transfer_engine = Mock()
    m.storage_engine = object()
    m.all_gpu_blocks = {(2, 3): object()}
    m.gpu_worker_key_mapping = {(2, 3): WorkerKey(2, 0)}
    return m


@pytest.mark.parametrize("action", ["fence", "recover"])
def test_control_resolves_dp_from_registration_key(action):
    m = manager()
    getattr(m.transfer_engine, f"{action}_gpu_route").return_value = {"safe": True, "generation": 4}
    response = m.handle_gpu_control({"type": f"{action}_gpu_route", "registration_key": (2, 3),
                                    "generation": 3, "dp_client_id": 999})
    assert response == {"ok": True, "safe": True, "generation": 4}
    args = (2,) if action == "fence" else (2, 3)
    getattr(m.transfer_engine, f"{action}_gpu_route").assert_called_once_with(*args)


def test_unknown_exporter_cannot_fence_another_instance():
    m = manager()
    with pytest.raises(KeyError, match="Unknown registration"):
        m.handle_gpu_control({"type": "fence_gpu_route", "registration_key": (999, 0)})
    m.transfer_engine.fence_gpu_route.assert_not_called()


def test_client_uses_fresh_req_connections_and_reports_server_errors(tmp_path):
    endpoint = f"ipc://{tmp_path}/control"
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.setsockopt(zmq.RCVTIMEO, 5000)
    server.bind(endpoint)
    client = KVTPClient.__new__(KVTPClient)
    client.gpu_control_socket = SimpleNamespace(context=context)
    client._gpu_control_endpoint = endpoint
    client.dp_client_id, client.intra_client_id = 2, 3
    requests, errors = [], []

    def respond():
        try:
            for reply in ({"ok": True, "safe": True, "generation": 1},
                          {"ok": False, "error": "quarantined"},
                          {"ok": True, "safe": True, "generation": 2}):
                requests.append(server.recv_pyobj())
                server.send_pyobj(reply)
        except BaseException as exc:
            errors.append(exc)

    thread = Thread(target=respond)
    thread.start()
    try:
        assert client.gpu_route_control("fence")["generation"] == 1
        with pytest.raises(RuntimeError, match="quarantined"):
            client.gpu_route_control("recover", 1)
        assert client.gpu_route_control("recover", 1)["generation"] == 2
        thread.join(timeout=5)
        assert not thread.is_alive() and not errors
        assert [request["registration_key"] for request in requests] == [(2, 3)] * 3
        with pytest.raises(ValueError, match="Unknown"):
            client.gpu_route_control("bad")
    finally:
        server.close(linger=0)
        context.destroy(linger=0)
