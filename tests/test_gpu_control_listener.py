"""The GPU control REP socket must be served in every deployment mode.

``TransferManager.__init__`` binds ``<gpu_register_port>_control``
unconditionally, but only the subprocess mode's selector loop used to drain
it.  In thread mode and remote mode the endpoint therefore accepted the
client's REQ and then never answered, turning a vLLM sleep/wake call into a
silent stall for the client's full 120s RCVTIMEO.
"""
import tempfile
import threading

import pytest
import zmq

from flexkv.transfer_manager import TransferManager


@pytest.fixture
def control_endpoint():
    with tempfile.NamedTemporaryFile(suffix="_control") as handle:
        yield f"ipc://{handle.name}"


def _listening_manager(endpoint):
    """A TransferManager with only the control-plane bits wired up."""
    manager = TransferManager.__new__(TransferManager)
    manager.context = zmq.Context(1)
    manager.gpu_control_port = endpoint
    manager.gpu_control_socket = manager.context.socket(zmq.REP)
    manager.gpu_control_socket.bind(endpoint)
    manager._gpu_control_shutdown = threading.Event()
    manager._gpu_control_thread = None
    return manager


def _request(endpoint, payload, timeout_ms=5000):
    context = zmq.Context(1)
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    try:
        socket.send_pyobj(payload)
        return socket.recv_pyobj()
    finally:
        socket.close()
        context.term()


def test_listener_answers_control_requests(control_endpoint):
    manager = _listening_manager(control_endpoint)
    seen = []
    manager.handle_gpu_control = lambda request: (
        seen.append(request) or {"ok": True, "echo": request["value"]}
    )

    manager.start_gpu_control_listener()
    try:
        # Without a listener this recv blocks until RCVTIMEO and raises.
        assert _request(control_endpoint, {"value": 1}) == {"ok": True, "echo": 1}
        # REQ/REP is strictly alternating: a second round-trip proves the
        # listener loops rather than serving exactly one request.
        assert _request(control_endpoint, {"value": 2}) == {"ok": True, "echo": 2}
    finally:
        manager.stop_gpu_control_listener()
        manager.gpu_control_socket.close()
        manager.context.term()

    assert seen == [{"value": 1}, {"value": 2}]


def test_listener_reports_handler_failure_instead_of_hanging(control_endpoint):
    """A raising handler must still send a reply.

    REQ/REP is a lockstep socket: skipping the reply would wedge the client
    for its whole RCVTIMEO and leave the REP socket unable to accept the next
    request.
    """
    manager = _listening_manager(control_endpoint)

    def _boom(request):
        raise RuntimeError("suspend failed")

    manager.handle_gpu_control = _boom

    manager.start_gpu_control_listener()
    try:
        response = _request(control_endpoint, {"type": "suspend_gpu"})
    finally:
        manager.stop_gpu_control_listener()
        manager.gpu_control_socket.close()
        manager.context.term()

    assert response["ok"] is False
    assert "suspend failed" in response["error"]


def test_stop_is_idempotent_and_start_does_not_double_spawn(control_endpoint):
    manager = _listening_manager(control_endpoint)
    manager.handle_gpu_control = lambda request: {"ok": True}

    manager.start_gpu_control_listener()
    thread = manager._gpu_control_thread
    manager.start_gpu_control_listener()
    assert manager._gpu_control_thread is thread

    try:
        manager.stop_gpu_control_listener()
        assert not thread.is_alive()
        manager.stop_gpu_control_listener()
    finally:
        manager.gpu_control_socket.close()
        manager.context.term()
