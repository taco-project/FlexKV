"""Unit tests for the is_ready() request/response protocol.

``KVTaskEngine.is_ready()`` raises when the TransferManager subprocess died
during startup. Every caller polls in a ``while not is_ready(): sleep()`` loop,
and the server's run loop wraps handlers in a blanket ``except Exception``, so
an uncaught raise inside the handler would return no reply at all and leave the
client blocked forever in a synchronous ``recv_pyobj()``. The two halves tested
here are what turn that hang back into an error:

  * the server answers with ``Response(is_ready=False, error_msg=...)``
  * the client raises on a response carrying ``error_msg``

Run with: pytest tests/test_server_is_ready_protocol.py -q
"""
import pytest

from flexkv.server.client import KVDPClient
from flexkv.server.request import IsReadyRequest, Response
from flexkv.server.server import KVServer

pytestmark = pytest.mark.unit


class _FakeSocket:
    def __init__(self, inbox=None):
        self.sent = []
        self._inbox = list(inbox or ())

    def send_pyobj(self, obj):
        self.sent.append(obj)

    def recv_pyobj(self):
        return self._inbox.pop(0)


class _FakeClientManager:
    def __init__(self, socket):
        self._socket = socket

    def get_zmq(self, dp_client_id):
        return self._socket


class _FakeEngine:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def is_ready(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeServer:
    # Stands in for KVServer without its __init__, which binds zmq sockets and
    # spawns the task engine.
    def __init__(self, engine):
        self.kv_task_engine = engine
        self.socket = _FakeSocket()
        self.client_manager = _FakeClientManager(self.socket)

    _handle_is_ready_request = KVServer._handle_is_ready_request


class _FakeClient:
    def __init__(self, response):
        self.dp_client_id = 3
        self.send_to_server = _FakeSocket()
        self.recv_from_server = _FakeSocket([response])

    is_ready = KVDPClient.is_ready


# --------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------

def test_server_reports_ready():
    server = _FakeServer(_FakeEngine(result=True))
    server._handle_is_ready_request(IsReadyRequest(dp_client_id=3))
    (response,) = server.socket.sent
    assert response.dp_client_id == 3
    assert response.is_ready is True
    assert response.error_msg is None


def test_server_reports_not_ready_without_error():
    server = _FakeServer(_FakeEngine(result=False))
    server._handle_is_ready_request(IsReadyRequest(dp_client_id=3))
    (response,) = server.socket.sent
    assert response.is_ready is False
    # Still polling is not a failure: the client must keep waiting.
    assert response.error_msg is None


def test_server_always_replies_when_is_ready_raises():
    # The failure this guards: no reply at all, and a client blocked in recv.
    server = _FakeServer(_FakeEngine(exc=RuntimeError("subprocess exited")))
    server._handle_is_ready_request(IsReadyRequest(dp_client_id=3))
    (response,) = server.socket.sent
    assert response.is_ready is False
    assert "subprocess exited" in response.error_msg


# --------------------------------------------------------------------------
# client side
# --------------------------------------------------------------------------

def test_client_returns_ready_flag():
    assert _FakeClient(Response(dp_client_id=3, is_ready=True)).is_ready() is True
    assert _FakeClient(Response(dp_client_id=3, is_ready=False)).is_ready() is False


def test_client_sends_its_own_id():
    client = _FakeClient(Response(dp_client_id=3, is_ready=True))
    client.is_ready()
    (req,) = client.send_to_server.sent
    assert isinstance(req, IsReadyRequest)
    assert req.dp_client_id == 3


def test_client_raises_on_server_error():
    # Returning False here would make the caller's polling loop spin forever on
    # a server that can never become ready.
    client = _FakeClient(
        Response(dp_client_id=3, is_ready=False, error_msg="subprocess exited")
    )
    with pytest.raises(RuntimeError, match="subprocess exited"):
        client.is_ready()
