"""``KVManager`` must only call methods its two backends actually expose.

``KVManager`` dispatches every public operation to one of two objects that
share no base class: ``KVDPClient`` (server_client_mode) or ``KVTaskEngine``.
Nothing checks that the two agree, so renaming a method on one side --
``cancel_task`` -> ``cancel_tasks`` -- leaves the other call site raising
AttributeError at runtime, and only in the mode that CI happens not to
exercise.

Each test drives the real ``KVManager`` method against an autospecced
backend, which raises AttributeError for any method the real class does not
define.  That is what makes this catch a rename rather than restating one.
"""
from unittest.mock import create_autospec

import pytest

from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVTaskEngine
from flexkv.server.client import KVDPClient

pytestmark = pytest.mark.unit


def _manager(server_client_mode: bool) -> KVManager:
    """A KVManager with both backends stubbed and no real resources."""
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode = server_client_mode
    manager.server_launch_mode = "embedded"
    manager.owns_mps = False
    manager.server_handle = None
    # spec_set: attribute access outside the real class raises, so a call to a
    # method that was renamed away fails here instead of silently passing.
    manager.dp_client = create_autospec(KVDPClient, spec_set=True, instance=True)
    manager.kv_task_engine = create_autospec(
        KVTaskEngine, spec_set=True, instance=True
    )
    return manager


@pytest.mark.parametrize("server_client_mode", [True, False],
                         ids=["server_client_mode", "engine_mode"])
def test_cancel_reaches_backend(server_client_mode):
    """Regression: KVManager.cancel called the removed ``cancel_task``."""
    manager = _manager(server_client_mode)

    manager.cancel(7)

    backend = manager.dp_client if server_client_mode else manager.kv_task_engine
    backend.cancel_tasks.assert_called_once_with([7])
    # The scalar overload must reach the backend as a list.
    other = manager.kv_task_engine if server_client_mode else manager.dp_client
    assert not other.method_calls


@pytest.mark.parametrize("server_client_mode", [True, False],
                         ids=["server_client_mode", "engine_mode"])
def test_cancel_accepts_a_list(server_client_mode):
    manager = _manager(server_client_mode)

    manager.cancel([1, 2, 3])

    backend = manager.dp_client if server_client_mode else manager.kv_task_engine
    backend.cancel_tasks.assert_called_once_with([1, 2, 3])


@pytest.mark.parametrize("server_client_mode", [True, False],
                         ids=["server_client_mode", "engine_mode"])
def test_wait_reaches_backend(server_client_mode):
    manager = _manager(server_client_mode)
    # KVManager.wait is annotated -> Dict, and the Cython build enforces it,
    # so the stub has to return a real dict.
    manager.dp_client.wait.return_value = {}
    manager.kv_task_engine.wait.return_value = {}

    manager.wait(5, timeout=1.0)

    backend = manager.dp_client if server_client_mode else manager.kv_task_engine
    assert backend.wait.call_count == 1


@pytest.mark.parametrize("server_client_mode", [True, False],
                         ids=["server_client_mode", "engine_mode"])
def test_is_ready_reaches_backend(server_client_mode):
    manager = _manager(server_client_mode)

    manager.is_ready()

    backend = manager.dp_client if server_client_mode else manager.kv_task_engine
    assert backend.is_ready.call_count == 1


@pytest.mark.parametrize("server_client_mode", [True, False],
                         ids=["server_client_mode", "engine_mode"])
def test_start_reaches_backend(server_client_mode):
    manager = _manager(server_client_mode)

    manager.start()

    if server_client_mode:
        manager.dp_client.start_server_and_register.assert_called_once_with()
    else:
        manager.kv_task_engine.start.assert_called_once_with()


@pytest.mark.parametrize(
    "server_client_mode,server_launch_mode",
    [(True, "embedded"), (True, "external"), (False, "embedded")],
    ids=["server_client-embedded", "server_client-external", "engine_mode"],
)
def test_shutdown_reaches_backend(server_client_mode, server_launch_mode):
    manager = _manager(server_client_mode)
    manager.server_launch_mode = server_launch_mode

    manager.shutdown()

    if not server_client_mode:
        manager.kv_task_engine.shutdown.assert_called_once_with()
    elif server_launch_mode == "external":
        manager.dp_client.unregister.assert_called_once_with()
    else:
        manager.dp_client.shutdown.assert_called_once_with()
