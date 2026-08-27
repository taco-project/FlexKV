"""The names external frameworks import from us, checked from this side.

Everything here is load-bearing for a *different repo*: sglang's FlexKV
connector (``sglang/srt/mem_cache/storage/flexkv/``) imports these paths and
reads these attributes. That makes them a contract, and it makes them the one
kind of breakage the rest of this suite structurally cannot catch -- there is
no importer of them inside this tree, so a rename passes every test here and
fails only when a server is launched.

Both entries below were found exactly that way, by launching one:

* ``flexkv.transfer.layerwise`` was renamed to ``layer_eventfd`` when the
  layerwise *worker* dissolved into ``CompletionContract.PER_LAYER``. The
  connector's import sits in a ``try/except ImportError`` that re-raises as
  "FlexKV is not installed", so the rename presented as a missing package.
* ``ModelConfig.use_mla`` became the ``kv_dim`` field. The connector reads the
  old name inside the GPU-registration path, which retries on any exception --
  so it presented as 360 identical "GPU register retry" lines and a startup
  timeout.

Neither is a name this repo uses itself. That is the point.
"""
import importlib

import pytest

from flexkv.common.config import ModelConfig

pytestmark = pytest.mark.unit


# The exact import lines in sglang's flexkv_connector.py, as (module, name).
SGLANG_IMPORTS = [
    ("flexkv.common.request", "KVResponseStatus"),
    ("flexkv.common.storage", "KVCacheLayout"),
    ("flexkv.common.storage", "KVCacheLayoutType"),
    ("flexkv.integration.config", "FlexKVConfig"),
    ("flexkv.kvmanager", "KVManager"),
    ("flexkv.server.client", "KVTPClient"),
    ("flexkv.transfer.layerwise", "build_layerwise_eventfd_socket_path"),
    ("flexkv.transfer_manager", "TransferManagerOnRemote"),
]


@pytest.mark.parametrize("module_path,name", SGLANG_IMPORTS,
                         ids=[f"{m.rsplit('.', 1)[-1]}.{n}"
                              for m, n in SGLANG_IMPORTS])
def test_sglang_can_still_import_it(module_path, name):
    module = importlib.import_module(module_path)
    assert hasattr(module, name), (
        f"sglang's connector does `from {module_path} import {name}`. "
        f"Re-export it from the old path even if the definition moved.")


def test_use_mla_tracks_kv_dim():
    """MLA is one KV tensor per token; plain MHA is two (K and V)."""
    assert ModelConfig(kv_dim=1).use_mla is True
    assert ModelConfig(kv_dim=2).use_mla is False


def test_use_mla_is_derived_not_stored():
    """A settable copy could disagree with ``kv_dim``, and the connector uses
    it to choose the registration layout while everything here sizes buffers
    from ``kv_dim`` -- they would silently describe different memory."""
    config = ModelConfig(kv_dim=2)
    with pytest.raises(AttributeError):
        config.use_mla = True
