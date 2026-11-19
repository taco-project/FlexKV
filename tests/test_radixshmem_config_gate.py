"""radix_shmem serves replicas that each sit on one node: a replica spanning
nodes keeps a block's KV on all of them, which the radix-servers cannot see."""
import pytest

import flexkv.server.shm_radix_bootstrap as bootstrap
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
from flexkv.kvmanager import KVManager

pytestmark = pytest.mark.unit


class _Attached(Exception):
    pass


def _attach(*args, **kwargs):
    raise _Attached


@pytest.fixture(autouse=True)
def _radixshmem(monkeypatch):
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "enable_radixshmem", True)
    monkeypatch.setattr(bootstrap, "adopt_radix_server", _attach)


@pytest.mark.parametrize("tp_size,pp_size", [(16, 1), (8, 2)], ids=["tp", "pp"])
def test_a_replica_spanning_nodes_is_refused(tp_size, pp_size):
    model_config = ModelConfig(tp_size=tp_size, pp_size=pp_size, nnodes=2)

    with pytest.raises(ValueError, match="replica spanning nodes"):
        KVManager(model_config, CacheConfig(), dp_client_id=0)


def test_node_local_dp_ranks_reach_the_radix_server():
    model_config = ModelConfig(dp_size=2, nnodes=2, local_dp_size=1,
                               enable_dp_attention=True)

    with pytest.raises(_Attached):
        KVManager(model_config, CacheConfig(), dp_client_id=0)
