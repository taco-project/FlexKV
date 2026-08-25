# External backend adapters for FlexKV.
# Currently provides the mooncake-store distributed KV cache backend.
from flexkv.external.mooncake_store_utils import (
    MooncakeStoreConfig,
    MooncakeStoreClient,
    MooncakeStoreCacheEngine,
)
from flexkv.external.mooncake_fault_inject import (
    inject_mooncake_fault,
    is_mooncake_fault_inject_enabled,
    reset_mooncake_fault_rng,
)

__all__ = [
    "MooncakeStoreConfig",
    "MooncakeStoreClient",
    "MooncakeStoreCacheEngine",
    "inject_mooncake_fault",
    "is_mooncake_fault_inject_enabled",
    "reset_mooncake_fault_rng",
]


def __getattr__(name):
    if name in ("MooncakeStoreConfig", "MooncakeStoreClient", "MooncakeStoreCacheEngine"):
        from . import mooncake_store_utils as _m
        return getattr(_m, name)
    if name in (
        "inject_mooncake_fault",
        "is_mooncake_fault_inject_enabled",
        "reset_mooncake_fault_rng",
    ):
        from . import mooncake_fault_inject as _m
        return getattr(_m, name)
    raise AttributeError(name)