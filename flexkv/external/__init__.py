# External backend adapters for FlexKV.
# Currently provides the mooncake-store distributed KV cache backend.
from flexkv.external.mooncake_store_utils import (
    MooncakeStoreConfig,
    MooncakeStoreClient,
    MooncakeStoreCacheEngine,
)

__all__ = ["MooncakeStoreConfig", "MooncakeStoreClient", "MooncakeStoreCacheEngine"]
def __getattr__(name):
    if name in __all__:
        from . import mooncake_store_utils as _m
        return getattr(_m, name)
    raise AttributeError(name)