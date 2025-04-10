__version__ = "0.1.0"

from flexkv.core.block import BlockStatus, BlockMeta
from flexkv.core.evictor import EvictionPolicy
from flexkv.core.kv_manager import KVManager

__all__ = [
    "__version__",
    "BlockStatus",
    "BlockMeta",
    "EvictionPolicy",
    "KVManager",
]
