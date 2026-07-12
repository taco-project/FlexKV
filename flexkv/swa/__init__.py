from flexkv.swa.swa_host_pool import SWAHostPool

# SWA is NODE-MOUNTED on the Full-KV radix tree (see csrc/radix_tree.cpp and
# flexkv/cache/radixtree.py). The radix nodes own the SWA slot / tombstone /
# lock; this host pool only supplies the slot bytes + free-list. Eviction is
# unified with Full-KV through the one tree.

__all__ = [
    "SWAHostPool",
]
