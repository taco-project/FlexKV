"""SWA Connector — Bridge between SGLang FlexKVConnector and FlexKV SWARadixManager.

Provides high-level SWA operations that work with token_ids,
converting to/from radix tree nodes internally. This is the integration
layer that the SGLang connector (or KVManager) calls for SWA store/load.
"""
from typing import Optional, Union, List, TYPE_CHECKING

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_radix_manager import SWARadixManager

if TYPE_CHECKING:
    from flexkv.cache.radixtree import RadixNode, RadixTreeIndex


class SWAConnector:
    """Bridge between SGLang FlexKVConnector and FlexKV SWARadixManager.

    Provides high-level SWA operations that work with token_ids
    (converting to/from radix tree nodes internally).

    Lifecycle:
        1. On request finish (store):
           connector calls store_swa(token_ids, swa_data) -> stores SWA snapshot

        2. On prefix match (load):
           connector calls check_swa(token_ids) -> bool
           connector calls load_swa(token_ids) -> data or None

    This connector is meant to be instantiated once per KVManager and kept
    alive for the duration of the server process.
    """

    def __init__(self, swa_manager: SWARadixManager, radix_tree: 'RadixTreeIndex'):
        """Initialize SWAConnector.

        Args:
            swa_manager: The SWARadixManager that handles pool operations.
            radix_tree: The radix tree index used for prefix matching.
                       Used to locate the leaf node for given token_ids.
        """
        self._swa_manager = swa_manager
        self._radix_tree = radix_tree

    @property
    def swa_manager(self) -> SWARadixManager:
        return self._swa_manager

    @property
    def config(self) -> SWAPoolConfig:
        return self._swa_manager.config

    def store_swa(
        self,
        token_ids: Union['torch.Tensor', np.ndarray],
        swa_data: Union['torch.Tensor', np.ndarray, bytes],
        leaf_node: Optional['RadixNode'] = None,
    ) -> bool:
        """Called by connector on request finish. Stores SWA for the prefix.

        Finds the leaf node for the given token_ids in the radix tree
        and stores the SWA data on it.

        Args:
            token_ids: The full token sequence that was just processed.
            swa_data: Raw SWA snapshot data (sliding window attention state).
            leaf_node: Optional pre-resolved leaf node. If None, we look it up
                      via the radix tree.

        Returns:
            True if successfully stored, False otherwise.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        node = leaf_node
        if node is None:
            node = self._find_leaf_node(token_ids)

        if node is None:
            return False

        return self._swa_manager.swa_put(node, swa_data)

    def load_swa(
        self,
        token_ids: Union['torch.Tensor', np.ndarray],
        leaf_node: Optional['RadixNode'] = None,
    ) -> Optional[Union['torch.Tensor', np.ndarray]]:
        """Called by connector on prefix match. Returns SWA data or None.

        Args:
            token_ids: The token prefix that was matched.
            leaf_node: Optional pre-resolved leaf node.

        Returns:
            SWA data buffer (CPU tensor/ndarray) or None if not available.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        node = leaf_node
        if node is None:
            node = self._find_leaf_node(token_ids)

        if node is None:
            return None

        return self._swa_manager.swa_load_back(node)

    def check_swa(
        self,
        token_ids: Union['torch.Tensor', np.ndarray],
        leaf_node: Optional['RadixNode'] = None,
    ) -> bool:
        """Check if SWA is available for given prefix.

        Verifies that the trailing window_size tokens of the matched
        prefix all have SWA data stored (no tombstones in the window).

        Args:
            token_ids: The token prefix to check.
            leaf_node: Optional pre-resolved leaf node.

        Returns:
            True if SWA data is available for the trailing window.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        node = leaf_node
        if node is None:
            node = self._find_leaf_node(token_ids)

        if node is None:
            return False

        # Build the path from root to this node
        path = self._build_path_to_node(node)
        return self._swa_manager.check_swa_trailing(path)

    def lock_swa(
        self,
        token_ids: Union['torch.Tensor', np.ndarray],
        leaf_node: Optional['RadixNode'] = None,
    ) -> None:
        """Lock SWA data for active request (prevents eviction).

        Args:
            token_ids: The token prefix being actively used.
            leaf_node: Optional pre-resolved leaf node.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        node = leaf_node
        if node is None:
            node = self._find_leaf_node(token_ids)

        if node is not None:
            self._swa_manager.swa_lock(node)

    def unlock_swa(
        self,
        token_ids: Union['torch.Tensor', np.ndarray],
        leaf_node: Optional['RadixNode'] = None,
    ) -> None:
        """Unlock SWA data after request finishes.

        Args:
            token_ids: The token prefix that was being used.
            leaf_node: Optional pre-resolved leaf node.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        node = leaf_node
        if node is None:
            node = self._find_leaf_node(token_ids)

        if node is not None:
            self._swa_manager.swa_unlock(node)

    @property
    def stats(self) -> dict:
        """Return SWA statistics."""
        return self._swa_manager.stats

    # --- Internal helpers -----------------------------------------------------

    def _find_leaf_node(self, token_ids: np.ndarray) -> Optional['RadixNode']:
        """Find the deepest matching node in the radix tree for given token_ids.

        Uses the radix tree's match_prefix to locate the last matched node.
        Returns None if no match found (empty tree or no prefix match).
        """
        try:
            from flexkv.common.block import SequenceMeta
            seq = SequenceMeta(
                token_ids=token_ids,
                tokens_per_block=self._radix_tree.tokens_per_block,
            )
            seq.gen_hashes()
        except (ImportError, TypeError, AttributeError):
            # Fallback: build a minimal sequence-like object with hashlib
            import hashlib as _hl
            tpb = self._radix_tree.tokens_per_block
            num_blocks = len(token_ids) // tpb
            block_hashes = np.zeros(num_blocks, dtype=np.int64)
            for i in range(num_blocks):
                blk = token_ids[i * tpb: (i + 1) * tpb]
                h = _hl.sha256(blk.tobytes()).hexdigest()
                block_hashes[i] = int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF

            from flexkv.common.hash_utils import HashType

            class _FakeSeq:
                def __init__(self, hashes, tpb_):
                    self.block_hashes = hashes
                    self.num_blocks = len(hashes)
                    self.tokens_per_block = tpb_
                def gen_hashes(self):
                    pass
                def get_hash(self, idx):
                    if idx < len(self.block_hashes):
                        return HashType(int(self.block_hashes[idx]))
                    return HashType(0)

            seq = _FakeSeq(block_hashes, tpb)

        match_result = self._radix_tree.match_prefix(seq, update_cache_info=False)

        if match_result.is_empty():
            return None

        # Return the last matched node (deepest in the tree for this prefix)
        return match_result.last_node

    def _build_path_to_node(self, node: 'RadixNode') -> List['RadixNode']:
        """Build path from root to the given node (inclusive).

        Returns list ordered [root, ..., parent, node].
        """
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path
