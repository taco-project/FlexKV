"""Release-build indexing contracts; CPU tests do not build the Cython wheel."""
import ast
import itertools
import contextlib
import io
import unittest

import numpy as np
from submission_test_support import ROOT, method, tr


class CythonIndexingTests(unittest.TestCase):
    def test_no_constant_negative_subscripts_in_repaired_modules(self):
        for name in ('common/transfer.py', 'integration/config.py', 'kvtask.py', 'transfer/worker.py'):
            tree = ast.parse((ROOT/'flexkv'/name).read_text())
            unsafe = [node.lineno for node in ast.walk(tree)
                if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.UnaryOp)
                and isinstance(node.slice.op, ast.USub)
                and isinstance(node.slice.operand, ast.Constant)]
            self.assertEqual(unsafe, [], name)

    def test_long_block_debug_format_preserves_tail(self):
        graph = tr.TransferOpGraph()
        graph.add_transfer_op(tr.TransferOp(graph_id=graph.graph_id,
            transfer_type=tr.TransferType.H2D, src_block_ids=np.arange(20),
            dst_block_ids=np.arange(100, 120)))
        formatted = graph.visualize()
        self.assertIn('...19]', formatted)
        self.assertIn('...119]', formatted)

    def test_list_prefix_bound_is_nonnegative_and_preserves_original_slice(self):
        for size in range(33):
            first = list(range(size))
            self.assertEqual(first[:max(0, len(first)-1)], first[:-1])
        tree = ast.parse((ROOT/'flexkv/common/transfer.py').read_text())
        slices = [node for node in ast.walk(tree) if isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name) and node.value.id == 'first'
            and isinstance(node.slice, ast.Slice)]
        self.assertTrue(slices)
        self.assertFalse(any(isinstance(node.slice.upper, ast.UnaryOp)
            and isinstance(node.slice.upper.op, ast.USub) for node in slices))

    def test_debug_format_keeps_original_prefix_at_length_boundaries(self):
        for count in (0,1,7,8,9,20,257):
            graph = tr.TransferOpGraph()
            graph.add_transfer_op(tr.TransferOp(graph_id=graph.graph_id,
                transfer_type=tr.TransferType.H2D, src_block_ids=np.arange(count),
                dst_block_ids=np.arange(100,100+count)))
            with contextlib.redirect_stdout(io.StringIO()):
                text = graph.visualize()
            expected = str(list(range(count))) if count <= 8 else f'[0, 1, 2]...{count-1}] (n={count})'
            self.assertIn(expected, text)

    def test_cp_owner_ranges_cover_empty_sparse_and_repeated_owners(self):
        build = method(ROOT/'flexkv/transfer/worker.py', 'tpGPUCPUTransferWorker',
                       '_build_contiguous_owner_ranges', {})
        for owners in ([], [0], [0, 0, 1, 1, 1, 7], [0, 7, 0],
                       [i//10 for i in range(78)]):
            ranges = build(owners)
            reconstructed = []
            for start, count, owner in ranges:
                self.assertEqual(start, len(reconstructed))
                reconstructed.extend([owner]*count)
            self.assertEqual(reconstructed, owners)
            self.assertEqual(len(ranges), len(list(itertools.groupby(owners))))


if __name__ == '__main__':
    unittest.main()
