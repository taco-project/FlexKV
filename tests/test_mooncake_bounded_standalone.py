"""Run original upstream #307 test bodies, replacing import plumbing only."""
import ast
from pathlib import Path
from types import SimpleNamespace
from threading import Lock
import pytest
from mooncake_test_support import MooncakeStoreClient, MooncakeStoreConfig

path = Path(__file__).with_name('test_mooncake_bounded_io.py')
nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)]
tree = ast.Module(body=[ast.ImportFrom(module='__future__',
    names=[ast.alias(name='annotations')], level=0)] + nodes, type_ignores=[])
exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), globals())

if __name__ == '__main__':
    raise SystemExit(pytest.main(['--noconftest', '-q', __file__]))
