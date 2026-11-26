"""CPU-only contracts for the SGLang standard KV registration path."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def connector():
    tree = ast.parse((ROOT / "flexkv/integration/sglang/connector.py").read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "FlexKVConnector")
    names = {"_standard_kv_geometry", "_build_indexer_layer_group_specs",
             "_register_standard_to_server"}
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for method in methods:
        method.returns = None
        for node in ast.walk(method):
            if isinstance(node, ast.arg):
                node.annotation = None
    namespace = dict(KVCacheLayout=SimpleNamespace, LayerGroupSpec=SimpleNamespace,
                     KVCacheLayoutType=SimpleNamespace(LAYERFIRST="layerfirst"), logger=Mock())
    exec(compile(ast.Module(body=methods, type_ignores=[]), "connector.py", "exec"), namespace)
    instance = SimpleNamespace(page_size=64, _label="test", tp_client=Mock(),
                               model_config=SimpleNamespace(kv_dim=2, num_kv_heads=16),
                               rank_info=SimpleNamespace(num_layers_per_pp_stage=2))
    for name in names:
        setattr(instance, name, MethodType(namespace[name], instance))
    return instance


def tensor(shape, contiguous=True):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype="float16",
                           is_contiguous=lambda: contiguous)


@pytest.mark.parametrize("shape", [(320, 2, 8), (5, 2, 64, 8)])
@pytest.mark.parametrize("with_indexer", [False, True])
def test_standard_registration_counts_pages_once(shape, with_indexer):
    engine = connector()
    buffers = [tensor(shape), tensor(shape)]
    indexer = SimpleNamespace(logical_layer_count=2, layer_indices=[0, 1],
                              buffers=[tensor((5, 192)), tensor((5, 192))]) if with_indexer else None
    engine._register_standard_to_server(buffers, indexer)
    kwargs = engine.tp_client.register_to_server.call_args.kwargs
    layout = kwargs["kv_layout"]
    assert (layout.num_block, layout.tokens_per_block, layout.num_head, layout.head_size) == (5, 64, 2, 8)
    assert layout.num_kv_heads == 16
    assert layout.kv_dim == 2
    if with_indexer:
        assert kwargs["gpu_layouts"][1].num_block == 5
        assert kwargs["handles_per_group"] == [buffers, indexer.buffers]
        main_group, index_group = kwargs["layer_groups"]
        assert (main_group.num_kv_heads, main_group.head_size) == (2, 8)
        assert (index_group.head_size, index_group.compress_ratio) == (192, 64)
    else:
        assert kwargs["kv_caches"] is buffers


@pytest.mark.parametrize("buffers,error", [
    ([tensor((5, 2, 32, 8))], "page axis"),
    ([tensor((5, 2, 64, 8), False)], "contiguous"),
    ([tensor((5, 2, 64, 8)), tensor((6, 2, 64, 8))], "identical shapes"),
    ([tensor((5, 8))], "3D token-major or 4D"),
    ([tensor((1, 5, 2, 64, 8))], "3D token-major or 4D"),
])
def test_invalid_geometry_fails_before_gpu_registration(buffers, error):
    engine = connector()
    with pytest.raises(ValueError, match=error):
        engine._register_standard_to_server(buffers)
    engine.tp_client.register_to_server.assert_not_called()


def test_three_dimensional_partial_page_behavior_is_unchanged():
    engine = connector()
    assert engine._standard_kv_geometry([tensor((321, 2, 8))]) == (5, 2, 8)
