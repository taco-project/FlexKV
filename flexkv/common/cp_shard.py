"""Node-local byte layouts for physically sharded CP cache layers.

Framework identities stay global until GPU registrations have arrived.  The
transfer manager then localizes the model and its corresponding handle tables.
"""

import copy
from dataclasses import replace


def cp_node_cache_model(model, node_rank):
    if not (getattr(model, "layer_shard_enabled", False) and model.nnodes > 1):
        return model, None
    if getattr(model, "cp_cache_node_rank", None) is not None:
        raise ValueError("CP cache model has already been localized")
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

    if (model.pp_size != 1 or model.dp_size != 1 or model.tp_size != 1
            or model.instance_num != 1 or model.num_kv_heads != 1
            or model.kv_dim != 1
            or GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode == "all_write"
            or model.cp_size < model.nnodes or model.cp_size % model.nnodes
            or type(node_rank) is not int or not 0 <= node_rank < model.nnodes
            or not model.layer_groups
            or len(model.h2d_layer_holders or ()) != model.num_layers):
        raise ValueError(
            "Cross-node physical CP requires PP=DP=attention-TP=1 and complete groups/holders"
        )
    model._validate_layer_groups()
    per_node = model.cp_size // model.nnodes
    first = node_rank * per_node
    end = first + per_node
    owners = model.h2d_layer_holders
    if any(type(owner) is not int or not 0 <= owner < model.cp_size for owner in owners):
        raise ValueError("Global CP holder outside the framework group")
    selections = []
    groups = []
    for group in model.layer_groups:
        selected = [i for i, original in enumerate(group.layer_indices)
                    if first <= owners[original] < end]
        if not selected:
            raise ValueError("Cross-node CP group has no local layers; unsupported layout")
        selections.append(selected)
        groups.append(replace(
            group, num_layers=len(selected),
            layer_indices=[group.layer_indices[i] for i in selected],
        ))
    local = replace(
        model, layer_groups=groups, cp_cache_node_rank=node_rank,
        h2d_layer_holders=[owner - first if first <= owner < end else -1
                           for owner in owners],
    )
    local.freeze()
    return local, selections


def slice_cp_gpu_groups(blocks, layouts, selections):
    if selections is None:
        return blocks, layouts
    if len(blocks) != len(selections) or len(layouts) != len(selections):
        raise ValueError("CP GPU registration group count differs from local layout")
    selected_blocks = []
    selected_layouts = []
    for group, layout, indices in zip(blocks, layouts, selections):
        if len(group) != layout.num_layer:
            raise ValueError("Physical CP requires separate GPU handles per layer")
        if (not indices or any(type(i) is not int for i in indices)
                or indices != sorted(set(indices))
                or min(indices) < 0 or max(indices) >= len(group)):
            raise ValueError("CP layer selection is outside the GPU handle array")
        selected_blocks.append([group[i] for i in indices])
        local = copy.copy(layout)
        local.num_layer = len(indices)
        local._kv_shape = None
        local._compute_kv_shape()
        selected_layouts.append(local)
    return selected_blocks, selected_layouts


def slice_cp_storage_handle(handle, blocks, layouts, selections):
    selected, selected_layouts = slice_cp_gpu_groups(blocks, layouts, selections)
    flat = [tensor for group in selected for tensor in group]
    local_handle = replace(handle, data=flat, kv_layout=selected_layouts[0])
    return local_handle, selected, selected_layouts
