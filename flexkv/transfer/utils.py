from collections import defaultdict
from typing import Tuple, List, Dict
import torch
import numpy as np

def group_blocks_by_node_and_type(
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
    remote_block_node_ids: List[int],
) -> Dict[int, Dict[str, List[int]]]:
    '''
    group the blocks by remote node id and remote block source type
    Parameters:
        src_block_ids (torch.Tensor): the source block ids on remote node
        dst_block_ids (torch.Tensor): the destination block ids on local node
        remote_block_node_ids (List[int]): the remote node ids for each block
        remote_block_src_types (List[int]): the remote block source types for each block
    Returns:
        Dict[Tuple[int, int], Dict[str, List[int]]]: the grouped blocks
        1st key: (node_id, src_type)
        2nd key: "src" or "dst"
        value: List of block ids
    '''
    groups = defaultdict(lambda: {"src": [], "dst": []})

    for src, dst, node_id in zip(
        src_block_ids, dst_block_ids, remote_block_node_ids
    ):
        groups[node_id]["src"].append(src)
        groups[node_id]["dst"].append(dst)

    return dict(groups)

def group_blocks_by_node_and_segment(
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
    remote_block_node_ids: List[int],
) -> Dict[int, List[Dict[str, List[int]]]]:
    '''
    Group by node_id and divide blocks with consecutive source/dst into subsegments.
    Parameters:
        src_block_ids (torch.Tensor): source block ids
        dst_block_ids (torch.Tensor): target block ids
        remote_block_node_ids (List[int]): the remote node ids for each block
    Returns:
        Dict[node_id, List[Dict[str, List[int]]]]:
            {
                node_id: [
                    {"src": [...], "dst": [...]},
                    ...
                ]
            }
    '''
    groups = defaultdict(list)
    tmp = defaultdict(lambda: {"src": [], "dst": []})
    for src, dst, node_id in zip(src_block_ids.tolist(), dst_block_ids.tolist(), remote_block_node_ids):
        tmp[node_id]["src"].append(src)
        tmp[node_id]["dst"].append(dst)

    for node_id, pair in tmp.items():
        sorted_pairs = sorted(zip(pair["src"], pair["dst"]), key=lambda x: (x[0], x[1]))

        current_src_segment = []
        current_dst_segment = []

        last_src = None
        last_dst = None

        for src, dst in sorted_pairs:
            if last_src is not None and last_dst is not None:
                if src == last_src + 1 and dst == last_dst + 1:
                    # src and dst are continuous
                    current_src_segment.append(src)
                    current_dst_segment.append(dst)
                else:
                    # Disconnect and save the current segment
                    groups[node_id].append({"src": current_src_segment, "dst": current_dst_segment})
                    current_src_segment = [src]
                    current_dst_segment = [dst]
            else:
                current_src_segment.append(src)
                current_dst_segment.append(dst)

            last_src = src
            last_dst = dst

        # Save the last segment
        if current_src_segment:
            groups[node_id].append({"src": current_src_segment, "dst": current_dst_segment})

    return dict(groups)

class RemoteSSD2HMetaInfo:
    task_id: int
    cpu_block_ids: np.ndarray
    ssd_block_ids: np.ndarray
    peer_engine_addr: str # the mooncake engine address of the node that initiates the transfer, used for write data back to the node.
    peer_cpu_base_ptr: int # the cpu buffer base ptr of the peer node, used for calculating the dst ptrs.
    data_size: int
    layer_id: int
    layer_granularity: int
    
    def __init__(self, task_id, cpu_block_ids, ssd_block_ids, peer_engine_addr, peer_cpu_base_ptr, data_size, layer_id, layer_granularity):
        self.task_id = task_id
        self.cpu_block_ids = cpu_block_ids
        self.ssd_block_ids = ssd_block_ids
        self.peer_engine_addr = peer_engine_addr
        self.peer_cpu_base_ptr = peer_cpu_base_ptr
        self.data_size = data_size
        self.layer_id = layer_id
        self.layer_granularity = layer_granularity
    
    def from_dict(self, data) -> "RemoteSSD2HMetaInfo":
        return RemoteSSD2HMetaInfo(
            task_id = data.get("task_id"),
            cpu_block_ids=data.get("cpu_block_ids"),
            ssd_block_ids=data.get("ssd_block_ids"),
            peer_engine_addr=data.get("peer_engine_addr"),
            peer_cpu_base_ptr = data.get("peer_cpu_base_ptr"),
            data_size=data.get("data_size"),
            layer_id = data.get("layer_id"),
            layer_granularity = data.get("layer_granularity")
        )
        