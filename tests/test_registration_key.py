import sys
import types

from flexkv.common.config import ModelConfig, RankInfo
from flexkv.server.request import RegisterTPClientRequest


# Registration tests do not need the CUDA/liburing transfer workers.
transfer_engine_module = types.ModuleType("flexkv.transfer.transfer_engine")
transfer_engine_module.TransferEngine = object
sys.modules["flexkv.transfer.transfer_engine"] = transfer_engine_module

from flexkv.transfer_manager import TransferManager


def _request(dp_client_id: int, intra_client_id: int, device_id: int):
    return RegisterTPClientRequest(
        dp_client_id=dp_client_id,
        pp_rank=0,
        intra_client_id=intra_client_id,
        device_id=device_id,
        handles=[],
        gpu_layout=object(),
    )


def _empty_transfer_manager() -> TransferManager:
    manager = TransferManager.__new__(TransferManager)
    manager.all_gpu_layouts = {}
    manager.all_gpu_blocks = {}
    manager.gpu_worker_key_mapping = {}
    manager.gpu_device_id_mapping = {}
    manager.all_gpu_layouts_per_group = {}
    manager.all_gpu_blocks_per_group = {}
    manager.all_swa_gpu_blocks = {}
    manager.all_swa_gpu_layouts = {}
    manager.all_swa_gpu_layouts_per_group = {}
    manager.all_swa_gpu_blocks_per_group = {}
    manager.swa_layer_groups = None
    manager.model_config = ModelConfig()
    return manager


def test_intra_client_id_flattens_pp_and_effective_tp_rank():
    model_config = ModelConfig(tp_size=4, pp_size=2, attn_cp_size=2)
    rank_info = RankInfo(
        model_config=model_config,
        pp_rank=1,
        tp_rank=3,
        attn_cp_rank=1,
    )

    assert model_config.effective_tp_size == 4
    assert rank_info.effective_tp_rank == 3
    assert rank_info.intra_client_id == 7


def test_registration_key_distinguishes_replicas_from_cuda_device_id():
    manager = _empty_transfer_manager()
    first = _request(dp_client_id=0, intra_client_id=0, device_id=0)
    second = _request(dp_client_id=1, intra_client_id=0, device_id=0)

    manager._handle_gpu_blocks_registration(first)
    manager._handle_gpu_blocks_registration(second)

    assert set(manager.all_gpu_blocks) == {(0, 0), (1, 0)}
    assert manager.gpu_device_id_mapping == {(0, 0): 0, (1, 0): 0}


def test_duplicate_registration_key_is_rejected():
    manager = _empty_transfer_manager()
    first = _request(dp_client_id=2, intra_client_id=3, device_id=4)
    duplicate = _request(dp_client_id=2, intra_client_id=3, device_id=5)

    manager._handle_gpu_blocks_registration(first)
    manager._handle_gpu_blocks_registration(duplicate)

    assert len(manager.all_gpu_blocks) == 1
    assert manager.gpu_device_id_mapping[(2, 3)] == 4
