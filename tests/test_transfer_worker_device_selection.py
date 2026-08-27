"""The CUDA device must be bound before the first cudaHostRegister.

Registering pinned host memory attributes it to whatever device is current;
get that wrong and the pages are pinned against the wrong context, which
shows up much later as a slow (or failing) transfer on a device that never
registered them.  So these pin the *order*: set_device first, register second.

Both cases go through one worker now -- tp_group_size=1 is a TP group of one,
not a separate class -- so the interesting difference is only which device the
handles name.
"""
import pytest
import torch

from flexkv.transfer import worker
from flexkv.transfer.workers import runtime


class RegistrationStopped(Exception):
    pass


def stop_at_first_host_registration(monkeypatch, events):
    monkeypatch.setattr(
        torch.cuda,
        "set_device",
        lambda device: events.append(("set_device", device)),
    )

    def register(_tensor):
        events.append(("register", None))
        raise RegistrationStopped

    # Patch where the *name is resolved*, not where the worker is imported
    # from. Both calls this test intercepts live in ``TransferWorkerBase`` /
    # ``ensure_cuda_device`` in ``workers.runtime``; ``flexkv.transfer.worker``
    # is a re-export façade, so patching an attribute on it would rebind
    # nothing the worker actually looks up and the real cudaHostRegister would
    # run -- a green test that no longer tests anything.
    monkeypatch.setattr(runtime, "cudaHostRegister", register)


def _handle_on(device: str):
    return type("Handle", (), {"device": torch.device(device)})()


@pytest.mark.parametrize(
    "gpu_blocks, tp_group_size, expected_device",
    [
        ([[_handle_on("cuda:3")]], 1, 3),
        ([[_handle_on("cuda:4")], [_handle_on("cuda:4")]], 2, 4),
    ],
    ids=["tp1", "tp2"],
)
def test_worker_selects_device_before_host_registration(
    monkeypatch, gpu_blocks, tp_group_size, expected_device
):
    events = []
    stop_at_first_host_registration(monkeypatch, events)

    with pytest.raises(RegistrationStopped):
        worker.GPUCPUTransferWorker(
            worker_id=0,
            transfer_conn=None,
            finished_ops_queue=None,
            op_buffer_tensor=torch.empty(0),
            gpu_blocks=gpu_blocks,
            cpu_blocks=torch.empty(0),
            gpu_kv_layouts=[],
            cpu_kv_layout=None,
            dtype=torch.bfloat16,
            tp_group_size=tp_group_size,
        )

    assert events == [("set_device", expected_device), ("register", None)]


def test_there_is_no_separate_tp_worker():
    """tp==1 is num_gpus==1 inside TPTransferThreadGroup, not another class.

    If this name comes back, so do the two _transfer_impl / two
    _control_suspend_gpu / two nvcomp strategies it used to carry -- and the
    bug where only one of the pair got a fix.
    """
    import flexkv.transfer.workers.gpu_cpu as gpu_cpu

    assert not hasattr(worker, "tpGPUCPUTransferWorker")
    # The façade only re-exports what it is told to, so ask the module that
    # would actually have to define the class.
    assert not hasattr(gpu_cpu, "tpGPUCPUTransferWorker")
