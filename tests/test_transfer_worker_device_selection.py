import pytest
import torch

from flexkv.transfer import worker


class RegistrationStopped(Exception):
    pass


def stop_at_first_host_registration(monkeypatch, events):
    monkeypatch.setattr(
        worker.torch.cuda,
        "set_device",
        lambda device: events.append(("set_device", device)),
    )

    def register(_tensor):
        events.append(("register", None))
        raise RegistrationStopped

    monkeypatch.setattr(worker, "cudaHostRegister", register)


def test_gpu_cpu_worker_selects_device_before_host_registration(monkeypatch):
    events = []
    stop_at_first_host_registration(monkeypatch, events)

    with pytest.raises(RegistrationStopped):
        worker.GPUCPUTransferWorker(
            worker_id=0,
            transfer_conn=None,
            finished_ops_queue=None,
            op_buffer_tensor=torch.empty(0),
            gpu_blocks=[],
            cpu_blocks=torch.empty(0),
            gpu_kv_layout=None,
            cpu_kv_layout=None,
            dtype=torch.bfloat16,
            gpu_device_id=3,
        )

    assert events == [("set_device", 3), ("register", None)]


def test_tp_worker_selects_registered_device_before_host_registration(monkeypatch):
    events = []
    stop_at_first_host_registration(monkeypatch, events)
    handle = type("Handle", (), {"device": torch.device("cuda:4")})()

    with pytest.raises(RegistrationStopped):
        worker.tpGPUCPUTransferWorker(
            worker_id=0,
            transfer_conn=None,
            finished_ops_queue=None,
            op_buffer_tensor=torch.empty(0),
            gpu_blocks=[[handle], [handle]],
            cpu_blocks=torch.empty(0),
            gpu_kv_layouts=[],
            cpu_kv_layout=None,
            dtype=torch.bfloat16,
            tp_group_size=2,
        )

    assert events == [
        ("set_device", 4),
        ("register", None),
    ]
