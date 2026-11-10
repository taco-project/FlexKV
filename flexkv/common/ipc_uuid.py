"""Portable torch IPC device identity without global monkey patches."""

import inspect

import torch


def device_uuid(device) -> str:
    value = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if value is None or not str(value).strip():
        raise RuntimeError("CUDA backend does not expose a device UUID for IPC")
    return str(value)


def device_from_uuid(uuid: str) -> torch.device:
    for index in range(torch.cuda.device_count()):
        if device_uuid(index) == uuid:
            return torch.device("cuda", index)
    raise RuntimeError(f"IPC owner device UUID is not visible in this process: {uuid}")


def rebuild_by_uuid(rebuild_func, args, device_index, uuid):
    device = device_from_uuid(uuid)
    rebuilt_args = list(args)
    if device_index is not None:
        rebuilt_args[device_index] = device.index
    with torch.cuda.device(device):
        return rebuild_func(*rebuilt_args)


def wrap_rebuild(rebuild_func, args, device):
    """Preserve external UUID wrappers; otherwise locate torch's device argument."""
    uuid = device_uuid(device)
    parameters = list(inspect.signature(rebuild_func).parameters)
    if "storage_device" in parameters:
        index = parameters.index("storage_device")
        if index >= len(args):
            raise RuntimeError("Unsupported torch CUDA IPC rebuild arguments")
        if isinstance(args[index], str):
            if args[index] != uuid:
                raise RuntimeError("IPC rebuild UUID differs from the owner device")
            index = None
        elif not isinstance(args[index], int):
            raise RuntimeError("Unsupported torch CUDA IPC storage_device type")
    elif len(args) > 6 and isinstance(args[6], str) and args[6] == uuid:
        index = None
    else:
        raise RuntimeError(
            "Unsupported torch CUDA IPC rebuild signature; cannot map device UUID safely")
    return rebuild_by_uuid, (rebuild_func, tuple(args), index, uuid)
