"""Manual check: shareable HugePage pool survives spawn and is reclaimed on SIGKILL.

Not collected by pytest (leading underscore). Run inside a container with a
2 MiB HugePage pool:  python tests/hugepage/_e2e_sigkill_check.py
"""
import multiprocessing as mp
import os
import signal
import sys
import time

import torch

PAGE = 2 * 1024 * 1024
NPAGES = 64


def free_pages() -> int:
    """Pages neither faulted in nor reserved by an existing mapping."""
    base = "/sys/kernel/mm/hugepages/hugepages-2048kB/"
    with open(base + "free_hugepages") as f, open(base + "resv_hugepages") as r:
        return int(f.read()) - int(r.read())


def worker(handle, ready, done):
    from flexkv.storage.allocator import materialize_worker_tensor
    t = materialize_worker_tensor(handle)
    assert int(t[0].item()) == 0x11 and int(t[-1].item()) == 0x22, "worker sees other pages"
    t[1] = 0x33
    ready.set()
    done.wait()


def owner(q):
    from flexkv.storage.allocator import alloc_hugepage_tensor, get_worker_hugepage_handle
    ctx = mp.get_context("spawn")
    t = alloc_hugepage_tensor(NPAGES * PAGE, torch.uint8, PAGE, shareable=True)
    t[0], t[-1] = 0x11, 0x22
    handle = get_worker_hugepage_handle(t, t.numel(), t.dtype)
    ready, done = ctx.Event(), ctx.Event()
    p = ctx.Process(target=worker, args=(handle, ready, done), daemon=True)
    p.start()
    assert ready.wait(120), "worker did not attach"
    assert int(t[1].item()) == 0x33, "owner does not see worker write"
    q.put((os.getpid(), p.pid, handle.path, os.readlink(handle.path)))
    time.sleep(3600)


def main():
    from flexkv.storage.allocator import alloc_hugepage_tensor  # noqa: F401  (import check)
    before = free_pages()
    q = mp.get_context("spawn").Queue()
    o = mp.get_context("spawn").Process(target=owner, args=(q,))
    o.start()
    owner_pid, worker_pid, path, target = q.get(timeout=180)
    during = free_pages()
    hp_files = os.listdir("/dev/hugepages") if os.path.isdir("/dev/hugepages") else None
    print(f"owner={owner_pid} worker={worker_pid} path={path} -> {target}")
    print(f"/dev/hugepages: {'not mounted' if not os.path.ismount('/dev/hugepages') else hp_files}")

    os.kill(owner_pid, signal.SIGKILL)
    o.join()
    time.sleep(0.5)
    worker_alive = os.path.exists(f"/proc/{worker_pid}")
    after_owner = free_pages()
    if worker_alive:
        os.kill(worker_pid, signal.SIGKILL)
    deadline = time.time() + 10
    while os.path.exists(f"/proc/{worker_pid}") and time.time() < deadline:
        time.sleep(0.1)
    time.sleep(0.5)
    after_all = free_pages()

    print(f"free: before={before} running={during} owner_killed={after_owner} "
          f"(worker_alive={worker_alive}) all_killed={after_all}")
    assert before - during >= NPAGES, "pool not on HugePages"
    assert target.startswith("/memfd:flexkv_hugepage")
    assert after_all == before, "HugePages leaked after SIGKILL"
    print("E2E OK")


if __name__ == "__main__":
    sys.exit(main())
