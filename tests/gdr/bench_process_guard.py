"""Linux batch supervisor: clean only descendants carrying a recorded run token.

No process-name matching and no process-group-wide kill. pidfds prevent PID reuse
races. An advisory lock prevents one live batch from cleaning another live batch.
"""
import argparse
import ctypes
import platform
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import stat
import subprocess
import sys
import time

TOKEN_KEY = "FLEXKV_BENCH_RUN_TOKEN"
PARENT_KEY = "FLEXKV_BENCH_SUPERVISOR_PID"


def _pidfd_syscall(number, *args):
    # Linux v6.1 syscall_64.tbl and asm-generic/unistd.h: open=434, send=424.
    # Restrict the fallback to known 64-bit ABIs; never guess on other targets.
    if (not sys.platform.startswith("linux") or ctypes.sizeof(ctypes.c_void_p) != 8
            or platform.machine().lower() not in ("x86_64", "aarch64")):
        raise RuntimeError("pidfd fallback requires Linux x86_64/aarch64; use Python with native pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, sig):
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        return native(fd, sig)
    return _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig),
                          ctypes.c_void_p(), ctypes.c_uint(0))


def check_pidfd_support():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Safe process cleanup requires Linux")
    try:
        probe = pidfd_open(os.getpid())
        try:
            pidfd_send_signal(probe, 0)  # Permission/capability probe, no signal delivered.
        finally:
            os.close(probe)
    except OSError as exc:
        raise RuntimeError(f"pidfd kernel/container probe failed: errno={exc.errno} ({exc.strerror}); "
                           "check host kernel pidfd support and container seccomp policy; no benchmark started") from exc


def matches(pid, token, proc=Path("/proc")):
    """Exact initial environment marker + current user, never command substrings."""
    try:
        directory = proc / str(pid)
        if directory.stat().st_uid != os.getuid():
            return False
        expected = (TOKEN_KEY + "=" + token).encode()
        return expected in (directory / "environ").read_bytes().split(b"\0")
    except (FileNotFoundError, ProcessLookupError):
        return False


def tagged_pids(token):
    for item in Path("/proc").iterdir():
        if item.name.isdigit() and int(item.name) != os.getpid():
            try:
                if matches(int(item.name), token):
                    yield int(item.name)
            except PermissionError:
                # Unreadable processes are never targets. Their identity cannot
                # be established; do not broaden selection to names or groups.
                continue


def send_checked(pid, token, sig):
    try:
        fd = pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        # Open the pidfd first, then verify the marker. If PID is recycled while
        # checking /proc, this descriptor still refers only to the original task.
        if matches(pid, token):
            print(f"[bench-guard] {signal.Signals(sig).name} owned pid={pid}", flush=True)
            pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


def cleanup(token):
    # Repeat enumeration to include children created during shutdown. Give each
    # wave TERM before eventually using KILL; only exact token matches qualify.
    signaled = set()
    for sig, timeout in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 3.0)):
        deadline = time.monotonic() + timeout
        while True:
            pids = list(tagged_pids(token))
            if not pids:
                return
            for pid in pids:
                key = (pid, sig)
                if key not in signaled:
                    send_checked(pid, token, sig)
                    signaled.add(key)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    if list(tagged_pids(token)):
        raise RuntimeError("Owned processes remain; record retained; refusing a new batch")


def state_directory(script, root):
    digest = hashlib.sha256((str(script) + "\0" + str(root)).encode()).hexdigest()[:24]
    path = Path("/tmp") / f"flexkv-bench-{os.getuid()}-{digest}"
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError(f"Unsafe state directory: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    check_pidfd_support()
    script, root = Path(args.script).resolve(), Path(args.root).resolve()
    directory = state_directory(script, root)
    lock = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This script/root already has an active supervised batch; nothing was killed")
        record = directory / "run.json"
        if record.exists():
            if record.is_symlink():
                raise RuntimeError("Refusing symlink run record")
            previous = json.loads(record.read_text())
            token = previous["token"]
            if (previous.get("script") != str(script) or previous.get("root") != str(root)
                    or not isinstance(token, str) or len(token) != 64
                    or any(c not in "0123456789abcdef" for c in token)):
                raise RuntimeError("Invalid run record; refusing cleanup")
            print("[bench-guard] checking recorded previous run for leftovers", flush=True)
            cleanup(token)
        token = secrets.token_hex(32)
        temporary = directory / "run.new"
        # Directory is private to this user. Replacing the record is atomic.
        temporary.write_text(json.dumps(dict(token=token, script=str(script), root=str(root))))
        temporary.replace(record)
        env = dict(os.environ, **{TOKEN_KEY: token, PARENT_KEY: str(os.getpid())})
        child = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupted)
        try:
            child = subprocess.Popen(["bash", str(script)], env=env, start_new_session=True)
            print(f"[bench-guard] supervised batch pid={child.pid}; state={directory}", flush=True)
            result = child.wait()
        except KeyboardInterrupt:
            result = 130
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            cleanup(token)
            if child is not None:
                child.wait(timeout=3)
            record.unlink()
        return result if result >= 0 else 128 - result
    finally:
        os.close(lock)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[bench-guard] {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
