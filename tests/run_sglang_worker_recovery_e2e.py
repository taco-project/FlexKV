"""Opt-in DeepSeek V4.1 TP4 HTTP recovery test in an existing SGLang environment.

Starts only its own SGLang process tree. It gracefully terminates only the
owned H2D worker, then asserts HTTP correctness, bypass and host-cache recovery.
Example (inside the existing GPU container):
  python tests/run_sglang_worker_recovery_e2e.py --output /data/cache/sglang/te-recovery-run
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request

import psutil


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="/data/DeepSeek-v4.1-flash")
    parser.add_argument("--port", type=int, default=32229)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    config = out / "flexkv.yaml"
    config.write_text("cpu_cache_gb: 16\nswa:\n  num_cpu_slots: 512\n")
    base = f"http://127.0.0.1:{args.port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLEXKV_")}
    env.update(PYTHONPATH=f"{repo}:/data/cache/sglang/sglang/python", FLEXKV_SERVER_CLIENT_MODE="1",
        FLEXKV_SERVER_LAUNCH_MODE="embedded", FLEXKV_SERVER_RECV_PORT=f"ipc:///tmp/flexkv-resilience-{args.port}",
        FLEXKV_INSTANCE_NUM="1", FLEXKV_INSTANCE_ID="0", FLEXKV_ENABLE_METRICS="0", FLEXKV_READY_TIMEOUT_S="600",
        FLEXKV_ENABLE_MPS="0", FLEXKV_SGLANG_RECOVERY_INTERVAL_S="5",
        SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="1", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
        "--host", "127.0.0.1", "--port", str(args.port), "--tp", "4", "--ep", "4", "--base-gpu-id", "0",
        "--context-length", "8192", "--max-total-tokens", "8192", "--max-running-requests", "2",
        "--chunked-prefill-size", "2048", "--mem-fraction-static", "0.98", "--swa-full-tokens-ratio", "0.3",
        "--moe-runner-backend", "marlin", "--fp4-gemm-backend", "marlin", "--fp8-gemm-backend", "triton",
        "--enable-flexkv", "--enable-cache-report", "--flexkv-config-file", str(config),
        "--served-model-name", "DeepSeek-V4.1-Flash-NVFP4", "--reasoning-parser", "deepseek-v41",
        "--tool-call-parser", "deepseekv41", "--disable-cuda-graph", "--skip-server-warmup",
        "--watchdog-timeout", "300", "--random-seed", "20261107"]
    source_manifest = {"base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
                       "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()}
    files = subprocess.check_output(["git", "diff", "--name-only"], cwd=repo, text=True).splitlines()
    files += subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines()
    source_manifest["files"] = {path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
                                for path in sorted(set(files)) if (repo / path).is_file()}
    (out / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2))
    rows = []
    summary = {"command": command, "injection": "SIGTERM to owned H2D worker; explicit drain acknowledgement",
               "passed": False}
    log_path = out / "server.log"
    log = log_path.open("w")
    root = subprocess.Popen(command, env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    owned = {}

    def record(event, **fields):
        item = {"time": time.time(), "event": event, **fields}
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item), flush=True)

    def snapshot():
        try:
            for process in psutil.Process(root.pid).children(recursive=True):
                owned[process.pid] = process.create_time()
        except psutil.Error:
            pass

    def server_log():
        return log_path.read_text(errors="replace")

    def request(index, phase):
        text = f"Recovery document {index}.\n" + (
            f"Record {index}: The lake is blue, the field is green, and the tree is tall.\n" * 72
        ) + "\nReturn only the number 739220. Do not explain."
        body = {"model": "DeepSeek-V4.1-Flash-NVFP4", "messages": [{"role": "user", "content": text}],
                "temperature": 0, "max_tokens": 16, "stream": False,
                "chat_template_kwargs": {"thinking": False}}
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=90) as response:
            value = json.loads(response.read())
            assert response.status == 200
        answer = value["choices"][0]["message"].get("content", "").strip()
        assert answer == "739220", value
        item = {"index": index, "phase": phase, "latency_s": time.monotonic() - started, "response": value}
        with (out / "requests.jsonl").open("a") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        record("request", phase=phase, index=index, latency_s=item["latency_s"])
        return item

    try:
        record("launch", pid=root.pid)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            snapshot()
            if root.poll() is not None:
                raise RuntimeError(f"SGLang exited during startup: {root.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise TimeoutError("SGLang did not become ready")
        record("ready")
        for index in range(12):
            rows.append(request(index, "cold"))
        for index in range(4):
            rows.append(request(index, "host_restore_before_fault"))
        before = server_log()
        assert re.search(r"operation=load .*status=success", before), "No baseline host H2D observed"
        matches = re.findall(r"Transfer worker created worker_id=0 pid=(\d+)", before)
        assert len(matches) == 1, matches
        victim_pid = int(matches[0])
        snapshot()
        assert victim_pid in owned, "Refuse to signal a process outside this test's tree"
        victim = psutil.Process(victim_pid)
        assert victim.create_time() == owned[victim_pid]
        victim.send_signal(signal.SIGTERM)
        record("fault_injected", worker_id=0, pid=victim_pid)
        # This prompt was evicted from GPU but remains in the CPU tree.
        rows.append(request(4, "fault_load_and_bypass"))
        rows.append(request(100, "new_request_during_bypass"))
        after_fault = server_log()
        assert "FlexKV bypass entered" in after_fault, "The failure did not activate bypass"
        assert max(row["latency_s"] for row in rows[-2:]) < 30, "Fault still costs a full load timeout"
        time.sleep(6)
        for index in (5, 6, 7, 8):
            rows.append(request(index, "recovered_host_restore"))
        recovered = server_log()
        assert "FlexKV GPU route recovered generation=1" in recovered, "No collective recovery observed"
        recovery_offset = recovered.index("FlexKV GPU route recovered generation=1")
        assert re.search(r"operation=load .*status=success", recovered[recovery_offset:]), "No H2D after recovery"
        with ThreadPoolExecutor(max_workers=4) as pool:
            rows.extend(pool.map(lambda i: request(i % 12, "concurrent_after_recovery"), range(24)))
        assert root.poll() is None, "SGLang died after recovery"
        snapshot()
        summary.update(passed=True, requests=len(rows), max_latency_s=max(row["latency_s"] for row in rows),
                       worker_pid=victim_pid, root_alive_before_cleanup=True)
        record("passed", requests=len(rows))
    except Exception as exc:
        summary["error"] = repr(exc)
        record("failed", error=repr(exc))
        raise
    finally:
        snapshot()
        if root.poll() is None:
            root.terminate()
            try:
                root.wait(timeout=45)
            except subprocess.TimeoutExpired:
                root.kill()
                root.wait(timeout=10)
        remaining = []
        for pid, created in owned.items():
            try:
                p = psutil.Process(pid)
                if p.create_time() == created and p.status() != psutil.STATUS_ZOMBIE:
                    p.terminate()
                    remaining.append(p)
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(remaining, timeout=15)
        for p in alive:
            p.kill()
        _, alive = psutil.wait_procs(alive, timeout=10)
        summary.update(root_exit=root.returncode, remaining_owned_processes=[p.pid for p in alive])
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        log.close()


if __name__ == "__main__":
    main()
