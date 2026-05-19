# FlexKV Two-Node Direct-Mode e2e — How to Run

This harness validates **distributed KV cache sharing** across two physical
hosts (146 ↔ 129) **without touching sglang** — it drives FlexKV directly
via ``KVManager`` + ``KVTPClient`` inside ``benchmark_dist_direct.py``.

## Why direct mode (not ``benchmark_dist_kvcache.py``)

* No ``KVServer`` subprocess → simpler lifecycle, no residual IPC socket at
  ``/tmp/flexkv_server`` to clean up.
* Uses the same ``get_match`` / ``put_async`` main path that §2.1 wires
  through to ``_sharing_domain_gate_get`` and ``_notify_sd_ready_on_put``,
  so any breakage here is a real breakage.
* GPU footprint is tiny (~40 MB / GPU) — safe to run alongside a live
  sglang serving that already owns most of the memory.

## Conflict isolation with the running sglang process

| Resource | sglang (GLM-5-FP8) | this benchmark |
|---|---|---|
| Mooncake engine TCP port | **5555** (sglang Transfer) | **5556 on 146** / **5557 on 129** |
| Redis logical DB | 0 (mooncake keys ``mooncake/*``) | 0 (mooncake) + **DB 1 for flexkv keys** |
| Redis key prefixes (DB 1) | — | ``sd:*``, ``instance:*``, ``node:*`` … |
| GPU VRAM | most of it | one GPU, ~40 MB |
| IPC sockets | ``/tmp/flexkv_server`` | **none** (direct mode) |

So the only shared resources are the physical Redis server (different DB)
and the RDMA NICs (different QP). Neither overlaps in state.

## Quick checklist before launching

1. **Redis reachable + password OK**:
   ```bash
   redis-cli -h 10.206.0.9 -p 6379 -a 123456 PING        # → PONG
   ```
2. **DB 1 is clean (first time only)**:
   ```bash
   redis-cli -h 10.206.0.9 -p 6379 -a 123456 -n 1 DBSIZE # → (integer) 0
   # If non-zero and you know it's our leftover state, wipe it:
   #   redis-cli -h 10.206.0.9 -p 6379 -a 123456 -n 1 FLUSHDB
   # Do NOT touch DB 0 — mooncake + sglang live there.
   ```
3. **Mooncake ports 5556 / 5557 free** on the respective hosts:
   ```bash
   ss -ltn | grep -E ':5556|:5557'                       # should be empty
   ```
4. **FlexKV built with FLEXKV_ENABLE_P2P=1** on both hosts:
   ```bash
   python3 -c 'from flexkv.cache.redis_meta import dist_available; print(dist_available())'
   # → True
   ```

## Run

### Step A — on host **146** (10.206.0.9), start **PUT-only**

```bash
cd /data1/phaedonsun/flexkv/FlexKV

export PYTHONPATH=/data1/phaedonsun/flexkv/FlexKV
export LD_LIBRARY_PATH=/data1/phaedonsun/flexkv/FlexKV/build/lib
export CUDA_VISIBLE_DEVICES=0          # pick any single free GPU

python3 benchmarks/dist_benchmark/benchmark_dist_direct.py \
    --config benchmarks/dist_benchmark/twonode_direct_146.yml \
    --mode put-only \
    --batch-size 1 --sequence-length 256 \
    --seed 42 \
    --rebuild-interval-ms 20
```

Expected final line before the process idles:
```
Data published to Redis. Press Enter to shutdown (keep running for other nodes to GET)...
```

### Step B — on host **129** (10.206.0.13), start **GET-only with same seed**

```bash
cd /data1/phaedonsun/flexkv/FlexKV

export PYTHONPATH=/data1/phaedonsun/flexkv/FlexKV
export LD_LIBRARY_PATH=/data1/phaedonsun/flexkv/FlexKV/build/lib
export CUDA_VISIBLE_DEVICES=0          # different physical GPU, same index is fine

python3 benchmarks/dist_benchmark/benchmark_dist_direct.py \
    --config benchmarks/dist_benchmark/twonode_direct_129.yml \
    --mode get-only \
    --batch-size 1 --sequence-length 256 \
    --seed 42 \
    --rebuild-interval-ms 20
```

## Success criteria

In the 129 log look for:

```
--- GET Phase ---
  GET: 256/256 tokens, data_size: 0.000 GB, cache_ratio: 100.00% ...
```

A non-zero ``cache_ratio`` means the 129 instance:
1. Found the 146 instance via the shared Redis (``instance:*`` discovery)
2. Resolved the 146 peer SD's ``node_id`` from the aggregate radix
3. Issued a Mooncake RDMA read against 146's mooncake engine @ 5556
4. Received KV data that matches byte-for-byte what 146 PUT

If ``cache_ratio: 0.00%``:
* Check the 129 log for ``[DistReuse]`` lines — the §2.1 gate ruled it out.
* Check ``KEYS sd:*`` in Redis DB 1 — the 146 side should have published
  ``sd:<…>:block:<nid>:<hash>`` keys.
* Check Mooncake connectivity by running the ``transfer_engine_bench``
  binary between 146:5556 and 129:5557.

## Teardown

* 146: press Ctrl-C in the PUT-only terminal (the Ctrl-C handler calls
  ``kvmanager.shutdown()`` which releases Mooncake + Redis state).
* 129: the GET-only run exits on its own; its ``atexit`` hook tears
  KVManager down.
* Optionally wipe Redis DB 1 between runs:
  ```bash
  redis-cli -h 10.206.0.9 -p 6379 -a 123456 -n 1 FLUSHDB
  ```
