# CPU + SSD KV pool across nodes with PD disaggregation (radixshmem)

This guide shows how to deploy FlexKV with SGLang so that every prefill (P) and decode (D) node shares one
CPU KV pool and one SSD KV pool across the cluster. Chinese version: [cpu_ssd_pool_pd_zh.md](cpu_ssd_pool_pd_zh.md).

## 1. What you get

- **CPU tier**: each node's host memory is a slot pool managed by its own `radix-server` (data mode). A request on
  any node can hit a prefix held in another node's CPU and pull it over RDMA.
- **SSD tier**: each node keeps its SSD blocks in local files; a second `radix-server` per node (index-only mode)
  indexes them. All nodes see one global SSD index, so a prefix written by one node can be read by another.
- **PD**: both P and D run FlexKV. D stores the KV it generated when a request finishes (and optionally every N
  tokens during decode), so the next turn's prefill on any P can reuse it.
- **Whole-prefix prefetch**: when a request is queued, FlexKV prefetches its whole cached prefix into local CPU
  from local SSD, peer CPU and peer SSD. The GPU load (H2D) at schedule time then reads only local CPU.
- **SSD global dedup** (optional, recommended): before writing a prefix to SSD, FlexKV checks the global SSD
  index. If a peer already holds the prefix, the write is skipped; if a peer holds the first part, only the
  remainder is written to that peer's SSD. This avoids keeping the same KV on several nodes' SSDs (for example
  P and D both storing the same conversation).

```
            node A (P)                                   node B (D)
 GPU ─H2D/D2H─ CPU pool (radix-server, data) ◄─RDMA pull─► CPU pool (radix-server, data)
                 │  ▲                                          │  ▲
             H2DISK│  │DISK2H                                 ...
                 ▼  │
            SSD files (radix-server, index) ◄─NIXL read/write─► SSD files (radix-server, index)

 global indexes: CPU cluster (etcd + RDMA)       SSD cluster (etcd + RDMA)
```

Transfer types you will see in FlexKV logs: `H2D`/`D2H` (GPU<->local CPU), `H2DISK`/`DISK2H` (local CPU<->local
SSD), peer CPU pull (radixshmem `pull_async`), `PEERSSD2H` (peer SSD -> local CPU) and `H2PEERSSD` (local CPU ->
peer SSD, dedup writes).

## 2. Versions

| Component | Branch | Notes |
|---|---|---|
| FlexKV | `linhu-nv/FlexKV` `linhu/rsm-ssd-pool` | this branch |
| SGLang | `linhu-nv/sglang` `linhu/flexkv-pd-rsm-ssd` | FlexKV PD decode integration, decode store stride, two fixes (see section 8) |
| radixshmem | `feat/pd-support` (validated at `5e32312`) | build with RDMA and the Mooncake transfer backend |
| NIXL | 1.4 or 1.5 Python wheel with the UCX and POSIX plugins | used for peer SSD reads and writes |

Use the same NIXL / Mooncake / UCX versions on all nodes.

## 3. Prerequisites

- RDMA NICs (InfiniBand or RoCE) between all nodes. For RoCE set the GID index (`--gid-idx 3` for radixshmem,
  `MC_GID_INDEX=3` for Mooncake, `UCX_IB_GID_INDEX` if UCX does not pick it).
- One etcd endpoint reachable from all nodes.
- A local NVMe filesystem for the SSD files on every node.
- A directory **shared by all nodes** for FlexKV's SSD manifests and NIXL contact files
  (`FLEXKV_PEER_SSD_MANIFEST_DIR`), e.g. on NFS. Each node writes small JSON files there at startup and peers read
  them to find its SSD files and NIXL endpoint. Without a shared filesystem, sync the directory between nodes.
- Clocks synchronized (NTP): NIXL requests carry deadlines.
- Build radixshmem (`cd radixshmem/python && pip install -e .`, see its README for RDMA, gRPC and Mooncake
  dependencies) and FlexKV (`pip install -e .` in this repo; it needs a C++20 compiler).

## 4. Deployment

Start order: etcd, then both radix-servers on every node, then SGLang (P and D) on every node, then the router.
Example below: 2 nodes with 4 GPUs each, node A = prefill at `10.0.0.10`, node B = decode at `10.0.0.20`, etcd at
`10.0.0.10:2379`. `N` is the total number of nodes (P + D) in the cluster.

### 4.1 etcd

```bash
etcd --name kv --data-dir /var/lib/etcd-kv \
  --listen-client-urls http://10.0.0.10:2379 --advertise-client-urls http://10.0.0.10:2379
```

### 4.2 radix-servers (two per node)

Both clusters use the same etcd but different `--cluster-id`. Use the same `--node-name` for the two servers of a
node; it is also the name FlexKV publishes its manifest under.

```bash
# On node A (repeat on every node with its own --node-name / --rpc-address / NIC)
# CPU pool: data mode. --data-bytes is the CPU tier capacity of this node.
# --swa-ratio is required for SWA models (e.g. DeepSeek-V4.x); 0 otherwise.
radix-server --name /kv_cpu --data-bytes 200G --swa-ratio 0.2 \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_cpu --node-name nodeA \
  --rpc-address 10.0.0.10 --index-dev mlx5_0 --gid-idx 3 \
  --transfer-dev mlx5_0 --transfer-backend mooncake

# SSD index: index-only mode (no --data-bytes). Slot ids are FlexKV SSD block ids.
radix-server --name /kv_ssd \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_ssd --node-name nodeA \
  --rpc-address 10.0.0.10 --index-dev mlx5_0 --gid-idx 3
```

(`radix-server` is `python -m radixshmem.cli`.) The SSD index server gets its geometry (number of SSD blocks and SWA
slots) from the FlexKV process on its node, so the SSD cluster becomes ready only after FlexKV has started on all
`N` nodes. Do not start decode-side radix-servers with `--decode` for this pipeline; that mode (prefill eviction
write-back to decode) was not validated together with it.

### 4.3 FlexKV config file

```yaml
# /etc/flexkv/nodeA.yaml
cpu_cache_gb: 8              # ignored in radixshmem mode: CPU capacity = radix-server --data-bytes
ssd_cache_gb: 1000           # SSD tier capacity of this node
ssd_cache_dir: /nvme/flexkv  # local NVMe directory for the SSD files
swa:
  num_ssd_slots: 4096        # SSD SWA slots (SWA models only)
```

### 4.4 Environment (per SGLang instance)

```bash
# radixshmem tiers
export FLEXKV_ENABLE_RADIXSHMEM=1
export FLEXKV_CPU_LAYOUT=BLOCKFIRST
export FLEXKV_RADIXSHMEM_SERVER_NAME=/kv_cpu
export FLEXKV_RADIXSHMEM_SSD_SERVER_NAME=/kv_ssd
export FLEXKV_PEER_SSD_MANIFEST_DIR=/shared/flexkv-manifests

# peer SSD data plane over NIXL
export FLEXKV_REMOTE_SSD_READER=flexkv.transfer.workers.peer_ssd_nixl:make_reader
export FLEXKV_NIXL_SSD_IP=10.0.0.10        # this node's IP reachable by peers
export FLEXKV_NIXL_SSD_PORT=5810           # FULL pool; SWA pool uses port+1
export FLEXKV_NIXL_UCX_DEVICES=mlx5_0
export UCX_TLS=rc,ud,self,cuda_copy

# SSD global dedup (recommended)
export FLEXKV_SSD_WRITE_MODE=global_dedup

# process / startup
export FLEXKV_ENABLE_MPS=0
export FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_nodeA   # unique per instance on a host
export FLEXKV_READY_TIMEOUT_S=3600                      # slow first start (model load, cluster rendezvous)
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export MC_GID_INDEX=3                                   # RoCE

# decode node only (optional): also store every 512 generated tokens
export SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE=512
```

### 4.5 SGLang (DeepSeek-V4.1, attention DP4 + MoE EP4)

```bash
# prefill (node A)
python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1 --trust-remote-code \
  --tp 4 --dp 4 --enable-dp-attention --ep-size 4 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --page-size 256 --mem-fraction-static 0.93 --chunked-prefill-size 4096 \
  --no-enable-decoder-swa-bounded-replay --enable-cache-report \
  --enable-flexkv --flexkv-config-file /etc/flexkv/nodeA.yaml \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 --disaggregation-bootstrap-port 8995 \
  --host 0.0.0.0 --port 30000

# decode (node B): same, with --disaggregation-mode decode and without the bootstrap port.
# --enable-flexkv on decode turns on --disaggregation-decode-enable-radix-cache automatically.
```

Then start the SGLang PD router with the prefill (`http://10.0.0.10:30000`, bootstrap port 8995) and decode
endpoints as usual. Other models work the same way; drop the DSv4-specific flags.

### 4.6 Check that it works

1. All radix-servers log ready; every FlexKV instance logs that it attached the CPU and the SSD radix-server and
   (with NIXL) that its peer-SSD endpoint is serving.
2. `ls $FLEXKV_PEER_SSD_MANIFEST_DIR` shows `<node>.json` plus `<node>.nixl-full.json` (and `nixl-swa`) for every node.
3. Send a long prompt to P, push it out of P's CPU (or use another P), send it again: SGLang reports cached
   tokens and FlexKV logs `DISK2H` / `PEERSSD2H` / peer pulls; the answer is the same as the first time.
4. Metrics: `flexkv_py_transfer_blocks_total{transfer_type=...}` grows for the transfer types in use.

## 5. Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `FLEXKV_RADIXSHMEM_SSD_SERVER_NAME` | empty | `--name` of this node's index-only SSD radix-server; required when `ssd_cache_gb > 0` in radixshmem mode |
| `FLEXKV_PEER_SSD_MANIFEST_DIR` | empty | shared directory for SSD manifests and NIXL contact files; required for peer SSD |
| `FLEXKV_REMOTE_SSD_READER` | shared-file reader | `module:factory` of the peer SSD transport. `flexkv.transfer.workers.peer_ssd_nixl:make_reader` = NIXL; the default reads peers' SSD files directly and only works if they are on a shared filesystem |
| `FLEXKV_NIXL_SSD_IP` / `FLEXKV_NIXL_SSD_PORT` | auto / 5600 | address peers use for this node's NIXL endpoint (FULL pool port; SWA uses +1) |
| `FLEXKV_NIXL_UCX_DEVICES` | UCX default | NICs for NIXL (UCX) |
| `FLEXKV_NIXL_SSD_TIMEOUT_S` | 30 | deadline of one peer SSD request |
| `FLEXKV_NIXL_SSD_RETRY_S` | 10 | back-off before retrying a peer whose endpoint failed |
| `FLEXKV_NIXL_SSD_DIRECT_READ` | 1 | serve SSD reads with O_DIRECT (aligned span for blocks that are not 4 KiB-aligned) |
| `FLEXKV_NIXL_POSIX_URING` | true | NIXL POSIX plugin uses io_uring |
| `FLEXKV_SSD_WRITE_MODE` | `local` | `global_dedup` enables SSD global dedup |
| `FLEXKV_SSD_DEDUP_LOCAL_FRACTION` | 0 | fraction of SSD writes that skip the global check and write locally (spreads write load) |
| `FLEXKV_SSD_DEDUP_PEER_BACKOFF_S` | 10 | after a failed remote write, write locally instead of to that peer for this long |
| `FLEXKV_RADIXSHMEM_SSD_READY_TIMEOUT_S` | 3600 | how long FlexKV waits for the SSD cluster to become ready |
| `SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE` | 0 | D also stores every N generated tokens (page-aligned); 0 = store at finish only. SWA models store FULL KV only mid-decode |

## 6. Sizing and operations

- **CPU pool size**: the CPU pool is also the staging area for SSD reads, peer pulls and SSD writes. Size it to at
  least *concurrent requests x new blocks per request*; when it is too small, FlexKV silently skips stores and
  prefetches (no errors, but hit rate collapses).
- **Restart rule**: restart a node's two radix-servers together with that node's SGLang/FlexKV. A radix-server
  restarted alone waits for FlexKV to configure it again and the node's SSD tier stays off until FlexKV restarts.
  After a FlexKV crash, restart its radix-servers too (pinned slots of the dead process are not reclaimed).
- **First start**: the SSD cluster is ready only after FlexKV has configured the SSD index server on all nodes, so
  keep the startup timeouts in 4.4 when nodes load models at different speeds.
- **Several instances per host**: give each its own `FLEXKV_SERVER_RECV_PORT`, `FLEXKV_NIXL_SSD_PORT` (two ports
  each), radix-server names and, with layerwise transfer, `FLEXKV_LAYERWISE_EVENTFD_SOCKET`.
- **Failures**: a peer that does not answer within the deadline is backed off; reads fall back to recompute and
  dedup writes fall back to local writes. A NIXL peer-SSD worker process that dies is restarted automatically
  (up to 8 times per engine).
- DeepSeek-V4.x does not support `--disaggregation-enable-kv-checksum`.

## 7. Validation and performance (summary)

Validated on H20 nodes with Qwen3-8B (1P1D) and DeepSeek-V4.1 (1P1D, and 2P1D across two machines with attention
DP4 + MoE EP4): 1 h and 4 h runs with random and sticky routing, 20% client aborts, hot-prefix bursts, P/D restart,
NIXL endpoint stall, 64k-100k contexts, layerwise on/off, forced retraction, decode store stride, dedup on/off. All
with 0 unexpected request errors, correct needle answers and no garbled outputs on cache hits.

| Metric | Result |
|---|---|
| Cross-node duplicate SSD blocks, local vs `global_dedup` | DeepSeek-V4.1 2P1D 60.6% -> 1.7%; Qwen3-8B 1P1D 37-41% -> 0-14% |
| SSD bytes written | -34% to -60% with `global_dedup` |
| Throughput / TTFT / TPOT, `global_dedup` vs local | within 2% (DeepSeek-V4.1 1P1D, c8-c32, 3 alternating repeats) |
| Peer SSD read (NIXL, cross-machine, 1.67-9.4 MB blocks) | 6-27 GB/s depending on request size and concurrency |
| TTFT, DeepSeek-V4.1, 16k prompt (median of 5) | recompute 4572 ms; local CPU 189; local SSD 206; peer CPU 262; peer SSD 197 |
| TTFT, DeepSeek-V4.1, 64k prompt (median of 5) | recompute 20390 ms; local CPU 217; local SSD 262; peer CPU 222; peer SSD 228 |

## 8. Known limitations

- A radix-server restarted on its own does not rejoin until FlexKV restarts (section 6).
- With `global_dedup`, P and D writing the same new prefix at the same moment can still both write it (a few
  percent duplicates); prefixes concentrate on the node that first stored them, which shifts some SSD reads from
  local to peer.
- The SGLang branch also fixes two SGLang issues found under this pipeline: a SWA double free when a chunked
  prefill is aborted after a FlexKV load-back, and the PD retract re-bootstrap reusing the original request id and
  bootstrap room (upstream issue, also without FlexKV).
