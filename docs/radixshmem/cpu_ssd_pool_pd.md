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
- **Attention-DP direct mode** (optional, `FLEXKV_DP_DIRECT=1`): every DP rank runs its own FlexKV engine on the
  radixshmem index instead of queueing behind one KVServer per node.
- **SSD files shared on a node**: every FlexKV engine of a node (the DP ranks of direct mode, several instances)
  shares one set of SSD files under `ssd_cache_dir` and one SSD index, and the SSD cache stays valid across FlexKV
  restarts (with the radix-server still up). The radix-server only indexes; it never touches the SSD files.
- **KV events for the dynamo router** (optional): the CPU-pool radix-server publishes the node's block placement;
  the dynamo KV router (`--shared-cache-type radixshmem`) learns which node's CPU pool holds a prefix and prefers it.

```
            node A (P)                                   node B (D)
 GPU ─H2D/D2H─ CPU pool (radix-server, data) ◄─RDMA pull─► CPU pool (radix-server, data)
                 │  ▲                                          │  ▲
             H2DISK│  │DISK2H                                 ...
                 ▼  │
            SSD files (radix-server, index) ◄─NIXL read/write─► SSD files (radix-server, index)

 global indexes: CPU cluster (etcd + gRPC + Redis)   SSD cluster (etcd + gRPC + Redis)
```

Transfer types you will see in FlexKV logs: `H2D`/`D2H` (GPU<->local CPU), `H2DISK`/`DISK2H` (local CPU<->local
SSD), peer CPU pull (radixshmem `pull_async`), `PEERSSD2H` (peer SSD -> local CPU) and `H2PEERSSD` (local CPU ->
peer SSD, dedup writes).

## 2. Versions

| Component | Branch | Notes |
|---|---|---|
| FlexKV | `linhu-nv/FlexKV` `linhu/rsm-ssd-pool-flexkv-owned` | this branch (upstream main + this work) |
| SGLang | `linhu-nv/sglang` `linhu/flexkv-1116-xing-pd-ssdpool-kvevent` (upstream main + XingLiu1 `agent/flexkv-dsv4-main`) | FlexKV PD decode integration, decode store stride, hybrid batched layerwise restore, caller block hashes, a sync before non-layerwise restores, fixes (see section 9) |
| radixshmem | `linhu-nv/radixshmem` `linhu/index-id` (based on `main`) | KV events, index id (`NodeInfo.index_id`); build with RDMA, gRPC and the Mooncake transfer backend |
| dynamo (optional) | `linhu-nv/dynamo` `linhu/radixshmem-shared-cache-1116` (based on `main`) | only for the dynamo KV router with KV events |
| NIXL | 1.4 or 1.5 Python wheel with the UCX and POSIX plugins | used for peer SSD reads and writes |

Use the same NIXL / Mooncake / UCX versions on all nodes.

## 3. Prerequisites

- RDMA NICs (InfiniBand or RoCE) between all nodes. For RoCE set the GID index (`MC_GID_INDEX=3` for Mooncake,
  `UCX_IB_GID_INDEX` if UCX does not pick it). The radixshmem index runs on gRPC and the Redis RHT by default and
  opens no RDMA device; only an RDMA index needs `RADIXSHMEM_INDEX_DEV` / `RADIXSHMEM_INDEX_GID_IDX` (the old
  `--index-dev` / `--gid-idx` flags are gone).
- A Redis Cluster for radixshmem's global route table (RHT, `RADIXSHMEM_RHT_BACKEND=redis`; see the Redis Cluster
  guide under radixshmem `docs/`).
- One etcd endpoint reachable from all nodes.
- A local NVMe filesystem for the SSD files on every node.
- A directory **shared by all nodes** for FlexKV's SSD manifests and NIXL contact files
  (`FLEXKV_PEER_SSD_MANIFEST_DIR`), e.g. on NFS. Each node writes small JSON files there at startup and peers read
  them to find its SSD files and NIXL endpoint. Without a shared filesystem, sync the directory between nodes.
- Clocks synchronized (NTP): NIXL requests carry deadlines.
- Build radixshmem (`pip install -e .` at its repository root, see its README for RDMA, gRPC and Mooncake
  dependencies) and FlexKV (`pip install -e .` in this repo; it needs a C++20 compiler). For KV events to the dynamo
  router (4.7), install radixshmem with `pip install -e ".[kv-events]"` (adds pyzmq and msgpack): a radix-server
  started with `--kv-events-endpoint` fails at start-up without them.

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
# Every radix-server process carries the Redis RHT settings (rank 0's take effect, the others read them from etcd):
export RADIXSHMEM_RHT_BACKEND=redis RADIXSHMEM_REDIS_ADDRS=10.0.0.10:7001,10.0.0.10:7002,10.0.0.10:7003
radix-server --name /kv_cpu --data-bytes 200G --swa-ratio 0.2 \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_cpu --node-name nodeA \
  --rpc-address 10.0.0.10 --rpc-port 24610 \
  --transfer-dev mlx5_0 --transfer-backend mooncake \
  --kv-events-endpoint 'tcp://*:5691'            # optional: KV events for the dynamo router (4.7)

# SSD index: index-only mode (no --data-bytes). Slot ids are FlexKV's SSD block ids; FlexKV creates the SSD files.
radix-server --name /kv_ssd \
  --expected-min-nodes 2 --bootstrap-timeout 3600 \
  --registry etcd://10.0.0.10:2379 --cluster-id kv_ssd --node-name nodeA \
  --rpc-address 10.0.0.10 --rpc-port 24611
```

(`radix-server` is `python -m radixshmem.cli`.) The SSD index server gets its geometry (the number of SSD blocks and
SWA slots) from the FlexKV process on its node, so the SSD cluster becomes ready only after FlexKV has started on all
`N` nodes. Do not start decode-side radix-servers with `--decode` for this pipeline; that mode (prefill eviction
write-back to decode) was not validated together with it.

### 4.3 FlexKV config file

```yaml
# /etc/flexkv/nodeA.yaml
cpu_cache_gb: 8              # ignored in radixshmem mode: CPU capacity = radix-server --data-bytes
ssd_cache_gb: 1000           # SSD tier capacity of this node
ssd_cache_dir: /nvme/flexkv  # local NVMe directory for the SSD files (shared by every FlexKV of the node)
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

# layerwise transfer with batched restore (recommended)
export FLEXKV_ENABLE_LAYERWISE_TRANSFER=1
export FLEXKV_BATCH_LAYERWISE_RESTORE=1
export FLEXKV_LAYERWISE_EVENTFD_SOCKET=/tmp/flexkv_lw_nodeA.sock   # unique per instance on a host

# process / startup
export FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_nodeA   # unique per instance on a host
export FLEXKV_READY_TIMEOUT_S=3600                      # slow first start (model load, cluster rendezvous)
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export MC_GID_INDEX=3                                   # RoCE

# DeepSeek-V4.1: SWA snapshots at chunk boundaries on a 4-page grid, for decoder SWA bounded replay (on by default; higher throughput)
export SGLANG_FLEXKV_SWA_GRID_PAGES=4

# decode node only (optional): also store every 512 generated tokens
export SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE=512
```

### 4.5 SGLang (DeepSeek-V4.1, attention DP4 + MoE EP4)

```bash
# prefill (node A)
python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1 --trust-remote-code \
  --tp 4 --dp 4 --enable-dp-attention --ep-size 4 \
  --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 \
  --page-size 256 --mem-fraction-static 0.92 --chunked-prefill-size 4096 \
  --enable-cache-report \
  --enable-flexkv --flexkv-config-file /etc/flexkv/nodeA.yaml \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3 --disaggregation-bootstrap-port 8995 \
  --host 0.0.0.0 --port 30000

# decode (node B): same, with --disaggregation-mode decode, --mem-fraction-static 0.93 (section 5) and without the bootstrap port.
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

### 4.7 Optional: attention-DP direct mode, KV-event routing (dynamo)

Either can be turned on alone.

**Direct mode** (`FLEXKV_DP_DIRECT=1` in each SGLang instance's environment): with attention DP every DP rank runs
its own FlexKV engine instead of going through one KVServer per node. A DP instance spanning nodes must run its own
DP ranks on every node. Each rank serves its own NIXL peer-SSD endpoint: rank k takes ports `PORT+2k` and
`PORT+2k+1` from `FLEXKV_NIXL_SSD_PORT`.

**KV-event routing**: needs the dynamo branch; start workers with `python -m dynamo.sglang` and the router with
`python -m dynamo.frontend`.

```bash
# CPU-pool radix-server with --kv-events-endpoint 'tcp://*:5691' (4.2); radixshmem installed with the [kv-events] extra (pyzmq, msgpack; section 3)

# environment of every dynamo.sglang worker
export FLEXKV_HASH_SCHEME=sglang-sha256-i64-v1                  # FlexKV names blocks by SGLang's page hashes, which the router recomputes
export DYN_RADIXSHMEM_KV_EVENTS_ENDPOINT=tcp://10.0.0.10:5691   # this node's CPU-pool radix-server event endpoint, as the router reaches it
# an attention-DP instance spanning nodes sets DYN_RADIXSHMEM_KV_EVENTS_ENDPOINTS=<one per node, in node-rank order>

python3 -m dynamo.sglang <the SGLang arguments of 4.5> \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:24710","topic":"kv"}'

# router (frontend)
export DYN_RADIXSHMEM_REMOTE_HIT_WEIGHT=0.9   # credit of a block only another node holds, relative to one on the worker's node (default 0.5)
python3 -m dynamo.frontend --http-port 8000 --router-mode kv --router-kv-events \
  --router-track-prefill-tokens --shared-cache-type radixshmem
```

It works when the router logs `subscribed to radixshmem` and `worker nodes`, and `shared-cache lookup ... held=N` with
N > 0. `FLEXKV_HASH_SCHEME` is part of the radixshmem geometry: every FlexKV of a cluster must use the same value or
it is refused at attach. Do not give the SSD index radix-server `--kv-events-endpoint` (the router reads the CPU pool
only).

## 5. Configuration guide

Three levels: get the **minimal configuration** running, deploy with the **recommended configuration**, then **tune per scenario**. Each switch lists its cost and how to observe it.

### 5.1 Minimal configuration

The deployment steps in section 4 are the minimal configuration: two radix-servers per node, the FlexKV config file, the environment of 4.4 (without the optional parts) and `--enable-flexkv` on SGLang. Section 4.6 tells you it works.

### 5.2 Recommended configuration

| Setting | Recommended | Notes |
|---|---|---|
| CPU pool `radix-server --data-bytes` | as large as host memory allows, and at least *concurrent requests x new blocks per request* | The CPU pool is also the staging area for SSD reads, peer pulls and SSD writes. When it is too small FlexKV silently skips stores and prefetches (no errors, lower hit rate); the log shows `skipped: CPU ... slots available` / `no CPU room to stage` |
| SWA pool share `--swa-ratio` (SWA models) | 0.2 | raise it if many sessions run out of SWA slots |
| SSD capacity `ssd_cache_gb` / `swa.num_ssd_slots` | disk size / 4096 | — |
| Peer SSD transport `FLEXKV_REMOTE_SSD_READER` | `flexkv.transfer.workers.peer_ssd_nixl:make_reader` | the default implementation only works when the SSD files are on a shared filesystem |
| NIXL NICs `FLEXKV_NIXL_UCX_DEVICES` / `UCX_TLS` | RDMA NIC on the CPU pool's NUMA node / `rc,ud,self,cuda_copy` | — |
| SSD global dedup `FLEXKV_SSD_WRITE_MODE` | `global_dedup` | DeepSeek-V4.1: cross-node SSD duplicates 60.6% -> 1.7%, SSD writes -60%, throughput/TTFT/TPOT within 2% |
| Share of writes kept local `FLEXKV_SSD_DEDUP_LOCAL_FRACTION` | 0 | see the scenarios below |
| Layerwise transfer `FLEXKV_ENABLE_LAYERWISE_TRANSFER` | 1 | overlaps loading with compute layer by layer; give each instance on a host its own `FLEXKV_LAYERWISE_EVENTFD_SOCKET` |
| Batched layerwise restore `FLEXKV_BATCH_LAYERWISE_RESTORE` | 1 | merges the restores of a batch into a few transfers instead of one request after another; up to 1.8x faster with TP>1 and short hits, neutral otherwise |
| Decode store stride `SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE` | short outputs: 0; agentic / long outputs (thousands of tokens): 512-1024 | D's KV becomes reusable earlier and the store at finish writes only the tail; costs some D2H during decode |
| Peer back-off `FLEXKV_SSD_DEDUP_PEER_BACKOFF_S` / `FLEXKV_NIXL_SSD_RETRY_S` | 10 / 10 | usually unchanged |
| SSD O_DIRECT | automatic (default) | reads and writes bypass the page cache at the device alignment (512 B on most NVMe); Applies when the block size is a multiple of that alignment: DeepSeek-V4.1 SWA blocks (5990400 B) and FULL blocks under TP (1671680 B) read cold at 18-22 GB/s locally instead of 5; with attention DP the FULL block is 417920 B (not 512-aligned) and stays on the page cache (peer reads use aligned O_DIRECT spans either way). Override with `FLEXKV_NIXL_SSD_DIRECT_READ=0` / `FLEXKV_NIXL_SSD_DIRECT_ALIGN` when debugging |
| radixshmem CPU dedup after pulls (`RADIXSHMEM_DISABLE_AUTO_DEDUP`) | default (on) | validated with the default |
| radix-server `--decode` (prefill eviction written back to decode) | off | not validated with this pipeline yet |
| FlexKV process mode | direct mode `FLEXKV_DP_DIRECT=1` at high, prefill-heavy request rates; otherwise default (`dp_size>1`: one KVServer per node, DP ranks are clients) | direct mode removes the KVServer queue: on Qwen3-8B DP8 a FlexKV call's p90 dropped from 3-11 ms to 0.3 ms, prefill-heavy throughput +3%-10%, decode-heavy unchanged. In direct mode and with several instances per node (`FLEXKV_INSTANCE_NUM>1`) the engines share the node's SSD files: keep `ssd_cache_gb` / `ssd_cache_dir` the same |
| Routing | session-sticky if available | sticky routing cuts cross-node traffic to about 1/3 of random routing; peer SSD reads near 0 |
| Startup timeouts `FLEXKV_READY_TIMEOUT_S` / `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` / `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | 3600 | the first start waits for every node |
| SGLang (DeepSeek-V4.1) | `--mem-fraction-static 0.92 --chunked-prefill-size 4096 --page-size 256`, keep decoder SWA bounded replay on (the default; do not pass `--no-enable-decoder-swa-bounded-replay`), `SGLANG_FLEXKV_SWA_GRID_PAGES=4` | bounded replay matters a lot for throughput but takes about 1 GB more GPU memory at run time, hence 0.92 instead of 0.93 (H20, TP4). Except on a PD decode node: on H20 it has about 1 GB less free after loading the weights than a prefill node, and at 0.92 the SWA pool for `--max-running-requests 64` does not fit (start-up fails with `SWA pool cap ... leaves no room`), so run decode at 0.93 (validated). Do not use `--disaggregation-enable-kv-checksum` (not supported for DeepSeek-V4.x) |

### 5.3 Tuning per scenario

- **A few nodes take much more SSD writes or space** (hot prefixes concentrate on the node that stored them first): set `FLEXKV_SSD_DEDUP_LOCAL_FRACTION` to 0.1-0.3 so that share of writes skips the global check and writes locally, at the cost of some duplicates.
- **Network bandwidth is tight and disk space is not**: `FLEXKV_SSD_WRITE_MODE=local`.
- **CPU hit rate below expectation**: look for the CPU slot warnings above; raise `--data-bytes` or lower concurrency.
- **Long outputs / multi-turn agentic workloads**: enable the decode store stride (512-1024).

### 5.4 What to watch

- Transfers: `flexkv_py_transfer_blocks_total{transfer_type=...}`, especially `DISK2H`, `PEERSSD2H` (peer SSD reads) and `H2PEERSSD` (dedup writes to a peer).
- Dedup: one line per minute `[FlexKV-IO] operation=ssd_dedup act=stats` with `puts_skipped` (whole prefix already on a peer), `puts_remote` (remainder written to a peer), `fallback_*` (wrote locally instead) and `remote_write_failed`.
- Hits: SGLang `cached_tokens` (`--enable-cache-report`).
- Health: radix-server health in its log; `transfer failed` and worker death/restart lines (with class, pid, exit code) in the FlexKV log.

## 6. Configuration reference

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
| `FLEXKV_NIXL_SSD_DIRECT_ALIGN` | probed | O_DIRECT alignment of the SSD files used by the NIXL endpoint (normally probed, 512 on most NVMe) |
| `FLEXKV_ENABLE_LAYERWISE_TRANSFER` | 0 | load KV layer by layer, overlapped with compute |
| `FLEXKV_LAYERWISE_EVENTFD_SOCKET` | — | eventfd socket of layerwise transfer; unique per instance on a host |
| `FLEXKV_BATCH_LAYERWISE_RESTORE` | 0 | merge the layerwise restores of the requests admitted in one batch |
| `FLEXKV_RADIXSHMEM_SSD_READY_TIMEOUT_S` | 3600 | how long FlexKV waits for the SSD cluster to become ready |
| `SGLANG_FLEXKV_DECODE_OFFLOAD_STRIDE` | 0 | D also stores every N generated tokens (page-aligned); 0 = store at finish only. SWA models store FULL KV only mid-decode. A PD decode server counts from the end of the prompt, so its first store carries the prompt with new output |
| `FLEXKV_DP_DIRECT` | 0 | 1 = one FlexKV engine per DP rank under attention DP (see 4.7) |
| `FLEXKV_HASH_SCHEME` | unset | `sglang-sha256-i64-v1` = name blocks by SGLang's page hashes (needed for KV-event routing) |
| `DYN_RADIXSHMEM_KV_EVENTS_ENDPOINT(S)` | unset | dynamo worker: the event endpoint of this node's (or each node's, in node order) CPU-pool radix-server |
| `DYN_RADIXSHMEM_REMOTE_HIT_WEIGHT` | 0.5 | dynamo frontend: credit of a block only another node holds |

## 7. Sizing and operations

- **CPU pool size**: the CPU pool is also the staging area for SSD reads, peer pulls and SSD writes. Size it to at
  least *concurrent requests x new blocks per request*; when it is too small, FlexKV silently skips stores and
  prefetches (no errors, but hit rate collapses).
- **Restart rule**: restart a node's two radix-servers together with that node's SGLang/FlexKV. A radix-server
  restarted alone waits for FlexKV to configure it again and the node's SSD tier stays off until FlexKV restarts.
  After a FlexKV crash, restart its radix-servers too (pinned slots of the dead process are not reclaimed).
- **SSD cache across FlexKV restarts**: the SSD files have fixed names (`flexkv_rsm_ssd_*.bin` in `ssd_cache_dir`) and
  `flexkv_rsm_ssd.json` next to them records which SSD index (radixshmem `index_id`) describes them and their layout.
  When both still match at start-up FlexKV keeps the SSD index; after an SSD radix-server restart, a change of
  `ssd_cache_gb` or of the model, or recreated files, it clears the SSD index (log `clearing ... blocks of SSD index`).
- **First start**: the SSD cluster is ready only after FlexKV has configured the SSD index server on all nodes, so
  keep the startup timeouts in 4.4 when nodes load models at different speeds.
- **Several instances per host**: give each its own `FLEXKV_SERVER_RECV_PORT`, `FLEXKV_NIXL_SSD_PORT` (two ports
  each), radix-server names and, with layerwise transfer, `FLEXKV_LAYERWISE_EVENTFD_SOCKET`.
- **Failures**: a peer that does not answer within the deadline is backed off; reads fall back to recompute and
  dedup writes fall back to local writes. A NIXL peer-SSD worker process that dies is restarted automatically
  (up to 8 times per engine).
- DeepSeek-V4.x does not support `--disaggregation-enable-kv-checksum`.

## 8. Validation and performance (summary)

Validated on H20 nodes with Qwen3-8B (1P1D) and DeepSeek-V4.1 (1P1D, and 2P1D across two machines with attention
DP4 + MoE EP4): 1 h and 4 h runs with random and sticky routing, 20% client aborts, hot-prefix bursts, P/D restart,
NIXL endpoint stall, 64k-100k contexts, layerwise on/off (including batched restore across machines), forced retraction, decode store stride, dedup on/off. All
with 0 unexpected request errors, correct needle answers and no garbled outputs on cache hits.


The branch set of section 2 (2026-11-23, on upstream main) was re-run on the same kind of H20 nodes with DeepSeek-V4.1
on two nodes (TP4/DP4/EP4, bounded replay on, `SGLANG_FLEXKV_SWA_GRID_PAGES=4`, `--mem-fraction-static 0.92`): direct mode
+ FlexKV's shared SSD files + peer SSD + `global_dedup` + KV-event routing (dynamo) and the server-client mode, 952
requests each with 0 errors, every needle answer right and a 78.5% hit rate; the router's per-node block counts matched
the radix-servers exactly. After restarting only FlexKV/SGLang (radix-servers kept running) the SSD index was kept and
the same requests (480) ran with 0 errors at a 93.5% hit rate (79.8% before the restart). PD 1P1D (layerwise off,
overlap on; decode node at `--mem-fraction-static 0.93`, see section 5): 2833 requests, 0 errors, every needle answer
right, no garbled output.

| Metric | Result |
|---|---|
| Cross-node duplicate SSD blocks, local vs `global_dedup` | DeepSeek-V4.1 2P1D 60.6% -> 1.7%; Qwen3-8B 1P1D 37-41% -> 0-14% |
| SSD bytes written | -34% to -60% with `global_dedup` |
| Throughput / TTFT / TPOT, `global_dedup` vs local | within 2% (DeepSeek-V4.1 1P1D, c8-c32, 3 alternating repeats) |
| Peer SSD read (NIXL, cross-machine, 1.67-9.4 MB blocks) | 6-27 GB/s depending on request size and concurrency |
| TTFT, DeepSeek-V4.1, 16k prompt (median of 5) | recompute 4572 ms; local CPU 189; local SSD 206; peer CPU 262; peer SSD 197 |
| TTFT, DeepSeek-V4.1, 64k prompt (median of 5) | recompute 20390 ms; local CPU 217; local SSD 262; peer CPU 222; peer SSD 228 |

## 9. Known limitations

- A radix-server restarted on its own does not rejoin until FlexKV restarts (section 7).
- With `global_dedup`, P and D writing the same new prefix at the same moment can still both write it (a few
  percent duplicates); prefixes concentrate on the node that first stored them, which shifts some SSD reads from
  local to peer.
- The SGLang branch also fixes SGLang issues found under this pipeline: the PD retract re-bootstrap reusing the
  original request id and bootstrap room (upstream issue, also without FlexKV); under the overlap scheduler, a
  non-layerwise restore's H2D racing the previous batch's unfinished kernels on the same slots (rare garbled output;
  the restore now syncs first); the detokenizer crashing at start when `SGLANG_LOGGING_CONFIG_PATH` is set (upstream
  did not import `logging.config`). The fix for the SWA double free when a chunked prefill is aborted after a load-back
  is part of XingLiu1's branch.
