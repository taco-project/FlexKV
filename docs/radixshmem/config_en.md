# Radixshmem mode configuration

[中文文档](config_zh.md)

FlexKV can use radixshmem for its CPU tier: the index, SlotStore and cross-node
pulls. Configuration is split between the operator, who starts one `radix-server`
per node with the storage budget and cluster settings, and FlexKV, which enables
and attaches to the server.

The first FlexKV client sends block and SWA geometry to the server. The server
plans slot counts from `--data-bytes` and `--swa-ratio`; every FlexKV process
adopts those counts as `num_cpu_blocks` and, when enabled, `swa.num_slots`.
`cpu_cache_gb` does not size this tier. An explicitly configured CPU size is
reported as a warning; otherwise the notice is informational. FlexKV does not
start or size the server. See `flexkv/server/shm_radix_bootstrap.py` and the
radixshmem repository's README, API documentation and `radix-server --help`.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLEXKV_ENABLE_RADIXSHMEM` | `0` | Set to `1` before the first FlexKV import to enable this mode. |
| `FLEXKV_RADIXSHMEM_SERVER_NAME` | `/flexkv` | Server's `--name`; a shared-memory name starting with `/`. The gRPC endpoint is derived from that name. |
| `FLEXKV_CPU_LAYOUT` | `BLOCKFIRST` | Must be `BLOCKFIRST`: one SlotStore slot is one contiguous CPU block. |
| `FLEXKV_INSTANCE_NUM` / `FLEXKV_INSTANCE_ID` | `1` / `0` | Identify engines sharing a server on one node. |

This mode supplies only the CPU tier. Disable `enable_ssd`, `enable_remote`,
`enable_p2p_cpu` and `enable_p2p_ssd`; `KVManager` checks these combinations.
Cross-node reuse is handled by a clustered radix-server (etcd and RDMA).
For one DP and one instance, KVTaskEngine runs in the engine process and TE is
its child. Multiple DP clients or instances use a per-node KVServer whose
KVTaskEngine and TE attach to radix-server.

## Fixed parameters

The following are constants in `flexkv/server/shm_radix_bootstrap.py`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `READY_TIMEOUT_S` | 600 | Total wait for server reachability and readiness, including late startup, prefault and cluster rendezvous. Keep the server's bootstrap timeout below this. |
| `PREFETCH_TIMEOUT_MS` | 5000 | Server-side timeout for one peer pull; the job completes with the local hits available at expiry. |
| `PREFETCH_MAX_INFLIGHT` | 128 | Maximum peer pulls in flight per KVTaskEngine; new prefetches skip peer lookup at the limit. |
| `MAX_OUTSTANDING` | 256 | Maximum outstanding peer-pull jobs per `RadixClient`. FlexKV submits nonblocking; at the limit the new job completes with local hits. |

## Geometry and slot counts

FlexKV sends `block_size` from `CacheConfig.tokens_per_block`, `full_slot_bytes`
from the BLOCKFIRST CPU block layout, and, when SWA is enabled, `swa_slot_bytes`
and `swa_window_blocks`. `slot_align` is the largest power of two no greater
than 4096 that divides every slot size, so the SlotStore stride equals the block
size. `register_chunk_tokens=0` leaves the server's configured registration
chunk in effect; FlexKV converts its published value to blocks, with at least one
block.

The server plans SWA slots as `floor(swa_ratio * data_bytes / swa_stride)` and
uses the remaining budget for FULL slots. If a required pool would have zero
slots, or an SWA model uses a zero SWA ratio, configuration is rejected. Once
published, FlexKV adopts the slot counts and the cluster rank. KVManager and
KVTaskEngine adopt them, and TE adopts them again after its block-count
recomputation so that the server's counts remain authoritative.

Attachers check block size, pool slot bytes and stride, and SWA window against
their own layout. Slot counts themselves come from the server. Clients sharing
a server must supply compatible geometry; a mismatch is rejected. In a cluster,
nodes must agree on geometry, but may have different byte budgets and slot counts.

## Startup

### Install

Follow the radixshmem repository for installation and building Mooncake or NIXL
backends. For an editable Python installation:

```bash
cd radixshmem/python && pip install -e .
```

Start one `radix-server` on each node. Supply `--name` explicitly: radix-server's
default name is `/radixshmem`, whereas FlexKV attaches to `/flexkv` by default.

### Single node

```bash
# Operator, once per node. For a model with SWA, add e.g. --swa-ratio 0.2.
radix-server --name /flexkv --data-bytes 1024G

# In the engine environment, before importing FlexKV:
export FLEXKV_ENABLE_RADIXSHMEM=1
export FLEXKV_CPU_LAYOUT=BLOCKFIRST
# Unset FLEXKV_RADIXSHMEM_SERVER_NAME to attach to /flexkv.
```

The server waits for a client's geometry and creates its regions before becoming
ready. It may start after the engine; FlexKV retries within `READY_TIMEOUT_S`.

### Multi-node cluster

For four nodes coordinated by etcd, run the following on each node with its own
`--rpc-address`. Choose appropriate control- and data-plane devices:

```bash
radix-server --name /flexkv --data-bytes 1024G --swa-ratio 0.2 \
    --expected-min-nodes 4 --registry etcd://10.0.0.1:2379 \
    --rpc-address 10.0.0.10 --index-dev mlx5_bond_0 \
    --transfer-dev mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
```

`--transfer-backend` selects Mooncake (default) or NIXL; NIXL also needs a
`--transfer-port`. Inspect compiled backends with
`python -c "import radixshmem; print(radixshmem._data.TRANSFER_BACKENDS)"`.
The peer-reachable address can instead be derived using `--rpc-interface`.
Use distinct `--cluster-id` values when multiple clusters share etcd. Keep an
increased `--bootstrap-timeout` below FlexKV's 600-second ready timeout. Cluster
nodes publish and compare geometry through etcd; a node waiting for geometry
may adopt one already published by another node.

### Two nodes on one machine for testing

Use different `--name` and `--node-name` values, `--rpc-address 127.0.0.1`,
a separate `FLEXKV_RADIXSHMEM_SERVER_NAME` for each FlexKV process, and distinct
`FLEXKV_SERVER_RECV_PORT` values.

### Multiple engines on one node

Engines sharing a radix-server must use the same server name and compatible
model, page size and SWA geometry. Set `FLEXKV_INSTANCE_NUM=2` for both, and
`FLEXKV_INSTANCE_ID=0` and `1` respectively. Multiple instances use
server-client mode; client 0 starts the embedded KVServer unless an external
server is used with `FLEXKV_SERVER_LAUNCH_MODE=external`. Its TE waits for the expected GPU registrations, so both engines
must start. Multi-instance with node-local DP (multi-node DP attention) is not
supported; deployment must avoid this combination.

## Names

| Object | Name |
| --- | --- |
| Index shared memory | `--name`; cluster mode adds `_<node_name>` internally, while clients use `--name`. |
| SlotStore shared memory | `<name>_data`, unless changed by `--data-name`. |
| gRPC socket | Derived from the name under `/dev/shm`; keep the server's default `--endpoint`. |
| etcd keys | `radix/<cluster_id>/...`. |

## Errors

The server name must start with `/` and contain no whitespace. KVManager rejects
SSD, remote and FlexKV P2P combinations. Geometry construction rejects a
non-BLOCKFIRST CPU layout and an enabled SWA window smaller than one block.
Attach reports unreachable or not-ready servers after the 600-second budget;
server configuration failures and incompatible geometry are reported as errors.
In cluster mode index attachment retries unreachable RHT shards until the ready
budget expires. Check the server's `last_error` for bootstrap and sizing failures.
