# DeepSeek-V4 compress-state I/O in FlexKV

[中文文档](dsv4_compress_state_io_zh.md)

## Background and scope

DeepSeek-V4 DSA attention reads C4 KV and indexer KV as well as two runtime
score buffers: `compress_state_pools[i].kv_score_buffer.kv_score` and
`indexer_compress_state_pools[i].kv_score_buffer.kv_score`. Restoring KV alone
can leave stale or empty score state. FlexKV offloads and restores snapshots of
both states together with SWA KV; it does not recompute the states after restore.
Whether restored state has the desired semantics in subsequent decode still
requires the CP8 model-accuracy validation described below.

The KV storage layout is MLA-shaped (`kv_dim=1`, combined K/V buffer, one KV
head shared across TP ranks), while the attention computation is MHA. FlexKV
uses the storage layout to determine `kv_dim`.

## SWA pages and sidecar groups

The score addresses are determined by SWA physical pages and their ring sizes,
not by main-KV slot mapping. SGLang translates full-KV token slots using
`translate_loc_from_full_to_swa`. PUT and GET share one `swa_slot_mapping`, so
SWA KV, attention state and indexer state use the same physical page ID.

The connector discovers both state pools and checks consistent `ring_size`,
contiguous 2D tensors, per-group shape and dtype, equal SWA/FlexKV page sizes,
page-size divisibility by ring size, and enough state rows for all SWA pages.
Missing or incomplete pools and invalid layouts are rejected rather than
silently treated as valid state.

| Group | GPU data | Rows per FlexKV page | `compress_ratio` |
| --- | --- | ---: | ---: |
| SWA KV | `swa_kv_pool` | `swa_page_size` | 1 |
| C4 attention state | attention `kv_score` | `ring_size` | `swa_page_size / ring_size` |
| C4 indexer state | indexer `kv_score` | `ring_size` | `swa_page_size / ring_size` |

Each state group keeps its own dtype and head size. For a 256-token page and an
8-row ring, the state ratio is 32 and each page transfers eight score rows. The
connector passes matching `swa_layer_groups`, `swa_gpu_layouts`, and
`swa_handles_per_group`; the three registration fields must be supplied
together. Once groups are actually constructed, it sets the internal
`cache_config.swa.multi_group` flag.

## Configuration and page mapping

`swa_multi_group` defaults to enabled for the relevant DeepSeek-V4 path when
omitted. `true` enables SWA KV plus attention and indexer state; `false` keeps
SWA KV I/O but skips state registration and I/O. When using environment-backed
configuration instead of a configuration file, set `FLEXKV_SWA_MULTI_GROUP=0`
or `1`. The internal `swa.multi_group` flag only becomes true when state groups
were built successfully.

For PUT and GET, `_build_swa_slot_mapping` takes main token slots, translates
them to SWA slots, removes the unmapped slot-0 sentinel prefix, and passes the
CPU `int64` mapping to `kv_manager.launch(..., swa_slot_mappings=...)`. FlexKV
folds this into physical page IDs. The state groups share the SWA mapping.

## Registration and storage layout

The client serializes CUDA handles in `RegisterTPClientRequest`. TransferManager
collects the group handles and layouts, checks group specs across GPUs, arranges
them by `WorkerKey`, and passes them to StorageEngine and TransferEngine. The
uniform SWA path remains in use when the optional group fields are absent.

SWA multi-group requires `FLEXKV_CPU_LAYOUT=BLOCKFIRST`. Host storage represents
one page as a byte-flat block containing SWA KV, attention state and indexer
state. Its block byte size is calculated as the sum of each group's layers,
KV dimension, rows per page, heads, head size and dtype size, including the TP
layout. The second dimension of a byte-flat `KVCacheLayout.kv_shape` is already
in bytes and must not be multiplied by dtype size again.

GPU groups retain their own tensor dtype and layout. The GPU/CPU worker uses
per-group strides and host offsets for D2H and H2D. CPU, SSD and remote storage
use the same byte-flat page representation; CPU/SSD and CPU/remote I/O treat a
page as an opaque block. GDS uses GPU group metadata and the byte-flat SSD layout.

## PUT and GET ordering

A PUT allocates destination pages with `put_match`, binds main and translated
SWA slot mappings, and launches main-KV D2H and SWA/state D2H. SWA KV and its
states occupy the same SWA page block. They can then be written to lower tiers.

A GET obtains main and SWA/state source pages, binds destination slots, reads
from SSD or remote storage into host memory when needed, and copies each group
to the GPU using its registered offsets. Non-layerwise GET returns to SGLang
after the entire task completes.

With layerwise transfer enabled, main KV and SWA/state H2D are fused into the
layerwise worker. Relevant disk-to-host work precedes GPU restoration. For a C4
layer, its ready event is not sent until that layer's state has been written;
a layer without state waits on its main and SWA KV. The current implementation
has no separate `swa_multi_layer` switch. `swa_multi_group` still controls state
sidecar registration.

## Constraints and validation

- An SWA pool and configuration are required; state uses its physical page IDs.
- SWA and FlexKV page sizes must match, and page size must be divisible by ring size.
- CPU layout must be BLOCKFIRST; all three group registration fields must be present together.
- Group specs must agree across participating GPUs.
- Disable SGLang's `FLEXKV_P3_CLEAR` for restore testing; clearing after restore overwrites state.
- `swa_multi_group: false` is SWA-only. Missing state is not automatically reconstructed or replaced with zeros.

Check the `FlexKV-DSv4-State` preparation and group-registration logs and verify
that host bytes per block match the sum of the three groups. Compare page-level
checksums of both score buffers before GPU PUT, in the host block after D2H, and
after GPU H2D restore. `tests/test_swa_state_sidecars.py` exercises byte-flat
packing and layerwise dependency behavior. A CP8 fresh → flush → restore model
accuracy run is still needed for the deployed revisions; unit tests alone do
not establish model-output correctness.

Relevant code includes `flexkv/integration/sglang/connector.py`,
`flexkv/server/{client,request}.py`, `flexkv/transfer_manager.py`,
`flexkv/storage/storage_engine.py`, `flexkv/common/{storage,transfer}.py`,
`flexkv/transfer/transfer_engine.py`, `flexkv/transfer/workers/`, and
`flexkv/kvtask.py`. SGLang's matching adapter owns its scheduler/cache hooks.
