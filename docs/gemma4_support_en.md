# Gemma4 31B: heterogeneous KV groups in FlexKV

[中文文档](gemma4_support.md)

This document explains the original Gemma4 integration and its historical
validation. The described runs do not validate every current FlexKV/vLLM
revision, GPU, topology, backend or model layout. Revalidate the exact deployed
combination, including cache hits and output correctness.

## Background

The documented Gemma4 31B configuration has 50 sliding-window layers with
16 KV heads of dimension 256 and 10 full-attention layers with 4 KV heads of
dimension 512. At BF16 this is 16,384 and 8,192 bytes per token per respective
layer, about 900 KiB per token in aggregate. Some vLLM layers share a KV tensor;
registration deduplicates tensors by address.

The historical integration used Triton attention because head dimensions differ.
Its GPU layout `[num_blocks, 2, block_size, num_kv_heads, head_size]` differs
from the FlashAttention layout `[2, num_blocks, block_size, num_kv_heads,
head_size]`. The relevant backend, connector and cache-manager behavior depend on
the vLLM revision; check the deployed version rather than assuming these historical
choices apply to every version.

## FlexKV design

`LayerGroupSpec` in `flexkv/common/config.py` describes the geometry and original
layer indices of a group. `ModelConfig.layer_groups` carries the groups and the
cache-capacity calculation accounts for their actual geometry. For a BLOCKFIRST
multi-group host layout, each block contains the groups' layer data in sequence;
group offsets and strides describe access to each part. Do not assume one
uniform layer or KV stride for the complete heterogeneous block.

The vLLM worker registers deduplicated GPU tensors, group specifications, GPU
layouts and handles. The request/client protocol carries the registration to
TransferManager. After registration supplies the actual group geometry, the
storage engine can size CPU/SSD blocks based on that geometry. TransferEngine
passes group information to the GPU/CPU, CPU/SSD and GDS workers. The current
implementation uses a unified `GPUCPUTransferWorker` for single-GPU and TP;
see `flexkv/transfer/workers/gpu_cpu.py` and the other worker modules for the
current implementation rather than the historical pseudocode in the Chinese
record.

The original work also processed pending D2H completions before matching new
requests, so data whose transfer has completed can be published before a later
lookup. Actual task ownership and publication behavior must be checked against
the current `kvtask.py` and server implementation.

## Historical issues and evidence

The documented dp1 run exposed a GPU stride mismatch: one layout assumed K/V
and block axes in the opposite order from the actual Triton tensor. It was
corrected by deriving strides from the physical tensor. A second issue sized
host capacity from maximum dimensions before group registration, overestimating
per-token bytes. In that historical 16 GB configuration, accounting for the
groups changed the reported CPU capacity from 546 to 1191 blocks. A dp2 run
also exposed delayed processing of completed D2H operations when the scheduler
was idle. These figures and remedies describe that implementation and run,
not expected numbers for a different model or build.

## Reproduction and checks

The historical environment used two large-memory GPUs, a Gemma4 31B model,
vLLM with Gemma4 and KV connector support, and a built FlexKV extension. It
configured `FLEXKV_CPU_CACHE_GB=16`, `FLEXKV_SSD_CACHE_GB=0`, and the native
library path. The Chinese record contains the original TP1/DP1 and TP1/DP2
launch commands and example request. Paths and engine flags must be adapted to
the actual installation.

Send repeated requests and inspect group registration, calculated block count,
D2H completion, and H2D reload logs. Compare output against a suitable cold
baseline as well as repeated cache-reuse runs; a successful launch or two matching
responses alone does not establish byte or model correctness. The historical
`test_dp2_kv_reuse.sh` is not present in this repository and must not be assumed
to run from the documented path.

The code contains CPU/SSD and GDS multi-group paths, but their presence is not
end-to-end validation for every Gemma4 configuration. Validate the intended
backend, dtype, GPU, parallel topology and CPU layout separately, including
non-BLOCKFIRST layouts where applicable.
