#!/bin/bash
# The operator's radix-server for one node, no cluster: nothing about the model
# on the command line. FlexKV's first client brings the geometry (page size,
# bytes per block, SWA page + window); the server plans the slot counts from the
# budget and FlexKV adopts them (docs/radixshmem/config_zh.md section 3).
#   --swa-ratio is needed only for models with an SWA pool (DeepSeek-V4 etc.).
set -eu
radix-server --name "${NAME:-/flexkv}" \
  --data-bytes "${DATA_BYTES:-64G}" \
  ${SWA_RATIO:+--swa-ratio "$SWA_RATIO"} \
  ${HUGEPAGE_PATH:+--hugepage-path "$HUGEPAGE_PATH"}
