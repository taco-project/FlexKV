#!/usr/bin/env bash
set -euo pipefail

# Query RDMA device limits for outstanding RDMA READ / ATOMIC.
# Same as AAA_MyWorkspace/examples/check_rdma_read_limit.sh.
# Use this to size -D / TX_DEPTH_LIST for the XDR benches.

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh"

apply_special_platform_env

TMP_C="$(mktemp /tmp/check_rdma_caps.XXXXXX.c)"
TMP_BIN="${TMP_C%.c}"

cleanup() {
    rm -f "${TMP_C}" "${TMP_BIN}"
}
trap cleanup EXIT

cat > "${TMP_C}" <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <infiniband/verbs.h>

int main(void)
{
    int num_devices = 0;
    struct ibv_device **dev_list = ibv_get_device_list(&num_devices);

    if (!dev_list) {
        perror("ibv_get_device_list");
        return 1;
    }

    if (num_devices == 0) {
        fprintf(stderr, "No RDMA devices found.\n");
        ibv_free_device_list(dev_list);
        return 1;
    }

    printf("Found %d RDMA device(s)\n\n", num_devices);

    for (int i = 0; i < num_devices; ++i) {
        const char *name = ibv_get_device_name(dev_list[i]);

        struct ibv_context *ctx = ibv_open_device(dev_list[i]);
        if (!ctx) {
            fprintf(stderr, "Failed to open device: %s\n", name);
            continue;
        }

        struct ibv_device_attr attr;
        if (ibv_query_device(ctx, &attr)) {
            fprintf(stderr, "ibv_query_device failed: %s\n", name);
            ibv_close_device(ctx);
            continue;
        }

        printf("========================================\n");
        printf("Device: %s\n", name);
        printf("FW:     %s\n", attr.fw_ver);
        printf("----------------------------------------\n");
        printf("max_qp_init_rd_atom : %d\n", attr.max_qp_init_rd_atom);
        printf("max_qp_rd_atom      : %d\n", attr.max_qp_rd_atom);
        printf("max_res_rd_atom     : %d\n", attr.max_res_rd_atom);
        printf("max_qp_wr           : %d\n", attr.max_qp_wr);
        printf("max_sge_rd          : %d\n", attr.max_sge_rd);
        printf("\n");
        printf("Interpretation:\n");
        printf("  Initiator outstanding RDMA Read/Atomic per QP <= %d\n",
               attr.max_qp_init_rd_atom);
        printf("  Responder RDMA Read/Atomic resources per QP   <= %d\n",
               attr.max_qp_rd_atom);
        printf("\n");

        ibv_close_device(ctx);
    }

    ibv_free_device_list(dev_list);
    return 0;
}
EOF

echo "[1/2] Building..."

if ! command -v gcc >/dev/null 2>&1; then
    echo "ERROR: gcc not found"
    exit 1
fi

if ! pkg-config --exists libibverbs 2>/dev/null; then
    echo "WARNING: pkg-config cannot find libibverbs."
    echo "Trying gcc -libverbs directly..."
    gcc -O2 "${TMP_C}" -o "${TMP_BIN}" -libverbs
else
    gcc -O2 "${TMP_C}" -o "${TMP_BIN}" \
        $(pkg-config --cflags --libs libibverbs)
fi

echo "[2/2] Querying RDMA devices..."
echo
"${TMP_BIN}"
