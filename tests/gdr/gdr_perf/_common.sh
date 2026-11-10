#!/usr/bin/env bash
# Shared helpers for some special machine types and scenarios XDR / GPUDirect-RDMA tests.
# Source this file; do not execute it directly.

die()
{
    echo "ERROR: $*" >&2
    exit 1
}

apply_special_platform_env()
{
    # XPU Direct RDMA (Kunlunxin equivalent of NVIDIA GDR).
    export BKCL_ENABLE_XDR="${BKCL_ENABLE_XDR:-1}"
    export BKCL_RDMA_VERBS="${BKCL_RDMA_VERBS:-1}"
    export BKCL_TREE_THRESHOLD="${BKCL_TREE_THRESHOLD:-1048576}"
    export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-OAM_ID}"

    # some special machine types and scenarios XMLIR enumerates devices via XPU_VISIBLE_DEVICES. Leaving this
    # unset (or importing torch.cuda without it) can abort with
    #   device=1, num_gpus=
    # These tests do not import torch; the vars still match klx launch scripts.
    export XPU_VISIBLE_DEVICES="${XPU_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

    # Required for XPU HBM registration / XDR.
    export CUDA_ENABLE_P2P_NO_UVA="${CUDA_ENABLE_P2P_NO_UVA:-1}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-${GID_INDEX:-3}}"
    export GID_INDEX="${GID_INDEX:-3}"

    export MLX5_SCATTER_TO_CQE="${MLX5_SCATTER_TO_CQE:-0}"

    # XPU runtime (libxpurt), not NVIDIA CUDA.
    local pyver xmlir
    pyver="python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.10)"
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        xmlir="${CONDA_PREFIX}/lib/${pyver}/site-packages/torch_xmlir"
        export LD_LIBRARY_PATH="${xmlir}:${xmlir}/xre/so:${CONDA_PREFIX}/lib/${pyver}/site-packages/xtorch_ops:${LD_LIBRARY_PATH:-}"
    fi
}

list_rdma_devs()
{
    if [[ -d /sys/class/infiniband ]]; then
        ls /sys/class/infiniband 2>/dev/null | tr ' ' '\n'
    fi
}

first_rdma_dev()
{
    if [[ -n "${RDMA_DEV:-}" ]]; then
        printf '%s\n' "$RDMA_DEV"
        return
    fi
    if [[ -n "${IB_DEV:-}" ]]; then
        printf '%s\n' "$IB_DEV"
        return
    fi

    local dev gpu="${GPU_ID:-0}"
    # This some special machine types and scenarios topology is 1:1: XPU i <-> mlx5_bond_i <-> bond i.
    for dev in "mlx5_bond_${gpu}" "mlx5_${gpu}" mlx5_bond_0 mlx5_0 mlx5_1; do
        if [[ -e "/sys/class/infiniband/${dev}" ]]; then
            printf '%s\n' "$dev"
            return
        fi
    done

    dev="$(list_rdma_devs | head -n1 || true)"
    [[ -n "$dev" ]] || die "No RDMA device found under /sys/class/infiniband"
    printf '%s\n' "$dev"
}

ibdev_to_netdev()
{
    local ibdev="$1"

    if command -v ibdev2netdev >/dev/null 2>&1; then
        ibdev2netdev 2>/dev/null | awk -v d="$ibdev" '
            $1 == d {
                for (i = 1; i <= NF; i++) {
                    if ($i == "==>") {
                        print $(i + 1)
                        exit
                    }
                }
            }
        '
        return
    fi

    local netdev
    for netdev in /sys/class/infiniband/"$ibdev"/device/net/*; do
        if [[ -e "$netdev" ]]; then
            basename "$netdev"
            return
        fi
    done
}

roce_ip_for_dev()
{
    local ibdev="$1"
    local netdev ip

    if [[ -n "${LOOPBACK_IP:-}" ]]; then
        printf '%s\n' "$LOOPBACK_IP"
        return
    fi

    netdev="$(ibdev_to_netdev "$ibdev")"
    if [[ -n "$netdev" ]]; then
        ip="$(
            ip -4 addr show dev "$netdev" 2>/dev/null \
                | awk '/inet / {print $2}' \
                | cut -d/ -f1 \
                | head -n1
        )"
        if [[ -n "$ip" ]]; then
            printf '%s\n' "$ip"
            return
        fi
    fi

    # Bonded devices: mlx5_bond_0 -> bond0
    local suffix="${ibdev##*_}"
    if ip link show "bond${suffix}" >/dev/null 2>&1; then
        ip="$(
            ip -4 addr show dev "bond${suffix}" 2>/dev/null \
                | awk '/inet / {print $2}' \
                | cut -d/ -f1 \
                | head -n1
        )"
        if [[ -n "$ip" ]]; then
            printf '%s\n' "$ip"
            return
        fi
    fi

    return 1
}

build_test_xdr()
{
    local src="$1"
    local bin="$2"
    local cc="${CC:-cc}"

    [[ -f "$src" ]] || die "missing $src"
    if [[ -x "$bin" && "$bin" -nt "$src" ]]; then
        return 0
    fi
    echo "[compile] $cc -O2 -Wall $src -o $bin -libverbs -ldl"
    if ! "$cc" -O2 -Wall "$src" -o "$bin" -libverbs -ldl; then
        die "compile failed. Install libibverbs-dev (infiniband/verbs.h) and retry."
    fi
}

prefer_xmlir_cudart()
{
    [[ -n "${CONDA_PREFIX:-}" ]] || return 0
    local pyver xmlir_cudart
    pyver="python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3.10)"
    xmlir_cudart="${CONDA_PREFIX}/lib/${pyver}/site-packages/torch_xmlir/xre/so/libcudart.so"
    if [[ -f "$xmlir_cudart" ]]; then
        export XMLIR_CUDART="$xmlir_cudart"
    fi
}

bond_slaves()
{
    local ibdev="$1"
    local netdev slave

    if [[ -n "${SLAVE0:-}" || -n "${SLAVE1:-}" ]]; then
        printf '%s\n' ${SLAVE0:-} ${SLAVE1:-}
        return
    fi

    netdev="$(ibdev_to_netdev "$ibdev" || true)"
    if [[ -z "$netdev" ]]; then
        local suffix="${ibdev##*_}"
        if ip link show "bond${suffix}" >/dev/null 2>&1; then
            netdev="bond${suffix}"
        fi
    fi

    if [[ -n "$netdev" && -f "/sys/class/net/${netdev}/bonding/slaves" ]]; then
        for slave in $(cat "/sys/class/net/${netdev}/bonding/slaves"); do
            printf '%s\n' "$slave"
        done
        return
    fi

    printf '%s\n' eth1 eth2
}

list_xpus()
{
    if command -v xpu-smi >/dev/null 2>&1; then
        xpu-smi -L 2>/dev/null || xpu-smi info 2>/dev/null || true
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null || true
        return
    fi
    echo "(no xpu-smi / nvidia-smi; not importing torch on some special machine types and scenarios XMLIR)"
}

counter()
{
    local dev="$1"
    local key="$2"

    ethtool -S "$dev" 2>/dev/null |
    awk -F': ' -v key="$key" '
    {
        name=$1
        gsub(/^[ \t]+|[ \t]+$/, "", name)
        if (name == key) {
            value=$2
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            found=1
        }
    }
    END {
        if (found)
            print value
        else
            print 0
    }'
}

snapshot_nics()
{
    local prefix="$1"
    shift
    local dev
    for dev in "$@"; do
        [[ -e "/sys/class/net/${dev}" ]] || continue
        counter "$dev" tx_bytes   > "${prefix}.${dev}.tx"
        counter "$dev" rx_bytes   > "${prefix}.${dev}.rx"
        counter "$dev" tx_packets > "${prefix}.${dev}.txpkt"
        counter "$dev" rx_packets > "${prefix}.${dev}.rxpkt"
    done
}

show_slave_deltas()
{
    local before="$1"
    local after="$2"
    shift 2
    local dev
    echo "---------- physical slave delta ----------"
    for dev in "$@"; do
        show_delta "$before" "$after" "$dev"
    done
}

show_delta()
{
    local before="$1"
    local after="$2"
    local dev="$3"

    [[ -f "${before}.${dev}.tx" ]] || return 0

    local tx0 rx0 txp0 rxp0
    local tx1 rx1 txp1 rxp1

    tx0=$(cat "${before}.${dev}.tx")
    rx0=$(cat "${before}.${dev}.rx")
    txp0=$(cat "${before}.${dev}.txpkt")
    rxp0=$(cat "${before}.${dev}.rxpkt")

    tx1=$(cat "${after}.${dev}.tx")
    rx1=$(cat "${after}.${dev}.rx")
    txp1=$(cat "${after}.${dev}.txpkt")
    rxp1=$(cat "${after}.${dev}.rxpkt")

    local dtx=$((tx1 - tx0))
    local drx=$((rx1 - rx0))
    local dtxp=$((txp1 - txp0))
    local drxp=$((rxp1 - rxp0))

    awk \
        -v dev="$dev" \
        -v tx="$dtx" \
        -v rx="$drx" \
        -v txp="$dtxp" \
        -v rxp="$drxp" \
        'BEGIN {
            printf "%-8s TX=%8.3f GiB RX=%8.3f GiB TXpkt=%d RXpkt=%d\n",
                   dev,
                   tx / 1024 / 1024 / 1024,
                   rx / 1024 / 1024 / 1024,
                   txp,
                   rxp
        }'
}

parse_xdr_bw()
{
    awk '
    $1 ~ /^[0-9]+$/ && NF >= 6 {
        line=$0
    }
    END {
        print line
    }
    ' "$1"
}

# SIZE_LIST / QP_LIST / TX_DEPTH_LIST / ITERS / BIN / RDMA_DEV / GPU_ID /
# GID_INDEX / RESULT_DIR / CSV_FILE / HOST_ONLY=0|1
run_xdr_sweep()
{
    local opcode="$1"
    local extra=()
    local size qp depth log rc line bw mpps total

    [[ "${HOST_ONLY:-0}" == 1 ]] && extra+=(-H)

    mkdir -p "${RESULT_DIR}"
    echo "size_bytes,qps,tx_depth,total_outstanding,bw_gbps,msg_rate_mpps" \
        > "${CSV_FILE}"

    for size in $SIZE_LIST; do
        for qp in $QP_LIST; do
            for depth in $TX_DEPTH_LIST; do
                total=$((qp * depth))
                log="${RESULT_DIR}/size_${size}_qp_${qp}_d_${depth}.log"

                echo
                echo "============================================================"
                echo "opcode=${opcode}  size=${size}  qp=${qp}  tx_depth=${depth}"
                echo "============================================================"

                set +e
                "$BIN" \
                    -d "$RDMA_DEV" \
                    -g 0 \
                    -x "$GID_INDEX" \
                    -s "$size" \
                    -n "$ITERS" \
                    -q "$qp" \
                    -D "$depth" \
                    "${extra[@]}" \
                    "$opcode" \
                    >"$log" 2>&1
                rc=$?
                set -e

                if [[ "$rc" -ne 0 ]]; then
                    echo "ERROR: rc=$rc"
                    tail -n 30 "$log" || true
                    echo "${size},${qp},${depth},${total},ERROR,ERROR" >> "${CSV_FILE}"
                    continue
                fi

                line="$(parse_xdr_bw "$log")"
                if [[ -z "$line" ]]; then
                    echo "WARNING: cannot parse BW from $log"
                    tail -n 20 "$log" || true
                    echo "${size},${qp},${depth},${total},ERROR,ERROR" >> "${CSV_FILE}"
                    continue
                fi

                bw="$(echo "$line" | awk '{print $(NF-1)}')"
                mpps="$(echo "$line" | awk '{print $NF}')"
                echo "BW      : ${bw} Gbps"
                echo "MsgRate : ${mpps} Mpps"
                echo "${size},${qp},${depth},${total},${bw},${mpps}" >> "${CSV_FILE}"
            done
        done
    done
}
