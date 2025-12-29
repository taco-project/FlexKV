# FlexKV NVMe-oF User Guide

NVMe-oF target offload enables DRAM-bounce-buffer-free datapath between any remote SSD and GPU. This document explains how to configure and enable this functionality in FlexKV.

## 1. Environment Setup and Verification

### 1.1 Make sure OS kernel is qualified

The OS kernel should be 

```bash
$ cat /boot/config-$(uname -r) | grep P2PDMA
CONFIG_PCI_P2PDMA=y
$ cat /boot/config-$(uname -r) | grep NVME
…
CONFIG_NVME_FABRICS=m
CONFIG_NVME_RDMA=m
…
CONFIG_NVME_TARGET=m
CONFIG_NVME_TARGET_PASSTHRU=y
CONFIG_NVME_TARGET_LOOP=m
CONFIG_NVME_TARGET_RDMA=m
…
```

### 1.2 Install [libnvme](https://github.com/linux-nvme/libnvme)

### 1.3 Install [nvme-cli](https://github.com/linux-nvme/nvme-cli)

```bash
# Verify successful installation
$ nvme version
nvme version 2.16 (git 2.16)
libnvme version 1.16 (git 1.16)
$ nvme list
Node                  Generic               SN  Model  Namespace  Usage                      Format           FW Rev
--------------------- --------------------- --- ------ ---------- -------------------------- ---------------- -------
/dev/nvme0n1          /dev/ng0n1            …   …      0x1        899.54  GB /   6.40  TB      4 KiB +  0 B   …
/dev/nvme1n1          /dev/ng1n1            …   …      0x1        898.72  GB /   6.40  TB      4 KiB +  0 B   …
/dev/nvme2n1          /dev/ng2n1            …   …      0x1        902.14  GB /   6.40  TB      4 KiB +  0 B   …
/dev/nvme3n1          /dev/ng3n1            …   …      0x1        900.79  GB /   6.40  TB      4 KiB +  0 B   …
```

### 1.4 Install [DOCA](https://developer.nvidia.com/networking/doca)

Assume servers are equipped with NVIDIA [ConnectX](https://www.nvidia.com/en-us/networking/ethernet-adapters/)-6 Dx or newer NICs. We recommend installing [DOCA-Host](https://docs.nvidia.com/doca/sdk/doca-host-installation-and-upgrade/) with [doca-all profile](https://docs.nvidia.com/doca/sdk/doca-profiles/).

> [!NOTE]
> NVIDIA transitioned from MLNX_OFED to DOCA_OFED, part of DOCA-Host. It is no longer necessary to append the `--with-nvmf` flag when installing DOCA-Host.

### 1.5 Configure extra I/O queues for NVMe devices

```bash
$ modprobe nvme num_p2p_queues=1

# Verify queue configuration
$ cat /sys/module/nvme/parameters/num_p2p_queues
1
$ cat /sys/block/<nvme>/device/num_p2p_queues # E.g., nvme0n1
1
```

### 1.6 Load NVMe target modules

```bash
$ modprobe nvmet
$ modprobe nvmet-rdma
```

> [!TIP]
> For the sake of performance, explicitly specify `offload_mem_start`, `offload_mem_size` and `offload_buffer_size` parameters of module `nvmet_rdma`.


### 1.7 Configure devices for NVMe-oF target offload

As shown in setp [1.3](#13-install-nvme-cli), data disks are `/dev/nvme0n1`, `/dev/nvme1n1`, `/dev/nvme2n1` and `/dev/nvme3n1`. Suppose they form a RAID0 group, then all of them participate in KV caching and should be enabled for NVMe-oF target offload.

```bash
$ export FLEXKV_ENABLED_NVMET="nvme0n1,nvme1n1,nvme2n1,nvme3n1"
```

> [!WARNING]
> NVMe device names do not persist across reboots. FlexKV will automatically identify NVMe devices and NICs by their PCIe addresses at configuration time.

## 2. Using NVMe-oF in FlexKV

## References

- [Future-Proof Your Networking Stack with NVIDIA DOCA-OFED](https://developer.nvidia.com/blog/future-proof-your-networking-stack-with-nvidia-doca-ofed/)
- [HowTo Configure NVMe over Fabrics (NVMe-oF) Target Offload](https://enterprise-support.nvidia.com/s/article/howto-configure-nvme-over-fabrics--nvme-of--target-offload)
- [spdk/spdk Issue #900](https://github.com/spdk/spdk/issues/900)