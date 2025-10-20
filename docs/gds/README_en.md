# FlexKV GDS (GPU Direct Storage) User Guide

This document explains how to configure and use GPU Direct Storage (GDS) functionality in FlexKV to enable direct data transfer between GPU and SSD.

---

## Table of Contents

- [1. Environment Setup and Verification](#1-environment-setup-and-verification)
- [2. Using GDS in FlexKV](#2-using-gds-in-flexkv)
- [3. Troubleshooting and Debugging](#3-troubleshooting-and-debugging)
- [4. References](#4-references)

---

## 1. Environment Setup and Verification

### 1.1 Install GDS Driver

```bash
# Install NVIDIA GDS package
sudo apt install nvidia-gds

# Load GDS kernel module
sudo modprobe nvidia_fs

# Verify module is loaded
lsmod | grep nvidia_fs

# Set device permissions
sudo chmod 666 /dev/nvidia-fs*
```

### 1.2 Configure File System

GDS requires file systems to be mounted with `data=ordered mode` (ext4).

#### Check Current Mount Status

```bash
# View current filesystem mount information
mount | grep -E "ext4"

# Check if mounted in ordered mode
dmesg | grep "mounted filesystem"
```

#### Configure ext4 Filesystem (Recommended)

```bash
# Method 1: Temporary remount
sudo mount -o remount,data=ordered /path/to/mount/point

# Method 2: Permanent configuration (edit /etc/fstab)
sudo vim /etc/fstab
# Add or modify mount options:
# /dev/nvme0n1  /mnt/nvme0  ext4  rw,relatime,data=ordered  0  2

# Apply configuration
reboot

# Verify mount options
mount | grep nvme
```

### 1.3 Verify GDS Functionality

#### Use Official Tools for Verification

```bash
# Run GDS check tool
/usr/local/cuda/gds/tools/gdscheck.py -p
```

#### Test GDS Read/Write Performance

```bash
# Generate test file (1GB)
/usr/local/cuda/gds/tools/gdsio -D ./ -w 1 -I 1 -x 1 -s 1G -i 1M

# Perform read/write test using GPU 0
/usr/local/cuda/gds/tools/gdsio -D ./ -d 0 -w 1 -s 1G -i 1M -I 0 -x 0
```

## 2. Using GDS in FlexKV

### 2.1 Running in Docker

To use GDS in a Docker container, special configuration is required:

```bash
docker run -itd \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --gpus=all \
    --network=host \
    --privileged \
    --cap-add=CAP_SYS_PTRACE \
    --cap-add=IPC_LOCK \
    -v /run/udev:/run/udev \
    -v /dev:/dev \
    -v /mnt:/mnt \
    --name flexkv \
    nvcr.io/nvidia/pytorch:24.05-py3 \
    /bin/bash
```

### 2.2 Configure FlexKV to Use GDS

Configuration example after compilation `config.json`:

```json
{
    "cache_config": {
          "enable_ssd": False,
          "enable_gds": True,
          "num_gds_blocks": 10000000,
          "gds_cache_dir": ["./gdstest"]
    },
}
```

---

## 3. Troubleshooting and Debugging

### 3.1 Enable GDS Logging

Edit `/etc/cufile.json`:

```json
{
    ...
    "logging": {
        "dir": "/path/to/log_dir/",
        "level": "INFO"
    },
    ...
    "properties": {
        "allow_compat_mode": false,
    },
}
```
**Important Notes:**
- `allow_compat_mode`: Set to `false` to force GDS usage, `true` to fall back to compatibility mode (via CPU) when GDS is unavailable
- `logging.level`: Options are `ERROR|WARN|INFO|DEBUG|TRACE`, adjust for troubleshooting

---


### 3.2 Common Issues

#### Issue 1: GDS Falls Back to Compatibility Mode

**Check Method:**
```bash
# View cuFile logs
tail -f /var/log/cufile/cufile.log

# If you see "using compatibility mode", GDS is not working properly
```

---

## 4. References

### Official Documentation

- [NVIDIA GPUDirect Storage Official Documentation](https://docs.nvidia.com/gpudirect-storage/)
- [GDS Installation Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#installing-gpudirect-storage)
- [GDS Filesystem Requirements](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mounting-a-local-file-system-for-gds)
- [GDS Supported GPU Models](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#supported-gpus)

---
