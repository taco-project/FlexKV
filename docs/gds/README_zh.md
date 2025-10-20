# FlexKV GDS (GPU Direct Storage) 使用指南

本文档介绍如何在 FlexKV 中配置和使用 GPU Direct Storage (GDS) 功能，实现 GPU 与 SSD 之间的直接数据传输。

---

## 目录

- [一、环境配置与验证](#一环境配置与验证)
- [二、在 FlexKV 中使用 GDS](#二在-flexkv-中使用-gds)
- [三、故障排查与调试](#三故障排查与调试)
- [四、参考资料](#四参考资料)

---

## 一、环境配置与验证

### 1.1 安装 GDS 驱动

```bash
# 安装 NVIDIA GDS 包
sudo apt install nvidia-gds

# 加载 GDS 内核模块
sudo modprobe nvidia_fs

# 验证模块已加载
lsmod | grep nvidia_fs

# 设置设备权限
sudo chmod 666 /dev/nvidia-fs*
```

### 1.2 配置文件系统

GDS 要求文件系统以 `data=ordered mode` 挂载（ext4）。

#### 检查当前挂载状态

```bash
# 查看当前文件系统挂载信息
mount | grep -E "ext4"

# 检查是否以 ordered 模式挂载
dmesg | grep "mounted filesystem"
```

#### 配置 ext4 文件系统（推荐）

```bash
# 方法 1：临时重新挂载
sudo mount -o remount,data=ordered /path/to/mount/point

# 方法 2：永久配置（编辑 /etc/fstab）
sudo vim /etc/fstab
# 添加或修改挂载选项：
# /dev/nvme0n1  /mnt/nvme0  ext4  rw,relatime,data=ordered  0  2

# 使配置生效
reboot

# 验证挂载选项
mount | grep nvme
```

### 1.3 验证 GDS 功能

#### 使用官方工具检查

```bash
# 运行 GDS 检查工具
/usr/local/cuda/gds/tools/gdscheck.py -p
```

#### 测试 GDS 读写性能

```bash
# 生成测试文件（1GB）
/usr/local/cuda/gds/tools/gdsio -D ./ -w 1 -I 1 -x 1 -s 1G -i 1M

# 使用 GPU 0 进行读写测试
/usr/local/cuda/gds/tools/gdsio -D ./ -d 0 -w 1 -s 1G -i 1M -I 0 -x 0
```
## 二、在 FlexKV 中使用 GDS

### 2.1 在 Docker 中运行

在 Docker 容器中使用 GDS，需要特殊配置：

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

### 2.2 配置 FlexKV 使用 GDS

编译后config例子 `config.json`：

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

## 三、故障排查与调试

### 3.1 启用 GDS 日志

编辑 `/etc/cufile.json`：

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
**重要说明：**
- `allow_compat_mode`: 设为 `false` 强制使用 GDS，`true` 则在 GDS 不可用时回退到兼容模式（通过 CPU）
- `logging.level`: 可选 `ERROR|WARN|INFO|DEBUG|TRACE`，用于故障排查时调整

---


### 3.2 常见问题排查

#### 问题 1: GDS 回退到兼容模式

**检查方法：**
```bash
# 查看 cuFile 日志
tail -f /var/log/cufile/cufile.log

# 如果看到 "using compatibility mode"，说明 GDS 未正常工作
```

---

## 四、参考资料

### 官方文档

- [NVIDIA GPUDirect Storage 官方文档](https://docs.nvidia.com/gpudirect-storage/)
- [GDS 安装指南](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#installing-gpudirect-storage)
- [GDS 文件系统要求](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#mounting-a-local-file-system-for-gds)
- [GDS 支持的 GPU 型号](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#supported-gpus)

---
