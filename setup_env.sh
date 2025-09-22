#!/bin/bash
# FlexKV 环境设置脚本
# 设置 FlexKV 运行所需的环境变量

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXKV_ROOT="$SCRIPT_DIR"

# 设置库路径
export LD_LIBRARY_PATH="$FLEXKV_ROOT/build/lib:$FLEXKV_ROOT/flexkv/lib:$LD_LIBRARY_PATH"

# 自动检测并加入 PyTorch 的 C++ 动态库目录（包含 libc10.so 等）
PY_BIN=python3
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  PY_BIN=python
fi

# 尝试多种方式检测 PyTorch 库路径
TORCH_LIB_DIR=""
if command -v "$PY_BIN" >/dev/null 2>&1; then
  TORCH_LIB_DIR="$($PY_BIN -c 'import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null)"
fi

# 如果上述方法失败，尝试常见的 PyTorch 安装路径
if [ -z "$TORCH_LIB_DIR" ] || [ ! -d "$TORCH_LIB_DIR" ]; then
  for torch_path in \
    "$HOME/.local/lib/python3.6/site-packages/torch/lib" \
    "/usr/local/lib/python3.6/site-packages/torch/lib" \
    "/usr/lib/python3.6/site-packages/torch/lib" \
    "$HOME/.local/lib/python3.7/site-packages/torch/lib" \
    "/usr/local/lib/python3.7/site-packages/torch/lib"; do
    if [ -d "$torch_path" ]; then
      TORCH_LIB_DIR="$torch_path"
      break
    fi
  done
fi

if [ -n "$TORCH_LIB_DIR" ] && [ -d "$TORCH_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$LD_LIBRARY_PATH"
  echo "  PyTorch 库路径: $TORCH_LIB_DIR"
else
  echo "Warning: 未能检测到 PyTorch 动态库目录，flexkv.c_ext 可能无法导入 (缺少 libc10.so)。" 1>&2
fi

# 自动检测并加入“已安装”的 FlexKV 动态库目录（包含 libxxhash.so）
FLEXKV_SITE_LIB=""
if command -v "$PY_BIN" >/dev/null 2>&1; then
  FLEXKV_SITE_LIB="$($PY_BIN -c 'import os, sys;\
try:\n import flexkv;\n d=os.path.join(os.path.dirname(flexkv.__file__), "lib");\n print(d)\nexcept Exception as e:\n print("")' 2>/dev/null)"
fi

if [ -n "$FLEXKV_SITE_LIB" ] && [ -d "$FLEXKV_SITE_LIB" ]; then
  # 避免重复添加；无论是否与本地 $FLEXKV_ROOT/flexkv/lib 相同，都加入以兼容“仅安装包存在 .so”的情况
  export LD_LIBRARY_PATH="$FLEXKV_SITE_LIB:$LD_LIBRARY_PATH"
  echo "  FlexKV 安装库路径: $FLEXKV_SITE_LIB"
fi

# 设置 Python 路径
export PYTHONPATH="$FLEXKV_ROOT:${PYTHONPATH:-}"

echo "FlexKV 环境变量已设置："
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""
echo "现在可以运行 FlexKV 测试程序了！"
