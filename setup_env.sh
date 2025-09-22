#!/bin/bash
# FlexKV 环境设置脚本
# 设置 FlexKV 运行所需的环境变量

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLEXKV_ROOT="$(dirname "$SCRIPT_DIR")"

# 设置库路径
export LD_LIBRARY_PATH="$FLEXKV_ROOT/build/lib:$FLEXKV_ROOT/flexkv/lib:$LD_LIBRARY_PATH"

# 设置 Python 路径
export PYTHONPATH="$FLEXKV_ROOT:$PYTHONPATH"

echo "FlexKV 环境变量已设置："
echo "  LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""
echo "现在可以运行 FlexKV 测试程序了！"
