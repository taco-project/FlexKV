#!/bin/bash

# ============================================================================
# 停止所有 FlexKV vLLM Serving 节点
# ============================================================================

echo "============================================================================"
echo " 停止所有 vLLM Serving 节点"
echo "============================================================================"

# 查找所有相关进程
echo ""
echo "正在查找 vLLM serving 进程..."

# 查找主进程
main_pids=$(ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print $2}')

if [ -z "$main_pids" ]; then
    echo "未找到运行中的 vLLM serving 主进程"
else
    echo "找到主进程:"
    ps aux | grep 'vllm.entrypoints.openai.api_server' | grep -v grep | awk '{print "  PID: "$2" - "$11" "$12" "$13}'
fi

# 查找所有相关的子进程
echo ""
echo "查找所有相关的子进程..."
all_vllm_pids=$(ps aux | grep -E '(vllm|VllmWorker|VLLM::Engine)' | grep -v grep | awk '{print $2}')

if [ -n "$all_vllm_pids" ]; then
    echo "找到 $(echo "$all_vllm_pids" | wc -w) 个相关进程"
fi

# 停止进程
echo ""
echo "============================================================================"
echo " 开始停止进程..."
echo "============================================================================"

# 方法1：优雅地停止主进程（这会尝试停止子进程）
if [ -n "$main_pids" ]; then
    echo ""
    echo "步骤 1/4: 发送 SIGTERM 信号给主进程..."
    for pid in $main_pids; do
        echo "  停止 PID: $pid"
        kill -15 $pid 2>/dev/null
    done
    
    # 等待进程响应
    echo "  等待 5 秒..."
    sleep 5
fi

# 方法2：强制停止所有 vLLM 相关进程
echo ""
echo "步骤 2/4: 强制停止所有 vLLM 相关进程..."
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null && echo "  已停止主进程"
pkill -9 -f 'VllmWorker' 2>/dev/null && echo "  已停止 VllmWorker 进程"
pkill -9 -f 'VLLM::Engine' 2>/dev/null && echo "  已停止 VLLM::Engine 进程"

sleep 2

# 方法3：停止剩余的 Python 进程（与 vLLM 相关的）
echo ""
echo "步骤 3/4: 检查剩余的相关进程..."
remaining_pids=$(ps aux | grep -E '(vllm|VllmWorker|VLLM)' | grep -v grep | awk '{print $2}')

if [ -n "$remaining_pids" ]; then
    echo "  发现剩余进程，强制停止..."
    for pid in $remaining_pids; do
        echo "    强制停止 PID: $pid"
        kill -9 $pid 2>/dev/null
    done
    sleep 2
else
    echo "  ✓ 没有剩余进程"
fi

# 方法4：清理僵尸进程（通过杀死父进程）
echo ""
echo "步骤 4/4: 检查僵尸进程..."
zombie_count=$(ps aux | grep -E 'defunct|Z' | grep -v grep | wc -l)
if [ $zombie_count -gt 0 ]; then
    echo "  发现 $zombie_count 个僵尸进程"
    echo "  僵尸进程通常会在父进程清理后自动消失"
    
    # 查找僵尸进程的父进程
    zombie_parents=$(ps -eo pid,ppid,stat,cmd | grep -E ' Z ' | awk '{print $2}' | sort -u)
    if [ -n "$zombie_parents" ]; then
        echo "  尝试停止僵尸进程的父进程..."
        for ppid in $zombie_parents; do
            if [ "$ppid" != "1" ]; then
                echo "    停止父进程 PID: $ppid"
                kill -9 $ppid 2>/dev/null
            fi
        done
    fi
else
    echo "  ✓ 没有僵尸进程"
fi

# 最终检查
echo ""
echo "============================================================================"
echo " 最终状态检查"
echo "============================================================================"
sleep 2

final_check=$(ps aux | grep -E '(vllm|VllmWorker|VLLM::Engine)' | grep -v grep | wc -l)
if [ $final_check -eq 0 ]; then
    echo "✓ 所有 vLLM 进程已成功停止"
else
    echo "⚠ 警告: 仍有 $final_check 个相关进程在运行"
    echo ""
    echo "剩余进程列表:"
    ps aux | grep -E '(vllm|VllmWorker|VLLM)' | grep -v grep | awk '{print "  PID "$2": "$11" "$12" "$13}'
    echo ""
    echo "如需手动清理，运行:"
    echo "  kill -9 \$(ps aux | grep -E '(vllm|VllmWorker|VLLM)' | grep -v grep | awk '{print \$2}')"
fi

# 显示清理建议
echo ""
echo "============================================================================"
echo " 清理临时文件"
echo "============================================================================"
echo ""
echo "手动清理命令:"
echo "  # 清理 IPC socket 文件"
echo "  rm -rf /tmp/flexkv_server_*"
echo ""
echo "  # 清理 SSD 缓存"
echo "  rm -rf /workspace/FlexKV/flexkv_ssd*/"
echo ""
echo "  # 清理配置文件（可选）"
echo "  cd /raid/fly/FlexKV/multi-nodes"
echo "  rm -f node*.yml mooncake_config_node*.json"
echo ""
echo "  # 清理 Redis（可选）"
echo "  redis-cli -h 10.6.131.12 -p 6379 -a redis-serving-passwd FLUSHALL"
echo "  redis-cli -h 10.6.131.12 -p 6380 -a redis-serving-passwd FLUSHALL"
echo ""
echo "============================================================================"
