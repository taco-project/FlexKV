#!/usr/bin/env python
"""调试 stop() 方法卡住的问题"""

import time
import sys
import torch

print("=== 开始调试 stop() 方法 ===")

# 导入必要的模块
print("1. 导入模块...")
from flexkv.cache.radix_remote import LocalRadixTree, DistributedRadixTree
from flexkv.cache.redis_meta import RedisMeta
print("   [OK] 模块导入成功")

# 创建 RedisMeta
print("\n2. 创建 RedisMeta...")
redis_meta = RedisMeta(host="127.0.0.1", port=6379, local_ip="127.0.0.1")
print("   [OK] RedisMeta 创建成功")

print("\n3. 初始化 RedisMeta...")
node_id = redis_meta.init_meta()
if node_id is None:
    print("   [ERROR] RedisMeta 初始化失败")
    sys.exit(1)
print(f"   [OK] RedisMeta 初始化成功，node_id={node_id}")

# 创建 LocalRadixTree
print("\n4. 创建 LocalRadixTree...")
local_tree = LocalRadixTree(
    tokens_per_block=4,
    max_num_blocks=1000,
    lease_ttl_ms=10000,
    renew_lease_ms=0,
    refresh_batch_size=32,
    idle_sleep_ms=10
)
print("   [OK] LocalRadixTree 创建成功")

# 启动
print("\n5. 启动 LocalRadixTree...")
channel = redis_meta.get_redis_meta_channel()
if not local_tree.start(channel):
    print("   [ERROR] LocalRadixTree 启动失败")
    sys.exit(1)
print("   [OK] LocalRadixTree 启动成功")

# 等待一小段时间
print("\n6. 等待 1 秒...")
time.sleep(1)
print("   [OK] 等待完成")

# 测试 stop
print("\n7. 调用 local_tree.stop()...")
print("   提示：如果这里卡住，说明 pthread_join 阻塞在等待后台线程")
sys.stdout.flush()

start_time = time.time()
local_tree.stop()
stop_time = time.time()

print(f"   [OK] stop() 完成，耗时 {stop_time - start_time:.3f} 秒")

print("\n=== 测试完成 ===")
