#!/usr/bin/env python
"""调试同时停止多个 RadixTree 的问题"""

import time
import sys
import torch

print("=== 测试同时停止多个 RadixTree ===\n")

from flexkv.cache.radix_remote import LocalRadixTree, DistributedRadixTree
from flexkv.cache.redis_meta import RedisMeta

# 创建两个 RedisMeta
print("1. 创建 RedisMeta...")
redis_meta1 = RedisMeta(host="127.0.0.1", port=6379, local_ip="127.0.0.1")
redis_meta2 = RedisMeta(host="127.0.0.1", port=6379, local_ip="127.0.0.2")
node_id1 = redis_meta1.init_meta()
node_id2 = redis_meta2.init_meta()
print(f"   [OK] node_id1={node_id1}, node_id2={node_id2}")

# 创建4个RadixTree
print("\n2. 创建 4 个 RadixTree...")
local_tree1 = LocalRadixTree(
    tokens_per_block=4,
    max_num_blocks=1000,
    lease_ttl_ms=10000,
    renew_lease_ms=2,
    refresh_batch_size=64,
    idle_sleep_ms=1
)

local_tree2 = LocalRadixTree(
    tokens_per_block=4,
    max_num_blocks=1000,
    lease_ttl_ms=10000,
    renew_lease_ms=2,
    refresh_batch_size=64,
    idle_sleep_ms=1
)

distributed_tree1 = DistributedRadixTree(
    tokens_per_block=4,
    max_num_blocks=1000,
    node_id=node_id1,
    refresh_batch_size=32,
    rebuild_interval_ms=1,
    idle_sleep_ms=1,
    lease_renew_ms=2
)

distributed_tree2 = DistributedRadixTree(
    tokens_per_block=4,
    max_num_blocks=1000,
    node_id=node_id2,
    refresh_batch_size=32,
    rebuild_interval_ms=1,
    idle_sleep_ms=1,
    lease_renew_ms=2
)
print("   [OK] 4 个 RadixTree 创建完成")

# 启动所有树
print("\n3. 启动所有 RadixTree...")
channel1 = redis_meta1.get_redis_meta_channel()
channel2 = redis_meta2.get_redis_meta_channel()
channel3 = redis_meta1.get_redis_meta_channel()
channel4 = redis_meta2.get_redis_meta_channel()

local_tree1.start(channel1)
local_tree2.start(channel2)
distributed_tree1.start(channel3)
distributed_tree2.start(channel4)
print("   [OK] 所有 RadixTree 启动完成")

# 插入一些数据
print("\n4. 插入测试数据...")
physical_blocks1 = torch.tensor([1001, 1002, 1003, 1004], dtype=torch.long)
block_hashes1 = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.long)

physical_blocks2 = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.long)
block_hashes2 = torch.tensor([3001, 3002, 3003, 3004], dtype=torch.long)

node1 = local_tree1.insert(physical_blocks1, block_hashes1, 4, 4, True, None, -1, -1)
if node1:
    ret = local_tree1.insert_and_publish(node1)
    if not ret:
        print("   [ERROR] LocalRadixTree1 insert_and_publish失败")
    else:
        print("   [OK] LocalRadixTree1 insert_and_publish成功")

node2 = local_tree2.insert(physical_blocks2, block_hashes2, 4, 4, True, None, -1, -1)
if node2:
    ret = local_tree2.insert_and_publish(node2)
    if not ret:
        print("   [ERROR] LocalRadixTree2 insert_and_publish失败")
    else:
        print("   [OK] LocalRadixTree2 insert_and_publish成功")

# 等待数据同步
print("\n5. 等待 3 秒让数据同步...")
time.sleep(3)
print("   [OK] 等待完成")

# 依次停止，并记录每个耗时
print("\n6. 依次停止所有 RadixTree...")
sys.stdout.flush()

start = time.time()
print("   - 停止 distributed_tree1...")
sys.stdout.flush()
distributed_tree1.stop()
t1 = time.time() - start
print(f"     耗时: {t1:.3f} 秒")
sys.stdout.flush()

start = time.time()
print("   - 停止 distributed_tree2...")
sys.stdout.flush()
distributed_tree2.stop()
t2 = time.time() - start
print(f"     耗时: {t2:.3f} 秒")
sys.stdout.flush()

start = time.time()
print("   - 停止 local_tree1...")
sys.stdout.flush()
local_tree1.stop()
t3 = time.time() - start
print(f"     耗时: {t3:.3f} 秒")
sys.stdout.flush()

start = time.time()
print("   - 停止 local_tree2...")
sys.stdout.flush()
local_tree2.stop()
t4 = time.time() - start
print(f"     耗时: {t4:.3f} 秒")
sys.stdout.flush()

print(f"\n   [OK] 所有停止完成，总耗时: {t1+t2+t3+t4:.3f} 秒")

print("\n=== 测试完成 ===")
