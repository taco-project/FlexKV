#!/usr/bin/env python3
"""
FlexKV DistributedRadixTree 基本功能测试
验证 FlexKV 编译安装成功，DistributedRadixTree 基本功能正常
"""

import sys
import os
import torch
import time
import threading
from typing import Optional, List, Dict, Any

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_environment():
    """检查运行环境"""
    print("=== 检查运行环境 ===")
    
    # 检查 Python 版本
    python_version = sys.version_info
    print(f"Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查 PyTorch 版本
    try:
        torch_version = torch.__version__
        print(f"PyTorch 版本: {torch_version}")
    except Exception as e:
        print(f"[ERROR] PyTorch 版本检查失败: {e}")
        return False
    
    # 检查 CUDA 可用性
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 可用性: {cuda_available}")
    
    # 检查环境变量
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'flexkv' in ld_library_path.lower() or 'build/lib' in ld_library_path:
        print("[OK] LD_LIBRARY_PATH 包含 FlexKV 库路径")
    else:
        print("[WARN] LD_LIBRARY_PATH 可能缺少 FlexKV 库路径")
    
    return True

def test_imports():
    """测试模块导入"""
    print("\n=== 测试模块导入 ===")
    try:
        import flexkv
        print("[OK] flexkv 模块导入成功")
        
        from flexkv import c_ext
        print("[OK] flexkv.c_ext 模块导入成功")
        
        from flexkv.cache.radix_remote import LocalRadixTree, DistributedRadixTree
        print("[OK] LocalRadixTree 和 DistributedRadixTree 导入成功")
        
        # PCFSCacheEngine 已被重构为 HierarchyLRCacheEngine
        try:
            from flexkv.cache.pcfs_cache_engine import HierarchyLRCacheEngine
            print("[OK] HierarchyLRCacheEngine 导入成功")
        except ImportError as e:
            print(f"[WARN] HierarchyLRCacheEngine 导入失败: {e}")
        
        from flexkv.cache.redis_meta import RedisMeta, RedisMetaChannel
        print("[OK] RedisMeta 和 RedisMetaChannel 导入成功")
        
        return True
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        return False

def test_distributed_radix_tree():
    """测试 DistributedRadixTree 功能"""
    print("\n=== 测试 DistributedRadixTree 功能 ===")
    try:
        from flexkv.cache.radix_remote import DistributedRadixTree
        
        # 创建 DistributedRadixTree 实例
        drt = DistributedRadixTree(
            tokens_per_block=4,
            max_num_blocks=1000,
            node_id=1,
            refresh_batch_size=32,
            rebuild_interval_ms=1000,
            idle_sleep_ms=1,
            lease_renew_ms=1
        )
        print("[OK] DistributedRadixTree 创建成功")
        
        # 测试基本操作
        test_tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        
        # 测试查找（DistributedRadixTree 没有 insert 方法）
        try:
            match_result = drt.match_prefix(test_tokens, len(test_tokens), False)
            print(f"[OK] 查找操作成功: {match_result}")
        except Exception as e:
            print(f"[WARN] 查找操作失败: {e}")
        
        # 测试其他方法
        try:
            is_empty = drt.is_empty()
            print(f"[OK] is_empty 方法成功: {is_empty}")
        except Exception as e:
            print(f"[WARN] is_empty 方法失败: {e}")
        
        return True
    except Exception as e:
        print(f"[ERROR] DistributedRadixTree 测试失败: {e}")
        return False

def test_local_radix_tree():
    """测试 LocalRadixTree 功能"""
    print("\n=== 测试 LocalRadixTree 功能 ===")
    try:
        from flexkv.cache.radix_remote import LocalRadixTree
        
        # 创建 LocalRadixTree 实例
        lrt = LocalRadixTree(
            tokens_per_block=4,
            max_num_blocks=1000,
            lease_ttl_ms=10000,
            renew_lease_ms=2,
            refresh_batch_size=64,
            idle_sleep_ms=1
        )
        print("[OK] LocalRadixTree 创建成功")
        
        # 测试基本操作
        test_tokens = torch.tensor([5, 6, 7, 8], dtype=torch.long)
        
        # 测试查找
        try:
            lrt.insert(test_tokens, test_tokens, 4, 4, True, None, -1, -1)
            match_result = lrt.match_prefix(test_tokens, len(test_tokens), False)
            if match_result is None:
                print(f"[WARN] 查找操作失败: {match_result}")
                return False
            elif match_result.num_matched_blocks == 0:
                print(f"[WARN] 查找操作失败: {match_result}")
                return False
            print(f"[OK] 查找操作成功: {match_result}")
        except Exception as e:
            print(f"[WARN] 查找操作失败: {e}")
            return False
        
        # 测试其他方法
        try:
            is_empty = lrt.is_empty()
            if is_empty:
                print(f"[WARN] 查找操作失败: {is_empty}")
                return False
            total_nodes = lrt.total_node_num()
            if total_nodes == 0:
                print(f"[WARN] 查找操作失败: {total_nodes}")
                return False
            print(f"[OK] is_empty 方法成功: {is_empty}")
            print(f"[OK] total_node_num 方法成功: {total_nodes}")
        except Exception as e:
            print(f"[WARN] 其他方法失败: {e}")
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] LocalRadixTree 测试失败: {e}")
        return False

def test_pcfs_cache_engine():
    """测试 HierarchyLRCacheEngine 功能"""
    print("\n=== 测试 HierarchyLRCacheEngine 功能 ===")
    try:
        from flexkv.cache.pcfs_cache_engine import HierarchyLRCacheEngine
        from flexkv.common.transfer import DeviceType
        
        # 创建 HierarchyLRCacheEngine 实例
        cache_engine = HierarchyLRCacheEngine(
            num_total_blocks=1000,
            tokens_per_block=4,
            evict_ratio=0.1,
            device_type=DeviceType.CPU
        )
        print("[OK] HierarchyLRCacheEngine 创建成功")
        
        # 测试基本属性
        print(f"  - num_total_blocks: {cache_engine.num_total_blocks}")
        print(f"  - tokens_per_block: {cache_engine.tokens_per_block}")
        print(f"  - evict_ratio: {cache_engine.evict_ratio}")
        
        return True
    except Exception as e:
        print(f"[ERROR] HierarchyLRCacheEngine 测试失败: {e}")
        return False

def test_distributed_radix_tree_integration():
    """测试分布式RadixTree集成功能"""
    print("\n=== 测试分布式RadixTree集成功能 ===")
    try:
        from flexkv.cache.radix_remote import LocalRadixTree, DistributedRadixTree
        from flexkv.cache.redis_meta import RedisMeta
        import redis
        
        # 步骤0: 清理Redis中的历史数据
        print("步骤0: 清理Redis历史数据...")
        try:
            r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)
            # 删除所有 block:* 和 node:* keys
            block_keys = list(r.scan_iter("block:*"))
            node_keys = list(r.scan_iter("node:*"))
            if block_keys:
                r.delete(*block_keys)
            if node_keys:
                r.delete(*node_keys)
            print(f"[OK] Redis 清理完成 - 删除了 {len(block_keys)} 个block keys 和 {len(node_keys)} 个node keys")
        except Exception as e:
            print(f"[WARN] Redis 清理失败: {e}")
        
        # 使用时间戳生成唯一的 IP 地址
        import time
        timestamp = int(time.time() * 1000) % 100000
        ip_suffix1 = (timestamp % 250) + 1
        ip_suffix2 = ((timestamp + 1) % 250) + 1
        local_ip1 = f"10.0.{ip_suffix1 // 250}.{ip_suffix1 % 250}"
        local_ip2 = f"10.0.{ip_suffix2 // 250}.{ip_suffix2 % 250}"
        
        # 步骤1: 创建两个RedisMeta实例
        print("步骤1: 创建RedisMeta实例...")
        print(f"  使用唯一IP: local_ip1={local_ip1}, local_ip2={local_ip2}")
        redis_meta1 = RedisMeta(host="127.0.0.1", port=6379, local_ip=local_ip1)
        redis_meta2 = RedisMeta(host="127.0.0.1", port=6379, local_ip=local_ip2)
        print(f"[OK] RedisMeta实例创建成功 - Meta1({local_ip1}), Meta2({local_ip2})")
        
        # 步骤2: 初始化RedisMeta
        print("步骤2: 初始化RedisMeta...")
        node_id1 = redis_meta1.init_meta()
        if node_id1 is None:
            raise RuntimeError("RedisMeta1初始化失败，无法获取node_id")
        
        node_id2 = redis_meta2.init_meta()
        if node_id2 is None:
            raise RuntimeError("RedisMeta2初始化失败，无法获取node_id")
        
        print(f"[OK] RedisMeta初始化成功 - Node1: {node_id1}, Node2: {node_id2}")
        
        # 步骤3: 创建2个LocalRadixTree和2个DistributedRadixTree实例
        print("步骤3: 创建RadixTree实例...")
        
        # 创建LocalRadixTree实例
        local_tree1 = LocalRadixTree(
            tokens_per_block=4,
            max_num_blocks=1000,
            lease_ttl_ms=100000,
            renew_lease_ms=2,
            refresh_batch_size=64,
            idle_sleep_ms=1
        )
        
        local_tree2 = LocalRadixTree(
            tokens_per_block=4,
            max_num_blocks=1000,
            lease_ttl_ms=100000,
            renew_lease_ms=2,
            refresh_batch_size=64,
            idle_sleep_ms=1
        )
        
        # 创建DistributedRadixTree实例
        distributed_tree1 = DistributedRadixTree(
            tokens_per_block=4,
            max_num_blocks=1000,
            node_id=node_id1,
            refresh_batch_size=32,
            rebuild_interval_ms=3000,  # 增加到3秒，避免频繁的 Redis 操作阻塞 stop()
            idle_sleep_ms=100,  # 增加idle时间，让 stop() 更快响应
            lease_renew_ms=2
        )
        
        distributed_tree2 = DistributedRadixTree(
            tokens_per_block=4,
            max_num_blocks=1000,
            node_id=node_id2,
            refresh_batch_size=32,
            rebuild_interval_ms=3000,  # 增加到3秒，避免频繁的 Redis 操作阻塞 stop()
            idle_sleep_ms=100,  # 增加idle时间，让 stop() 更快响应
            lease_renew_ms=2
        )
        print("[OK] RadixTree实例创建成功 - 2个LocalRadixTree, 2个DistributedRadixTree")
        
        # 步骤4: 获取RedisMetaChannel并同时启动所有RadixTree
        print("步骤4: 启动所有RadixTree...")
        channel1 = redis_meta1.get_redis_meta_channel()
        if not channel1:
            raise RuntimeError("RedisMeta1获取RedisMetaChannel失败")
        channel2 = redis_meta2.get_redis_meta_channel()
        if not channel2:
            raise RuntimeError("RedisMeta2获取RedisMetaChannel失败")
        channel3 = redis_meta1.get_redis_meta_channel()
        if not channel3:
            raise RuntimeError("RedisMeta3获取RedisMetaChannel失败")
        channel4 = redis_meta2.get_redis_meta_channel()
        if not channel4:
            raise RuntimeError("RedisMeta4获取RedisMetaChannel失败")
        
        # 先启动 LocalRadixTree
        if not local_tree1.start(channel1):
            raise RuntimeError("LocalRadixTree1启动失败")
        if not local_tree2.start(channel2):
            raise RuntimeError("LocalRadixTree2启动失败")
        print("[OK] LocalRadixTree启动成功")
        
        # 步骤5: 创建测试数据 - 每个包含4个block
        print("步骤5: 创建测试数据...")
        
        # LocalRadixTree1的测试数据 - 4个block
        physical_blocks1 = torch.tensor([1001, 1002, 1003, 1004], dtype=torch.long)
        block_hashes1 = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.long)
        
        # LocalRadixTree2的测试数据 - 4个block
        physical_blocks2 = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.long)
        block_hashes2 = torch.tensor([3001, 3002, 3003, 3004], dtype=torch.long)
        
        print(f"[OK] 测试数据创建成功 - 每个包含4个block")
        print(f"  - LocalRadixTree1: physical_blocks={physical_blocks1.tolist()}, hashes={block_hashes1.tolist()}")
        print(f"  - LocalRadixTree2: physical_blocks={physical_blocks2.tolist()}, hashes={block_hashes2.tolist()}")
        
        # 步骤6: 向LocalRadixTree添加节点
        print("步骤6: 向LocalRadixTree添加节点...")
        
        # LocalRadixTree1使用insert方法
        print("  - LocalRadixTree1使用insert方法...")
        node1 = local_tree1.insert(
            physical_blocks1, block_hashes1, 4, 4, True, None, -1, -1
        )
        if node1 is not None:
            print(f"    [OK] LocalRadixTree1 insert成功，插入节点包含4个block")
            local_tree1.insert_and_publish(node1)
        else:
            print("    [WARN] LocalRadixTree1 insert返回None")
        
        # LocalRadixTree2使用insert_and_publish方法
        print("  - LocalRadixTree2使用insert_and_publish方法...")
        node2 = local_tree2.insert(
            physical_blocks2, block_hashes2, 4, 4, True, None, -1, -1
        )
        if node2 is not None:
            local_tree2.insert_and_publish(node2)
            print(f"    [OK] LocalRadixTree2 insert_and_publish成功，插入并发布节点包含4个block")
        else:
            print("    [WARN] LocalRadixTree2 insert返回None")
        
        # 步骤6.5: 等待数据发布到 Redis
        print("步骤6.5: 等待数据发布到 Redis. sleep 2 seconds...")
        time.sleep(2)
        print("[OK] 数据发布等待完成")
        
        # 步骤6.6: 启动 DistributedRadixTree
        print("步骤6.6: 启动 DistributedRadixTree...")
        if not distributed_tree1.start(channel3):
            raise RuntimeError("DistributedRadixTree1启动失败")
        if not distributed_tree2.start(channel4):
            raise RuntimeError("DistributedRadixTree2启动失败")
        print("[OK] DistributedRadixTree启动成功")
        
        # 步骤7: 等待数据同步
        print("步骤7: 等待 DistributedRadixTree 刷新. sleep 8 seconds...")
        time.sleep(8)  # 等待 DistributedRadixTree 完成至少两次刷新（rebuild_interval_ms=3000），确保数据被加载
        print("[OK] 数据同步等待完成")
        
        # 步骤7.5: 调试信息 - 检查 Redis 中的数据
        print("步骤7.5: 检查 Redis 中的数据...")
        print(f"  - node_id1: {node_id1}")
        print(f"  - node_id2: {node_id2}")
        
        # 步骤8: 详细验证结果
        print("步骤8: 验证结果...")
        
        # 验证LocalRadixTree状态
        print("LocalRadixTree状态:")
        lrt1_nodes = local_tree1.total_node_num()
        if lrt1_nodes == 0:
            raise RuntimeError("LocalRadixTree1 total_node_num失败")
        lrt1_cached = local_tree1.total_cached_blocks()
        if lrt1_cached == 0:
            raise RuntimeError("LocalRadixTree1 total_cached_blocks失败")
        lrt1_ready = local_tree1.total_ready_blocks()
        if lrt1_ready == 0:
            raise RuntimeError("LocalRadixTree1 total_ready_blocks失败")
        lrt1_unready = local_tree1.total_unready_blocks()
        if lrt1_unready != 0:
            raise RuntimeError("LocalRadixTree1 total_unready_blocks失败")
        
        lrt2_nodes = local_tree2.total_node_num()
        if lrt2_nodes == 0:
            raise RuntimeError("LocalRadixTree2 total_node_num失败")
        lrt2_cached = local_tree2.total_cached_blocks()
        if lrt2_cached == 0:
            raise RuntimeError("LocalRadixTree2 total_cached_blocks失败")
        lrt2_ready = local_tree2.total_ready_blocks()
        if lrt2_ready == 0:
            raise RuntimeError("LocalRadixTree2 total_ready_blocks失败")
        lrt2_unready = local_tree2.total_unready_blocks()
        if lrt2_unready != 0:
            raise RuntimeError("LocalRadixTree2 total_unready_blocks失败")
        
        print(f"  - LocalRadixTree1: 节点数={lrt1_nodes}, 缓存块数={lrt1_cached}, 就绪块数={lrt1_ready}, 未就绪块数={lrt1_unready}")
        print(f"  - LocalRadixTree2: 节点数={lrt2_nodes}, 缓存块数={lrt2_cached}, 就绪块数={lrt2_ready}, 未就绪块数={lrt2_unready}")
        
        # 验证DistributedRadixTree状态
        print("DistributedRadixTree状态:")
        drt1_empty = distributed_tree1.is_empty()
        if drt1_empty:
            raise RuntimeError("DistributedRadixTree1 is_empty失败")
        drt2_empty = distributed_tree2.is_empty()
        if drt2_empty:
            raise RuntimeError("DistributedRadixTree2 is_empty失败")
        print(f"  - DistributedRadixTree1: 是否为空={drt1_empty}")
        print(f"  - DistributedRadixTree2: 是否为空={drt2_empty}")
        
        # 测试前缀匹配功能
        print("测试前缀匹配功能...")
        test_hashes1 = torch.tensor([2001, 2002, 2003, 2004], dtype=torch.long)
        test_hashes2 = torch.tensor([3001, 3002, 3003, 3004], dtype=torch.long)
        
        # 在LocalRadixTree中测试匹配
        match_result1 = local_tree1.match_prefix(test_hashes1, len(test_hashes1), True)
        if match_result1 is None:
            raise RuntimeError("LocalRadixTree1 match_prefix失败")
        if match_result1.num_matched_blocks == 0:
            raise RuntimeError("LocalRadixTree1 match_prefix失败")
        match_result2 = local_tree2.match_prefix(test_hashes2, len(test_hashes2), True)
        if match_result2 is None:
            raise RuntimeError("LocalRadixTree2 match_prefix失败")
        if match_result2.num_matched_blocks == 0:
            raise RuntimeError("LocalRadixTree2 match_prefix失败")
        print(f"  - LocalRadixTree1匹配结果: 匹配块数={match_result1.num_matched_blocks if match_result1 else 0}")
        print(f"  - LocalRadixTree2匹配结果: 匹配块数={match_result2.num_matched_blocks if match_result2 else 0}")
        
        # 在DistributedRadixTree中测试匹配
        drt_match1 = distributed_tree1.match_prefix(test_hashes2, len(test_hashes2), True)
        print(f"  - DistributedRadixTree1匹配结果: {drt_match1}")
        if drt_match1:
            print(f"    匹配块数={drt_match1.num_matched_blocks}")
        if drt_match1 is None:
            raise RuntimeError("DistributedRadixTree1 match_prefix失败: 返回None")
        if drt_match1.num_matched_blocks == 0:
            raise RuntimeError(f"DistributedRadixTree1 match_prefix失败: 匹配块数为0, 查询hashes={test_hashes2.tolist()}")
        drt_match2 = distributed_tree2.match_prefix(test_hashes1, len(test_hashes1), True)
        if drt_match2 is None:
            raise RuntimeError("DistributedRadixTree2 match_prefix失败")
        if drt_match2.num_matched_blocks == 0:
            raise RuntimeError("DistributedRadixTree2 match_prefix失败")
        print(f"  - DistributedRadixTree1匹配结果: 匹配块数={drt_match1.num_matched_blocks if drt_match1 else 0}")
        print(f"  - DistributedRadixTree2匹配结果: 匹配块数={drt_match2.num_matched_blocks if drt_match2 else 0}")
        # 步骤9: 使用DistributedRadixTree加载Redis数据
        print("步骤9: 使用DistributedRadixTree加载Redis数据...")
        distributed_tree1.stop()
        distributed_tree2.stop()
        local_tree1.stop()
        local_tree2.stop()
        print("[OK] DistributedRadixTree停止")
        time.sleep(1)
        # DistributedRadixTree1刷新
        print("  - DistributedRadixTree1执行remote_tree_refresh...")
        refresh_result1 = distributed_tree1.remote_tree_refresh()
        if refresh_result1 is None:
            raise RuntimeError("DistributedRadixTree1 remote_tree_refresh失败")
        print(f"    [OK] DistributedRadixTree1 remote_tree_refresh完成")
        
        # DistributedRadixTree2刷新
        print("  - DistributedRadixTree2执行remote_tree_refresh...")
        refresh_result2 = distributed_tree2.remote_tree_refresh()
        if refresh_result2 is None:
            raise RuntimeError("DistributedRadixTree2 remote_tree_refresh失败")
        print(f"    [OK] DistributedRadixTree2 remote_tree_refresh完成")
        # 步骤10: 性能测试
        print("步骤10: 性能测试...")
        
        # 使用已创建的local_tree1和distributed_tree1进行性能测试
        num_operations = 100
        print(f"  - 开始性能测试，执行{num_operations}次操作...")
        
        # LocalRadixTree1性能测试
        print("  - LocalRadixTree1性能测试...")
        lrt_start_time = time.time()
        
        for i in range(num_operations):
            test_physical = torch.tensor([i % 100, (i + 1) % 100, (i + 2) % 100, (i + 3) % 100], dtype=torch.long)
            test_hashes = torch.tensor([(i + 1000) % 2000, (i + 1001) % 2000, (i + 1002) % 2000, (i + 1003) % 2000], dtype=torch.long)
            try:
                # 测试insert性能
                lrt_node = local_tree1.insert(test_physical, test_hashes, 4, 4, True, None, -1, -1)
                # 测试match_prefix性能
                lrt_match = local_tree1.match_prefix(test_hashes, len(test_hashes), True)
            except Exception as e:
                print(f"    [WARN] LocalRadixTree1性能测试中操作失败: {e}")
                break
        
        lrt_end_time = time.time()
        lrt_duration = lrt_end_time - lrt_start_time
        lrt_ops_per_sec = num_operations / lrt_duration if lrt_duration > 0 else 0
        
        print(f"    [OK] LocalRadixTree1性能测试完成:")
        print(f"      - 操作数量: {num_operations}")
        print(f"      - 总时间: {lrt_duration:.3f} 秒")
        print(f"      - 每秒操作数: {lrt_ops_per_sec:.0f}")
        
        # DistributedRadixTree1性能测试
        print("  - DistributedRadixTree1性能测试...")
        drt_start_time = time.time()
        
        for i in range(num_operations):
            test_hashes = torch.tensor([(i + 2000) % 3000, (i + 2001) % 3000, (i + 2002) % 3000, (i + 2003) % 3000], dtype=torch.long)
            try:
                # 测试match_prefix性能
                drt_match = distributed_tree1.match_prefix(test_hashes, len(test_hashes), True)
            except Exception as e:
                print(f"    [WARN] DistributedRadixTree1性能测试中操作失败: {e}")
                break
        
        drt_end_time = time.time()
        drt_duration = drt_end_time - drt_start_time
        drt_ops_per_sec = num_operations / drt_duration if drt_duration > 0 else 0
        
        print(f"    [OK] DistributedRadixTree1性能测试完成:")
        print(f"      - 操作数量: {num_operations}")
        print(f"      - 总时间: {drt_duration:.3f} 秒")
        print(f"      - 每秒操作数: {drt_ops_per_sec:.0f}")
        
        # 性能对比
        print("  - 性能对比:")
        if lrt_ops_per_sec > 0 and drt_ops_per_sec > 0:
            ratio = lrt_ops_per_sec / drt_ops_per_sec
            print(f"    LocalRadixTree1 vs DistributedRadixTree1: {ratio:.2f}x")
        
        print("[OK] 性能测试完成")
        
        # 步骤11: 清理资源
        print("步骤11: 清理资源...")
        local_tree1.stop()
        local_tree2.stop()
        distributed_tree1.stop()
        distributed_tree2.stop()
        redis_meta1.unregister_node()
        redis_meta2.unregister_node()
        print("[OK] 资源清理完成")
        
        # 验证测试结果
        success = True
        if lrt1_cached == 0 and lrt2_cached == 0:
            print("[WARN] LocalRadixTree没有缓存任何块，可能插入失败")
            success = False
        
        if drt1_empty and drt2_empty:
            print("[WARN] DistributedRadixTree为空，可能remote_tree_refresh失败")
            success = False
        
        if success:
            print("\n[SUCCESS] 分布式RadixTree集成测试完成，所有功能正常工作！")
            print(f"性能测试结果: LocalRadixTree1({lrt_ops_per_sec:.0f} ops/s), DistributedRadixTree1({drt_ops_per_sec:.0f} ops/s)")
        else:
            print("\n[WARN] 分布式RadixTree集成测试完成，但有一些警告")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] 分布式RadixTree集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_skipping():
    """测试 CUDA 跳过"""
    print("\n=== 测试 CUDA 跳过 ===")
    try:
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 可用性: {cuda_available}")
        
        if cuda_available:
            print("[WARN] CUDA 可用，但跳过 CUDA 相关测试")
        else:
            print("[OK] CUDA 不可用，自动跳过 CUDA 相关测试")
        
        # 测试 CPU 张量操作
        cpu_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        print(f"[OK] CPU 张量创建成功: {cpu_tensor}")
        
        return True
    except Exception as e:
        print(f"[ERROR] CUDA 跳过测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("FlexKV DistributedRadixTree 基本功能测试")
    print("=" * 60)
    
    # 记录测试结果
    test_results = []
    
    # 运行所有测试
    tests = [
        ("环境检查", check_environment),
        ("模块导入", test_imports),
        ("LocalRadixTree", test_local_radix_tree),
        ("DistributedRadixTree", test_distributed_radix_tree),
        ("HierarchyLRCacheEngine", test_pcfs_cache_engine),
        ("分布式RadixTree集成", test_distributed_radix_tree_integration),
        ("CUDA 跳过", test_cuda_skipping),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in test_results:
        status = "[OK] 通过" if result else "[ERROR] 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n总计: {passed} 个测试通过, {failed} 个测试失败")
    
    if failed == 0:
        print("\n[SUCCESS] FlexKV DistributedRadixTree 测试全部通过！")
        print("所有基本功能都正常工作，DistributedRadixTree 可以正常使用。")
        return 0
    else:
        print(f"\n[WARN] 有 {failed} 个测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
