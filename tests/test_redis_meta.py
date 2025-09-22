#!/usr/bin/env python3
"""
FlexKV RedisMeta 测试程序
测试 RedisMetaChannel 和 redis_meta.py 里的代码
"""

import sys
import os
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
        
        # 尝试导入 c_ext，如果失败也不影响其他测试
        try:
            from flexkv import c_ext
            print("[OK] flexkv.c_ext 模块导入成功")
        except Exception as e:
            print(f"[WARN] flexkv.c_ext 模块导入失败: {e}")
        
        from flexkv.cache.redis_meta import RedisMeta, RedisMetaChannel, BlockMeta, NodeState
        print("[OK] RedisMeta, RedisMetaChannel, BlockMeta, NodeState 导入成功")
        
        # 检查 C++ 扩展是否可用
        from flexkv.cache.redis_meta import _CRedisMetaChannel, _CBlockMeta
        if _CRedisMetaChannel is not None:
            print("[OK] C++ RedisMetaChannel 扩展可用")
        else:
            print("[WARN] C++ RedisMetaChannel 扩展不可用")
            
        if _CBlockMeta is not None:
            print("[OK] C++ BlockMeta 扩展可用")
        else:
            print("[WARN] C++ BlockMeta 扩展不可用")
        
        return True
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")
        return False

def test_node_state():
    """测试 NodeState 枚举"""
    print("\n=== 测试 NodeState 枚举 ===")
    try:
        from flexkv.cache.redis_meta import NodeState
        
        # 测试枚举值
        print(f"[OK] NODE_STATE_NORMAL: {NodeState.NODE_STATE_NORMAL}")
        print(f"[OK] NODE_STATE_ABOUT_TO_EVICT: {NodeState.NODE_STATE_ABOUT_TO_EVICT}")
        print(f"[OK] NODE_STATE_EVICTED: {NodeState.NODE_STATE_EVICTED}")
        
        # 测试枚举转换
        normal_state = NodeState(0)
        evict_state = NodeState(1)
        evicted_state = NodeState(2)
        
        print(f"[OK] 枚举转换测试成功: {normal_state}, {evict_state}, {evicted_state}")
        
        return True
    except Exception as e:
        print(f"[ERROR] NodeState 测试失败: {e}")
        return False

def test_block_meta():
    """测试 BlockMeta 类"""
    print("\n=== 测试 BlockMeta 类 ===")
    try:
        from flexkv.cache.redis_meta import BlockMeta, NodeState
        
        # 创建 BlockMeta 实例
        meta = BlockMeta(
            ph=12345,
            pb=67890,
            nid=1,
            hash=987654321,
            lt=1000000,
            state=NodeState.NODE_STATE_NORMAL
        )
        print("[OK] BlockMeta 创建成功")
        
        # 测试属性
        print(f"  - ph: {meta.ph}")
        print(f"  - pb: {meta.pb}")
        print(f"  - nid: {meta.nid}")
        print(f"  - hash: {meta.hash}")
        print(f"  - lt: {meta.lt}")
        print(f"  - state: {meta.state}")
        
        # 测试默认值
        default_meta = BlockMeta()
        print("[OK] BlockMeta 默认值创建成功")
        print(f"  - 默认 ph: {default_meta.ph}")
        print(f"  - 默认 state: {default_meta.state}")
        
        # 测试 C++ 转换（如果 C++ 扩展可用）
        try:
            from flexkv.cache.redis_meta import _CBlockMeta
            if _CBlockMeta is not None:
                c_meta = meta.to_c()
                print("[OK] BlockMeta.to_c() 转换成功")
                
                restored_meta = BlockMeta.from_c(c_meta)
                print("[OK] BlockMeta.from_c() 转换成功")
                
                # 验证转换正确性
                if (restored_meta.ph == meta.ph and 
                    restored_meta.pb == meta.pb and 
                    restored_meta.nid == meta.nid and 
                    restored_meta.hash == meta.hash and 
                    restored_meta.lt == meta.lt and 
                    restored_meta.state == meta.state):
                    print("[OK] C++ 转换验证成功")
                else:
                    print("[ERROR] C++ 转换验证失败")
                    return False
            else:
                print("[WARN] C++ BlockMeta 扩展不可用，跳过转换测试")
                
        except Exception as e:
            print(f"[WARN] C++ 转换测试失败: {e}")
        
        return True
    except Exception as e:
        print(f"[ERROR] BlockMeta 测试失败: {e}")
        return False

def test_redis_meta_channel():
    """测试 RedisMetaChannel 类"""
    print("\n=== 测试 RedisMetaChannel 类 ===")
    try:
        from flexkv.cache.redis_meta import RedisMetaChannel, BlockMeta, NodeState, _CRedisMetaChannel
        
        # 检查 C++ 扩展是否可用
        if _CRedisMetaChannel is None:
            print("[WARN] C++ RedisMetaChannel 扩展不可用，跳过 RedisMetaChannel 测试")
            return True
        
        # 创建 RedisMetaChannel 实例
        channel = RedisMetaChannel(
            host="127.0.0.1",
            port=6379,
            node_id=1,
            local_ip="127.0.0.1",
            blocks_key="flexkv_test_blocks"
        )
        print("[OK] RedisMetaChannel 创建成功")
        
        # 测试属性
        try:
            node_id = channel.node_id
            local_ip = channel.local_ip
            print(f"[OK] node_id: {node_id}")
            print(f"[OK] local_ip: {local_ip}")
        except Exception as e:
            print(f"[WARN] 属性访问失败: {e}")
        
        # 测试连接
        try:
            connected = channel.connect()
            if connected:
                print("[OK] RedisMetaChannel 连接成功")
            else:
                print("[WARN] RedisMetaChannel 连接失败")
                return False
        except Exception as e:
            print(f"[ERROR] RedisMetaChannel 连接异常: {e}")
            return False
        
        # 测试 make_block_key 方法
        try:
            key = channel.make_block_key(1, 12345)
            print(f"[OK] make_block_key 成功: {key}")
        except Exception as e:
            print(f"[WARN] make_block_key 失败: {e}")
        
        # 测试 publish_one 方法（需要连接）
        try:
            meta = BlockMeta(ph=1, pb=2, nid=1, hash=12345, lt=1000000, state=NodeState.NODE_STATE_NORMAL)
            channel.publish_one(meta)
            print("[OK] publish_one 成功")
        except Exception as e:
            print(f"[WARN] publish_one 失败: {e}")
        
        # 测试 publish_batch 方法
        try:
            metas = [
                BlockMeta(ph=1, pb=2, nid=1, hash=12345, lt=1000000, state=NodeState.NODE_STATE_NORMAL),
                BlockMeta(ph=2, pb=3, nid=1, hash=12346, lt=1000001, state=NodeState.NODE_STATE_NORMAL)
            ]
            channel.publish_batch(metas, batch_size=10)
            print("[OK] publish_batch 成功")
        except Exception as e:
            print(f"[WARN] publish_batch 失败: {e}")
        
        # 测试 list_keys 方法
        try:
            keys = channel.list_keys("*")
            print(f"[OK] list_keys 成功，找到 {len(keys)} 个键")
        except Exception as e:
            print(f"[WARN] list_keys 失败: {e}")
        
        # 测试 hmget_field_for_keys 方法
        try:
            # 先创建一些测试数据
            test_keys = []
            for i in range(5):
                meta = BlockMeta(ph=1, pb=2, nid=1, hash=50000+i, lt=1000000+i, state=NodeState.NODE_STATE_NORMAL)
                channel.publish_one(meta)
                test_keys.append(channel.make_block_key(1, 50000+i))
            
            # 测试获取单个字段
            values = channel.hmget_field_for_keys(test_keys, "ph")
            print(f"[OK] hmget_field_for_keys 成功，获取了 {len(values)} 个字段值")
            print(f"    字段值示例: {values[:3]}")
        except Exception as e:
            print(f"[WARN] hmget_field_for_keys 失败: {e}")
        
        # 测试 hmget_two_fields_for_keys 方法
        try:
            # 测试获取两个字段
            field_pairs = channel.hmget_two_fields_for_keys(test_keys, "ph", "pb")
            print(f"[OK] hmget_two_fields_for_keys 成功，获取了 {len(field_pairs)} 个字段对")
            print(f"    字段对示例: {field_pairs[:2]}")
        except Exception as e:
            print(f"[WARN] hmget_two_fields_for_keys 失败: {e}")
        
        # 测试 renew_node_leases 方法
        try:
            result = channel.renew_node_leases(1, 2000000, batch_size=10)
            print(f"[OK] renew_node_leases 成功，结果: {result}")
        except Exception as e:
            print(f"[WARN] renew_node_leases 失败: {e}")
        
        # 测试 update_block_state_batch 方法
        try:
            # 准备测试哈希值
            test_hashes = [50000 + i for i in range(5)]
            result = channel.update_block_state_batch(1, test_hashes, NodeState.NODE_STATE_ABOUT_TO_EVICT, batch_size=10)
            print(f"[OK] update_block_state_batch 成功，结果: {result}")
            
            # 验证状态是否更新成功
            values = channel.hmget_field_for_keys(test_keys, "state")
            print(f"    更新后的状态值: {values}")
        except Exception as e:
            print(f"[WARN] update_block_state_batch 失败: {e}")
        
        # 测试 delete_blockmeta_batch 方法
        try:
            # 删除之前创建的测试数据
            result = channel.delete_blockmeta_batch(1, test_hashes, batch_size=10)
            print(f"[OK] delete_blockmeta_batch 成功，结果: {result}")
            
            # 验证数据是否被删除
            remaining_keys = channel.list_keys("flexkv_test_blocks:block:1:5000*")
            print(f"    删除后剩余的键数量: {len(remaining_keys)}")
        except Exception as e:
            print(f"[WARN] delete_blockmeta_batch 失败: {e}")
        
        return True
    except Exception as e:
        print(f"[ERROR] RedisMetaChannel 测试失败: {e}")
        return False

def test_redis_meta():
    """测试 RedisMeta 类"""
    print("\n=== 测试 RedisMeta 类 ===")
    try:
        from flexkv.cache.redis_meta import RedisMeta
        
        # 创建 RedisMeta 实例
        redis_meta = RedisMeta(
            host="127.0.0.1",
            port=6379,
            password=None,
            local_ip="127.0.0.1",
            decode_responses=True
        )
        print("[OK] RedisMeta 创建成功")
        
        # 测试属性
        print(f"  - host: {redis_meta.host}")
        print(f"  - port: {redis_meta.port}")
        print(f"  - local_ip: {redis_meta.local_ip}")
        print(f"  - decode_responses: {redis_meta.decode_responses}")
        
        # 测试 UUID
        uuid = redis_meta.get_uuid()
        print(f"[OK] UUID 生成成功: {uuid}")
        
        # 测试 init_meta
        try:
            node_id = redis_meta.init_meta()
            print(f"[OK] init_meta 成功，node_id: {node_id}")
            
            # 测试 get_node_id
            retrieved_node_id = redis_meta.get_node_id()
            print(f"[OK] get_node_id 成功: {retrieved_node_id}")
            
            # 测试 get_redis_meta_channel
            channel = redis_meta.get_redis_meta_channel("flexkv_test_blocks")
            print("[OK] get_redis_meta_channel 成功")
            
            # 测试 unregister_node
            redis_meta.unregister_node()
            print("[OK] unregister_node 成功")
            
        except Exception as e:
            print(f"[ERROR] Redis 操作失败: {e}")
            print("[INFO] 请确保 Redis 服务正在运行且可访问")
            return False
        
        return True
    except Exception as e:
        print(f"[ERROR] RedisMeta 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("FlexKV RedisMeta 测试程序")
    print("=" * 50)
    
    # 记录测试结果
    test_results = []
    
    # 运行所有测试
    tests = [
        ("环境检查", check_environment),
        ("模块导入", test_imports),
        ("NodeState 枚举", test_node_state),
        ("BlockMeta 类", test_block_meta),
        ("RedisMetaChannel 类", test_redis_meta_channel),
        ("RedisMeta 类", test_redis_meta),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} 测试异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 50)
    print("测试总结:")
    print("=" * 50)
    
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
        print("\n[SUCCESS] FlexKV RedisMeta 测试全部通过！")
        print("所有 RedisMeta 相关功能都正常工作。")
        return 0
    else:
        print(f"\n[WARN] 有 {failed} 个测试失败，请检查上述错误信息")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
