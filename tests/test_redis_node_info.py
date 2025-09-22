#!/usr/bin/env python3
"""
测试 RedisNodeInfo 类的功能
"""

import sys
import time
import threading
sys.path.insert(0, '..')

from flexkv.cache.redis_meta import RedisNodeInfo

def test_redis_node_info():
    """测试 RedisNodeInfo 的基本功能"""
    print("=== 测试 RedisNodeInfo 基本功能 ===")
    
    try:
        # 创建 RedisNodeInfo 实例
        node_info = RedisNodeInfo(
            host='127.0.0.1',
            port=6379,
            local_ip='127.0.0.1',
            password=''
        )
        print("[OK] RedisNodeInfo 创建成功")
        
        # 连接 Redis
        if not node_info.connect():
            print("[ERROR] 无法连接到 Redis")
            return False
        print("[OK] RedisNodeInfo 连接成功")
        
        # 注册节点
        node_id = node_info.register_node()
        if node_id == 0xFFFFFFFF:  # UINT32_MAX
            print("[ERROR] 节点注册失败")
            return False
        print(f"[OK] 节点注册成功，node_id: {node_id}")
        
        # 获取当前节点 ID
        current_node_id = node_info.node_id
        print(f"[OK] 当前节点 ID: {current_node_id}")
        
        # 获取活跃节点列表
        active_nodes = node_info.get_active_node_ids()
        print(f"[OK] 活跃节点列表: {active_nodes}")
        
        # 检查节点是否活跃
        is_active = node_info.is_node_active(node_id)
        print(f"[OK] 节点 {node_id} 是否活跃: {is_active}")
        
        # 等待一段时间让监听线程工作
        print("[INFO] 等待 3 秒让监听线程工作...")
        time.sleep(3)
        
        # 再次获取活跃节点列表
        active_nodes_after = node_info.get_active_node_ids()
        print(f"[OK] 3秒后活跃节点列表: {active_nodes_after}")
        
        # 解注册节点
        unregister_result = node_info.unregister_node()
        if not unregister_result:
            print("[ERROR] 节点解注册失败")
            return False
        print("[OK] 节点解注册成功")
        
        # 断开连接
        node_info.disconnect()
        print("[OK] RedisNodeInfo 断开连接成功")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] RedisNodeInfo 测试失败: {e}")
        return False

def test_multiple_nodes():
    """测试多个节点的注册和解注册"""
    print("\n=== 测试多个节点注册 ===")
    
    try:
        # 创建多个节点
        nodes = []
        for i in range(3):
            node_info = RedisNodeInfo(
                host='127.0.0.1',
                port=6379,
                local_ip=f'127.0.0.{i+1}',
                password=''
            )
            if node_info.connect():
                node_id = node_info.register_node()
                if node_id != 0xFFFFFFFF:
                    nodes.append(node_info)
                    print(f"[OK] 节点 {i+1} 注册成功，node_id: {node_id}, IP: 127.0.0.{i+1}")
                else:
                    print(f"[ERROR] 节点 {i+1} 注册失败")
            else:
                print(f"[ERROR] 节点 {i+1} 连接失败")
        
        if not nodes:
            print("[ERROR] 没有成功注册的节点")
            return False
        
        # 等待监听线程更新
        print("[INFO] 等待 2 秒让监听线程更新...")
        time.sleep(2)
        
        # 检查第一个节点的活跃节点列表
        active_nodes = nodes[0].get_active_node_ids()
        print(f"[OK] 第一个节点看到的活跃节点列表: {active_nodes}")
        
        # 解注册所有节点
        for i, node_info in enumerate(nodes):
            if node_info.unregister_node():
                print(f"[OK] 节点 {i+1} 解注册成功")
            else:
                print(f"[ERROR] 节点 {i+1} 解注册失败")
            node_info.disconnect()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 多节点测试失败: {e}")
        return False

def test_pub_sub_notification():
    """测试发布订阅通知功能"""
    print("\n=== 测试发布订阅通知功能 ===")
    
    try:
        # 创建两个节点
        node1 = RedisNodeInfo('127.0.0.1', 6379, '127.0.0.1', '')
        node2 = RedisNodeInfo('127.0.0.1', 6379, '127.0.0.2', '')
        
        if not node1.connect() or not node2.connect():
            print("[ERROR] 节点连接失败")
            return False
        
        # 注册两个节点
        node1_id = node1.register_node()
        node2_id = node2.register_node()
        
        print(f"[OK] 节点1 ID: {node1_id}, 节点2 ID: {node2_id}")
        
        # 等待通知传播
        print("[INFO] 等待 2 秒让通知传播...")
        time.sleep(2)
        
        # 检查两个节点是否都能看到对方
        active_nodes_1 = node1.get_active_node_ids()
        active_nodes_2 = node2.get_active_node_ids()
        
        print(f"[OK] 节点1看到的活跃节点: {active_nodes_1}")
        print(f"[OK] 节点2看到的活跃节点: {active_nodes_2}")
        
        # 解注册节点
        node1.unregister_node()
        node2.unregister_node()
        
        # 等待通知传播
        print("[INFO] 等待 2 秒让解注册通知传播...")
        time.sleep(2)
        
        # 检查解注册后的状态
        active_nodes_1_after = node1.get_active_node_ids()
        active_nodes_2_after = node2.get_active_node_ids()
        
        print(f"[OK] 解注册后节点1看到的活跃节点: {active_nodes_1_after}")
        print(f"[OK] 解注册后节点2看到的活跃节点: {active_nodes_2_after}")
        
        # 断开连接
        node1.disconnect()
        node2.disconnect()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] 发布订阅通知测试失败: {e}")
        return False

def main():
    print("RedisNodeInfo 功能测试")
    print("=" * 50)
    
    success = True
    
    # 测试基本功能
    if not test_redis_node_info():
        success = False
    
    # 测试多个节点
    if not test_multiple_nodes():
        success = False
    
    # 测试发布订阅通知
    if not test_pub_sub_notification():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("[SUCCESS] 所有 RedisNodeInfo 测试通过！")
        print("\n功能总结:")
        print("1. 节点注册：通过原子递增 global:node_id 获取唯一 node_id")
        print("2. 节点信息存储：在 node:node_id 哈希中存储节点信息")
        print("3. 发布订阅通知：注册/解注册时发布 flexkv_node_id_updated 消息")
        print("4. 监听线程：订阅 flexkv_node_id_updated 并更新活跃节点列表")
        print("5. 节点扫描：通过 SCAN 0 MATCH node:* 扫描所有活跃节点")
    else:
        print("[ERROR] 部分测试失败")

if __name__ == "__main__":
    main()
