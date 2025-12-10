#!/usr/bin/env python3
"""
直接使用 JSONL 数据测试 vLLM API
支持多节点负载均衡
"""

import json
import time
import argparse
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import random

class VLLMBenchmark:
    def __init__(self, hosts, max_output_tokens=1024, model_name=None):
        """
        Args:
            hosts: list of (host, port) tuples
            max_output_tokens: 最大输出 token 数
            model_name: 模型名称（默认使用模型路径）
        """
        self.hosts = hosts
        self.base_urls = [f"http://{host}:{port}/v1" for host, port in hosts]
        self.max_output_tokens = max_output_tokens
        self.model_name = model_name or "/workspace/Qwen3-8B"
        self.host_counter = 0  # 用于 round-robin
        
    def get_next_url(self, strategy='round-robin'):
        """获取下一个 URL"""
        if strategy == 'round-robin':
            url = self.base_urls[self.host_counter % len(self.base_urls)]
            self.host_counter += 1
            return url
        elif strategy == 'random':
            return random.choice(self.base_urls)
        else:
            return self.base_urls[0]
    
    def send_request(self, item, thread_id, strategy='round-robin'):
        """发送单个请求"""
        base_url = self.get_next_url(strategy)
        
        # 构建消息
        system_prompt = item.get('system_prompt', 'You are a helpful assistant')
        user_prompt = item.get('prompt', '')
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                stream=True,
                timeout=300
            )
            
            first_token_time = None
            response_content = ""
            token_count = 0
            last_chunk = None
            
            if response.status_code == 200:
                for chunk_bytes in response.iter_lines():
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    
                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    
                    if chunk == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(chunk)
                        
                        if "usage" in data:
                            last_chunk = data
                        
                        if choices := data.get("choices"):
                            if first_token_time is None:
                                first_token_time = time.time()
                            
                            delta_content = choices[0]["delta"].get("content")
                            if delta_content:
                                response_content += delta_content
                                token_count += 1
                    except json.JSONDecodeError:
                        continue
                
                end_time = time.time()
                
                # 提取 usage 信息
                prompt_tokens = last_chunk["usage"]["prompt_tokens"] if last_chunk else 0
                completion_tokens = last_chunk["usage"]["completion_tokens"] if last_chunk else token_count
                
                ttft = first_token_time - start_time if first_token_time else 0
                total_time = end_time - start_time
                decode_time = end_time - first_token_time if first_token_time else 0
                
                result = {
                    'success': True,
                    'base_url': base_url,
                    'thread_id': thread_id,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'ttft': ttft,
                    'total_time': total_time,
                    'decode_time': decode_time,
                    'prefill_speed': prompt_tokens / ttft if ttft > 0 else 0,
                    'decode_speed': (completion_tokens - 1) / decode_time if decode_time > 0 else 0,
                    'start_time': start_time,
                    'first_token_time': first_token_time,
                    'end_time': end_time,
                }
                
                print(f"[Thread {thread_id}] URL: {base_url}, "
                      f"Prompt: {prompt_tokens} tokens, "
                      f"TTFT: {ttft:.3f}s, "
                      f"Prefill: {result['prefill_speed']:.1f} tok/s, "
                      f"Completion: {completion_tokens} tokens, "
                      f"Decode: {result['decode_speed']:.1f} tok/s")
                
                return result
            else:
                print(f"[Thread {thread_id}] ERROR: HTTP {response.status_code} from {base_url}")
                return {'success': False, 'error': f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"[Thread {thread_id}] ERROR: {e} from {base_url}")
            return {'success': False, 'error': str(e)}
    
    def run_benchmark(self, jsonl_file, num_requests, concurrency, strategy='round-robin', start_line=0):
        """运行 benchmark"""
        
        # 读取数据
        print(f"读取数据文件: {jsonl_file}")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            all_data = [json.loads(line) for line in f if line.strip()]
        
        print(f"总数据量: {len(all_data)}")
        print(f"起始行: {start_line}")
        print(f"请求数: {num_requests}")
        print(f"并发数: {concurrency}")
        print(f"分发策略: {strategy}")
        print(f"目标节点: {len(self.hosts)}")
        for i, (host, port) in enumerate(self.hosts, 1):
            print(f"  节点 {i}: {host}:{port}")
        print("")
        
        # 选择数据
        if start_line + num_requests > len(all_data):
            # 循环使用数据
            data_to_test = (all_data[start_line:] + all_data * ((num_requests // len(all_data)) + 1))[:num_requests]
        else:
            data_to_test = all_data[start_line:start_line + num_requests]
        
        # Shuffle 策略：打乱数据顺序
        if strategy == 'shuffle':
            print(f"📊 Shuffle 策略：打乱数据顺序后使用 round-robin 分发")
            import random as shuffle_random
            shuffle_random.shuffle(data_to_test)
            print(f"   数据已随机打乱")
            # 打乱后使用 round-robin
            actual_strategy = 'round-robin'
        else:
            actual_strategy = strategy
        
        print(f"实际测试数据量: {len(data_to_test)}")
        print("")
        
        # 并发测试
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.send_request, item, i, actual_strategy)
                for i, item in enumerate(data_to_test)
            ]
            
            for future in as_completed(futures):
                result = future.result()
                if result.get('success'):
                    results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 统计结果
        if not results:
            print("没有成功的请求！")
            return
        
        print("\n" + "="*80)
        print(" Benchmark 结果")
        print("="*80)
        
        # 基础统计
        total_requests = len(results)
        total_prompt_tokens = sum(r['prompt_tokens'] for r in results)
        total_completion_tokens = sum(r['completion_tokens'] for r in results)
        total_tokens = total_prompt_tokens + total_completion_tokens
        
        # 延迟统计
        ttfts = [r['ttft'] for r in results]
        ttfts.sort()
        
        decode_speeds = [r['decode_speed'] for r in results if r['decode_speed'] > 0]
        decode_speeds.sort()
        
        # 打印统计
        print(f"\n请求统计:")
        print(f"  总请求数: {total_requests}")
        print(f"  成功率: {total_requests / num_requests * 100:.1f}%")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  QPS: {total_requests / total_time:.2f}")
        
        print(f"\nToken 统计:")
        print(f"  总 Prompt Tokens: {total_prompt_tokens}")
        print(f"  总 Completion Tokens: {total_completion_tokens}")
        print(f"  总 Tokens: {total_tokens}")
        print(f"  平均 Prompt 长度: {total_prompt_tokens / total_requests:.1f}")
        print(f"  平均 Completion 长度: {total_completion_tokens / total_requests:.1f}")
        
        print(f"\n吞吐量:")
        print(f"  端到端吞吐: {total_tokens / total_time:.2f} tokens/s")
        print(f"  Prefill 吞吐: {total_prompt_tokens / sum(ttfts):.2f} tokens/s")
        print(f"  Decode 吞吐: {total_completion_tokens / sum(r['decode_time'] for r in results):.2f} tokens/s")
        
        print(f"\nTTFT (Time to First Token):")
        print(f"  平均: {sum(ttfts) / len(ttfts):.3f}s")
        print(f"  中位数: {ttfts[len(ttfts)//2]:.3f}s")
        print(f"  P95: {ttfts[int(len(ttfts)*0.95)]:.3f}s")
        print(f"  P99: {ttfts[int(len(ttfts)*0.99)]:.3f}s")
        
        print(f"\nDecode 速度:")
        print(f"  平均: {sum(decode_speeds) / len(decode_speeds):.2f} tokens/s")
        print(f"  中位数: {decode_speeds[len(decode_speeds)//2]:.2f} tokens/s")
        print(f"  P95: {decode_speeds[int(len(decode_speeds)*0.95)]:.2f} tokens/s")
        print(f"  P99: {decode_speeds[int(len(decode_speeds)*0.99)]:.2f} tokens/s")
        
        # 按节点统计
        print(f"\n每个节点的请求分布:")
        for base_url in self.base_urls:
            count = sum(1 for r in results if r['base_url'] == base_url)
            print(f"  {base_url}: {count} 请求 ({count/total_requests*100:.1f}%)")
        
        print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(description='vLLM JSONL Benchmark')
    parser.add_argument('--jsonl', type=str, required=True, help='JSONL 数据文件路径')
    parser.add_argument('--hosts', type=str, required=True, help='节点列表，格式: host1:port1,host2:port2')
    parser.add_argument('--num-requests', type=int, default=1000, help='请求数量')
    parser.add_argument('--concurrency', type=int, default=64, help='并发数')
    parser.add_argument('--max-tokens', type=int, default=1024, help='最大输出 token 数')
    parser.add_argument('--model', type=str, default='/workspace/Qwen3-8B', help='模型名称')
    parser.add_argument('--strategy', type=str, default='round-robin', 
                       choices=['round-robin', 'random', 'shuffle'], 
                       help='负载均衡策略: round-robin(轮询), random(随机), shuffle(打乱后轮询)')
    parser.add_argument('--start-line', type=int, default=0, help='起始行号')
    
    args = parser.parse_args()
    
    # 解析 hosts
    hosts = []
    for host_str in args.hosts.split(','):
        host, port = host_str.strip().split(':')
        hosts.append((host, int(port)))
    
    # 运行 benchmark
    benchmark = VLLMBenchmark(hosts, args.max_tokens, args.model)
    benchmark.run_benchmark(
        args.jsonl,
        args.num_requests,
        args.concurrency,
        args.strategy,
        args.start_line
    )

if __name__ == '__main__':
    main()

