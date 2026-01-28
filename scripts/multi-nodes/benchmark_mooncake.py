#!/usr/bin/env python3
"""
Mooncake Trace Benchmark 脚本
正确处理 hash_id 到 token 的映射，支持 prefix cache 测试

Mooncake trace 格式:
{
    "timestamp": 27482,        # 时间戳（毫秒）
    "input_length": 6955,      # 输入 token 数
    "output_length": 52,       # 输出 token 数
    "hash_ids": [46, 47, ...]  # hash ID 列表（每个代表一个 KV cache block）
}

设计思路：
- 每个 hash_id 对应一个固定的 token block（默认 512 tokens）
- 相同 hash_id 生成相同的 token 序列，确保 prefix cache 可复用
- 使用 prompt_token_ids 直接发送 token IDs（如果支持），否则用文本
"""

import json
import time
import asyncio
import aiohttp
import argparse
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from transformers import AutoTokenizer


@dataclass
class MooncakeRequest:
    """Mooncake 请求数据结构"""
    timestamp: int  # 毫秒
    input_length: int
    output_length: int
    hash_ids: List[int]
    request_id: int = 0


class TokenBlockGenerator:
    """
    Token Block 生成器
    每个 hash_id 映射到一个固定的 token block
    """
    
    def __init__(self, tokenizer_path: str, block_size: int = 512, seed: int = 42):
        """
        Args:
            tokenizer_path: tokenizer 路径
            block_size: 每个 hash block 的 token 数量
            seed: 随机种子
        """
        print(f"加载 tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.block_size = block_size
        self.seed = seed
        
        # 缓存：hash_id -> token_ids
        self.hash_to_tokens: Dict[int, List[int]] = {}
        
        # 获取词汇表大小，用于生成随机 token
        self.vocab_size = self.tokenizer.vocab_size
        
        # 排除特殊 token
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        
        print(f"Tokenizer 加载完成，词汇表大小: {self.vocab_size}, Block 大小: {block_size}")
    
    def _generate_block_tokens(self, hash_id: int) -> List[int]:
        """为特定 hash_id 生成固定的 token block"""
        if hash_id in self.hash_to_tokens:
            return self.hash_to_tokens[hash_id]
        
        # 使用 hash_id 作为随机种子，确保相同 hash_id 生成相同的 tokens
        rng = random.Random(self.seed + hash_id * 12345)
        
        tokens = []
        while len(tokens) < self.block_size:
            # 生成随机 token ID，避开特殊 token
            token_id = rng.randint(100, self.vocab_size - 1)
            if token_id not in self.special_token_ids:
                tokens.append(token_id)
        
        tokens = tokens[:self.block_size]
        self.hash_to_tokens[hash_id] = tokens
        return tokens
    
    def generate_prompt_tokens(self, hash_ids: List[int], target_length: int) -> List[int]:
        """
        根据 hash_ids 和目标长度生成 prompt token IDs
        
        Args:
            hash_ids: hash ID 列表
            target_length: 目标 token 数
            
        Returns:
            token ID 列表
        """
        # 拼接所有 hash block 的 tokens
        all_tokens = []
        for hash_id in hash_ids:
            block_tokens = self._generate_block_tokens(hash_id)
            all_tokens.extend(block_tokens)
        
        # 调整到目标长度
        if len(all_tokens) >= target_length:
            # 截断
            return all_tokens[:target_length]
        else:
            # 填充（使用最后一个 hash_id 的延续或随机 tokens）
            rng = random.Random(self.seed + target_length)
            while len(all_tokens) < target_length:
                token_id = rng.randint(100, self.vocab_size - 1)
                if token_id not in self.special_token_ids:
                    all_tokens.append(token_id)
            return all_tokens[:target_length]
    
    def tokens_to_text(self, token_ids: List[int]) -> str:
        """将 token IDs 转换为文本"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


class MooncakeBenchmark:
    """Mooncake Trace Benchmark 主类"""
    
    def __init__(
        self,
        hosts: List[tuple],
        tokenizer_path: str,
        block_size: int = 512,
        model_name: str = None,
        timeout: int = 300,
        use_token_ids: bool = False,  # 是否使用 prompt_token_ids
    ):
        self.hosts = hosts
        self.base_urls = [f"http://{host}:{port}/v1" for host, port in hosts]
        self.model_name = model_name or tokenizer_path
        self.timeout = timeout
        self.use_token_ids = use_token_ids
        self.host_counter = 0
        
        # Token 生成器
        self.token_generator = TokenBlockGenerator(tokenizer_path, block_size)
        
        # 统计信息
        self.results: List[Dict[str, Any]] = []
    
    def get_next_url(self, strategy: str = 'round-robin') -> str:
        """获取下一个目标 URL"""
        if strategy == 'round-robin':
            url = self.base_urls[self.host_counter % len(self.base_urls)]
            self.host_counter += 1
            return url
        elif strategy == 'random':
            return random.choice(self.base_urls)
        else:
            return self.base_urls[0]
    
    def get_url_by_hash(self, hash_ids: List[int]) -> str:
        """根据 hash_ids 选择节点"""
        if not hash_ids:
            return random.choice(self.base_urls)
        return self.base_urls[hash_ids[0] % len(self.base_urls)]
    
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        request: MooncakeRequest,
        base_url: str,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        """发送单个请求"""
        async with semaphore:
            
            # 生成 prompt tokens
            prompt_tokens = self.token_generator.generate_prompt_tokens(
                request.hash_ids,
                request.input_length
            )
            
            # 构建请求
            if self.use_token_ids:
                # 使用 prompt_token_ids（需要 vLLM 支持）
                payload = {
                    "model": self.model_name,
                    "prompt_token_ids": prompt_tokens,
                    "max_tokens": request.output_length,
                    "temperature": 0.7,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            else:
                # 转换为文本
                prompt_text = self.token_generator.tokens_to_text(prompt_tokens)
                payload = {
                    "model": self.model_name,
                    "prompt": prompt_text,
                    "max_tokens": request.output_length,
                    "temperature": 0.7,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
            
            start_time = time.time()
            first_token_time = None
            completion_tokens = 0
            prompt_tokens_actual = 0
            
            try:
                url = f"{base_url}/completions"
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response is None:
                        return {
                            'success': False,
                            'request_id': request.request_id,
                            'base_url': base_url,
                            'error': 'Response is None',
                            'timestamp': request.timestamp,
                        }
                    
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            'success': False,
                            'request_id': request.request_id,
                            'base_url': base_url,
                            'error': f"HTTP {response.status}: {error_text[:200]}",
                            'timestamp': request.timestamp,
                        }
                    
                    # 流式读取响应
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line or not line.startswith('data: '):
                            continue
                        
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            # 记录首 token 时间
                            choices = data.get('choices', [])
                            if first_token_time is None and choices:
                                # choices 可能是空列表（usage chunk）
                                if len(choices) > 0 and choices[0].get('text'):
                                    first_token_time = time.time()
                            
                            # 提取 usage 信息
                            usage = data.get('usage')
                            if usage:
                                prompt_tokens_actual = usage.get('prompt_tokens', 0)
                                completion_tokens = usage.get('completion_tokens', 0)
                        except json.JSONDecodeError:
                            continue
                    
                    end_time = time.time()
                    
                    # 计算指标
                    ttft = first_token_time - start_time if first_token_time else 0
                    total_time = end_time - start_time
                    decode_time = end_time - first_token_time if first_token_time else 0
                    
                    result = {
                        'success': True,
                        'request_id': request.request_id,
                        'base_url': base_url,
                        'timestamp': request.timestamp,
                        'target_input_length': request.input_length,
                        'target_output_length': request.output_length,
                        'actual_prompt_tokens': prompt_tokens_actual,
                        'actual_completion_tokens': completion_tokens,
                        'hash_ids': request.hash_ids,
                        'hash_prefix_len': len(request.hash_ids),
                        'ttft': ttft,
                        'total_time': total_time,
                        'decode_time': decode_time,
                        'prefill_speed': prompt_tokens_actual / ttft if ttft > 0 else 0,
                        'decode_speed': (completion_tokens - 1) / decode_time if decode_time > 0 and completion_tokens > 1 else 0,
                        'start_time': start_time,
                        'end_time': end_time,
                    }
                    
                    port = base_url.split(':')[-1].split('/')[0]
                    print(f"[Req {request.request_id:4d}] Port: {port}, "
                          f"Prompt: {prompt_tokens_actual:5d} tokens (target: {request.input_length}), "
                          f"TTFT: {ttft:.3f}s, "
                          f"Prefill: {result['prefill_speed']:7.1f} tok/s, "
                          f"Output: {completion_tokens:3d} tokens, "
                          f"Decode: {result['decode_speed']:6.1f} tok/s, "
                          f"HashBlocks: {len(request.hash_ids)}")
                    
                    return result
                    
            except asyncio.TimeoutError:
                return {
                    'success': False,
                    'request_id': request.request_id,
                    'base_url': base_url,
                    'error': 'Timeout',
                    'timestamp': request.timestamp,
                }
            except Exception as e:
                return {
                    'success': False,
                    'request_id': request.request_id,
                    'base_url': base_url,
                    'error': str(e),
                    'timestamp': request.timestamp,
                }
    
    async def run_benchmark(
        self,
        requests: List[MooncakeRequest],
        strategy: str = 'round-robin',
        max_concurrency: int = 64,
        time_scale: float = 1.0,
        use_timestamp: bool = True,
    ):
        """运行 benchmark"""
        print("="*80)
        print(" Mooncake Trace Benchmark")
        print("="*80)
        print(f"请求数量: {len(requests)}")
        print(f"负载均衡策略: {strategy}")
        print(f"最大并发数: {max_concurrency}")
        print(f"按时间戳发送: {use_timestamp}")
        print(f"Block 大小: {self.token_generator.block_size} tokens")
        if use_timestamp:
            print(f"时间缩放因子: {time_scale}x")
        print(f"目标节点:")
        for i, (host, port) in enumerate(self.hosts, 1):
            print(f"  节点 {i}: {host}:{port}")
        print("="*80)
        print()
        
        # 处理请求顺序
        if strategy == 'shuffle':
            # Shuffle 策略：打乱请求顺序，然后用 round-robin 分发
            # 这样相同前缀的请求更可能被分配到不同节点，测试跨节点 KV cache reuse
            print("📊 Shuffle 策略：打乱请求顺序后使用 round-robin 分发")
            sorted_requests = list(requests)
            random.shuffle(sorted_requests)
            # 打乱后使用 round-robin
            actual_strategy = 'round-robin'
        else:
            # 按 timestamp 排序
            sorted_requests = sorted(requests, key=lambda r: r.timestamp)
            actual_strategy = strategy
        
        base_timestamp = sorted_requests[0].timestamp if sorted_requests else 0
        
        semaphore = asyncio.Semaphore(max_concurrency)
        
        # 预先分配每个请求的目标 URL（避免并发时的竞争条件）
        request_urls = []
        for i, req in enumerate(sorted_requests):
            if actual_strategy == 'hash':
                url = self.get_url_by_hash(req.hash_ids)
            elif actual_strategy == 'round-robin':
                url = self.base_urls[i % len(self.base_urls)]
            elif actual_strategy == 'random':
                url = random.choice(self.base_urls)
            else:
                url = self.base_urls[0]
            request_urls.append(url)
        
        # 统计分发情况
        url_counts = {}
        for url in request_urls:
            url_counts[url] = url_counts.get(url, 0) + 1
        print(f"\n📊 请求分发统计:")
        for url, count in sorted(url_counts.items()):
            print(f"  {url}: {count} 请求 ({count/len(request_urls)*100:.1f}%)")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            benchmark_start_time = time.time()
            
            for i, req in enumerate(sorted_requests):
                if use_timestamp:
                    target_delay = (req.timestamp - base_timestamp) / 1000.0 * time_scale
                    elapsed = time.time() - benchmark_start_time
                    wait_time = target_delay - elapsed
                    
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                
                task = asyncio.create_task(
                    self.send_request(session, req, request_urls[i], semaphore)
                )
                tasks.append(task)
            
            print(f"\n所有请求已发送，等待完成...")
            results = await asyncio.gather(*tasks)
        
        total_time = time.time() - benchmark_start_time
        self._print_statistics(results, total_time)
        
        return results
    
    def _print_statistics(self, results: List[Dict], total_time: float):
        """打印统计结果"""
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        print("\n" + "="*80)
        print(" Benchmark 结果")
        print("="*80)
        
        print(f"\n📊 基础统计:")
        print(f"  总请求数: {len(results)}")
        print(f"  成功请求: {len(successful)}")
        print(f"  失败请求: {len(failed)}")
        print(f"  成功率: {len(successful)/len(results)*100:.1f}%")
        print(f"  总耗时: {total_time:.2f}s")
        print(f"  实际 QPS: {len(results)/total_time:.2f}")
        
        if not successful:
            print("\n⚠️ 没有成功的请求!")
            if failed:
                print("错误示例:")
                for f in failed[:5]:
                    print(f"  - Req {f['request_id']}: {f.get('error', 'Unknown error')}")
            return
        
        # Token 统计
        total_prompt_tokens = sum(r['actual_prompt_tokens'] for r in successful)
        total_completion_tokens = sum(r['actual_completion_tokens'] for r in successful)
        
        print(f"\n📝 Token 统计:")
        print(f"  总 Prompt Tokens: {total_prompt_tokens:,}")
        print(f"  总 Completion Tokens: {total_completion_tokens:,}")
        print(f"  平均 Prompt 长度: {total_prompt_tokens/len(successful):.1f}")
        print(f"  平均 Completion 长度: {total_completion_tokens/len(successful):.1f}")
        
        # 吞吐量
        total_ttft = sum(r['ttft'] for r in successful if r['ttft'] > 0)
        total_decode_time = sum(r['decode_time'] for r in successful if r['decode_time'] > 0)
        
        print(f"\n🚀 吞吐量:")
        print(f"  端到端吞吐: {(total_prompt_tokens + total_completion_tokens)/total_time:.2f} tokens/s")
        if total_ttft > 0:
            print(f"  Prefill 吞吐: {total_prompt_tokens/total_ttft:.2f} tokens/s")
        if total_decode_time > 0:
            print(f"  Decode 吞吐: {total_completion_tokens/total_decode_time:.2f} tokens/s")
        
        # TTFT 统计
        ttfts = sorted([r['ttft'] for r in successful if r['ttft'] > 0])
        if ttfts:
            print(f"\n⏱️ TTFT (Time to First Token):")
            print(f"  平均: {sum(ttfts)/len(ttfts):.3f}s")
            print(f"  中位数: {ttfts[len(ttfts)//2]:.3f}s")
            print(f"  P95: {ttfts[int(len(ttfts)*0.95)]:.3f}s")
            print(f"  P99: {ttfts[int(len(ttfts)*0.99)]:.3f}s")
            print(f"  最小: {ttfts[0]:.3f}s")
            print(f"  最大: {ttfts[-1]:.3f}s")
        
        # Decode 速度统计
        decode_speeds = sorted([r['decode_speed'] for r in successful if r['decode_speed'] > 0])
        if decode_speeds:
            print(f"\n⚡ Decode 速度:")
            print(f"  平均: {sum(decode_speeds)/len(decode_speeds):.2f} tokens/s")
            print(f"  中位数: {decode_speeds[len(decode_speeds)//2]:.2f} tokens/s")
        
        # 按节点统计
        print(f"\n🖥️ 每个节点的请求分布:")
        for base_url in self.base_urls:
            node_results = [r for r in successful if r['base_url'] == base_url]
            count = len(node_results)
            if count > 0:
                avg_ttft = sum(r['ttft'] for r in node_results) / count
                print(f"  {base_url}: {count} 请求 ({count/len(successful)*100:.1f}%), "
                      f"平均TTFT: {avg_ttft:.3f}s")
        
        # Hash prefix 分析
        hash_lens = [r['hash_prefix_len'] for r in successful]
        if hash_lens:
            print(f"\n🔗 Hash Block 分析:")
            print(f"  平均 block 数: {sum(hash_lens)/len(hash_lens):.1f}")
            print(f"  最小 block 数: {min(hash_lens)}")
            print(f"  最大 block 数: {max(hash_lens)}")
            
            # 统计唯一 prefix
            unique_prefixes = set()
            for r in successful:
                prefix_tuple = tuple(r['hash_ids'])
                unique_prefixes.add(prefix_tuple)
            print(f"  唯一 prefix 数量: {len(unique_prefixes)}")
            
            # 统计 prefix 共享情况
            prefix_counts = defaultdict(int)
            for r in successful:
                # 检查每个可能的前缀长度
                for prefix_len in range(1, len(r['hash_ids']) + 1):
                    prefix = tuple(r['hash_ids'][:prefix_len])
                    prefix_counts[prefix] += 1
            
            # 找出共享最多的前缀
            shared_prefixes = [(p, c) for p, c in prefix_counts.items() if c > 1]
            if shared_prefixes:
                shared_prefixes.sort(key=lambda x: -x[1])
                print(f"  共享的 prefix 数量: {len(shared_prefixes)}")
                print(f"  最常共享的 prefix: {shared_prefixes[0][1]} 次 (长度 {len(shared_prefixes[0][0])})")
        
        # 失败请求分析
        if failed:
            print(f"\n❌ 失败请求分析:")
            error_counts = defaultdict(int)
            for f in failed:
                error = f.get('error', 'Unknown')[:50]
                error_counts[error] += 1
            for error, count in sorted(error_counts.items(), key=lambda x: -x[1])[:5]:
                print(f"  {error}: {count} 次")
        
        print("\n" + "="*80)


def load_mooncake_trace(filepath: str, num_requests: int = None, start_line: int = 0) -> List[MooncakeRequest]:
    """加载 Mooncake trace 数据"""
    requests = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if num_requests and len(requests) >= num_requests:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                req = MooncakeRequest(
                    timestamp=data.get('timestamp', 0),
                    input_length=data.get('input_length', 100),
                    output_length=data.get('output_length', 50),
                    hash_ids=data.get('hash_ids', []),
                    request_id=len(requests),
                )
                requests.append(req)
            except json.JSONDecodeError as e:
                print(f"警告: 第 {i+1} 行 JSON 解析失败: {e}")
                continue
    
    return requests


def main():
    parser = argparse.ArgumentParser(
        description='Mooncake Trace Benchmark for vLLM + FlexKV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python benchmark_mooncake.py --trace /path/to/trace.jsonl \\
      --hosts 10.6.131.12:30001 \\
      --tokenizer /workspace/Qwen3-8B

  # 多节点 + hash 策略（测试 prefix cache）
  python benchmark_mooncake.py --trace /path/to/trace.jsonl \\
      --hosts 10.6.131.12:30001,10.6.131.12:30002 \\
      --tokenizer /workspace/Qwen3-8B \\
      --strategy hash \\
      --block-size 512

  # 高并发压测
  python benchmark_mooncake.py --trace /path/to/trace.jsonl \\
      --hosts 10.6.131.12:30001 \\
      --tokenizer /workspace/Qwen3-8B \\
      --no-timestamp --concurrency 128
        """
    )
    
    parser.add_argument('--trace', type=str, required=True,
                        help='Mooncake trace JSONL 文件路径')
    parser.add_argument('--hosts', type=str, required=True,
                        help='节点列表，格式: host1:port1,host2:port2')
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Tokenizer 路径（用于生成 token）')
    parser.add_argument('--block-size', type=int, default=512,
                        help='每个 hash block 的 token 数量（默认: 512）')
    parser.add_argument('--num-requests', type=int, default=None,
                        help='请求数量（默认: 全部）')
    parser.add_argument('--start-line', type=int, default=0,
                        help='起始行号')
    parser.add_argument('--concurrency', type=int, default=64,
                        help='最大并发数')
    parser.add_argument('--strategy', type=str, default='round-robin',
                        choices=['round-robin', 'random', 'hash', 'shuffle'],
                        help='负载均衡策略')
    parser.add_argument('--model', type=str, default=None,
                        help='模型名称（默认使用 tokenizer 路径）')
    parser.add_argument('--time-scale', type=float, default=1.0,
                        help='时间缩放因子')
    parser.add_argument('--no-timestamp', action='store_true',
                        help='不按时间戳发送')
    parser.add_argument('--use-token-ids', action='store_true',
                        help='使用 prompt_token_ids 而非文本（需要 vLLM 支持）')
    parser.add_argument('--timeout', type=int, default=300,
                        help='请求超时时间（秒）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果到 JSON 文件')
    
    args = parser.parse_args()
    
    # 解析 hosts
    hosts = []
    for host_str in args.hosts.split(','):
        host_str = host_str.strip()
        if ':' in host_str:
            host, port = host_str.split(':')
            hosts.append((host, int(port)))
        else:
            hosts.append((host_str, 8000))
    
    # 加载数据
    print(f"加载 Mooncake trace: {args.trace}")
    requests = load_mooncake_trace(args.trace, args.num_requests, args.start_line)
    print(f"加载了 {len(requests)} 个请求")
    
    if not requests:
        print("错误: 没有加载到任何请求!")
        return
    
    # 打印数据集统计
    print(f"\n📊 数据集统计:")
    if requests[-1].timestamp > requests[0].timestamp:
        print(f"  时间范围: {requests[0].timestamp}ms - {requests[-1].timestamp}ms")
        print(f"  持续时间: {(requests[-1].timestamp - requests[0].timestamp)/1000:.1f}s")
    avg_input = sum(r.input_length for r in requests) / len(requests)
    avg_output = sum(r.output_length for r in requests) / len(requests)
    print(f"  平均输入长度: {avg_input:.1f} tokens")
    print(f"  平均输出长度: {avg_output:.1f} tokens")
    avg_hash_len = sum(len(r.hash_ids) for r in requests) / len(requests)
    print(f"  平均 hash block 数: {avg_hash_len:.1f}")
    print(f"  预计每个 block: {args.block_size} tokens")
    print()
    
    # 创建 benchmark 实例
    benchmark = MooncakeBenchmark(
        hosts=hosts,
        tokenizer_path=args.tokenizer,
        block_size=args.block_size,
        model_name=args.model,
        timeout=args.timeout,
        use_token_ids=args.use_token_ids,
    )
    
    # 运行 benchmark
    use_timestamp = not args.no_timestamp and args.time_scale > 0
    results = asyncio.run(benchmark.run_benchmark(
        requests=requests,
        strategy=args.strategy,
        max_concurrency=args.concurrency,
        time_scale=args.time_scale,
        use_timestamp=use_timestamp,
    ))
    
    # 保存结果
    if args.output:
        # 移除不可序列化的字段
        serializable_results = []
        for r in results:
            sr = {k: v for k, v in r.items() if k != 'hash_ids' or isinstance(v, (list, tuple))}
            serializable_results.append(sr)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
