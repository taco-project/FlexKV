# P2P 跨实例 KV 复用 — 运维快速参考

> 给运维 / SRE 的 30 秒速读版。详细技术背景见
> `KNOWN_ISSUE_p2p_refcount_2026-05-14.md`。

## ⚠️ 一句话风险

启用 `enable_p2p_cpu=True` 后，跨实例 KV 复用依赖 **lease 时间窗口** 防止脏读，
**不依赖** 显式 refcount。配置不当会导致 LLM 输出乱码。

## ✅ 安全部署 4 步走

### 1. 强制环境变量

```bash
export FLEXKV_LEASE_TTL_MS=30000        # 必须 >= 10000
export FLEXKV_SAFETY_TTL_MS=100
export FLEXKV_RENEW_LEASE_MS=4000        # 必须 <= lease_ttl / 5
export FLEXKV_REBUILD_INTERVAL_MS=100
```

> ⚠️ **如果你设置 `FLEXKV_LEASE_TTL_MS < 10000`，FlexKV 会拒绝启动并报错**——
> 这是有意为之的硬约束，不要绕过。

### 2. 容量规划

master 端 CPU pool 必须能容纳：

```
peak_concurrent_requests × avg_seq_tokens / tokens_per_block × 2
```

**实测参考**（H800 + Qwen3-8B + tokens_per_block=16）：
- 并发 64 + 平均 200 blocks/req → **至少 25600 blocks**
- 不够会触发"高水位 evict"路径，lease 防线失效

### 3. 监控指标（必接）

| 指标 | 阈值 | 触发动作 |
|---|---|---|
| `cpu_pool_utilization` | > 95% 持续 60s | 告警，扩容 |
| `lease_meta_nullptr_count` | > 0 | **致命**，立即停 P2P |
| `mooncake_read_p99_latency_ms` | > 500ms | 告警 |
| `mooncake_read_failure_rate` | > 0.1% | 告警 |
| `cross_instance_hit_text_garbage_rate` | > 0.01% | **致命**，立即停 P2P |

### 4. 灰度策略

第一次上线建议：

1. **第 1 周**：开启 P2P 但只让 5% 流量走，业务侧加输出健康度抽样对比
2. **第 2 周**：流量提升到 50%，监控数据稳定无告警则继续
3. **第 3 周**：全量

---

## 🚨 出问题怎么办

### 现象 1：LLM 输出乱码 / 重复 token / 完全无关内容

**99% 是 lease 防线被击穿**。立即：

```bash
# 紧急停用 P2P，回退到本地命中
export FLEXKV_ENABLE_P2P_CPU=0
# 或在 sglang yaml 里 enable_p2p_cpu: false
# 重启服务
```

然后查 `lease_meta_nullptr_count` 监控，如果 > 0，说明已进入高水位强压路径，
**必须扩容 master 端 CPU pool 或启动 refcount glue 实现**（见 known-issue 文档 §5）。

### 现象 2：跨实例命中率突然降到 0

不是数据正确性问题，是 lease 过期或 Redis 心跳断了。查：

```bash
# 看 master 端
docker exec <flexkv> python3 -c '
import redis
r = redis.Redis(host="<redis>", port=6379, password="<pwd>", db=2)
keys = r.keys("sd:*:node:*")
print("active nodes:", [(k.decode(), r.ttl(k)) for k in keys])
'
# TTL < 5s 说明心跳要断了
```

### 现象 3：FlexKV 启动报错 `FLEXKV_LEASE_TTL_MS=... is below the safety floor`

按报错提示把环境变量调大到 >= 10000。**不要**改源码绕过这个 check。

---

## 🔍 健康度自检脚本

```bash
#!/bin/bash
# flexkv_p2p_healthcheck.sh — 在 master / peer 任意一台跑

echo "=== FlexKV P2P 健康度自检 ==="

# 1. 配置检查
LEASE_TTL=${FLEXKV_LEASE_TTL_MS:-30000}
if [ "$LEASE_TTL" -lt 10000 ]; then
  echo "❌ FLEXKV_LEASE_TTL_MS=$LEASE_TTL < 10000 (UNSAFE)"
  exit 1
fi
echo "✅ FLEXKV_LEASE_TTL_MS=$LEASE_TTL"

# 2. Redis 心跳
docker exec flexkv_distreuse python3 -c "
import redis
r = redis.Redis(host='${REDIS_HOST:-127.0.0.1}', port=6379, password='${REDIS_PWD}', db=2)
keys = r.keys('sd:*:node:*')
if not keys:
  print('❌ no active nodes in Redis')
  exit(1)
for k in sorted(keys):
  ttl = r.ttl(k)
  status = '✅' if ttl > 5 else '⚠️ ' if ttl > 0 else '❌'
  print(f'{status} {k.decode()}: TTL={ttl}s')
"

# 3. CPU pool 利用率（需要先实现 metrics 暴露端点）
# curl -s http://localhost:8080/metrics | grep -E 'cpu_pool_utilization|lease_meta_nullptr'

echo "=== 自检完成 ==="
```

---

## 📞 升级到无 lease 漏洞的版本

满足下面任一条件，立即联系 FlexKV 团队启动 **方案 A（refcount handshake）**：

1. `lease_meta_nullptr_count > 0` 在生产出现
2. `cross_instance_hit_text_garbage_rate > 0.01%`
3. 业务需要 `lease_ttl_ms < 10000`
4. PP > 1 或 tp_node_count > 1 的 multi-SD 部署
5. 多 peer 并发量 > 4

详细决策矩阵见 `KNOWN_ISSUE_p2p_refcount_2026-05-14.md` §5。

---

*最后更新：2026-05-14*
