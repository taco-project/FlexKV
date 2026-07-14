# `dpskv4_refactor` 合入 `main` 验证结果

执行说明见
[`dpskv4_main_merge_validation.md`](dpskv4_main_merge_validation.md)。所有结果必须绑定
被测 commit SHA；每个环境追加一个小节，不要覆盖已有记录。

## 汇总

| Commit | 环境 | Agent | Static | Build | Unit/Smoke | GPU | Framework | Multi-node | Soak | Overall |
|---|---|---|---|---|---|---|---|---|---|---|
| 待填写 | 待填写 | 待填写 | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

## 环境记录

### 待填写：环境名称

```text
Commit:
Tester/agent:
Date:
Host/GPU:
OS/driver/toolkit:
Python/PyTorch:
Framework versions:

Static checks: PASS / FAIL / BLOCKED
Default build: PASS / FAIL / BLOCKED
P2P build: PASS / FAIL / BLOCKED
GDS/NIXL build: PASS / FAIL / BLOCKED
nvCOMP build: PASS / FAIL / BLOCKED
Unit tests: passed / failed / skipped
Smoke tests: passed / failed / skipped
NVIDIA GPU tests: PASS / FAIL / BLOCKED
AMD GPU tests: PASS / FAIL / BLOCKED
SGLang DSv4: PASS / FAIL / BLOCKED
vLLM: PASS / FAIL / BLOCKED
TensorRT-LLM: PASS / FAIL / BLOCKED
Redis/Mooncake multi-node: PASS / FAIL / BLOCKED
1-hour soak: PASS / FAIL / BLOCKED
Performance delta:

Commands executed:
Failed command and full log/log link:
Minimal reproduction:
Suspected subsystem/owner:
Additional notes:
```

## 已确认问题

按以下格式追加，禁止只写结论：

```text
ID:
Commit:
Severity: blocker / high / medium / low
Environment:
Command:
Expected:
Actual:
Full log/log link:
Minimal reproduction:
Suspected files/subsystem:
Suggested fix (optional):
```
