# FlexKV 持续集成说明

[English](README_en.md)

本文介绍 GitHub Actions 工作流
[`publish.yml`](../../.github/workflows/publish.yml) 的执行环境、检查范围和明确不覆盖的
GPU/集成测试。

## 触发条件

`flexkv ci` 工作流在以下情况触发：

- 目标分支为 `main` 或 `dev` 的 Pull Request；
- 向 `main` 或 `dev` 分支推送代码。

工作流只有仓库只读权限（`contents: read`）。Pull Request 会构建并测试安装包，但拿不到
COS 凭据，也不会上传产物。受保护分支的 push 完成相同检查后，会将生成的 wheel 上传到
COS。

## Runner 和版本矩阵

| 项目 | 当前配置 |
| --- | --- |
| Runner | GitHub-hosted `ubuntu-22.04` |
| Python | `3.10` |
| PyTorch | `2.6.0` |
| CUDA toolkit | `12.4` |
| CUDA 架构 | `8.9`、`9.0+PTX` |
| 并行编译任务数 | `4` |

当前只有这一套构建配置，并不是真正的多版本矩阵。该 GitHub-hosted runner 是 CPU
runner：安装 CUDA 是为了编译 wheel，unit-test 阶段不依赖 GPU。

## CI 检查内容

`Build Wheel` job 按顺序执行：

1. Checkout Pull Request 或受保护分支的 commit。
2. 安装 Linux 编译依赖。
3. 安装 Python 3.10、CUDA 12.4 和 PyTorch 2.6.0。
4. 执行 `./build.sh --release`，其中包含 release 模式的 Cython 编译。
5. 安装 `requirements.txt`，然后强制安装刚构建出的 wheel。
6. 切换到 runner 临时目录并运行 CPU 单元测试。
7. 仅在 push 到 `main` 或 `dev` 时，将 `dist/` 上传至 COS。

测试刻意在源码目录外运行，使 Python import 指向已经安装的 wheel，确保 CI 验证的是
实际构建产物，而不是 checkout 目录中的 Python 源文件。

## CPU 单元测试筛选规则

CI 会查找所有包含以下模块级 marker 的 `tests/**/test_*.py` 文件：

```python
import pytest

pytestmark = pytest.mark.unit
```

如果一个匹配文件都没有，工作流会直接失败。筛选出的测试通过以下命令执行：

```bash
timeout 10m python -m pytest -q --maxfail=1 <selected-test-files>
```

要把新测试加入这个 CI 层级，需要：

1. 保证测试只依赖 CPU，不依赖 GPU、网络服务、存储设备或仓库 secrets。
2. 在模块级添加 `pytestmark = pytest.mark.unit`。
3. 确认测试可以在仓库目录外、使用已安装的 release wheel 运行通过。

## 在 Ubuntu 22.04 本地复现

在 FlexKV 仓库根目录执行以下命令。环境初始化脚本会安装系统依赖，可能需要 `sudo`
权限。

```bash
bash -x .github/workflows/scripts/env.sh
bash -x .github/workflows/scripts/cuda-install.sh 12.4 ubuntu-22.04
bash -x .github/workflows/scripts/pytorch-install.sh 3.10 2.6.0 12.4

TORCH_CUDA_ARCH_LIST="8.9 9.0+PTX" MAX_JOBS=4 ./build.sh --release
python -m pip install -r requirements.txt
python -m pip install --force-reinstall --no-deps dist/*.whl

mapfile -t unit_test_files < <(
  python - <<'PY'
from pathlib import Path

marker = "pytestmark = pytest.mark.unit"
for path in sorted(Path("tests").resolve().rglob("test_*.py")):
    if marker in path.read_text():
        print(path)
PY
)

ci_test_dir=$(mktemp -d)
cd "$ci_test_dir"
timeout 10m python -m pytest -q --maxfail=1 "${unit_test_files[@]}"
```

`timeout` 和 `mapfile` 是 GNU/Linux 命令。如需完全复现工作流，请使用 Ubuntu 环境。

## 参考耗时

2026 年 8 月 6 日的
[GitHub Actions run #31087143283](https://github.com/taco-project/FlexKV/actions/runs/31087143283)
执行成功，各阶段耗时取整后如下：

| 阶段 | 耗时 |
| --- | ---: |
| Linux 环境初始化 | 1 分 31 秒 |
| Python 初始化 | 30 秒 |
| 安装 CUDA | 5 分 06 秒 |
| 安装 PyTorch | 2 分 10 秒 |
| 构建 release wheel | 6 分 49 秒 |
| 安装 wheel 和测试依赖 | 5 秒 |
| CPU 单元测试步骤 | 4 秒 |
| Job 总耗时 | **17 分 05 秒** |

该次运行发现 12 个单元测试模块，结果为 `198 passed, 1 skipped in 1.01s`。总耗时只是
参考值，不是 SLA；runner 调度和依赖缓存状态都可能导致明显变化。

## 产物上传

COS 上传步骤只在 push 到 `main` 或 `dev` 时执行，需要配置以下仓库 secrets：

- `COS_SECRET_ID`；
- `COS_SECRET_KEY`；
- `COS_BUCKET`；
- `COS_ENDPOINT`。

工作流会把 `dist/` 上传到 `flexkv/<date>/<time>`。Pull Request 始终跳过日期生成和上传
步骤，因此 PR 中缺少 COS secrets 不应被判断为构建或测试失败。

## 当前未覆盖范围

当前工作流不验证：

- GPU 运行时正确性或 CUDA kernel 执行；
- 多 GPU、TP、DP、PP 或跨节点行为；
- SSD、GDS、RDMA、Mooncake、Redis 或 COS 数据往返；
- vLLM、SGLang、Dynamo 或 TensorRT-LLM 端到端服务；
- 性能回归或长稳测试；
- 其他 Python、PyTorch、CUDA 或 Ubuntu 版本。

这些路径在发布前仍需单独执行 GPU 和集成测试。

## 查看 Pull Request 状态

可以直接查看 GitHub Pull Request 页面，或使用 GitHub CLI：

```bash
gh pr checks <pr-number> --repo taco-project/FlexKV
```

发生失败时，应先定位第一个失败步骤并检查完整日志，尤其要区分构建/单元测试失败和依赖
secret 的产物上传失败；Pull Request 中跳过上传属于预期行为。
