# FlexKV CLI 使用文档

FlexKV CLI 是一个本地观察工具，用于查看配置、环境、进程、trace、SSD 目录和 Prometheus 指标。当前 CLI 不负责拉起或停止 FlexKV 运行时，也不修改推理进程状态。

当前实现没有 `flexkv version` 命令，因为 FlexKV 尚未提供独立的版本号元数据。`flexkv collect-env` 会显示当前代码仓的短 Git commit，可作为开发分支的辅助标识。

---

## 一、安装与入口

CLI 入口通过 `setup.py` 注册：

```python
entry_points={"console_scripts": ["flexkv=flexkv.cli.main:main"]}
```

在包含 CLI 实现的分支中使用可编辑安装：

```bash
cd /path/to/FlexKV
conda activate flexkv
pip install -e .

flexkv --help
```

如果修改的是 Python 源码，可编辑安装通常不需要重新安装；如果修改了 `setup.py` 中的入口点，则需要重新执行 `pip install -e .`。

---

## 二、命令总览

```text
flexkv
├── env             # 查看 FLEXKV_* 和 ENABLE_FLEXKV 环境变量
├── config          # 查看 / 校验本地配置
│   ├── show
│   └── validate
├── collect-env     # 收集排障用的环境摘要
├── status          # 查看本地进程、IPC 文件和 metrics 端口
├── trace           # 查看 trace 文件
│   ├── list
│   └── show
├── storage         # 查看 SSD 缓存目录占用
└── metrics         # 抓取本地 Prometheus 指标
```

| 命令 | 主要问题 | 是否需要运行时 |
|---|---|---|
| `flexkv env` | 当前 shell 里设置了哪些 FlexKV 变量？ | 否 |
| `flexkv config` | CLI 当前能解析到什么配置？ | 否 |
| `flexkv collect-env` | 排障时需要的基本环境和依赖是什么？ | 否 |
| `flexkv status` | 本机有哪些疑似 FlexKV 进程和端点？ | 部分；无运行时时仍可执行 |
| `flexkv trace` | trace 文件在哪里、内容是什么？ | 需要应用已写入 trace |
| `flexkv storage` | SSD 缓存目录占用了多少空间？ | 需要目录存在 |
| `flexkv metrics` | Python / C++ metrics 端点暴露了什么？ | 需要 metrics 端点已启动 |

---

## 三、公共参数

所有一级命令都支持：

| 参数 | 说明 |
|---|---|
| `--format terminal` | 默认输出格式，适合人工查看 |
| `--format json` | JSON 输出，适合脚本处理 |
| `-q, --quiet` | 抑制 stdout 输出 |

示例：

```bash
flexkv env --format json
flexkv status --format json
flexkv metrics --format json
```

JSON 输出只写 stdout；错误信息写 stderr。

---

## 四、环境与配置

### 4.1 `flexkv env`

列出当前 shell 中所有 `FLEXKV_*` 变量，以及兼容变量 `ENABLE_FLEXKV`：

```bash
flexkv env
flexkv env --format json
```

该命令读取的是 CLI 进程继承到的环境变量，不会读取其他正在运行进程的 `/proc/<pid>/environ`。

### 4.2 `flexkv config`

`config show` 可以省略，`flexkv config` 默认等价于 `flexkv config show`。

```bash
flexkv config
flexkv config show
flexkv config validate
flexkv config --format json
```

配置来源规则：

1. 如果设置 `FLEXKV_CONFIG_PATH`，读取该 JSON 或 YAML 文件中的 `UserConfig` 字段。
2. 否则通过 `FLEXKV_*` 环境变量和默认值构造 `UserConfig`。
3. 输出中还会附加 `env.enable_metrics`、`env.py_metrics_port`、`env.trace_file_path` 等全局环境配置。

配置文件支持 `.json`、`.yaml` 和 `.yml`：

```bash
export FLEXKV_CONFIG_PATH=/path/to/flexkv.yaml
flexkv config show
```

`config validate` 当前检查：

- `cpu_cache_gb` 必须大于 0
- `ssd_cache_gb` 必须大于等于 0
- 启用 SSD 时，`ssd_cache_gb` 必须大于 `cpu_cache_gb`
- 启用 SSD 时，每个 `ssd_cache_dir` 都必须是已存在目录

注意：当前实现会在终端输出违规项，但没有为校验失败设置独立的非零退出码。`config show` 展示的是 CLI 侧解析结果，不代表某个正在运行推理进程内部最终合并后的配置。

### 4.3 `flexkv collect-env`

收集排障摘要：

```bash
flexkv collect-env
flexkv collect-env --format json
```

包含 OS、Python 版本、当前代码仓短 commit、PyTorch / CUDA / GPU 摘要、NVIDIA 驱动、关键 Python 包版本，以及 `FLEXKV_*` 变量数量。为了控制输出大小，它不导出完整环境变量内容；需要逐项查看时使用 `flexkv env`。

---

## 五、本地运行状态

### 5.1 `flexkv status`

```bash
flexkv status
flexkv status --format json
```

当前实现包含三类信息：

1. 命令行中包含 `flexkv` 或 `FLEXKV` 的本地进程。
2. `/tmp/flexkv_server*` IPC 端点文件。
3. Python 和 C++ metrics 端口是否可连接。

进程识别基于命令行文本。SGLang 或 vLLM 进程即使加载了 FlexKV 模块，如果启动命令本身不包含 FlexKV 关键字，当前也不会被识别为 FlexKV 进程。读取其他进程的环境变量可能受权限限制。

### 5.2 `flexkv trace`

FlexKV 当前没有独立通用日志文件，也没有 `flexkv log` 命令。CLI 只查看应用通过 FlexKV tracer 写出的 trace 文件。

trace 路径不需要每次显式传入：

- 优先读取 `FLEXKV_TRACE_FILE_PATH`
- 未设置时使用 `./flexkv_trace.log`
- `trace list` 还会查找 `<trace-path>.*` 形式的轮转文件

```bash
# 查看默认路径和轮转文件
flexkv trace list

# 指向自定义 trace 文件
export FLEXKV_TRACE_FILE_PATH=/tmp/flexkv_trace.log
flexkv trace list

# 查看完整内容
flexkv trace show

# 只看最后 100 行
flexkv trace show -n 100

# 显式指定文件
flexkv trace show -f /tmp/flexkv_trace.log
```

`trace show` 原样输出文件内容，不做 JSON 解析或字段过滤。文件不存在时退出码为 1。

### 5.3 `flexkv storage`

```bash
flexkv storage
flexkv storage --format json
```

该命令递归统计 `ssd_cache_dir` 下普通文件的总大小和数量，并显示目录是否存在。多个目录使用分号分隔：

```bash
export FLEXKV_SSD_CACHE_DIR="/data/flexkv_ssd_0;/data/flexkv_ssd_1"
flexkv storage
```

当前实现从 `FLEXKV_SSD_CACHE_DIR` 环境变量读取目录；即使设置了 `FLEXKV_CONFIG_PATH`，`storage` 也不会从配置文件中读取 SSD 路径。

---

## 六、Metrics

### 6.1 启用指标端点

FlexKV metrics 由运行时进程暴露，CLI 只负责抓取和展示：

```bash
export FLEXKV_ENABLE_METRICS=1
export FLEXKV_PY_METRICS_PORT=8080
export FLEXKV_CPP_METRICS_PORT=8081
```

随后启动加载 FlexKV 的应用。Python 端点通常为：

```text
http://127.0.0.1:8080/metrics
```

C++ 端点默认为：

```text
http://127.0.0.1:8081/metrics
```

### 6.2 `flexkv metrics`

```bash
flexkv metrics
flexkv metrics --format json
flexkv metrics --py-port 8080
flexkv metrics --cpp-port 8081
```

端口解析优先级：

1. `--py-port` / `--cpp-port`
2. `FLEXKV_PY_METRICS_PORT` / `FLEXKV_CPP_METRICS_PORT`
3. 默认值 `8080` / `8081`

CLI 会解析 Prometheus text format，并输出指标名、labels 和数值。当前不按 `flexkv_py_` 或 `flexkv_cpp_` 前缀过滤，因此 Python 进程的 `python_*`、`process_*` 指标也会显示。

如果 Python 端点可用而 C++ 端点未启动，终端会显示 Python 指标，并在 C++ 部分提示 `endpoint not reachable`。如果两个端点都不可用，命令退出码为 1。

指标含义和 Prometheus / Grafana 部署方式见 [FlexKV Prometheus Metrics 文档](../monitoring/README_zh.md)。

---

## 七、扩展开发

CLI 采用扁平命令结构：

```text
flexkv/cli/
├── __init__.py
├── main.py              # 薄入口
└── commands/
    ├── __init__.py      # 自动发现 BaseCommand 子类
    ├── base.py          # BaseCommand、终端表格和 JSON 输出
    ├── env.py
    ├── config.py
    ├── collect_env.py
    ├── status.py
    ├── trace.py
    ├── storage.py
    └── metrics.py
```

新增命令时创建一个 Python 模块，继承 `BaseCommand`，并实现：

```python
class MyCommand(BaseCommand):
    def name(self) -> str:
        return "my-command"

    def help(self) -> str:
        return "Short command description."

    def add_arguments(self, parser) -> None:
        ...

    def execute(self, args) -> None:
        ...
```

`commands/__init__.py` 会通过 `pkgutil.iter_modules()` 自动发现并实例化非抽象 `BaseCommand` 子类，不需要手工维护注册列表。命令逻辑应尽量自包含；确需复用时，优先复用 `base.py` 中的输出工具或已有命令中的小型 helper，不额外引入 `core/` 中间层。

后续 `bench`、`server`、`kvcache` 等目录型命令组可以按子目录扩展，但应等对应运行时能力明确后再实现。
