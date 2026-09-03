# FlexKV CLI Documentation

The FlexKV CLI is a local inspection tool for configuration, environment, processes, traces, SSD directories, and Prometheus metrics. It does not launch or stop the FlexKV runtime, and it does not mutate inference-process state.

There is currently no `flexkv version` command because FlexKV does not yet expose independent version metadata. `flexkv collect-env` reports the current repository's short Git commit, which is useful as a development-branch identifier.

---

## 1. Installation and Entry Point

The console entry point is registered in `setup.py`:

```python
entry_points={"console_scripts": ["flexkv=flexkv.cli.main:main"]}
```

Install the branch containing the CLI in editable mode:

```bash
cd /path/to/FlexKV
conda activate flexkv
pip install -e .

flexkv --help
```

Editable installs normally pick up Python source changes immediately. Re-run `pip install -e .` after changing the entry-point definition in `setup.py`.

---

## 2. Command Overview

```text
flexkv
├── env             # Show FLEXKV_* and ENABLE_FLEXKV variables
├── config          # Show or validate local configuration
│   ├── show
│   └── validate
├── collect-env     # Collect an environment summary for troubleshooting
├── status          # Show local processes, IPC files, and metrics ports
├── trace           # Show FlexKV trace files
│   ├── list
│   └── show
├── storage         # Show SSD cache directory usage
└── metrics         # Scrape local Prometheus metrics
```

| Command | Question answered | Runtime required |
|---|---|---|
| `flexkv env` | Which FlexKV variables are set in the current shell? | No |
| `flexkv config` | What configuration can the CLI resolve? | No |
| `flexkv collect-env` | What are the basic environment and dependencies? | No |
| `flexkv status` | Which local processes and endpoints look FlexKV-related? | Partially; it still runs without a runtime |
| `flexkv trace` | Where are trace files, and what do they contain? | Requires an application to have written traces |
| `flexkv storage` | How much space do SSD cache directories use? | Requires the directories to exist |
| `flexkv metrics` | What do the Python/C++ metrics endpoints expose? | Requires metrics endpoints to be running |

---

## 3. Common Arguments

All top-level commands support:

| Argument | Description |
|---|---|
| `--format terminal` | Default human-readable output |
| `--format json` | JSON output for scripts |
| `-q, --quiet` | Suppress stdout output |

Examples:

```bash
flexkv env --format json
flexkv status --format json
flexkv metrics --format json
```

JSON output is written to stdout; errors are written to stderr.

---

## 4. Environment and Configuration

### 4.1 `flexkv env`

List all `FLEXKV_*` variables and the compatibility variable `ENABLE_FLEXKV` in the current shell:

```bash
flexkv env
flexkv env --format json
```

This command reads the environment inherited by the CLI process. It does not read `/proc/<pid>/environ` of other running processes.

### 4.2 `flexkv config`

`config show` may be omitted: `flexkv config` defaults to `flexkv config show`.

```bash
flexkv config
flexkv config show
flexkv config validate
flexkv config --format json
```

Configuration source rules:

1. If `FLEXKV_CONFIG_PATH` is set, load `UserConfig` fields from that JSON or YAML file.
2. Otherwise, construct `UserConfig` from `FLEXKV_*` environment variables and defaults.
3. The output additionally includes global environment settings such as `env.enable_metrics`, `env.py_metrics_port`, and `env.trace_file_path`.

Configuration files may use `.json`, `.yaml`, or `.yml`:

```bash
export FLEXKV_CONFIG_PATH=/path/to/flexkv.yaml
flexkv config show
```

`config validate` currently checks:

- `cpu_cache_gb` must be greater than 0
- `ssd_cache_gb` must be greater than or equal to 0
- When SSD is enabled, `ssd_cache_gb` must be greater than `cpu_cache_gb`
- When SSD is enabled, every `ssd_cache_dir` must be an existing directory

Note: violations are printed, but the current implementation does not assign a distinct non-zero exit code for validation failure. `config show` represents the CLI-side parsed configuration; it is not the final merged configuration inside an already-running inference process.

### 4.3 `flexkv collect-env`

Collect a troubleshooting summary:

```bash
flexkv collect-env
flexkv collect-env --format json
```

It includes the OS, Python version, current repository short commit, PyTorch/CUDA/GPU summary, NVIDIA driver, key Python package versions, and the number of `FLEXKV_*` variables. To keep output small, it does not export full environment-variable values; use `flexkv env` for individual variables.

---

## 5. Local Runtime State

### 5.1 `flexkv status`

```bash
flexkv status
flexkv status --format json
```

The current implementation reports:

1. Local processes whose command line contains `flexkv` or `FLEXKV`.
2. `/tmp/flexkv_server*` IPC endpoint files.
3. Whether the Python and C++ metrics ports accept connections.

Process detection is based on command-line text. An SGLang or vLLM process that loads the FlexKV module is not detected when its launch command does not contain a FlexKV keyword. Reading another process's environment may be restricted by permissions.

### 5.2 `flexkv trace`

FlexKV currently has no independent general-purpose log file and no `flexkv log` command. The CLI only views trace files written by the FlexKV tracer.

The trace path does not need to be passed explicitly:

- `FLEXKV_TRACE_FILE_PATH` is used first
- If unset, `./flexkv_trace.log` is used
- `trace list` also searches rotated files matching `<trace-path>.*`

```bash
# List the default file and rotated files
flexkv trace list

# Point to a custom trace file
export FLEXKV_TRACE_FILE_PATH=/tmp/flexkv_trace.log
flexkv trace list

# Show the complete file
flexkv trace show

# Show the last 100 lines
flexkv trace show -n 100

# Explicitly select a file
flexkv trace show -f /tmp/flexkv_trace.log
```

`trace show` prints file content verbatim; it does not parse JSON or filter fields. It exits with code 1 when the file does not exist.

### 5.3 `flexkv storage`

```bash
flexkv storage
flexkv storage --format json
```

This command recursively sums the size and count of regular files under `ssd_cache_dir` and reports whether each directory exists. Separate multiple directories with semicolons:

```bash
export FLEXKV_SSD_CACHE_DIR="/data/flexkv_ssd_0;/data/flexkv_ssd_1"
flexkv storage
```

The current implementation reads `FLEXKV_SSD_CACHE_DIR` from the environment. Even when `FLEXKV_CONFIG_PATH` is set, `storage` does not read SSD paths from that configuration file.

---

## 6. Metrics

### 6.1 Enable Metrics Endpoints

FlexKV metrics are exposed by runtime processes; the CLI only scrapes and renders them:

```bash
export FLEXKV_ENABLE_METRICS=1
export FLEXKV_PY_METRICS_PORT=8080
export FLEXKV_CPP_METRICS_PORT=8081
```

Then start an application with FlexKV loaded. The Python endpoint is typically:

```text
http://127.0.0.1:8080/metrics
```

The default C++ endpoint is:

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

Port resolution order:

1. `--py-port` / `--cpp-port`
2. `FLEXKV_PY_METRICS_PORT` / `FLEXKV_CPP_METRICS_PORT`
3. Defaults `8080` / `8081`

The CLI parses the Prometheus text format and displays metric names, labels, and values. It does not currently filter by the `flexkv_py_` or `flexkv_cpp_` prefix, so Python-process `python_*` and `process_*` metrics are also shown.

If the Python endpoint is available but the C++ endpoint is not running, the terminal shows Python metrics and reports `endpoint not reachable` for C++. If neither endpoint is available, the command exits with code 1.

See the [FlexKV Prometheus Metrics documentation](../monitoring/README_en.md) for metric definitions and Prometheus/Grafana deployment.

---

## 7. Extension Development

The CLI uses a flat command structure:

```text
flexkv/cli/
├── __init__.py
├── main.py              # Thin entry point
└── commands/
    ├── __init__.py      # Auto-discovers BaseCommand subclasses
    ├── base.py          # BaseCommand, table output, and JSON output
    ├── env.py
    ├── config.py
    ├── collect_env.py
    ├── status.py
    ├── trace.py
    ├── storage.py
    └── metrics.py
```

To add a command, create a Python module with a `BaseCommand` subclass:

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

`commands/__init__.py` discovers and instantiates non-abstract `BaseCommand` subclasses through `pkgutil.iter_modules()`, so no manual registry update is needed. Command implementations should remain self-contained. When reuse is necessary, prefer the output helpers in `base.py` or small helpers in existing commands instead of introducing a separate `core/` layer.

Future command groups such as `bench`, `server`, and `kvcache` can be added as subdirectories once their runtime capabilities are well defined.
