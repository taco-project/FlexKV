# Using FlexKV with SGLang

[中文文档](README_zh.md)

## Version compatibility

There are two different SGLang integration paths. Choose the one that matches
your model.

> Status last checked: August 6, 2026.

| Use case | SGLang version | Patch required? |
| --- | --- | --- |
| Standard models | SGLang `v0.5.16` or later, or a recent `main` | No |
| DeepSeek V4 | SGLang PR [#31781](https://github.com/sgl-project/sglang/pull/31781), pinned to commit [`ee0465a`](https://github.com/sgl-project/sglang/commit/ee0465a09196421a6e4d53a3103eccdef1dd32ac) | No local patch; use the pinned SGLang commit |

### Standard models: use upstream SGLang directly

The base FlexKV integration was merged into upstream SGLang by
[sglang#29701](https://github.com/sgl-project/sglang/pull/29701) and is included
in SGLang `v0.5.16` and later. Do **not** apply
`sglang_flexkv_connector.patch` to these versions.

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.16  # or use a newer release/main
```

### DeepSeek V4: pin the unmerged SGLang adaptation

FlexKV's DeepSeek V4 support is present on FlexKV `main` (merged by
[FlexKV#225](https://github.com/taco-project/FlexKV/pull/225)), but the matching
SGLang adaptation has **not** been merged upstream yet. Until
[sglang#31781](https://github.com/sgl-project/sglang/pull/31781) is merged, fetch
the PR and pin the commit below instead of using an SGLang release or `main`
alone:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/31781/head
git checkout -b flexkv-dsv4 ee0465a09196421a6e4d53a3103eccdef1dd32ac
```

Use this with the current FlexKV `main` branch:

```bash
git clone https://github.com/taco-project/FlexKV.git
cd FlexKV
git checkout main
```

The PR is still under development, so check its status before updating the
pinned SGLang commit. DeepSeek V4's unified-KV layout and `--enable-hisparse`
are not supported by that commit.

## Launch

After installing SGLang and FlexKV, create a FlexKV configuration file. For
example:

```yaml
# flexkv_config.yaml
cpu_cache_gb: 16
```

Launch SGLang with the built-in FlexKV backend:

```bash
python -m sglang.launch_server \
  --model-path <model> \
  --enable-flexkv \
  --flexkv-config-file /path/to/flexkv_config.yaml \
  # ... other SGLang arguments
```

The equivalent explicit backend option is
`--radix-cache-backend flexkv`. See SGLang's
[`flexkv/README.md`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/storage/flexkv/README.md)
for detailed configuration and validation instructions.

## About the patch in this directory

`sglang_flexkv_connector.patch` is retained only as a legacy reference for the
old pre-upstream integration. It is not required for supported SGLang releases,
and it must not be used for the DeepSeek V4 path above.
