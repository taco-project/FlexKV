# FlexKV Continuous Integration

[中文文档](README_zh.md)

This document describes the GitHub Actions workflow in
[`publish.yml`](../../.github/workflows/publish.yml), including where it runs,
what it validates, and what it deliberately leaves to GPU or integration
testing.

## When the workflow runs

The `flexkv ci` workflow runs for:

- pull requests targeting `main` or `dev`;
- pushes to `main` or `dev`.

The workflow has read-only repository permission (`contents: read`). Pull
requests build and test the package but do not receive COS credentials and do
not upload artifacts. Protected-branch pushes run the same validation and then
upload the generated wheel to COS.

## Runner and version matrix

| Item | Current value |
| --- | --- |
| Runner | GitHub-hosted `ubuntu-22.04` |
| Python | `3.10` |
| PyTorch | `2.6.0` |
| CUDA toolkit | `12.4` |
| CUDA architectures | `8.9`, `9.0+PTX` |
| Parallel build jobs | `4` |

This is currently a single build configuration, not a multi-version matrix.
The GitHub-hosted runner is a CPU runner: CUDA is installed to compile the
wheel, but the unit-test stage does not require a GPU.

## What the CI validates

The `Build Wheel` job performs these steps in order:

1. Checks out the pull request or protected-branch commit.
2. Installs the Linux build dependencies.
3. Installs Python 3.10, CUDA 12.4, and PyTorch 2.6.0.
4. Runs `./build.sh --release`, including release-mode Cython compilation.
5. Installs `requirements.txt` and then force-installs the newly built wheel.
6. Changes to the runner's temporary directory and runs the CPU unit tests.
7. On `main` or `dev` pushes only, uploads `dist/` to COS.

Running tests outside the source tree is intentional: imports must resolve to
the installed wheel, so the job validates the built artifact rather than the
checkout's Python files.

## CPU unit-test selection

CI discovers every `tests/**/test_*.py` file containing this module-level
marker:

```python
import pytest

pytestmark = pytest.mark.unit
```

If no matching files are found, the workflow fails. The selected files run
with:

```bash
timeout 10m python -m pytest -q --maxfail=1 <selected-test-files>
```

To add a test to this CI tier:

1. Keep it CPU-only and independent of GPUs, network services, storage
   devices, and repository secrets.
2. Add `pytestmark = pytest.mark.unit` at module scope.
3. Confirm it passes against an installed release wheel from outside the
   repository directory.

## Local reproduction on Ubuntu 22.04

Run the following from the FlexKV repository root. The environment setup
scripts install system packages and may require `sudo` access.

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

`timeout` and `mapfile` are GNU/Linux commands. Use an Ubuntu environment when
reproducing the workflow exactly.

## Reference duration

[GitHub Actions run #31087143283](https://github.com/taco-project/FlexKV/actions/runs/31087143283)
completed successfully on August 6, 2026. Rounded step durations were:

| Stage | Duration |
| --- | ---: |
| Linux environment setup | 1m 31s |
| Python setup | 30s |
| CUDA installation | 5m 06s |
| PyTorch installation | 2m 10s |
| Release-wheel build | 6m 49s |
| Wheel/dependency installation | 5s |
| CPU unit-test step | 4s |
| Complete job | **17m 05s** |

That run discovered 12 unit-test modules and reported `198 passed, 1 skipped in
1.01s`. The total duration is a reference, not an SLA; runner availability and
dependency caches can change it substantially.

## Artifact upload

The COS upload step runs only for pushes to `main` or `dev`. It requires these
repository secrets:

- `COS_SECRET_ID`;
- `COS_SECRET_KEY`;
- `COS_BUCKET`;
- `COS_ENDPOINT`.

The workflow uploads `dist/` to `flexkv/<date>/<time>`. Pull requests always
skip the date/time and upload steps, so missing COS secrets in a pull request
must not be treated as a build or test failure.

## What this workflow does not cover

The current workflow does not validate:

- GPU runtime correctness or CUDA kernel execution;
- multi-GPU, TP, DP, PP, or cross-node behavior;
- SSD, GDS, RDMA, Mooncake, Redis, or COS round trips;
- vLLM, SGLang, Dynamo, or TensorRT-LLM end-to-end serving;
- performance or soak regressions;
- other Python, PyTorch, CUDA, or Ubuntu versions.

Those paths require separate GPU and integration testing before release.

## Checking a pull request

Use the GitHub pull-request page, or query it with the GitHub CLI:

```bash
gh pr checks <pr-number> --repo taco-project/FlexKV
```

For a failure, inspect the first failed step and its complete log. In
particular, distinguish build or unit-test failures from secret-dependent
artifact upload failures; upload is expected to be skipped for pull requests.
