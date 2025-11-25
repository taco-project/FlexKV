#!/bin/bash
# Copied from vLLM github actions https://github.com/vllm-project/vllm/blob/main/.github/workflows/scripts/pytorch-install.sh

python_executable=python$1
pytorch_version=$2
cuda_version=$3

# Install torch
$python_executable -m pip install numpy ninja cython wheel typing typing-extensions dataclasses setuptools && conda clean -ya
$python_executable -m pip install torch=="${pytorch_version}+cu${cuda_version//./}" --extra-index-url "https://download.pytorch.org/whl/cu${cuda_version//./}"

# Print version information
$python_executable --version
$python_executable -c "import torch; print('PyTorch:', torch.__version__)"
$python_executable -c "import torch; print('CUDA:', torch.version.cuda)"
$python_executable -c "from torch.utils import cpp_extension; print (cpp_extension.CUDA_HOME)"
