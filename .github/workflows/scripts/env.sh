#!/bin/bash
# Copied from vLLM github actions https://github.com/vllm-project/vllm/blob/main/.github/workflows/scripts/env.sh

# This file installs common linux environment tools

export LANG=C.UTF-8

sudo    apt-get update && \
sudo    apt-get install -y --no-install-recommends \
        software-properties-common

sudo    apt-get install -y --no-install-recommends \
        build-essential \
        liburing-dev \
        git \
        cmake

# Remove github bloat files to free up disk space
sudo rm -rf "/usr/local/share/boost"
sudo rm -rf "$AGENT_TOOLSDIRECTORY"
sudo rm -rf "/usr/share/dotnet"
