#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG_YAML [--rounds N] [--duration SECONDS] [--dry-run]" >&2
  exit 2
fi

CONFIG="$1"
shift
cd "${REPO_ROOT}"
exec python3 -m benchmarks.flexkv_stress.main --config "${CONFIG}" "$@"
