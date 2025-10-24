#!/bin/bash
# FlexKV Environment Setup Script
# Source this file to set up the environment: source setup_env.sh

# Get the directory where this script is located
FLEXKV_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add FlexKV to Python path
export PYTHONPATH="$FLEXKV_ROOT:$PYTHONPATH"

# Display confirmation
echo "✓ FlexKV environment configured"
echo "  FLEXKV_ROOT: $FLEXKV_ROOT"
echo "  PYTHONPATH includes: $FLEXKV_ROOT"

# Test import
if python3 -c "import flexkv" 2>/dev/null; then
    echo "✓ FlexKV can be imported successfully"
else
    echo "✗ Failed to import FlexKV"
    return 1
fi

