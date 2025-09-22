#!/usr/bin/env bash
set -euo pipefail

echo "[FlexKV] Step 1: Uninstalling flexkv (if installed)"
if command -v pip3 >/dev/null 2>&1; then
  PIP=pip3
elif command -v pip >/dev/null 2>&1; then
  PIP=pip
else
  PIP="python3 -m pip"
fi

$PIP uninstall -y flexkv || true

echo "[FlexKV] Removing any flexkv.egg-link leftovers"
for p in \
  /usr/lib64/python3.6/site-packages \
  /usr/local/lib/python3.6/site-packages \
  /usr/local/lib64/python3.6/site-packages \
  /data/home/phaedonsun/.local/lib/python3.6/site-packages \
  /usr/lib/python3.6/site-packages; do
  if [ -f "$p/flexkv.egg-link" ]; then
    rm -f "$p/flexkv.egg-link" && echo "[FlexKV] removed $p/flexkv.egg-link" || true
  fi
done

echo "[FlexKV] Step 2: Cleaning repo build artifacts and compiled .so files"
rm -rf build build_temp dist .eggs *.egg-info || true

# Remove Python __pycache__
find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# Remove compiled extension .so files in package tree
find flexkv -maxdepth 2 -type f -name "*.so" -print -delete 2>/dev/null || true
find flexkv/lib -type f -name "*.so*" -print -delete 2>/dev/null || true

echo "[FlexKV] Step 3: Purging pip cache"
$PIP cache purge -q || true

echo "[FlexKV] Clean done. You can rebuild with: bash build.sh --no-cuda install"


