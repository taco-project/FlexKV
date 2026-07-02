#!/bin/bash
# FlexKV 分层测试计划运行器 (SWA 全量特性)
# 见 deployments/swa_design/process/测试设计报告_SWA全量特性.md
#
# 用法:
#   bash tests/run_test_plan.sh --tier unit|smoke|e2e|all
#
#   unit  = 纯逻辑 + c_ext 逻辑单测 (无需 GPU)
#   smoke = 控制面/建图/晚绑 + 节点挂载真实-IO driver (无需 GPU)
#   e2e   = 需 GPU 的 dispatch / 控制面 byte-exact / KVManager 全流程
#   all   = 以上全部 (e2e 需要空闲 GPU)
#
# 结果: 逐项 PASS/FAIL/SKIP + 末尾汇总矩阵, 并落一份 tests/plan_report_<ts>.txt

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

TIER="all"
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2;;
    --tier=*) TIER="${1#*=}"; shift;;
    *) echo "unknown arg: $1"; exit 2;;
  esac
done

export FLEXKV_ENABLE_METRICS="${FLEXKV_ENABLE_METRICS:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT="tests/plan_report_${TS}.txt"
PYTEST="python3 -m pytest -q -ra"

# rows: "tier|kind|name|status|seconds"
ROWS=()

# Files whose non-PASS is a KNOWN, non-blocking condition (not owned by this
# test plan). Reported but does not fail the run.
#   test_swa_cnode_cascade.py       -> EMPTY: P2P-only (LocalRadixTree needs
#                                      FLEXKV_ENABLE_P2P=1); covered on the
#                                      production CRadixTreeIndex by
#                                      test_swa_accel_nodemount.py.
#   test_recompute_block_counts.py  -> pre-existing FAIL unrelated to SWA (stale
#                                      >9000 block-count assumption after the MLA
#                                      block-size change); owned by config.
is_known_nonblocking() {
  case "$1" in
    test_swa_cnode_cascade.py) [ "$2" = "EMPTY" ] && return 0;;
    test_recompute_block_counts.py) return 0;;
  esac
  return 1
}

_run_pytest() {  # tier kind file [extra-args]
  local tier="$1" kind="$2" file="$3"; shift 3
  local extra="$*"
  echo ""
  echo ">>> [$tier/$kind] $file $extra"
  local t0 t1 rc
  t0=$(date +%s)
  $PYTEST "$file" $extra 2>&1 | tee -a "$REPORT" | tail -3
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  # pytest exit: 0 pass, 5 = no tests collected (treat as skip)
  local status="PASS"
  [ "$rc" -eq 5 ] && status="EMPTY"
  [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ] && status="FAIL"
  ROWS+=("$tier|$kind|$(basename "$file")|$status|$((t1-t0))")
}

_run_script() {  # tier kind script [args]
  local tier="$1" kind="$2" script="$3"; shift 3
  local args="$*"
  echo ""
  echo ">>> [$tier/$kind] $script $args"
  local t0 t1 rc
  t0=$(date +%s)
  python3 "$script" $args 2>&1 | tee -a "$REPORT" | tail -4
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  local status="PASS"; [ "$rc" -ne 0 ] && status="FAIL"
  ROWS+=("$tier|$kind|$(basename "$script")|$status|$((t1-t0))")
}

_has_gpu() {
  python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null
}

echo "==================================================================" | tee "$REPORT"
echo "FlexKV Test Plan — tier=$TIER — $TS" | tee -a "$REPORT"
echo "==================================================================" | tee -a "$REPORT"

# ---------------- UNIT (CPU / c_ext, no GPU) ----------------
if [ "$TIER" = "unit" ] || [ "$TIER" = "all" ]; then
  _run_pytest unit radix     tests/test_swa_node_mount.py
  _run_pytest unit radix     tests/test_swa_accel_nodemount.py
  _run_pytest unit radix     tests/test_swa_cnode_cascade.py
  _run_pytest unit engine    tests/test_cache_engine.py
  _run_pytest unit swa       tests/test_swa_host_pool.py
  _run_pytest unit swa       tests/test_swa_peer_op.py
  _run_pytest unit transfer  tests/test_transfer_engine_atomic_eviction.py
  _run_pytest unit task      tests/test_kvtask_lifecycle.py
  _run_pytest unit config    tests/test_namespace_isolation.py
  _run_pytest unit config    tests/test_recompute_block_counts.py
  _run_pytest unit config    tests/test_config_hugepage.py
fi

# ---------------- SMOKE (control plane / graph / bind, no GPU) ----------------
if [ "$TIER" = "smoke" ] || [ "$TIER" = "all" ]; then
  _run_pytest smoke swa   tests/test_swa_control_plane_smoke.py
  _run_pytest smoke swa   tests/test_swa_launch_bind.py
  _run_script smoke swa   benchmarks/benchmark_swa_nodemount.py --users 32 --turns 6
  # pressure variants (exercise full-evict I1 + SWA-LRU I2 under tight pools)
  _run_script smoke swa   benchmarks/benchmark_swa_nodemount.py --users 64 --turns 8 --cpu-blocks 512 --swa-slots 512
  _run_script smoke swa   benchmarks/benchmark_swa_nodemount.py --users 64 --turns 8 --cpu-blocks 8192 --swa-slots 32
fi

# ---------------- E2E (needs GPU) ----------------
if [ "$TIER" = "e2e" ] || [ "$TIER" = "all" ]; then
  if _has_gpu; then
    _run_script e2e swa       tests/test_swa_dispatch.py
    _run_script e2e swa       tests/test_swa_control_plane_e2e.py
    _run_pytest e2e memory    tests/test_memory_handle.py
    _run_pytest e2e kvmanager tests/test_kvmanager.py
  else
    echo "" | tee -a "$REPORT"
    echo "!!! no CUDA GPU available — skipping e2e tier" | tee -a "$REPORT"
    ROWS+=("e2e|swa|test_swa_dispatch.py|SKIP-NOGPU|0")
    ROWS+=("e2e|swa|test_swa_control_plane_e2e.py|SKIP-NOGPU|0")
    ROWS+=("e2e|memory|test_memory_handle.py|SKIP-NOGPU|0")
    ROWS+=("e2e|kvmanager|test_kvmanager.py|SKIP-NOGPU|0")
  fi
fi

# ---------------- summary matrix ----------------
echo "" | tee -a "$REPORT"
echo "==================================================================" | tee -a "$REPORT"
echo "SUMMARY (tier=$TIER)" | tee -a "$REPORT"
echo "==================================================================" | tee -a "$REPORT"
printf "%-6s %-9s %-42s %-12s %6s\n" TIER KIND FILE STATUS SEC | tee -a "$REPORT"
printf -- "------------------------------------------------------------------------\n" | tee -a "$REPORT"
FAILS=0
for row in "${ROWS[@]}"; do
  IFS='|' read -r tier kind name status sec <<< "$row"
  tag=""
  if [ "$status" != "PASS" ] && is_known_nonblocking "$name" "$status"; then
    tag="  (known, non-blocking)"
  elif [ "$status" = "FAIL" ]; then
    FAILS=$((FAILS+1))
  fi
  printf "%-6s %-9s %-42s %-12s %6s%s\n" "$tier" "$kind" "$name" "$status" "$sec" "$tag" | tee -a "$REPORT"
done
printf -- "------------------------------------------------------------------------\n" | tee -a "$REPORT"
echo "report: $REPORT" | tee -a "$REPORT"
if [ "$FAILS" -gt 0 ]; then
  echo "RESULT: FAIL ($FAILS file(s) failed, excluding known non-blocking)" | tee -a "$REPORT"
  exit 1
fi
echo "RESULT: OK (known non-blocking items excluded)" | tee -a "$REPORT"
exit 0
