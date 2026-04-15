# MUSA Support Feature — Test Plan

Test strategy, cases, and execution for the MUSA support feature.

---

## 1. Test Strategy

- **No impact on CUDA:** All existing CUDA tests and flows must behave as before when MUSA is not used.
- **TDD:** Build and backend-dispatch tests define expected behavior; MUSA transfer test validates stub (and future real impl) under `@pytest.mark.musa`.
- **Backend abstraction coverage:** `gpu_runtime` functions are tested for correct dispatch, error handling (MUSA selected but unavailable), and device/stream/IPC operations.
- **Environments:** Default CI may have no MUSA SDK or hardware; MUSA-specific tests are skippable or run only when MUSA is available.

---

## 2. Prerequisites & Environment Setup

### 2.1 MUSA Test Machine Setup

```bash
# 1. Verify MUSA driver and device
mthreads-gmi                        # Should list MUSA GPU(s)

# 2. Verify MUSA SDK
echo $MUSA_HOME                      # e.g. /usr/local/musa
ls $MUSA_HOME/bin/mcc                # MUSA compiler must exist

# 3. Verify torch_musa
python3 -c "import torch_musa; import torch; print(torch.musa.is_available())"
# Expected: True

# 4. Clone repo and apply patch
git clone <repo-url> FlexKV && cd FlexKV
git apply musa_flexkv_all.patch

# 5. Install dependencies (MUSA-specific)
pip install -r requirements-musa.txt

# 6. Build with MUSA enabled (debug mode skips Cython for faster iteration)
FLEXKV_USE_MUSA=1 FLEXKV_DEBUG=1 pip install -e .
```

### 2.2 Verify Build Output

```bash
# Check the MUSA extension was built
python3 -c "import flexkv.c_ext_musa; print('c_ext_musa imported OK')"

# Check HAS_MUSA_SDK attribute (True if mcc was found during build)
python3 -c "import flexkv.c_ext_musa as m; print('HAS_MUSA_SDK:', m.HAS_MUSA_SDK)"

# On a MUSA-only machine (no nvcc), verify c_ext was skipped
python3 -c "
try:
    import flexkv.c_ext
    print('c_ext imported (CUDA available)')
except ImportError:
    print('c_ext NOT built (expected on MUSA-only machine)')
"
```

---

## 3. Test Cases & Execution

### 3.1 Build Configuration Tests

**File:** `tests/test_musa_build.py`

| # | Test | Description | Condition |
|---|------|-------------|-----------|
| 1 | `test_default_build_includes_only_c_ext` | Without `FLEXKV_USE_MUSA`, `get_cpp_extension_names()` returns only `["flexkv.c_ext"]`. | Always run; temporarily clears `FLEXKV_USE_MUSA` if set. |
| 2 | `test_musa_build_includes_c_ext_musa` | With `FLEXKV_USE_MUSA=1`, extension list includes both `flexkv.c_ext` and `flexkv.c_ext_musa`. | Always run; sets env in test. |
| 3 | `test_musa_disabled_omits_c_ext_musa` | With `FLEXKV_USE_MUSA=0` or unset, `flexkv.c_ext_musa` is not in the list. | Always run. |

**Execute:**

```bash
# Run on any machine (no GPU required)
pytest tests/test_musa_build.py -v

# Expected output:
# test_default_build_includes_only_c_ext PASSED
# test_musa_build_includes_c_ext_musa PASSED
# test_musa_disabled_omits_c_ext_musa PASSED
```

---

### 3.2 Backend Dispatch Tests

**File:** `tests/test_gpu_backend_dispatch.py`

| # | Test | Description | Condition |
|---|------|-------------|-----------|
| 1 | `test_get_gpu_backend_returns_cuda_or_musa` | `get_gpu_backend()` returns `"cuda"` or `"musa"`. | Always run. |
| 2 | `test_cuda_available_uses_c_ext_by_default` | When CUDA is available and no override, `get_transfer_kv_blocks_module()` returns `flexkv.c_ext`. | Requires `torch.cuda.is_available()`. |
| 3 | `test_musa_override_selects_c_ext_musa_when_built` | With `FLEXKV_GPU_BACKEND=musa`, backend is `"musa"` and module has `transfer_kv_blocks`. | Always run. |
| 4 | `test_c_ext_musa_importable_when_built` | If `flexkv.c_ext_musa` is built, it can be imported and has `transfer_kv_blocks`. | Skip if `c_ext_musa` not built. |

**Execute:**

```bash
# On MUSA machine (with c_ext_musa built)
pytest tests/test_gpu_backend_dispatch.py -v

# On CUDA machine (to verify no regression)
pytest tests/test_gpu_backend_dispatch.py -v

# Expected: all applicable tests PASSED, others SKIPPED
```

---

### 3.3 GPU Runtime Abstraction Tests

**File:** `tests/test_gpu_runtime.py`

| # | Test | Description | Condition |
|---|------|-------------|-----------|
| 1 | `test_current_device_returns_int` | `gpu_runtime.current_device()` returns a non-negative int. | Requires GPU. |
| 2 | `test_device_count_positive` | `gpu_runtime.device_count()` returns a positive int. | Requires GPU. |
| 3 | `test_empty_cache_no_error` | `gpu_runtime.empty_cache()` runs without error. | Requires GPU. |
| 4 | `test_create_stream_returns_stream` | `gpu_runtime.create_stream()` returns a stream object. | Requires GPU. |
| 5 | `test_get_device_string_format` | `gpu_runtime.get_device_string(0)` returns `"cuda:0"` or `"musa:0"`. | Always run. |
| 6 | `test_musa_unavailable_raises_*` | When MUSA selected but `torch.musa` unavailable, all runtime functions raise `RuntimeError`. | Always run (mocks backend). |

**Execute:**

```bash
# On MUSA machine — validates real torch.musa dispatch
pytest tests/test_gpu_runtime.py -v

# Verify device string is "musa:0"
python3 -c "
from flexkv.common import gpu_runtime
print('Backend:', gpu_runtime.get_gpu_backend())
print('Device string:', gpu_runtime.get_device_string(0))
print('Device count:', gpu_runtime.device_count())
print('Current device:', gpu_runtime.current_device())
"
# Expected on MUSA: Backend: musa, Device string: musa:0
```

---

### 3.4 MUSA Transfer Tests

**File:** `tests/test_transfer_musa.py`

| # | Test | Description | Condition |
|---|------|-------------|-----------|
| 1 | `test_transfer_kv_blocks_musa_no_crash` | Calls `c_ext_musa.transfer_kv_blocks` with minimal tensors; no exception. | `@pytest.mark.musa`; skip if `c_ext_musa` not built. |

**Execute:**

```bash
# On MUSA machine with MUSA SDK build
pytest tests/test_transfer_musa.py -v -m musa

# Stub-only build (no MUSA SDK) — still should pass (no-op stub)
FLEXKV_USE_MUSA=1 pytest tests/test_transfer_musa.py -v -m musa

# Quick manual smoke test of the transfer binding
python3 -c "
import torch
import flexkv.c_ext_musa as m
print('transfer_kv_blocks available:', hasattr(m, 'transfer_kv_blocks'))
print('HAS_MUSA_SDK:', m.HAS_MUSA_SDK)
print('SSD transfer available:', hasattr(m, 'transfer_kv_blocks_ssd'))
print('Hasher available:', hasattr(m, 'Hasher'))
print('CRadixTreeIndex available:', hasattr(m, 'CRadixTreeIndex'))
"
```

---

### 3.5 Regression: Existing CUDA Behavior

Ensure existing tests pass with the new backend dispatch and no MUSA override.

**Execute:**

```bash
# Run on a CUDA machine to ensure no regression
pytest tests/test_kvmanager.py -v
pytest tests/test_memory_handle.py -v
pytest tests/test_cache_engine.py -v

# Full suite excluding MUSA-specific tests
pytest tests/ -v -m "not musa"
```

| Area | Command | Expected |
|------|---------|----------|
| Transfer / KV manager | `pytest tests/test_kvmanager.py -v` | Worker uses `c_ext`; same behavior as before. |
| Memory handle | `pytest tests/test_memory_handle.py -v` | IPC now via `gpu_runtime` but behavior identical on CUDA. |
| Cache engine | `pytest tests/test_cache_engine.py -v` | No change in behavior. |

---

## 4. End-to-End Validation on MUSA Machine

### 4.1 Build & Install

```bash
cd FlexKV
git checkout main
git apply musa_flexkv_all.patch

# MUSA-only build (no CUDA)
FLEXKV_USE_MUSA=1 FLEXKV_DEBUG=1 pip install -e .
```

### 4.2 Run All MUSA Tests

```bash
# Step 1: Build config tests (no GPU needed)
pytest tests/test_musa_build.py -v
# Expected: 3/3 PASSED

# Step 2: GPU runtime tests
pytest tests/test_gpu_runtime.py -v
# Expected: all PASSED (real MUSA dispatch)

# Step 3: Transfer stub/real tests
pytest tests/test_transfer_musa.py -v -m musa
# Expected: PASSED (stub or real kernel depending on build)

# Step 4: Full MUSA test suite in one command (use this on MUSA-only machines)
pytest tests/test_musa_build.py tests/test_gpu_runtime.py tests/test_transfer_musa.py -v
```

### 4.3 Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: flexkv.c_ext_musa` | Extension not built | Verify `FLEXKV_USE_MUSA=1` was set during `pip install` |
| `ImportError: libmusa.so` | MUSA runtime not in LD path | `export LD_LIBRARY_PATH=$MUSA_HOME/lib:$LD_LIBRARY_PATH` |
| `HAS_MUSA_SDK = False` | `mcc` not found during build | Verify `$MUSA_HOME/bin/mcc` exists and `MUSA_HOME` is set |
| `RuntimeError: MUSA backend ... torch.musa unavailable` | `torch_musa` not installed | `pip install torch_musa` or use `requirements-musa.txt` |
| `pip install` pulls `nvtx` or CUDA packages | Wrong requirements file | Use `pip install -r requirements-musa.txt` on MUSA machines |
| `error: flexkv.c_ext build failed` on MUSA-only machine | CUDA ext attempted without nvcc | Verify `FLEXKV_USE_MUSA=1` is set — this skips CUDA ext |
| `pytest tests/ -v -m musa` fails with `ModuleNotFoundError: flexkv.c_ext` or `nvtx` | Collection imports non-MUSA tests that need CUDA | Use explicit file list: `pytest tests/test_musa_build.py tests/test_gpu_runtime.py tests/test_transfer_musa.py -v -m musa` |

---

## 5. Environment Matrix

| Environment | FLEXKV_USE_MUSA | MUSA SDK / mcc | Expected |
|-------------|-----------------|----------------|----------|
| CI (CUDA only) | 0 or unset | No | Only `flexkv.c_ext` built; non-musa tests pass; musa tests skipped. |
| CI (MUSA stub) | 1 | No | `flexkv.c_ext_musa` built (C++ stub); build + dispatch tests pass. |
| MUSA dev (SDK) | 1 | Yes | Full MUSA build with `transfer_musa.mu`, TP; all MUSA tests pass. |
| MUSA dev (SDK + GDS) | 1 | Yes + muFile | Full build including GDS/muFile; GDS transfer tests can run. |
| MUSA-only (no CUDA) | 1 | Yes | `flexkv.c_ext` skipped; only `c_ext_musa` built; all MUSA tests pass. |

---

## 6. Running the Full Test Suite

```bash
# All tests (musa tests skip if c_ext_musa not built / no MUSA)
pytest tests/ -v

# Exclude MUSA-only tests (e.g. CI without MUSA)
pytest tests/ -v -m "not musa"

# Only MUSA-related tests (use this on MUSA-only machines)
pytest tests/test_musa_build.py tests/test_gpu_runtime.py tests/test_transfer_musa.py -v

# Only MUSA hardware tests — use explicit file list on MUSA-only machines
# (pytest tests/ -v -m musa fails there because collection imports test_cache_engine,
# test_kvmanager, test_namespace_isolation which require flexkv.c_ext or nvtx)
pytest tests/test_musa_build.py tests/test_gpu_runtime.py tests/test_transfer_musa.py -v -m musa
```

---

## 7. Pytest Marker

- **`musa`:** Marks tests that require or specifically target the MUSA backend (e.g. `c_ext_musa` or MUSA device).
- Defined in `pyproject.toml` under `[tool.pytest.ini_options]` → `markers`.
- Use `-m musa` to run only MUSA tests, `-m "not musa"` to exclude them.

---

## 8. Test-to-Phase Mapping

| Phase | Test Files | What is validated |
|-------|-----------|-------------------|
| Phase 1 | `test_musa_build.py`, `test_transfer_musa.py` | Build config, stub import |
| Phase 2 | `test_gpu_runtime.py` + all Phase 1 tests | Backend abstraction (device, stream, IPC, error handling), Python migration correctness |
| Phase 3 | All Phase 1+2 tests + regression suite | SSD/CFS/GDS bindings via `c_ext_musa`, full API parity |
