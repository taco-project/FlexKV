# FlexKV Utils

## subprocess.py - Safe Subprocess Creation

### 快速开始

```python
# 最简单的方式
from flexkv.utils.subprocess import safe_spawn_process

process = safe_spawn_process(target=my_function, args=(arg1, arg2))
process.start()
process.join()
```

### 为什么需要？

当 FlexKV 通过 MPI 启动时（`mpirun`），直接使用 `mp.Process()` 创建子进程会导致卡死。

### 两种 API

#### 1. `safe_spawn_process()` - 最简单 ⭐

```python
from flexkv.utils.subprocess import safe_spawn_process

p = safe_spawn_process(
    target=worker_func,
    args=(1, 2),
    kwargs={'option': 'value'},
    daemon=True
)
p.start()
p.join()
```

#### 2. `create_safe_process()` - 最灵活

```python
import torch.multiprocessing as mp
from flexkv.utils.subprocess import create_safe_process

mp_ctx = mp.get_context('spawn')
p = create_safe_process(
    mp_ctx,
    target=worker_func,
    args=(1, 2),
    daemon=True
)
p.start()
p.join()
```

### 迁移指南

#### 之前（可能卡住）

```python
import torch.multiprocessing as mp

mp_ctx = mp.get_context('spawn')
process = mp_ctx.Process(target=my_func, args=(arg1, arg2))
process.start()
```

#### 之后（安全）

```python
from flexkv.utils.subprocess import create_safe_process
import torch.multiprocessing as mp

mp_ctx = mp.get_context('spawn')
process = create_safe_process(mp_ctx, target=my_func, args=(arg1, arg2))
process.start()
```

### 完整文档

参见：
- [MPI_SUBPROCESS_GUIDE.md](../../docs/MPI_SUBPROCESS_GUIDE.md) - 详细指南
- [SUBPROCESS_CHANGES.md](../../SUBPROCESS_CHANGES.md) - 修改说明
- [examples/mpi_subprocess_demo.py](../../examples/mpi_subprocess_demo.py) - 示例代码
- [tests/test_mpi_subprocess.py](../../tests/test_mpi_subprocess.py) - 测试

### 常见问题

**Q: 不用 MPI 也需要用这个吗？**  
A: 推荐使用。即使不用 MPI，这个方法也完全兼容，没有副作用。

**Q: 会影响性能吗？**  
A: 几乎零开销，只是设置两个环境变量。

**Q: 可以在子进程中使用 MPI 吗？**  
A: 不行。这个方法专门阻止子进程初始化 MPI。

**Q: 父进程还能用 MPI 吗？**  
A: 可以！只有子进程被影响，父进程完全不受影响。

