# 远端 worker pipe 背压隔离

CPU↔Remote worker 的 pipe 由一个惰性启动的 FIFO sender 写入，慢远端进程不再通过
同步 `Connection.send()` 卡住共享 transfer scheduler。GPU↔CPU 和本地 SSD 的提交
路径保持原有行为。适用于当前统一的 `CPURemoteTransferWorker`，包括 Mooncake 和
PCFS backend。

队列最多保留 2048 个尚未发送的命令，shutdown sentinel 有独立保留位置。已接受
元数据的原 graph/buffer lease 一直保留到正常 worker 完成；队列满、断管或 sender
关闭时使用既有负完成协议，不重试、不制造成功 ACK。data、control、shutdown
共享同一个 writer，保持 FIFO。父侧 shutdown 仍由原有锁保护，多个 transfer-type
别名并发关闭同一个 handle 时只执行一次。

`pytest tests/test_remote_worker_pipe.py` 使用真实 pipe/进程验证背压、独立任务进度、
队列满、断管、控制消息顺序、优雅/强制退出和并发 shutdown。H2D 标签只携带元数据；
这些 CPU 检查不执行 GPU 拷贝、PCFS 或 RDMA，不代表设备故障恢复或性能验收。
