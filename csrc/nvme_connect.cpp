#include "nvme_connect.h"
#include <libnvme.h>
#include <folly/init/Init.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/experimental/coro/Task.h>
#include <folly/experimental/coro/BlockingWait.h>
#include <folly/experimental/coro/Collect.h>
#include <mutex>
#include <memory>
#include <algorithm>
#include <vector>
#include <utility>
#include <chrono>
#include <iostream>

static std::unique_ptr<folly::IOThreadPoolExecutor> g_executor; // Pool of EventBase loops
static std::unique_ptr<folly::CPUThreadPoolExecutor> g_blocking_executor; // For (blocking) write to /dev/nvme-fabrics
static std::once_flag g_init_flag;

void init_folly() {
    std::call_once(g_init_flag, []() {
        int threads = std::clamp(std::thread::hardware_concurrency(), 4, 32);
        g_executor = std::make_unique<folly::IOThreadPoolExecutor>(threads); // Create one EventBase (event loop) per thread
        g_blocking_executor = std::make_unique<folly::CPUThreadPoolExecutor>(threads);
    });
}

struct Target {
    int node_id;
    int idx; // RAID0 order
    std::string subsys_nqn;
    std::string ip;
    std::string port;
    std::string dev;
};

// Struct to hold the result of a connection attempt
struct Connection {
    int node_id;
    int idx;
    bool success;
    std::string local_view;

    std::string subsys_nqn;
    std::string ip;
    std::string port;
    std::string remote_dev;
};

folly::coro::Task<std::string> get_local_view_async(nvme_ctrl_t c) {
    // Retry logic for sysfs population race condition
    for (int i = 0; i < 5; i++) {
        nvme_ctrl_scan_namespace(c, NULL);
        nvme_ns_t n;
        nvme_ctrl_for_each_ns(c, n) {
            const char* name = nvme_ns_get_name(n);
            if (name) co_return std::string(name);
        }
        // Yield back to event Loop
        co_await folly::coro::sleep(std::chrono::milliseconds(10));
    }
    // Fallback
    const char* ctrl_name = nvme_ctrl_get_name(c);
    if (ctrl_name) {
        co_return std::string(ctrl_name) + "n1";
    }
    co_return "";
}

folly::coro::Task<Connection> connect_target_async(Target target) {
    // Switch to blocking executor
    co_await folly::coro::co_reschedule(g_blocking_executor.get());

    Connection result{
        .node_id = target.node_id,
        .idx = target.idx,
        .success = false,
        // .local_view = "",

        .subsys_nqn = target.subsys_nqn,
        .ip = target.ip,
        .port = target.port,
        .remote_dev = target.dev
    };
    // Create thread-local libnvme root. 
    nvme_root_t r = nvme_create_root(nullptr, LOG_ERR);
    if (!r) {
        co_return result;
    }
    nvme_host_t h = nvme_default_host(r);

    struct nvme_fabrics_config cfg = { 0 };
    cfg.nqn = const_cast<char*>(target.subsys_nqn.c_str());
    cfg.transport = const_cast<char*>("rdma");
    cfg.traddr = const_cast<char*>(target.ip.c_str());
    cfg.trsvcid = const_cast<char*>(target.port.c_str());

    nvme_ctrl_t c = nullptr;
    int ret = nvmf_add_ctrl(h, &c, &cfg); // Blocking bottleneck
    if (ret == 0 && c != nullptr) {
        co_await folly::coro::co_reschedule(g_executor.get());
        // Await the async name resolution (which yields during retries)
        result.local_view = co_await get_local_view_async(c);
        if (!result.local_view.empty()) {
            result.success = true;
        }
        // Switch back to blocking executor for cleanup
        co_await folly::coro::co_reschedule(g_blocking_executor.get());
    }
    // Free root and controller (c) while preserving connection
    nvme_free_tree(r);

    co_return result;
}

folly::coro::Task<std::vector<Connection>> run_connection_batch(std::vector<Target> all_targets) {
    std::vector<folly::coro::Task<Connection>> tasks;
    tasks.reserve(all_targets.size());

    for (const auto& t : all_targets) {
        tasks.push_back(connect_target_async(t));
    }
    auto results = co_await folly::coro::collectAllRange(std::move(tasks));
    
    // Unwrap Try<Connection>
    std::vector<Connection> unwrapped;
    unwrapped.reserve(results.size());
    for (auto& r : results) {
        unwrapped.push_back(*r);
    }
    co_return unwrapped;
}

namespace flexkv {

// We may some day revampt it to a std::exec version.
std::unordered_map<int, py::dict> nvme_connect(std::unordered_map<int, py::dict>& nvmets) {
    init_folly();

    // 1. Flatten inputs, must hold GIL
    std::vector<Target> all_targets;
    for (auto& [node_id, targets] : nvmets) { // node_id: int, targets: OrderedDict[str, Dict[str, str]]
        int raid0_order = 0;
        // Preserves the OrderedDict order (RAID0 geometry)
        for (auto subsys : targets) {
            Target t;
            t.node_id = node_id;
            t.idx = raid0_order++;
            t.subsys_nqn = py::str(subsys.first);
    
            py::dict details = subsys.second.cast<py::dict>();
            t.ip = details["ip"].cast<std::string>();
            t.port = details["port"].cast<std::string>();
            t.dev = details["dev"].cast<std::string>();
            
            all_targets.push_back(t);
        }
    }
    // 2. Connect NVMf targets
    std::vector<Connection> results;
    {
        py::gil_scoped_release release;
        results = folly::coro::blockingWait(
            run_connection_batch(std::move(all_targets)).scheduleOn(g_executor.get())
        );
    }
    // 3. Reconstruct output
    std::unordered_map<int, std::vector<Connection>> grouped_results;
    for (const auto& c : results) {
        if (c.success) {
            grouped_results[c.node_id].push_back(c);
        } else {
            std::cerr << "Error connecting to " << c.subsys_nqn << " at " << c.ip << std::endl;
        }
    }

    std::unordered_map<int, py::dict> result;
    for (auto& [node_id, targets] : grouped_results) {
        // Sort by RAID order
        std::sort(targets.begin(), targets.end(), [](const Connection& a, const Connection& b) {
            return a.idx < b.idx;
        });

        py::dict per_node; // OrderedDict
        for (const auto& t : targets) {
            per_node[t.local_view.c_str()] = py::make_tuple(
                t.remote_dev,
                t.subsys_nqn,
                t.ip,
                t.port
            );
        }
        result[node_id] = per_node;
    }

    return result;
}

}
