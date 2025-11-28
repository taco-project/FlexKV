#pragma once

#include <unordered_map>
#include <map>
#include <string>

namespace py = pybind11;

namespace flexkv {
/**
 * Connect NVMe-oF targets exported by all other nodes.
 * 
 * \param nvmets Dict[node ID, OrderDict[subsys name, Dict[str, str]]], where the innest dict is
 *      {'ip': IPv4 address, 'port': port number, 'dev': NVMe device name}
 * 
 * \return Dict[node ID, OrderDict[local view of NVMf target, Dict[NVMf target's real name,
 *      corresponding subsystem, corresponding IP address, corresponding port number]]]
 */
std::unordered_map<int, py::dict> nvme_connect(std::unordered_map<int, py::dict>& nvmets);

}
