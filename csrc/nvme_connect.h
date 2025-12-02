#pragma once

namespace py = pybind11;

namespace flexkv {
/**
 * Connect NVMe-oF targets exported by all other nodes.
 * 
 * \param nvmets Dict[node ID, Dict[subsys name, Dict[str, str]]], where the innest dict is {'ip':
 *      IPv4 address, 'port': port number, 'dev': NVMe device name}
 * 
 * \return Dict[node ID, Dict[NVMf target's real name, local view of NVMf target]]
 */
py::dict nvme_connect(py::dict nvmets);

}
