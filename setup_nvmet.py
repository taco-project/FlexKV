import os
import json
from typing import Dict, List, Tuple, Optional

def check_platform() -> None:
    import platform
    pass

def get_ifc_driver(ifc: str) -> Optional[str]:
    driver = f'/sys/class/net/{ifc}/device/driver'
    if os.path.exists(driver):
        return os.path.basename(os.path.realpath(driver))
    return None

def get_sysfs_path(path: str) -> Optional[str]:
    try:
        return os.path.realpath(path)
    except OSError:
        return None

def get_ip_map() -> Dict[str, str]:
    import sys
    import subprocess

    try:
        # ip -j addr
        result = subprocess.check_output(['ip', '-j', 'addr'], text=True)
        data = json.loads(result)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Is iproute2 installed?", file=sys.stderr)
        return {}

    ifc_info: Dict[str, Dict] = {}
    
    # 1st pass: Index all interfaces by name
    for entry in data:
        name: str = entry['ifname']
        # Extract IPv4 address (preferred) or IPv6
        ip_addr: str = None

        addr_info = entry.get('addr_info', [])
        # Prioritize IPv4
        v4 = [a['local'] for a in addr_info if a['family'] == 'inet']
        if v4:
            ip_addr = v4[0]
        else:
            # Fallback to IPv6 if no v4, excluding link-local if possible
            v6 = [a['local'] for a in addr_info if a['family'] == 'inet6' and not a.get('scope') == 'link']
            if v6:
                ip_addr = v6[0]

        ifc_info[name] = {
            'ip': ip_addr,
            'master': entry.get('master', None)
        }

    # 2nd pass: Resolve slaves to masters
    out: Dict[str, str] = {} # Physical dev -> IP address
    for name, info in ifc_info.items():
        if info['master']:
            # If this is a slave, look up the master's IP
            master_name = info['master']
            if master_name in ifc_info:
                out[name] = ifc_info[master_name]['ip']
            else:
                out[name] = None
        else:
            out[name] = info['ip']
            
    return out

def find_closest_nic(nvme_path: str, topo: List[Tuple[str, str, str]]) -> Optional[Tuple[str, str, str]]:
    closest_nic: Optional[Tuple[str, str, str]] = None
    longest_common = 0

    for nic_name, nic_path, ip_addr in topo:
        try:
            common = os.path.commonpath([nvme_path, nic_path])
            # The longer the common path, the closer they are in PCIe topology.
            if len(common) > longest_common:
                longest_common = len(common)
                closest_nic = (nic_name, ip_addr, common)
        except ValueError:
            continue

    return closest_nic

def associate_nvme_to_ip(nvme_list: List[str]) -> Dict[str, str]:
    '''Find IP addresses connected to given NVMe devices under the same PCIe root complex.

    Assumes at most 1 NIC and at most 1 NVMe device under a PCIe root complex.

    Args:
        nvme_list (List[str]): List of NVMe device names, e.g. ['nvme0n1', 'nvme1n1']

    Returns:
        Dict[str, str]: List of IP addresses associated with the NVMe devices (same order)
    '''
    # 1. Gather NIC physical paths
    nics: List[Tuple[str, str]] = []
    sys_net = '/sys/class/net'
    if os.path.exists(sys_net):
        for ifc in os.listdir(sys_net):
            # Check if it's an mlx5_core device
            driver = get_ifc_driver(ifc)
            if driver == 'mlx5_core':
                dev_path = get_sysfs_path(f"{sys_net}/{ifc}/device")
                if dev_path is not None:
                    nics.append((ifc, dev_path)) # Name, path
    # 2. Gather IP info
    ip_map: Dict[str, str] = get_ip_map()

    topo: List[Tuple[str, str, str]] = []
    for nic in nics:
        ip = ip_map.get(nic[0], None)
        topo.append((nic[0], nic[1], ip)) # NIC name, NIC path, IP address

    # 3. Process NVMe devices
    out: Dict[str, str] = {} # Key: IP address, value: common PCIe prefix
    for nvme in nvme_list:
        nvme_sys_path = get_sysfs_path(f'/sys/block/{nvme}/device')
        assert nvme_sys_path is not None, f'NVMe device /dev/{nvme} does not exist.'

        closest_nic = find_closest_nic(nvme_sys_path, topo)
        assert closest_nic is not None, \
            f'No connected NIC found for NVMe device /dev/{nvme} under the same PCIe root complex.'

        _, ip_addr, common_prefix = closest_nic
        out[ip_addr] = common_prefix
    return out

def enable_nvmet() -> None:
    assert os.path.exists('/sys/module/nvme/parameters/num_p2p_queues'), \
        'num_p2p_queues must be configured to enable NVMe-oF target offload.'
    with open('/sys/module/nvme/parameters/num_p2p_queues', 'r') as f:
        assert int(f.read().strip()) >= 1, \
            'num_p2p_queues must be at least 1 to enable NVMe-oF target offload.'

    nvmet_list: List[str] = os.environ['FLEXKV_ENABLED_NVMET'].strip().split(',')

    # Step 3 in https://enterprise-support.nvidia.com/s/article/howto-configure-nvme-over-fabrics--nvme-of--target-offload
    # Steps 1 and 2 are done manually beforehand.
    assert os.path.exists('/sys/kernel/config/nvmet/subsystems'), \
        'Dir /sys/kernel/config/nvmet/subsystems must exist to enable NVMf target offload'
    # NVMf subsystem name is _-prefixed device name, e.g. nvme0n1 -> _nvme0n1.
    subsys_list: List[str] = [f'_{dev}' for dev in nvmet_list]
    for subsys_name in subsys_list:
        os.mkdir(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}')

    # Step 4
    for subsys_name in subsys_list:
        with open(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}/attr_allow_any_host', 'w') as f:
            f.write('1')

    # Step 5
    for subsys_name in subsys_list:
        with open(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}/attr_offload', 'w') as f:
            f.write('1')

    # Step 6
    # NOTE: 1 offloading subsystem <-> 1 namespace
    nsid_list: List[str] = []
    import re
    for dev, subsys_name in zip(nvmet_list, subsys_list):
        match = re.search(r'nvme(\d+)n\d+', dev)
        if match:
            nsid = str(int(match.group(1)) + 1)
        else:
            print('WARN: NVMe device names do not follow nvmeXnY pattern.')
            nsid = '1'
        assert os.path.exists(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}/namespaces'), \
            f'Dir /sys/kernel/config/nvmet/subsystems/{subsys_name}/namespaces must exist to enable NVMf target offload'
        os.mkdir(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}/namespaces/{nsid}')
        nsid_list.append(nsid)

    # Step 7
    # E.g. SSD device nvme0n1 corresponds to offloading subsystem _nvme0n1 and namespace 0.
    # NOTE: 1 offloading subsystem <-> 1 namespace <-> 1 physical NVMe dev
    for dev, subsys_name, nsid in zip(nvmet_list, subsys_list, nsid_list):
        with open(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}/namespaces/{nsid}/device_path', 'w') as f:
            f.write(f'/dev/{dev}')

    # Step 8
    for subsys_name, nsid in zip(subsys_list, nsid_list):
        with open(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}/namespaces/{nsid}/enable', 'w') as f:
            f.write('1')

    # Step 9
    # NOTE: The offloading subsystem characterized by an IP address corresponds to a ConnectX NIC,
    #       which must be connected to the associated physical NVMe device under the same PCIe root
    #       complex.
    # NOTE: Assumes RoCE transport
    assert os.path.exists('/sys/kernel/config/nvmet/ports'), \
            'Dir /sys/kernel/config/nvmet/ports must exist to enable NVMf target offload'
    ip_dict: Dict[str, str] = associate_nvme_to_ip(nvmet_list)
    # In case NVMe device names do not follow nvmeXnY pattern s.t. namespaces are all '0', use
    # natural number as port name.
    if all(nsid == '1' for nsid in nsid_list):
        port_list = list(range(1, len(nvmet_list) + 1))
    else:
        port_list = nsid_list

    user_port_list: List[str] = []
    for port, ip_addr, subsys_name, dev in zip(port_list, ip_dict.keys(), subsys_list, nvmet_list):
        os.mkdir(f'/sys/kernel/config/nvmet/ports/{port}')

        print('Configure per-subsystem NVMf target offload port. 4420 by default. Can also use 49152 - 65535.')
        while True:
            user_port = input(f'Port number: ').strip()
            if not user_port:
                user_port = '4420'
                print(f'Use default port 4420 on offloading subsystem {subsys_name}.')
                break

            if user_port.isdigit():
                print(f'Use port {user_port} on offloading subsystem {subsys_name}.')
                break
            else:
                print("Invalid input. Please enter a numeric port number.")

        user_port_list.append(user_port)

        # 1 port <-> 1 offloading subsystem <-> 1 namespace <-> 1 physical NVMe dev
        with open(f'/sys/kernel/config/nvmet/ports/{port}/addr_trsvcid', 'w') as f:
            f.write(user_port)
        with open(f'/sys/kernel/config/nvmet/ports/{port}/addr_traddr', 'w') as f:
            print(f'NIC with IP address {ip_addr} and NVMe device {dev} are connected under the '
                  f'same PCIe root complex. Common PCIe address prefix: {ip_dict[ip_addr]}')
            f.write(ip_addr)
        with open(f'/sys/kernel/config/nvmet/ports/{port}/addr_trtype', 'w') as f:
            f.write('rdma')
        with open(f'/sys/kernel/config/nvmet/ports/{port}/addr_adrfam', 'w') as f:
            f.write('ipv4')

    # Dump NVMe-oF target offload config
    from pathlib import Path
    dir = Path(__file__).resolve().parent # $WORKSPACE/setup_nvmet.py
    assert os.path.exists(os.path.join(dir, 'flexkv/integration')), f'{dir}/flexkv/integration does not exist.'
    
    config_data: Dict[str, Dict[str, str]] = {}
    for ip_addr, user_port, subsys_name, dev in zip(ip_dict.keys(), user_port_list, subsys_list, nvmet_list):
        config_data[subsys_name] = {
            'ip': ip_addr,
            'port': user_port,
            'dev': dev
        }
    with open(os.path.join(dir, 'flexkv/integration/nvmet_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)

    # Step 10
    for port, subsys_name in zip(port_list, subsys_list):
        os.symlink(f'/sys/kernel/config/nvmet/subsystems/{subsys_name}',
                   f'/sys/kernel/config/nvmet/ports/{port}/subsystems/{subsys_name}')


if __name__ == "__main__":
    check_platform()
    enable_nvmet()
