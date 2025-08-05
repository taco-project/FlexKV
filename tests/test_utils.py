import time
import os
import shutil
from typing import List, Dict, Tuple
from multiprocessing import Process

import pytest
import torch

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


# Default configurations
DEFAULT_MODEL_CONFIG = {
    'num_layers': 4,  # Increased to work well for both kvmanager and transfer engine tests
    'num_kv_heads': 32,
    'head_size': 128,
    'dtype': torch.float16,
    'use_mla': False,
    'tp_size': 1,
    'dp_size': 1,
}

DEFAULT_CACHE_CONFIG = {
    'tokens_per_block': 16,
    'enable_cpu': True,
    'enable_ssd': True,
    'enable_remote': False,
    'num_cpu_blocks': 128,
    'num_ssd_blocks': 512,
    'num_remote_blocks': 512,  # Aligned with ssd_blocks
    'remote_cache_size_mode': "block_num",
    'remote_file_size': (1024*1024*1024),
    'remote_file_num': 16,
    'remote_file_prefix': "remote_cache",
    'use_gds': False,
    'enable_trace': False,
    'use_pinned_memory': False,
    'ssd_cache_dir': ["./ssd_cache", "./ssd_cache2/"],
    'ssd_cache_iouring_entries': 0,
    'remote_cache_path': ["remote_cache1", "remote_cache2"],
    'remote_config_custom': {
        "pcfs_fsid": "f_l91fz6",
        "pcfs_port": 31,
        "pcfs_ip": "172.21.16.177",
        "pcfs_parent_nodeid": 144115188075855883  # Using transfer engine value for consistency
    },
    'use_ce_transfer_h2d': False,
    'use_ce_transfer_d2h': True,
    'transfer_sms_h2d': 4,
    'transfer_sms_d2h': 4,
}

DEFAULT_TEST_CONFIG = {
    'num_gpu_blocks': 512,
    'requests_per_block': 16,
    'initial_write_ratio': 0.4,
    'use_server_client': False,
}

# Fixtures
@pytest.fixture
def model_config(request: pytest.FixtureRequest):
    """Create model configuration with optional parameters override"""
    param = request.param if hasattr(request, 'param') else {}
    cfg = dict(DEFAULT_MODEL_CONFIG, **param)
    return ModelConfig(**cfg)

@pytest.fixture
def cache_config(request: pytest.FixtureRequest):
    """Create cache configuration with optional parameters override"""
    param = request.param if hasattr(request, 'param') else {}
    cfg = dict(DEFAULT_CACHE_CONFIG, **param)
    return CacheConfig(**cfg)

@pytest.fixture
def test_config(request: pytest.FixtureRequest):
    """Create test configuration with optional parameters override"""
    param = request.param if hasattr(request, 'param') else {}
    cfg = dict(DEFAULT_TEST_CONFIG, **param)
    return cfg

# Utility functions
def generate_request_pair(idx: int, block_per_request, num_gpu_blocks, tokens_per_block, dp_size):
    """Generate a request pair with token_ids, block_ids, and dp_id"""
    start_idx = (idx * block_per_request) % num_gpu_blocks
    if start_idx + block_per_request >= num_gpu_blocks:
        start_idx = (
            (start_idx + block_per_request) % num_gpu_blocks
        )
    block_ids = torch.arange(
        start_idx,
        start_idx + block_per_request,
        dtype=torch.int64
    )
    token_ids = torch.randint(
        low=0,
        high=100,
        size=(block_per_request * tokens_per_block,),
        dtype=torch.int64
    )
    return token_ids, block_ids, idx % dp_size

def block_ids_2_slot_mapping(block_ids, tokens_per_block, actual_length=-1):
    """Convert block ids to slot mapping"""
    slot_mapping = block_ids.repeat_interleave(tokens_per_block) * tokens_per_block
    if actual_length == -1:
        actual_length = len(block_ids) * tokens_per_block
    return slot_mapping[:actual_length]

def create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks):
    """Create GPU KV layout"""
    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_size = model_config.head_size
    use_mla = model_config.use_mla
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block

    tpgroup_gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=model_config.use_mla
    )
    gpu_kv_layout = tpgroup_gpu_kv_layout.div_head(tp_size) if not use_mla else tpgroup_gpu_kv_layout
    return gpu_kv_layout

def generate_gpu_blocks_with_ground_truth(model_config, cache_config, test_config):
    """Generate GPU blocks with ground truth data for kvmanager tests"""
    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_size = model_config.head_size
    use_mla = model_config.use_mla
    tp_size = model_config.tp_size
    dp_size = model_config.dp_size
    dtype = model_config.dtype
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]

    tpgroup_gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=model_config.use_mla
    )
    gpu_blocks = {}
    dp_wise_gpu_blocks_gt = []
    for dp_id in range(dp_size):
        tp_group_tensors_gt = [
            torch.randn(size=tpgroup_gpu_kv_layout.kv_shape[1:], dtype=dtype)
            for _ in range(num_layers)
        ]
        head_per_tp = num_kv_heads // tp_size
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            device = (
                torch.device(f"cuda:{global_gpu_id}")
                if global_gpu_id < torch.cuda.device_count() else torch.device("cuda:0")
            )
            gpu_blocks[global_gpu_id] = []
            for layer_id in range(num_layers):
                if not use_mla:
                    start_head = tp_id * head_per_tp
                    end_head = (tp_id + 1) * head_per_tp
                else:
                    start_head = 0
                    end_head = num_kv_heads
                tp_tensor = tp_group_tensors_gt[layer_id][:, :, :, start_head:end_head, :].to(device)
                gpu_blocks[global_gpu_id].append(tp_tensor)
        dp_wise_gpu_blocks_gt.append(tp_group_tensors_gt)
    gpu_kv_layout = tpgroup_gpu_kv_layout.div_head(tp_size) if not use_mla else tpgroup_gpu_kv_layout
    return gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout

def verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla):
    """Verify data consistency between GPU blocks and ground truth"""
    head_per_tp = num_kv_heads // tp_size
    for dp_id in range(dp_size):
        gt = dp_wise_gpu_blocks_gt[dp_id]
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            gpu_tensors = gpu_blocks[global_gpu_id]
            for layer in range(num_layers):
                gpu_tensor = gpu_tensors[layer].cpu()
                if not use_mla:
                    start_head = tp_id * head_per_tp
                    end_head = (tp_id + 1) * head_per_tp
                else:
                    start_head = 0
                    end_head = num_kv_heads
                gt_tensor_slice = gt[layer][:, :, :, start_head:end_head, :]
                if not torch.allclose(gpu_tensor, gt_tensor_slice):
                    print(f"Mismatch at dp_id={dp_id}, tp_id={tp_id}, global_gpu_id={global_gpu_id}, layer={layer}")
                    print(f"GPU tensor shape: {gpu_tensor.shape}, GT slice shape: {gt_tensor_slice.shape}")
                assert torch.allclose(gpu_tensor, gt_tensor_slice), \
                    f"Mismatch at dp_id={dp_id}, tp_id={tp_id}, global_gpu_id={global_gpu_id}, layer={layer}"
    print("verify done")

def wait_for_transfer_completion(transfer_engine, expected_graph_ids: List[int], max_wait_time: float = 15.0) -> bool:
    """Wait for transfer graphs to complete and return success status"""
    completed_graph_ids = set()
    start_time = time.time()

    while len(completed_graph_ids) < len(expected_graph_ids) and (time.time() - start_time) < max_wait_time:
        results = transfer_engine.get_completed_graphs_and_ops(timeout=0.1)
        for graph_id, op_id in results:
            if op_id == -1:  # Graph completion
                completed_graph_ids.add(graph_id)
        time.sleep(0.001)

    return len(completed_graph_ids) == len(expected_graph_ids)

def cleanup_ssd_cache_dirs(cache_config):
    """Clean up SSD cache directories"""
    if hasattr(cache_config, 'ssd_cache_dir') and cache_config.ssd_cache_dir:
        for dir_path in cache_config.ssd_cache_dir:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

def skip_if_insufficient_gpus(required_gpus: int):
    """Skip test if insufficient GPUs available"""
    if torch.cuda.device_count() < required_gpus:
        pytest.skip(f"Need at least {required_gpus} CUDA devices")

def skip_if_no_cuda():
    """Skip test if no CUDA devices available"""
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available")

# Server-Client mode support functions
class KVManagerServerClient:
    """Server-Client wrapper for KVManager that manages server, tp_client, and dp_client processes"""

    def __init__(self, model_config, cache_config, gpu_kv_layout, gpu_blocks):
        import tempfile
        from flexkv.server.client import KVDPClient, KVTPClient
        from flexkv.server.server import KVServer

        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_kv_layout = gpu_kv_layout
        self.gpu_blocks = gpu_blocks

        # Create temporary IPC port for communication
        self.server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

        # Extract basic config parameters for server process
        server_config = {
            'num_layers': model_config.num_layers,
            'num_kv_heads': model_config.num_kv_heads,
            'head_size': model_config.head_size,
            'use_mla': model_config.use_mla,
            'tp_size': model_config.tp_size,
            'dp_size': model_config.dp_size,
            'dtype': str(model_config.dtype),
            'tokens_per_block': cache_config.tokens_per_block,
            'enable_cpu': cache_config.enable_cpu,
            'enable_ssd': cache_config.enable_ssd,
            'enable_remote': cache_config.enable_remote,
            'num_cpu_blocks': cache_config.num_cpu_blocks,
            'num_ssd_blocks': cache_config.num_ssd_blocks,
            'ssd_cache_dir': cache_config.ssd_cache_dir if hasattr(cache_config, 'ssd_cache_dir') else ["./ssd_cache"],
        }

        # Start server process
        self.server_process = Process(
            target=self._run_server,
            args=(self.server_recv_port, server_config),
            daemon=False
        )
        self.server_process.start()

        # Wait for server to start
        time.sleep(5)

        # Initialize dp_client
        self.dp_client = KVDPClient(self.server_recv_port, model_config)
        print("dp_client started")

        # Start tp_client processes
        self.tp_client_processes = []
        for tp_rank in range(model_config.tp_size):
            device_id = tp_rank + self.dp_client.dp_client_id * model_config.tp_size
            # Extract only the necessary basic types for tp_client
            tp_client_process = Process(
                target=KVManagerServerClient._run_tp_client,
                args=(self.dp_client.dp_client_id, tp_rank, device_id, self.server_recv_port,
                      model_config.num_layers, str(model_config.dtype), list(gpu_kv_layout.kv_shape[1:])),
                daemon=True
            )
            tp_client_process.start()
            self.tp_client_processes.append(tp_client_process)

        # Wait for tp clients to register
        time.sleep(5)

        self._server_client_mode = True

    def _run_server(self, server_recv_port, server_config):
        """Run server process"""
        from flexkv.server.server import KVServer
        from flexkv.common.config import ModelConfig, CacheConfig

        # Recreate config objects from basic parameters
        model_config = ModelConfig(
            num_layers=server_config['num_layers'],
            num_kv_heads=server_config['num_kv_heads'],
            head_size=server_config['head_size'],
            use_mla=server_config['use_mla'],
            tp_size=server_config['tp_size'],
            dp_size=server_config['dp_size'],
            dtype=torch.float16 if server_config['dtype'] == 'torch.float16' else torch.float32
        )

        cache_config = CacheConfig(
            tokens_per_block=server_config['tokens_per_block'],
            enable_cpu=server_config['enable_cpu'],
            enable_ssd=server_config['enable_ssd'],
            enable_remote=server_config['enable_remote'],
            num_cpu_blocks=server_config['num_cpu_blocks'],
            num_ssd_blocks=server_config['num_ssd_blocks'],
            ssd_cache_dir=server_config['ssd_cache_dir']
        )
        print("starting server ... ...")
        kvserver = KVServer(model_config, cache_config, server_recv_port)
        kvserver.run()
        print("server started")

    @staticmethod
    def _run_tp_client(dp_client_id, tp_rank, device_id, server_recv_port, num_layers, dtype_str, kv_shape):
        """Run tp_client process"""
        from flexkv.server.client import KVTPClient
        from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType

        tp_client = KVTPClient(server_recv_port, dp_client_id, device_id, tp_rank)
        # Convert dtype string back to torch dtype
        if dtype_str == "torch.float16":
            dtype = torch.float16
        elif dtype_str == "torch.float32":
            dtype = torch.float32
        else:
            dtype = torch.float16  # default

        # Create GPU blocks for this tp_rank in the tp_client process
        gpu_blocks_for_tp = []
        for layer_id in range(num_layers):
            gpu_blocks_for_tp.append(
                torch.rand(size=tuple(kv_shape), dtype=dtype).cuda(device_id)
            )

        # Create a simple layout for registration
        gpu_kv_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERWISE,
            num_layer=num_layers,
            num_block=kv_shape[1],  # Assuming this is the block dimension
            tokens_per_block=kv_shape[2],  # Assuming this is the tokens_per_block dimension
            num_head=kv_shape[3],  # Assuming this is the num_head dimension
            head_size=kv_shape[4],  # Assuming this is the head_size dimension
            is_mla=False
        )
        print("registering to server ... ...")
        tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)
        print("registered to server")
        # Keep the process running
        while True:
            time.sleep(1)

    def is_ready(self):
        """Check if the server-client system is ready"""
        return self.server_process.is_alive() and all(p.is_alive() for p in self.tp_client_processes)

    def start(self):
        """Start the server-client system (already started in __init__)"""
        return True

    def put_async(self, token_ids, slot_mapping, dp_id):
        """Put data to the server-client system"""
        return self.dp_client.put_async(token_ids, slot_mapping, token_mask=None)

    def get_async(self, token_ids, slot_mapping, layer_granularity, dp_id):
        """Get data from the server-client system"""
        return self.dp_client.get_async(token_ids, slot_mapping, token_mask=None)

    def wait_for_graph_finished(self, request):
        """Wait for graph to finish"""
        masks = self.dp_client.wait(request)
        time.sleep(0.2)
        return masks

    def wait(self, request_ids):
        """Wait for requests to complete"""
        masks = self.dp_client.wait(request_ids)
        return masks

    def shutdown(self):
        """Shutdown all processes"""
        print("Shutting down KVManagerServerClient...")

        # First, try to gracefully shutdown the server by sending a shutdown signal
        try:
            # Send a shutdown request to the server
            self.dp_client.shutdown()
            print("Sent shutdown request to server")

            # Wait a bit for graceful shutdown
            time.sleep(3)
        except Exception as e:
            print(f"Error sending shutdown request: {e}")

        # Terminate tp_client processes
        print("Terminating tp_client processes...")
        for tp_process in self.tp_client_processes:
            if tp_process.is_alive():
                tp_process.terminate()
                tp_process.join(timeout=5)
                if tp_process.is_alive():
                    print(f"Force killing tp_client process {tp_process.pid}")
                    tp_process.kill()
                    tp_process.join(timeout=2)

        # Terminate server process
        print("Terminating server process...")
        if self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=10)
            if self.server_process.is_alive():
                print(f"Force killing server process {self.server_process.pid}")
                self.server_process.kill()
                self.server_process.join(timeout=5)

        # Clean up temporary file
        import os
        if hasattr(self, 'server_recv_port') and self.server_recv_port.startswith('ipc://'):
            temp_file = self.server_recv_port[6:]  # Remove 'ipc://' prefix
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    print(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")

        print("KVManagerServerClient shutdown complete")

def create_kvmanager_with_mode(model_config, cache_config, gpu_kv_layout, gpu_blocks, use_server_client=False):
    """Create KVManager with optional server-client mode"""
    if use_server_client:
        print("Using server-client mode")
        return KVManagerServerClient(model_config, cache_config, gpu_kv_layout, gpu_blocks)
    else:
        from flexkv.kvmanager import KVManager
        return KVManager(model_config, cache_config, gpu_kv_layout, gpu_blocks)
