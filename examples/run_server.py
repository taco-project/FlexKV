import argparse
from transformers import AutoConfig, PretrainedConfig
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.server.server import KVServer
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import init_logger


logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # NAME
    parser.add_argument("--model-path", type=str, help="model path", default="")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-cpu-blocks", type=int, default=8192)
    parser.add_argument("--server-recv-port", type=str, default=None)
    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    hf_config = AutoConfig.from_pretrained(args.model_path)
    
    num_layers=hf_config.num_hidden_layers
    num_kv_heads=hf_config.num_key_value_heads
    head_size=(hf_config.head_dim if hasattr(hf_config, 'head_dim') 
                else hf_config.hidden_size//hf_config.num_attention_heads)
    element_size=hf_config.torch_dtype.itemsize
    use_mla=hf_config.architectures[0].startswith("Deepseek")
    
    # TODO: different model config may have different attribute name
    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        element_size=element_size,
        use_mla=use_mla,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
    )
    
    cpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=args.num_cpu_blocks,
        tokens_per_block=args.block_size,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=use_mla,
    )
    
    cache_config = CacheConfig(
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        cpu_kv_layout=cpu_kv_layout,
        use_gds=False,
        use_pinned_memory=True,
        tokens_per_block=args.block_size,
        num_cpu_blocks=args.num_cpu_blocks,
    )
    
    kvserver = KVServer(model_config, cache_config, args.server_recv_port)
    kvserver.run()