import argparse

from transformers import AutoConfig, PretrainedConfig

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.server.server import KVServer




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # NAME
    parser.add_argument("--enable-cpu",
                        action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--enable-ssd",
                        action=argparse.BooleanOptionalAction,
                        default=False,)
    parser.add_argument("--enable-remote",
                        action=argparse.BooleanOptionalAction,
                        default=False,)
    parser.add_argument("--model-path", type=str, help="model path", default="")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-cpu-blocks", type=int, default=8192)
    parser.add_argument("--num-ssd-blocks", type=int, default=8192)
    parser.add_argument("--num-remote-blocks", type=int, default=8192)
    parser.add_argument("--server-recv-port", type=str, default=None)
    parser.add_argument("--remote-cache-size-mode", type=str, default="block_num")
    parser.add_argument(
        "--ssd-cache-dir",
        type=str,
        nargs='+',
        default=[],
        help="SSD cache file paths (multiple paths supported, separated by spaces)"
    )
    parser.add_argument(
        "--remote-cache-path",
        type=str,
        nargs='+',
        default=[],
        help="remote cache paths (multiple paths supported, separated by spaces)"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    hf_config = AutoConfig.from_pretrained(args.model_path)

    num_layers=hf_config.num_hidden_layers
    if hasattr(hf_config, 'num_key_value_heads'):
        num_kv_heads=hf_config.num_key_value_heads
    elif hasattr(hf_config, 'num_attention_heads'):
        num_kv_heads=hf_config.num_attention_heads
    else:
        raise NotImplementedError
    head_size=(hf_config.head_dim if hasattr(hf_config, 'head_dim')
                else hf_config.hidden_size//hf_config.num_attention_heads)
    use_mla=hf_config.architectures[0].startswith("Deepseek")

    # TODO: different model config may have different attribute name
    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        use_mla=use_mla,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        dtype=hf_config.torch_dtype
    )

    cache_config = CacheConfig(
        enable_cpu=args.enable_cpu,
        enable_ssd=args.enable_ssd,
        enable_remote=args.enable_remote,
        enable_gds=False,
        tokens_per_block=args.block_size,
        num_cpu_blocks=args.num_cpu_blocks,
        num_ssd_blocks=args.num_ssd_blocks,
        num_remote_blocks=args.num_remote_blocks,
        ssd_cache_dir=args.ssd_cache_dir,
        remote_cache_size_mode=args.remote_cache_size_mode,
        remote_cache_path=args.remote_cache_path,
    )

    kvserver = KVServer(model_config, cache_config, args.server_recv_port)
    kvserver.run()
