timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
so_dir=/workspace/sdk
log_dir=/workspace/logs
mooncake_log_dir=$log_dir/mooncake_logs_${timestamp}
mkdir -p $mooncake_log_dir

export MOONCAKE_CONFIG_PATH="./mooncake_config_r.json"
export LD_LIBRARY_PATH=$so_dir/engine.cpython-310-x86_64-linux-gnu.so:$LD_LIBRARY_PATH
export PYTHONPATH=$so_dir:$PYTHONPATH
export MC_LOG_DIR=$mooncake_log_dir
export MC_REDIS_PASSWORD="redis-serving-passwd"
export MC_LEGACY_RPC_PORT_BINDING=12840
strings $so_dir/engine.cpython-310-x86_64-linux-gnu.so | grep Version
#python test_distributed_e2e_r.py
python3 test_distributed_p2p_interleaved_a.py