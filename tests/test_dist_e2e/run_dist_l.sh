timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
so_dir=/cfs_zhongwei/moritzxu/code/MoonCake-SDK
log_dir=/cfs_zhongwei/moritzxu/logs
mooncake_log_dir=$log_dir/mooncake_logs_${timestamp}
mkdir -p $mooncake_log_dir

export MOONCAKE_CONFIG_PATH="./mooncake_config_l.json"
export MC_LOG_DIR=$mooncake_log_dir
export LD_LIBRARY_PATH=$so_dir/engine.cpython-310-x86_64-linux-gnu.so:$LD_LIBRARY_PATH
export PYTHONPATH=$so_dir:$PYTHONPATH
export MC_REDIS_PASSWORD="yourpass"
export MC_LEGACY_RPC_PORT_BINDING=24620
strings $so_dir/engine.cpython-310-x86_64-linux-gnu.so | grep Version
python test_distributed_e2e_l.py