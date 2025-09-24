so_dir=/cfs_zhongwei/moritzxu/code/MoonCake-SDK
# so_dir=/cfs_zhongwei/rongwei/Mooncake-SDK/origin/
log_dir=/cfs_zhongwei/moritzxu/logs
mooncake_log_dir=$log_dir/mooncake_logs
mkdir -p $mooncake_log_dir

# export MOONCAKE_CONFIG_PATH="./mooncake_config.json"
export LD_LIBRARY_PATH=$so_dir/engine.cpython-310-x86_64-linux-gnu.so:$LD_LIBRARY_PATH
export PYTHONPATH=$so_dir:$PYTHONPATH
export MC_REDIS_PASSWORD="yourpass"
strings $so_dir/engine.cpython-310-x86_64-linux-gnu.so | grep Version
python test_distributed_transfer_l.py