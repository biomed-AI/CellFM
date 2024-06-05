#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
# export ASCEND_GLOBAL_LOG_LEVEL=2
# export SLOG_PRINT_TO_STDOUT=2
# export MS_ENABLE_FORMAT_MODE=1
# export MINDSPORE_DUMP_CONFIG='/share-nfs/w50035851/code/msver/dump.json'
# export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
dir=device_$3_$4
export HCCL_DETERMINISTIC=1
# rm -rf $dir
mkdir $dir
cp ./*.py $dir
# rm -rf /share-nfs/w50035851/analyse/hvg$3
# rm -rf log/fin*.txt
cd $dir
echo "start training"
ttl=8
port=8448
export MS_WORKER_NUM=$ttl          # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=127.0.0.1  # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=$port       # 设置Scheduler端口
# 循环启动8个Worker训练进程
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
python ./1B_$3.py --batch $1 --epoch $2 --dist --data $4 > scheduler.log 2>&1 &
for((i=1;i<$ttl;i++));
do
    export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    python ./1B_$3.py --batch $1 --epoch $2  --dist --data $4 > worker_$i.log 2>&1 &
    
done
export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
export MS_NODE_ID=0                      # 设置进程id，可选
python ./1B_$3.py --batch $1 --epoch $2 --dist --data $4