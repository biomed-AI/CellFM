#!/bin/bash

echo "=========================================="
echo "Please run the script as: "
echo "bash run_gpu_cluster.sh DATA_PATH"
echo "For example: bash run_gpu_cluster.sh /path/dataset"
echo "It is better to use the absolute path."
echo "==========================================="
# export ASCEND_GLOBAL_LOG_LEVEL=2
# export SLOG_PRINT_TO_STDOUT=2
export MS_ENABLE_FORMAT_MODE=1
export MS_HCCL_CM_INIT=1
export HCCL_DETERMINISTIC=1
# export MINDSPORE_DUMP_CONFIG='/share-nfs/w50035851/code/msver/dump.json'
data='cancer'
start=$4
dir=device$((start/8+1))
rm -rf log/fin*.txt
rm -rf $dir
mkdir $dir
cp ./*.py ./$dir
cd $dir
rm -rf rank*
rm *.log
date
echo "start training"
python combine.py 2>&1 &
ttl=24
num=8
ip=$3
batch=4
port=8448
# 循环启动8个Worker训练进程
export MS_WORKER_NUM=$ttl          # 设置集群中Worker进程数量为8
export MS_SCHED_HOST=61.47.2.$ip  # 设置Scheduler IP地址为本地环路地址
# export MS_SCHED_HOST=127.0.0.1  # 设置Scheduler IP地址为本地环路地址
export MS_SCHED_PORT=$port       # 设置Scheduler端口
export MS_ROLE=MS_SCHED             # 设置启动的进程为MS_SCHED角色
python ./1B_$5train.py --dist --data0 $1 --data1 $2 --batch $batch --data $data > scheduler.log 2>&1 &
for((i=1;i<$num;i++));
do
    export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
    export MS_NODE_ID=$i                      # 设置进程id，可选
    python ./1B_$5train.py --dist --data0 $1 --data1 $2 --batch $batch --data $data > worker_$i.log 2>&1 &
done
export MS_ROLE=MS_WORKER        # 设置启动的进程为MS_WORKER角色
export MS_NODE_ID=0                      # 设置进程id，可选
python ./1B_$5train.py --dist --data0 $1 --data1 $2 --batch $batch --data $data 