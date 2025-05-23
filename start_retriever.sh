#!/bin/bash

# 设置日志文件路径
log_file="retriever/logs/retrieval.log"

echo "[INFO] Killing existing retrieval_server.py processes..."
ps aux | grep retrieval_server.py | grep -v grep | awk '{print $2}' | xargs -r kill -9

echo "[INFO] Starting retriever script with nohup..."
mkdir -p "$(dirname "$log_file")"  # 确保日志目录存在
nohup bash retriever/retrieval_launch_e5_flat.sh > "$log_file" 2>&1 &

sleep 1  # 等待 nohup 启动
echo "[INFO] Tailing log file: $log_file"
tail -f "$log_file"
