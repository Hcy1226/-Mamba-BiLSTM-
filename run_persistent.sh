#!/bin/bash

# 1. 设置 Hugging Face 镜像 (防止网络中断)
export HF_ENDPOINT=https://hf-mirror.com

# 2. 提示信息
echo "=================================================="
echo "   Starting Persistent Training (Background Mode) "
echo "=================================================="

# 3. 使用 nohup 后台运行
# > training_nohup.log : 将输出保存到文件
# 2>&1                 : 将错误信息也保存到同一个文件
# &                    : 放入后台运行，即使 SSH 断开也不停止
nohup python run.py train --epochs 100 --batch_size 32 --lr 0.0001 --fine_tune > training_nohup.log 2>&1 &

# 4. 获取进程 ID
PID=$!
echo "Training is running in background. PID: $PID"
echo "Log file: training_nohup.log"
echo ""
echo "Command to check log: tail -f training_nohup.log"
echo "Command to stop:      kill $PID"
echo "=================================================="
