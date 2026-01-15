#!/bin/bash

# 1. 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 读取参数 (默认值)
MODEL=${1:-mamba_bilstm}   # 第1个参数: 模型名称 (默认 mamba_bilstm)
DATASET=${2:-Davis}        # 第2个参数: 数据集名称 (默认 Davis)

DATA_PATH="./data/${DATASET}.txt"
LOG_FILE="${DATASET}_${MODEL}_nohup.log"

echo "=================================================="
echo "   Persistent Training Launcher"
echo "   Model:   $MODEL"
echo "   Dataset: $DATASET"
echo "   Log:     $LOG_FILE"
echo "=================================================="

# 3. 后台运行
# 清理旧补丁
rm -f patch_manual.py patch_super_optimize.py

nohup python run.py train \
  --data "$DATA_PATH" \
  --dataset_name "$DATASET" \
  --model_name "$MODEL" \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.0001 \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started in background. PID: $PID"
echo "To watch progress, run:"
echo "  tail -f $LOG_FILE"
echo "To stop, run:"
echo "  kill $PID"
echo "=================================================="
