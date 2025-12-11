#!/bin/bash
# Chunk-level离线强化学习训练脚本

# 设置数据集路径
DATASET_DIR="data/lerobot/libero_spatial"

# 检查数据集是否存在
if [ ! -d "$DATASET_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATASET_DIR"
    echo "请确保数据集已正确转换到LeRobot格式"
    exit 1
fi

# 检测设备
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "检测到CUDA，使用GPU训练"
else
    DEVICE="cpu"
    echo "未检测到CUDA，使用CPU训练"
fi

# 运行训练
python -m model.scripts.train \
    --dataset-dir "$DATASET_DIR" \
    --action-horizon 10 \
    --action-dim 7 \
    --gamma 0.99 \
    --beta 1.0 \
    --lambda-v 1.0 \
    --lambda-pi 1.0 \
    --lr 3e-4 \
    --batch-size 32 \
    --num-epochs 100 \
    --log-interval 100 \
    --save-interval 10 \
    --device "$DEVICE"

