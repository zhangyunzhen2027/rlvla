#!/bin/bash
# 下载Pi0 checkpoints脚本

# 设置下载目录（默认：项目根目录下的checkpoints/pi0）
DOWNLOAD_DIR="${1:-checkpoints/pi0}"

echo "下载Pi0 checkpoints到: $DOWNLOAD_DIR"
echo ""

# 运行下载脚本
python -m model.scripts.download_pi0_checkpoints \
    --checkpoints pi05_libero pi05_base \
    --download-dir "$DOWNLOAD_DIR"

echo ""
echo "下载完成！Checkpoints保存在: $DOWNLOAD_DIR"

