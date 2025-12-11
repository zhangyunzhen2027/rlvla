"""
辅助工具函数
"""
import torch
import logging
from pathlib import Path
from typing import Optional


def setup_device(device: Optional[str] = None) -> torch.device:
    """
    设置计算设备
    
    Args:
        device: 设备名称 ('cpu' 或 'cuda')，如果为None则自动检测
    
    Returns:
        torch.device对象
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device_obj = torch.device(device)
    print(f"使用设备: {device_obj}")
    
    if device_obj.type == "cuda":
        print(f"  - CUDA设备: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA版本: {torch.version.cuda}")
    
    return device_obj


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO):
    """
    设置日志
    
    Args:
        log_dir: 日志目录，如果为None则不保存到文件
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_dir / "training.log")] if log_dir else []),
        ]
    )
    
    return logging.getLogger(__name__)

