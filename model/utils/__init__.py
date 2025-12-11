"""
工具模块：辅助函数和工具
"""
from .helpers import setup_device, setup_logging

# 可选：导入checkpoint下载工具
try:
    from .download_checkpoints import download_pi0_checkpoints, get_checkpoint_path, PI0_CHECKPOINTS
    __all__ = [
        'setup_device',
        'setup_logging',
        'download_pi0_checkpoints',
        'get_checkpoint_path',
        'PI0_CHECKPOINTS',
    ]
except ImportError:
    __all__ = [
        'setup_device',
        'setup_logging',
    ]

