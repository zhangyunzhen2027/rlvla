"""
Chunk-level离线强化学习模块

这是一个完整的离线强化学习框架，用于改进预训练的Pi0策略。
"""
from .data import ChunkDataset
from .models import Critic, ValueNetwork, PolicyNetwork
from .training import ChunkLevelOfflineRL
from .integration import Pi0PolicyWrapper
from .config import DEFAULT_CONFIG, get_default_config

__version__ = "0.1.0"

__all__ = [
    # 数据
    'ChunkDataset',
    # 模型
    'Critic',
    'ValueNetwork',
    'PolicyNetwork',
    # 训练
    'ChunkLevelOfflineRL',
    # 集成
    'Pi0PolicyWrapper',
    # 配置
    'DEFAULT_CONFIG',
    'get_default_config',
]
