"""
Value网络：V函数，估计状态价值
"""
import torch
import torch.nn as nn
from typing import Dict

from .encoders import ObservationEncoder


class ValueNetwork(nn.Module):
    """Value网络：V(x_t) - 估计状态x_t的价值"""
    
    def __init__(
        self,
        image_size: int = 256,
        state_dim: int = 8,
        hidden_dim: int = 512,
    ):
        super().__init__()
        
        # Observation编码器
        self.obs_encoder = ObservationEncoder(
            image_size=image_size,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            feature_dim=hidden_dim,
        )
        
        # V值预测头
        self.v_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_t: observation字典
        Returns:
            v_value: (B, 1) V值估计
        """
        obs_feat = self.obs_encoder(x_t)  # (B, hidden_dim)
        v_value = self.v_head(obs_feat)  # (B, 1)
        return v_value

