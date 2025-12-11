"""
Critic网络：Q函数，估计动作chunk的价值
"""
import torch
import torch.nn as nn
from typing import Dict

from .encoders import ObservationEncoder, ActionEncoder


class Critic(nn.Module):
    """Critic网络：Q(x_t, A_t) - 估计执行动作chunk A_t从状态x_t的期望回报"""
    
    def __init__(
        self,
        image_size: int = 256,
        state_dim: int = 8,
        action_horizon: int = 10,
        action_dim: int = 7,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
        # Observation编码器
        self.obs_encoder = ObservationEncoder(
            image_size=image_size,
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            feature_dim=hidden_dim,
        )
        
        # 动作编码器
        self.action_encoder = ActionEncoder(
            action_horizon=action_horizon,
            action_dim=action_dim,
            feature_dim=hidden_dim,
        )
        
        # Q值预测头
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x_t: Dict[str, torch.Tensor], A_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: observation字典
            A_t: (B, H, action_dim) 动作chunk
        Returns:
            q_value: (B, 1) Q值估计
        """
        # 编码observation和action
        obs_feat = self.obs_encoder(x_t)  # (B, hidden_dim)
        act_feat = self.action_encoder(A_t)  # (B, hidden_dim)
        
        # 融合并预测Q值
        combined = torch.cat([obs_feat, act_feat], dim=-1)  # (B, hidden_dim * 2)
        q_value = self.q_head(combined)  # (B, 1)
        
        return q_value

