"""
Policy网络：策略网络，生成动作chunk分布
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .encoders import ObservationEncoder


class PolicyNetwork(nn.Module):
    """
    Policy网络：π(A_t | x_t) - 改进的Pi0策略
    
    这个网络学习在给定observation下生成动作chunk的分布
    我们使用高斯分布，输出均值和标准差
    """
    
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
        
        # 动作生成头（输出均值和标准差）
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_horizon * action_dim),
        )
        
        self.std_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_horizon * action_dim),
            nn.Softplus(),  # 确保标准差为正
        )
        
        # 添加小的常数以确保数值稳定性
        self.min_std = 1e-6
    
    def forward(self, x_t: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: observation字典
        Returns:
            mean: (B, H, action_dim) 动作chunk的均值
            std: (B, H, action_dim) 动作chunk的标准差
        """
        obs_feat = self.obs_encoder(x_t)  # (B, hidden_dim)
        
        mean = self.mean_head(obs_feat)  # (B, H * action_dim)
        mean = mean.view(-1, self.action_horizon, self.action_dim)  # (B, H, action_dim)
        
        std = self.std_head(obs_feat)  # (B, H * action_dim)
        std = std.view(-1, self.action_horizon, self.action_dim)  # (B, H, action_dim)
        std = std + self.min_std  # 确保数值稳定性
        
        return mean, std
    
    def sample(self, x_t: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从策略分布中采样动作chunk"""
        mean, std = self.forward(x_t)
        dist = torch.distributions.Normal(mean, std)
        A_t = dist.sample()
        return A_t
    
    def log_prob(self, x_t: Dict[str, torch.Tensor], A_t: torch.Tensor) -> torch.Tensor:
        """计算动作chunk的对数概率"""
        mean, std = self.forward(x_t)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(A_t).sum(dim=-1).sum(dim=-1)  # (B,)
        return log_prob

