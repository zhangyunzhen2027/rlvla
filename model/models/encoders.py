"""
编码器模块：图像、观察和动作编码器
"""
import torch
import torch.nn as nn
from typing import Dict


class ImageEncoder(nn.Module):
    """图像编码器：使用CNN提取图像特征"""
    
    def __init__(self, image_size: int = 256, feature_dim: int = 256):
        super().__init__()
        self.image_size = image_size
        self.feature_dim = feature_dim
        
        # 简单的CNN编码器
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),  # 256 -> 64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
        )
        
        # 计算特征维度
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 8 * 8, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) 图像tensor
        Returns:
            features: (B, feature_dim) 图像特征
        """
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ObservationEncoder(nn.Module):
    """Observation编码器：编码图像、状态和语言指令"""
    
    def __init__(
        self,
        image_size: int = 256,
        state_dim: int = 8,
        prompt_dim: int = 128,
        hidden_dim: int = 512,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        
        # 图像编码器（用于base和wrist图像）
        self.image_encoder = ImageEncoder(image_size, feature_dim // 2)
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        
        # 语言指令编码器（简单的embedding）
        self.prompt_encoder = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
    
    def forward(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observation: 包含 'image', 'wrist_image', 'state', 'prompt' 的字典
        Returns:
            encoded: (B, feature_dim) 编码后的observation特征
        """
        # 编码图像
        base_img_feat = self.image_encoder(observation['image'])
        wrist_img_feat = self.image_encoder(observation['wrist_image'])
        img_feat = torch.cat([base_img_feat, wrist_img_feat], dim=-1)  # (B, feature_dim)
        
        # 编码状态
        state_feat = self.state_encoder(observation['state'])  # (B, hidden_dim // 2)
        
        # 编码语言指令（这里简化处理，实际可以使用预训练的语言模型）
        # 对于prompt，我们使用简单的embedding
        prompt = observation.get('prompt', '')
        # 简化：使用prompt的长度作为特征（实际应该使用语言模型）
        prompt_feat = torch.zeros(observation['state'].shape[0], self.hidden_dim // 2, 
                                  device=observation['state'].device)
        
        # 融合所有特征
        combined = torch.cat([img_feat, state_feat, prompt_feat], dim=-1)  # (B, feature_dim + hidden_dim)
        encoded = self.fusion(combined)  # (B, feature_dim)
        
        return encoded


class ActionEncoder(nn.Module):
    """动作chunk编码器"""
    
    def __init__(self, action_horizon: int, action_dim: int, feature_dim: int = 256):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        
        # 使用MLP编码动作chunk
        self.encoder = nn.Sequential(
            nn.Linear(action_horizon * action_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
    
    def forward(self, A_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A_t: (B, H, action_dim) 动作chunk
        Returns:
            encoded: (B, feature_dim) 编码后的动作特征
        """
        B = A_t.shape[0]
        A_flat = A_t.view(B, -1)  # (B, H * action_dim)
        encoded = self.encoder(A_flat)
        return encoded

