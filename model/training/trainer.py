"""
训练器模块：Chunk-level离线强化学习训练器
"""
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import ChunkDataset
from ..models import Critic, ValueNetwork, PolicyNetwork
from .losses import compute_critic_loss, compute_value_loss, compute_policy_loss

# 可选：导入Pi0集成
try:
    from ..integration.pi0 import Pi0PolicyWrapper
    PI0_AVAILABLE = True
except ImportError:
    try:
        from integration.pi0 import Pi0PolicyWrapper
        PI0_AVAILABLE = True
    except ImportError:
        PI0_AVAILABLE = False
        Pi0PolicyWrapper = None


class ChunkLevelOfflineRL:
    """
    Chunk-level离线强化学习框架
    
    实现：
    1. Critic学习 (L_Q)
    2. Value函数学习 (L_V)
    3. Policy学习 (L_π)
    """
    
    def __init__(
        self,
        dataset_dir: str,
        action_horizon: int = 10,
        action_dim: int = 7,
        gamma: float = 0.99,
        beta: float = 1.0,
        lambda_v: float = 1.0,
        lambda_pi: float = 1.0,
        lr: float = 3e-4,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_pi0_actions: bool = False,
        pi0_checkpoint: Optional[str] = None,
        pi0_config: Optional[str] = None,
        init_policy_from_pi0: bool = False,
        pi0_checkpoints_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_dir: LeRobot数据集目录
            action_horizon: 动作chunk长度 H
            action_dim: 动作维度
            gamma: 折扣因子
            beta: advantage weighting温度参数
            lambda_v: value损失权重
            lambda_pi: policy损失权重
            lr: 学习率
            batch_size: batch大小
            device: 设备 ('cpu' 或 'cuda')
            use_pi0_actions: 是否使用Pi0生成动作chunk
            pi0_checkpoint: Pi0 checkpoint路径
            pi0_config: Pi0配置名称
            init_policy_from_pi0: 是否用Pi0初始化策略网络
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 超参数
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.gamma = gamma
        self.beta = beta
        self.lambda_v = lambda_v
        self.lambda_pi = lambda_pi
        self.batch_size = batch_size
        
        # 如果使用Pi0，确保checkpoints已下载
        if use_pi0_actions and pi0_config:
            if pi0_checkpoints_dir is None:
                # 默认使用项目根目录下的checkpoints/pi0
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                pi0_checkpoints_dir = str(project_root / "checkpoints" / "pi0")
            
            # 尝试下载checkpoint（如果不存在）
            try:
                from ..utils.download_checkpoints import get_checkpoint_path
                print(f"检查Pi0 checkpoint: {pi0_config}")
                local_path = get_checkpoint_path(pi0_config, pi0_checkpoints_dir)
                if local_path.exists():
                    print(f"✓ 找到本地checkpoint: {local_path}")
                    if pi0_checkpoint is None:
                        pi0_checkpoint = str(local_path)
            except Exception as e:
                print(f"警告: 无法获取本地checkpoint，将使用远程路径: {e}")
        
        # 加载数据集
        print("加载数据集...")
        self.dataset = ChunkDataset(
            dataset_dir=dataset_dir,
            action_horizon=action_horizon,
            gamma=gamma,
            device=str(self.device),
            use_pi0=use_pi0_actions,
            pi0_checkpoint=pi0_checkpoint,
            pi0_config=pi0_config,
            pi0_checkpoints_dir=pi0_checkpoints_dir,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        
        # 初始化网络
        print("初始化网络...")
        self.critic = Critic(
            image_size=256,
            state_dim=8,
            action_horizon=action_horizon,
            action_dim=action_dim,
            hidden_dim=512,
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            image_size=256,
            state_dim=8,
            hidden_dim=512,
        ).to(self.device)
        
        self.policy = PolicyNetwork(
            image_size=256,
            state_dim=8,
            action_horizon=action_horizon,
            action_dim=action_dim,
            hidden_dim=512,
        ).to(self.device)
        
        # 如果指定，用Pi0初始化策略网络
        if init_policy_from_pi0 and PI0_AVAILABLE and Pi0PolicyWrapper is not None:
            print("使用Pi0初始化策略网络...")
            self._init_policy_from_pi0(pi0_checkpoint, pi0_config)
        
        # 优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # 训练统计
        self.step = 0
        self.losses = {
            'critic': [],
            'value': [],
            'policy': [],
            'total': [],
        }
    
    def _init_policy_from_pi0(self, pi0_checkpoint: Optional[str], pi0_config: Optional[str]):
        """使用Pi0模型初始化策略网络"""
        if pi0_config is None:
            pi0_config = "pi05_libero"
        if pi0_checkpoint is None:
            pi0_checkpoint = "gs://openpi-assets/checkpoints/pi05_libero"
        
        try:
            pi0_policy = Pi0PolicyWrapper(
                checkpoint_dir=pi0_checkpoint,
                config_name=pi0_config,
                device=str(self.device),
            )
            print("Pi0策略加载成功，但策略网络初始化需要手动实现权重映射")
            print("当前策略网络使用随机初始化")
        except Exception as e:
            print(f"警告: 无法加载Pi0进行初始化: {e}")
            print("策略网络将使用随机初始化")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """执行一个训练步骤"""
        losses = {}
        
        # 1. 训练Critic
        self.critic_optimizer.zero_grad()
        critic_loss = compute_critic_loss(
            self.critic,
            self.value_net,
            batch,
            self.gamma,
            self.action_horizon,
            self.device,
        )
        critic_loss.backward()
        self.critic_optimizer.step()
        losses['critic'] = critic_loss.item()
        
        # 2. 训练Value网络
        self.value_optimizer.zero_grad()
        value_loss = compute_value_loss(
            self.critic,
            self.value_net,
            self.policy,
            batch,
            self.device,
        )
        value_loss.backward()
        self.value_optimizer.step()
        losses['value'] = value_loss.item()
        
        # 3. 训练Policy
        self.policy_optimizer.zero_grad()
        policy_loss = compute_policy_loss(
            self.critic,
            self.value_net,
            self.policy,
            batch,
            self.beta,
            self.device,
        )
        policy_loss.backward()
        self.policy_optimizer.step()
        losses['policy'] = policy_loss.item()
        
        # 总损失
        total_loss = critic_loss + self.lambda_v * value_loss + self.lambda_pi * policy_loss
        losses['total'] = total_loss.item()
        
        # 更新统计
        self.step += 1
        for key, value in losses.items():
            self.losses[key].append(value)
        
        return losses
    
    def train(self, num_epochs: int = 100, log_interval: int = 100, save_interval: int = 1000):
        """训练循环"""
        print(f"\n开始训练，共 {num_epochs} 个epochs")
        print(f"总batch数: {len(self.dataloader)}")
        print(f"超参数:")
        print(f"  - action_horizon: {self.action_horizon}")
        print(f"  - action_dim: {self.action_dim}")
        print(f"  - gamma: {self.gamma}")
        print(f"  - beta: {self.beta}")
        print(f"  - lambda_v: {self.lambda_v}")
        print(f"  - lambda_pi: {self.lambda_pi}")
        print(f"  - batch_size: {self.batch_size}")
        print()
        
        for epoch in range(num_epochs):
            epoch_losses = {'critic': [], 'value': [], 'policy': [], 'total': []}
            
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(pbar):
                losses = self.train_step(batch)
                
                # 更新进度条
                for key, value in losses.items():
                    epoch_losses[key].append(value)
                
                if (batch_idx + 1) % log_interval == 0:
                    avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
                    pbar.set_postfix(avg_losses)
            
            # 每个epoch结束时的平均损失
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"\nEpoch {epoch+1} 完成:")
            for key, value in avg_losses.items():
                print(f"  {key}: {value:.6f}")
            
            # 保存checkpoint
            if (epoch + 1) % save_interval == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(epoch + 1)
    
    def save_checkpoint(self, epoch: int, checkpoint_dir: str = "checkpoints"):
        """保存模型checkpoint"""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'critic_state_dict': self.critic.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'losses': self.losses,
            'hyperparameters': {
                'action_horizon': self.action_horizon,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'beta': self.beta,
                'lambda_v': self.lambda_v,
                'lambda_pi': self.lambda_pi,
            },
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        self.step = checkpoint['step']
        self.losses = checkpoint.get('losses', {'critic': [], 'value': [], 'policy': [], 'total': []})
        
        print(f"Checkpoint已加载: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Step: {self.step}")

