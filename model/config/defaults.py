"""
默认配置
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """训练配置"""
    action_horizon: int = 10
    action_dim: int = 7
    gamma: float = 0.99
    beta: float = 1.0
    lambda_v: float = 1.0
    lambda_pi: float = 1.0
    lr: float = 3e-4
    batch_size: int = 32
    num_epochs: int = 100
    log_interval: int = 100
    save_interval: int = 10


@dataclass
class NetworkConfig:
    """网络配置"""
    image_size: int = 256
    state_dim: int = 8
    hidden_dim: int = 512
    feature_dim: int = 512


@dataclass
class Pi0Config:
    """Pi0集成配置"""
    use_pi0_actions: bool = False
    pi0_checkpoint: Optional[str] = None
    pi0_config: Optional[str] = "pi05_libero"
    init_policy_from_pi0: bool = False


@dataclass
class DefaultConfig:
    """默认配置集合"""
    training: TrainingConfig
    network: NetworkConfig
    pi0: Pi0Config
    
    @classmethod
    def create(cls):
        return cls(
            training=TrainingConfig(),
            network=NetworkConfig(),
            pi0=Pi0Config(),
        )


DEFAULT_CONFIG = DefaultConfig.create()


def get_default_config() -> DefaultConfig:
    """获取默认配置"""
    return DEFAULT_CONFIG

