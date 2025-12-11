"""
损失函数模块：实现所有训练损失
"""
import torch
import torch.nn as nn
from typing import Dict


def compute_critic_loss(
    critic,
    value_net,
    batch: Dict[str, torch.Tensor],
    gamma: float,
    action_horizon: int,
    device: torch.device,
) -> torch.Tensor:
    """
    计算Critic损失: L_Q = E[(Q(x_t, A_t) - y_t)^2]
    
    其中 y_t = r_t + γ^H * V(x_{t+H})
    """
    # 处理observation，排除字符串类型的prompt
    x_t = {}
    for k, v in batch['x_t'].items():
        if k != 'prompt' and isinstance(v, torch.Tensor):
            x_t[k] = v.to(device)
        elif k == 'prompt':
            x_t[k] = v  # 保留字符串
    
    A_t = batch['A_t'].to(device)
    r_t = batch['r_t'].to(device)
    
    x_next = {}
    for k, v in batch['x_next'].items():
        if k != 'prompt' and isinstance(v, torch.Tensor):
            x_next[k] = v.to(device)
        elif k == 'prompt':
            x_next[k] = v  # 保留字符串
    
    done = batch['done'].to(device)
    
    # 计算Q值
    Q_pred = critic(x_t, A_t).squeeze(-1)  # (B,)
    
    # 计算TD target: y_t = r_t + γ^H * V(x_{t+H}) * (1 - done)
    with torch.no_grad():
        V_next = value_net(x_next).squeeze(-1)  # (B,)
        y_t = r_t + (gamma ** action_horizon) * V_next * (1 - done)
    
    # Critic损失
    loss = nn.functional.mse_loss(Q_pred, y_t)
    
    return loss


def compute_value_loss(
    critic,
    value_net,
    policy,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """
    计算Value损失: L_V = E[(V(x_t) - E_{A_t~π}[Q(x_t, A_t) - log π(A_t|x_t)])^2]
    
    使用soft Bellman backup
    """
    x_t = {}
    for k, v in batch['x_t'].items():
        if k != 'prompt' and isinstance(v, torch.Tensor):
            x_t[k] = v.to(device)
        elif k == 'prompt':
            x_t[k] = v  # 保留字符串
    
    # 计算V(x_t)
    V_pred = value_net(x_t).squeeze(-1)  # (B,)
    
    # 从策略中采样动作chunk并计算期望
    with torch.no_grad():
        # 采样多个动作chunk来估计期望
        n_samples = 5
        Q_values = []
        log_probs = []
        
        for _ in range(n_samples):
            A_t_sampled = policy.sample(x_t)  # (B, H, action_dim)
            Q_val = critic(x_t, A_t_sampled).squeeze(-1)  # (B,)
            log_prob = policy.log_prob(x_t, A_t_sampled)  # (B,)
            
            Q_values.append(Q_val)
            log_probs.append(log_prob)
        
        # 计算期望: E[Q - log π]
        Q_values = torch.stack(Q_values, dim=0)  # (n_samples, B)
        log_probs = torch.stack(log_probs, dim=0)  # (n_samples, B)
        
        # 使用重要性采样或直接平均
        target = (Q_values - log_probs).mean(dim=0)  # (B,)
    
    # Value损失
    loss = nn.functional.mse_loss(V_pred, target)
    
    return loss


def compute_policy_loss(
    critic,
    value_net,
    policy,
    batch: Dict[str, torch.Tensor],
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """
    计算Policy损失: L_π = -E[w_t * log π(A_t | x_t)]
    
    其中 w_t = exp(β * (Q(x_t, A_t) - V(x_t)))
    """
    x_t = {}
    for k, v in batch['x_t'].items():
        if k != 'prompt' and isinstance(v, torch.Tensor):
            x_t[k] = v.to(device)
        elif k == 'prompt':
            x_t[k] = v  # 保留字符串
    
    A_t = batch['A_t'].to(device)
    
    # 计算advantage: A = Q(x_t, A_t) - V(x_t)
    with torch.no_grad():
        Q_val = critic(x_t, A_t).squeeze(-1)  # (B,)
        V_val = value_net(x_t).squeeze(-1)  # (B,)
        advantage = Q_val - V_val
    
    # 计算权重: w_t = exp(β * advantage)
    weights = torch.exp(beta * advantage)  # (B,)
    
    # 归一化权重（可选，用于稳定性）
    weights = weights / (weights.mean() + 1e-8)
    
    # 计算策略对数概率
    log_prob = policy.log_prob(x_t, A_t)  # (B,)
    
    # Policy损失（负对数似然，加权）
    loss = -(weights * log_prob).mean()
    
    return loss

