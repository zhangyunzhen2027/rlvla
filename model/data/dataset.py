"""
数据集加载器：将LeRobot格式的数据转换为chunk-level transitions
"""
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
import torch
from torch.utils.data import Dataset

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


def load_image_from_parquet(image_data) -> np.ndarray:
    """从parquet中的图像数据加载图像"""
    if isinstance(image_data, dict):
        if 'bytes' in image_data:
            image_bytes = image_data['bytes']
        elif 'path' in image_data:
            path = image_data['path']
            if Path(path).exists():
                return np.array(Image.open(path))
            else:
                raise ValueError(f"图像路径不存在: {path}")
        else:
            raise ValueError(f"无法解析图像数据: {image_data.keys()}")
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        raise ValueError(f"未知的图像数据类型: {type(image_data)}")
    
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
        raise ValueError(f"无法加载图像: {e}")


class ChunkDataset(Dataset):
    """
    Chunk-level数据集，将frame-level轨迹转换为chunk-level transitions
    
    每个transition包含:
    - x_t: chunk开始时的observation (image, state, prompt)
    - A_t: Pi0生成的动作chunk (H x action_dim)
    - r_t: chunk-level reward
    - x_{t+H}: chunk执行后的observation
    - done: 任务完成标志
    """
    
    def __init__(
        self,
        dataset_dir: str,
        action_horizon: int = 10,
        gamma: float = 0.99,
        device: str = "cpu",
        use_pi0: bool = False,
        pi0_checkpoint: Optional[str] = None,
        pi0_config: Optional[str] = None,
        pi0_checkpoints_dir: Optional[str] = None,
    ):
        """
        Args:
            dataset_dir: LeRobot数据集目录路径
            action_horizon: 动作chunk的长度 H
            gamma: 折扣因子
            device: 设备 ('cpu' 或 'cuda')
            use_pi0: 是否使用Pi0生成动作chunk（而不是使用演示数据中的动作）
            pi0_checkpoint: Pi0 checkpoint路径
            pi0_config: Pi0配置名称（如 "pi05_libero"）
            pi0_checkpoints_dir: 本地Pi0 checkpoints目录
        """
        self.dataset_dir = Path(dataset_dir)
        self.action_horizon = action_horizon
        self.gamma = gamma
        self.device = device
        self.use_pi0 = use_pi0 and PI0_AVAILABLE
        
        # 如果使用Pi0，加载Pi0策略
        self.pi0_policy = None
        if self.use_pi0:
            if Pi0PolicyWrapper is None:
                print("警告: Pi0集成不可用，将使用演示数据中的动作")
                self.use_pi0 = False
            else:
                print("使用Pi0生成动作chunk...")
                self.pi0_policy = Pi0PolicyWrapper(
                    checkpoint_dir=pi0_checkpoint,
                    config_name=pi0_config,
                    device=device,
                    checkpoints_dir=pi0_checkpoints_dir,
                )
        
        # 加载元数据
        self.info = self._load_info()
        self.episodes = self._load_episodes()
        self.tasks = self._load_tasks()
        
        # 构建chunk-level transitions
        self.transitions = self._build_transitions()
        
        print(f"数据集加载完成:")
        print(f"  - 总episodes: {len(self.episodes)}")
        print(f"  - 总transitions: {len(self.transitions)}")
        print(f"  - Action horizon: {self.action_horizon}")
        print(f"  - Action dim: {self.info['features']['actions']['shape'][0]}")
    
    def _load_info(self) -> dict:
        """加载info.json"""
        info_path = self.dataset_dir / "meta" / "info.json"
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def _load_episodes(self) -> List[dict]:
        """加载episodes.jsonl"""
        episodes_path = self.dataset_dir / "meta" / "episodes.jsonl"
        episodes = []
        with open(episodes_path, 'r') as f:
            for line in f:
                episodes.append(json.loads(line))
        return episodes
    
    def _load_tasks(self) -> List[dict]:
        """加载tasks.jsonl"""
        tasks_path = self.dataset_dir / "meta" / "tasks.jsonl"
        tasks = []
        with open(tasks_path, 'r') as f:
            for line in f:
                tasks.append(json.loads(line))
        return tasks
    
    def _load_episode_data(self, episode_idx: int) -> pd.DataFrame:
        """加载单个episode的parquet数据"""
        data_path_template = self.info['data_path']
        chunk_idx = 0  # 假设只有一个chunk
        episode_path = self.dataset_dir / "data" / data_path_template.format(
            episode_chunk=f"{chunk_idx:03d}",
            episode_index=f"{episode_idx:06d}"
        )
        
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode文件不存在: {episode_path}")
        
        table = pq.read_table(episode_path)
        df = table.to_pandas()
        return df
    
    def _build_transitions(self) -> List[dict]:
        """构建chunk-level transitions"""
        transitions = []
        
        for episode in self.episodes:
            episode_idx = episode['episode_index']
            episode_length = episode['length']
            # 从episode中获取task信息
            episode_tasks = episode.get('tasks', [])
            if episode_tasks and len(episode_tasks) > 0:
                task_text = episode_tasks[0]  # 使用第一个task描述
            else:
                task_idx = episode.get('task_index', 0)
                task_text = self.tasks[task_idx]['task'] if task_idx < len(self.tasks) else ""
            
            # 加载episode数据
            df = self._load_episode_data(episode_idx)
            
            # 提取数据
            images = []
            wrist_images = []
            states = []
            actions = []
            rewards = []
            
            for idx in range(len(df)):
                row = df.iloc[idx]
                
                # 加载图像
                img = load_image_from_parquet(row['image'])
                wrist_img = load_image_from_parquet(row['wrist_image'])
                
                images.append(img)
                wrist_images.append(wrist_img)
                
                # 提取state和action
                state = np.array(row['state'], dtype=np.float32)
                if isinstance(state, (list, np.ndarray)) and len(state) > 0:
                    if isinstance(state[0], (list, np.ndarray)):
                        state = np.array(state[0], dtype=np.float32)
                states.append(state)
                
                action = np.array(row['actions'], dtype=np.float32)
                if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                    if isinstance(action[0], (list, np.ndarray)):
                        action = np.array(action[0], dtype=np.float32)
                actions.append(action)
                
                # 提取reward
                reward = row['reward']
                if isinstance(reward, (list, np.ndarray)):
                    reward = float(reward[0]) if len(reward) > 0 else 0.0
                else:
                    reward = float(reward)
                rewards.append(reward)
            
            # 转换为numpy数组
            images = np.array(images)
            wrist_images = np.array(wrist_images)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards, dtype=np.float32)
            
            # 构建chunk-level transitions
            # 对于每个可能的chunk起始位置
            for t in range(0, episode_length - self.action_horizon + 1):
                # 提取chunk
                chunk_end = min(t + self.action_horizon, episode_length)
                
                # Observation at chunk start: x_t
                x_t = {
                    'image': images[t],
                    'wrist_image': wrist_images[t],
                    'state': states[t],
                    'prompt': task_text,
                }
                
                # Action chunk: A_t = (a_t, a_{t+1}, ..., a_{t+H-1})
                if self.use_pi0 and self.pi0_policy is not None:
                    # 使用Pi0生成动作chunk
                    try:
                        A_t = self.pi0_policy.generate_action_chunk(x_t)
                        # 确保形状正确
                        if A_t.shape[0] != self.action_horizon:
                            # 如果Pi0生成的chunk长度不匹配，进行截断或填充
                            if A_t.shape[0] > self.action_horizon:
                                A_t = A_t[:self.action_horizon]
                            else:
                                padding = np.tile(A_t[-1:], (self.action_horizon - A_t.shape[0], 1))
                                A_t = np.concatenate([A_t, padding], axis=0)
                    except Exception as e:
                        print(f"警告: Pi0生成动作失败，使用演示动作: {e}")
                        # 回退到使用演示动作
                        A_t = actions[t:chunk_end]
                        if len(A_t) < self.action_horizon:
                            padding = np.tile(A_t[-1:], (self.action_horizon - len(A_t), 1))
                            A_t = np.concatenate([A_t, padding], axis=0)
                else:
                    # 使用演示数据中的动作
                    A_t = actions[t:chunk_end]
                    # 如果chunk不足H步，用最后一个动作填充
                    if len(A_t) < self.action_horizon:
                        padding = np.tile(A_t[-1:], (self.action_horizon - len(A_t), 1))
                        A_t = np.concatenate([A_t, padding], axis=0)
                
                # Chunk-level reward: r_t = max(b_{t+k}) for k in [0, H-1]
                # 根据论文公式: r^{chunk}_t = 1(max_{k in [0,H-1]} b_{t+k} = 1)
                chunk_rewards = rewards[t:chunk_end]
                r_t = float(np.max(chunk_rewards))  # 如果chunk中任何一帧有reward=1，则chunk reward=1
                
                # Observation after chunk: x_{t+H}
                next_t = min(t + self.action_horizon, episode_length - 1)
                x_next = {
                    'image': images[next_t],
                    'wrist_image': wrist_images[next_t],
                    'state': states[next_t],
                    'prompt': task_text,
                }
                
                # Done flag: 检查是否是episode结束
                done = (chunk_end >= episode_length - 1)
                
                transitions.append({
                    'episode_idx': episode_idx,
                    't': t,
                    'x_t': x_t,
                    'A_t': A_t,
                    'r_t': r_t,
                    'x_next': x_next,
                    'done': done,
                })
        
        return transitions
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个chunk-level transition"""
        transition = self.transitions[idx]
        
        # 转换为tensor
        x_t = {
            'image': torch.from_numpy(transition['x_t']['image']).permute(2, 0, 1).float() / 255.0,  # HWC -> CHW, [0,1]
            'wrist_image': torch.from_numpy(transition['x_t']['wrist_image']).permute(2, 0, 1).float() / 255.0,
            'state': torch.from_numpy(transition['x_t']['state']).float(),
            'prompt': transition['x_t']['prompt'],
        }
        
        A_t = torch.from_numpy(transition['A_t']).float()
        
        r_t = torch.tensor(transition['r_t'], dtype=torch.float32)
        
        x_next = {
            'image': torch.from_numpy(transition['x_next']['image']).permute(2, 0, 1).float() / 255.0,
            'wrist_image': torch.from_numpy(transition['x_next']['wrist_image']).permute(2, 0, 1).float() / 255.0,
            'state': torch.from_numpy(transition['x_next']['state']).float(),
            'prompt': transition['x_next']['prompt'],
        }
        
        done = torch.tensor(transition['done'], dtype=torch.float32)
        
        return {
            'x_t': x_t,
            'A_t': A_t,
            'r_t': r_t,
            'x_next': x_next,
            'done': done,
        }

