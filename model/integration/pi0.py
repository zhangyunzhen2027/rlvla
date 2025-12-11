"""
Pi0模型集成：加载Pi0策略并用于生成动作chunk
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Optional

# 添加openpi路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
openpi_src = project_root / "pi0_interface" / "openpi" / "src"
if openpi_src.exists():
    if str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))

try:
    # openpi is an optional dependency with runtime path injection
    from openpi.training import config as _config  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    from openpi.policies import policy_config as _policy_config  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    from openpi.shared import download  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
    OPENPI_AVAILABLE = True
except ImportError as e:
    OPENPI_AVAILABLE = False
    error_msg = str(e)
    _config = None
    _policy_config = None
    download = None
    
    # 提供更详细的错误信息
    if "etils" in error_msg or "flax" in error_msg or "jax" in error_msg:
        print(f"警告: 无法导入openpi模块，缺少依赖: {error_msg}")
        print(f"openpi需要额外的依赖（etils, flax, jax等）。")
        print(f"请确保pi0_interface/openpi已正确安装依赖。")
        print(f"如果使用uv: cd pi0_interface/openpi && uv sync")
        print(f"如果使用pip: 请参考pi0_interface/openpi/pyproject.toml安装所有依赖")
    else:
        print(f"警告: 无法导入openpi模块: {error_msg}")
        print(f"请确保pi0_interface/openpi路径正确且已安装依赖")

# 导入checkpoint下载工具
try:
    from ..utils.download_checkpoints import get_checkpoint_path, PI0_CHECKPOINTS
    DOWNLOAD_UTILS_AVAILABLE = True
except ImportError:
    try:
        from utils.download_checkpoints import get_checkpoint_path, PI0_CHECKPOINTS
        DOWNLOAD_UTILS_AVAILABLE = True
    except ImportError:
        DOWNLOAD_UTILS_AVAILABLE = False
        get_checkpoint_path = None
        PI0_CHECKPOINTS = {}


class Pi0PolicyWrapper:
    """
    Pi0策略包装器，用于生成动作chunk
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        config_name: Optional[str] = None,
        device: str = "cpu",
        checkpoints_dir: Optional[str] = None,
    ):
        """
        Args:
            checkpoint_dir: Pi0 checkpoint目录路径（可以是gs://路径或本地路径）
            config_name: 配置名称（如 "pi05_libero"）
            device: 设备 ('cpu' 或 'cuda')
            checkpoints_dir: 本地checkpoints目录（如果指定，会优先使用本地checkpoint）
        """
        if not OPENPI_AVAILABLE:
            raise ImportError("openpi模块不可用，无法加载Pi0策略")
        
        self.device = device
        
        # 如果没有指定checkpoint，使用默认的LIBERO checkpoint
        if checkpoint_dir is None and config_name is None:
            config_name = "pi05_libero"
            checkpoint_dir = "gs://openpi-assets/checkpoints/pi05_libero"
        
        if config_name is None:
            raise ValueError("必须指定 config_name 或 checkpoint_dir")
        
        # 如果指定了checkpoints_dir，尝试从本地获取checkpoint
        if checkpoints_dir is not None and DOWNLOAD_UTILS_AVAILABLE:
            try:
                local_checkpoint = get_checkpoint_path(config_name, checkpoints_dir)
                if local_checkpoint.exists():
                    checkpoint_dir = str(local_checkpoint)
                    print(f"使用本地checkpoint: {checkpoint_dir}")
            except Exception as e:
                print(f"警告: 无法从本地获取checkpoint，将使用远程路径: {e}")
        
        # 如果checkpoint_dir是gs://路径，设置OPENPI_DATA_HOME以便下载到指定目录
        if checkpoint_dir.startswith("gs://") and checkpoints_dir is not None:
            original_env = os.environ.get("OPENPI_DATA_HOME")
            os.environ["OPENPI_DATA_HOME"] = str(Path(checkpoints_dir).resolve())
            try:
                # download.maybe_download会自动下载并返回本地路径
                checkpoint_dir = str(download.maybe_download(checkpoint_dir))
            finally:
                if original_env is not None:
                    os.environ["OPENPI_DATA_HOME"] = original_env
                elif "OPENPI_DATA_HOME" in os.environ:
                    del os.environ["OPENPI_DATA_HOME"]
        
        print(f"加载Pi0策略: {config_name}")
        print(f"Checkpoint: {checkpoint_dir}")
        
        # 获取配置
        train_config = _config.get_config(config_name)
        
        # 创建策略
        self.policy = _policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
            pytorch_device=device,
        )
        
        print(f"Pi0策略加载完成")
        print(f"  - Action horizon: {train_config.model.action_horizon}")
        print(f"  - Action dim: {train_config.model.action_dim}")
    
    def generate_action_chunk(self, observation: Dict) -> np.ndarray:
        """
        使用Pi0生成动作chunk
        
        Args:
            observation: 包含 'image', 'wrist_image', 'state', 'prompt' 的字典
        
        Returns:
            action_chunk: (action_horizon, action_dim) 动作chunk
        """
        # 转换observation格式为Pi0期望的格式
        pi0_obs = {
            "observation/image": observation['image'],
            "observation/wrist_image": observation['wrist_image'],
            "observation/state": observation['state'],
            "prompt": observation.get('prompt', ''),
        }
        
        # 使用Pi0生成动作
        result = self.policy.infer(pi0_obs)
        actions = result['actions']  # (action_horizon, action_dim)
        
        return actions
    
    def generate_action_chunks_batch(self, observations: list) -> np.ndarray:
        """
        批量生成动作chunk
        
        Args:
            observations: observation列表
        
        Returns:
            action_chunks: (batch_size, action_horizon, action_dim) 动作chunk数组
        """
        action_chunks = []
        for obs in observations:
            action_chunk = self.generate_action_chunk(obs)
            action_chunks.append(action_chunk)
        
        return np.array(action_chunks)

