"""
将 LIBERO HDF5 格式的数据集转换为 LeRobot 格式。

这个脚本专门用于处理 HDF5 格式的 LIBERO 数据集（与 RLDS 格式不同）。

Usage:
uv run examples/libero/convert_libero_hdf5_to_lerobot.py --data_dir /path/to/your/data

指定输出目录：
uv run examples/libero/convert_libero_hdf5_to_lerobot.py --data_dir /path/to/your/data --output-dir /path/to/output

如果你想将数据集推送到 Hugging Face Hub，可以使用以下命令：
uv run examples/libero/convert_libero_hdf5_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

默认情况下，转换后的数据集将保存到 $HF_LEROBOT_HOME 目录。
"""

import json
import os
import sys
from pathlib import Path
import shutil

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

# 延迟导入 LeRobotDataset，以便在设置环境变量后再导入
REPO_NAME = "libero_spatial"  # 输出数据集的名称，也用于 Hugging Face Hub


def resize_image(image, size):
    """调整图像大小"""
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def get_language_instruction(hdf5_file):
    """从 HDF5 文件中提取语言指令"""
    if "problem_info" in hdf5_file["data"].attrs:
        problem_info = json.loads(hdf5_file["data"].attrs["problem_info"])
        return problem_info.get("language_instruction", "")
    return ""


def convert_hdf5_to_lerobot(
    data_dir: str, 
    *, 
    output_dir: str | None = None,
    push_to_hub: bool = False
):
    """
    将 HDF5 格式的 LIBERO 数据集转换为 LeRobot 格式
    
    Args:
        data_dir: 包含 HDF5 文件的目录路径
        output_dir: 输出目录路径（如果为 None，则使用默认的 HF_LEROBOT_HOME）
        push_to_hub: 是否推送到 Hugging Face Hub
    """
    data_dir = Path(data_dir)
    
    # 设置输出目录 - 必须在导入 LeRobotDataset 之前设置环境变量
    if output_dir is not None:
        output_base = Path(output_dir).resolve()
        # 临时设置环境变量来改变 LeRobot 的输出路径
        original_hf_home = os.environ.get("HF_LEROBOT_HOME")
        os.environ["HF_LEROBOT_HOME"] = str(output_base)
        output_path = output_base / REPO_NAME
    else:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        output_path = HF_LEROBOT_HOME / REPO_NAME
        original_hf_home = None
    
    # 现在导入 LeRobotDataset（环境变量已设置）
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    # 清理输出目录中任何现有的数据集
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 HDF5 文件
    hdf5_files = list(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise ValueError(f"在 {data_dir} 中未找到 HDF5 文件")
    
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件")
    
    # 创建 LeRobot 数据集，定义要存储的特征
    # OpenPi 假设本体感觉存储在 `state` 中，动作存储在 `action` 中
    # LeRobot 假设图像数据的 dtype 是 `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,  # LIBERO 数据通常以 10fps 记录
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # ee_pos (3) + ee_ori (4) + gripper (1) = 8
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),  # LIBERO 使用 7D 动作（位置控制）
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 遍历所有 HDF5 文件并写入 LeRobot 数据集
    total_episodes = 0
    for hdf5_file_path in tqdm(hdf5_files, desc="转换文件"):
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            # 获取语言指令
            language_instruction = get_language_instruction(hdf5_file)
            
            # 获取所有演示
            demos = list(hdf5_file["data"].keys())
            demos.sort()  # 确保顺序一致
            
            print(f"\n处理文件: {hdf5_file_path.name}")
            print(f"  语言指令: {language_instruction}")
            print(f"  演示数量: {len(demos)}")
            
            # 遍历每个演示
            for demo_key in tqdm(demos, desc=f"  转换 {hdf5_file_path.name}", leave=False):
                demo_group = hdf5_file[f"data/{demo_key}"]
                
                # 获取观察和动作
                obs_group = demo_group["obs"]
                actions = demo_group["actions"][:]  # shape: (T, 7)
                
                # 获取图像
                agentview_rgb = obs_group["agentview_rgb"][:]  # shape: (T, 128, 128, 3)
                eye_in_hand_rgb = obs_group["eye_in_hand_rgb"][:]  # shape: (T, 128, 128, 3)
                
                # 构建状态向量 (ee_pos + ee_ori + gripper)
                ee_pos = obs_group["ee_pos"][:]  # shape: (T, 3)
                ee_ori = obs_group["ee_ori"][:]  # shape: (T, 3) - 欧拉角
                
                # 处理 ee_ori：如果是 3 维（欧拉角），转换为 4 维（四元数）
                # 这里简单地将欧拉角填充为 4 维，实际使用时可能需要转换
                if ee_ori.shape[-1] == 3:
                    # 将欧拉角转换为四元数（简化处理：使用前 3 个值，w=1）
                    # 注意：这不是真正的欧拉角到四元数转换，只是保持维度一致
                    ee_ori_4d = np.zeros((ee_ori.shape[0], 4))
                    ee_ori_4d[:, :3] = ee_ori
                    ee_ori_4d[:, 3] = 1.0  # w 分量
                    ee_ori = ee_ori_4d
                elif ee_ori.shape[-1] != 4:
                    raise ValueError(f"意外的 ee_ori 形状: {ee_ori.shape}")
                
                gripper_states = obs_group["gripper_states"][:]  # shape: (T, 2) 或 (T, 1) 或 (T,)
                # 取第一个值作为 gripper 状态（通常是开/关）
                if gripper_states.ndim == 1:
                    gripper_states = gripper_states[:, None]
                elif gripper_states.shape[-1] > 1:
                    # 如果有多个值，只取第一个
                    gripper_states = gripper_states[:, 0:1]
                
                # 组合状态
                state = np.concatenate([ee_pos, ee_ori, gripper_states], axis=-1)  # shape: (T, 8)
                
                # 确保所有数组长度一致
                num_steps = len(actions)
                assert len(agentview_rgb) == num_steps, f"图像长度 {len(agentview_rgb)} 与动作长度 {num_steps} 不匹配"
                assert len(eye_in_hand_rgb) == num_steps, f"手腕图像长度 {len(eye_in_hand_rgb)} 与动作长度 {num_steps} 不匹配"
                assert len(state) == num_steps, f"状态长度 {len(state)} 与动作长度 {num_steps} 不匹配"
                
                # 写入每一帧
                for step_idx in range(num_steps):
                    dataset.add_frame(
                        {
                            "image": resize_image(agentview_rgb[step_idx], (256, 256)),
                            "wrist_image": resize_image(eye_in_hand_rgb[step_idx], (256, 256)),
                            "state": state[step_idx].astype(np.float32),
                            "actions": actions[step_idx].astype(np.float32),
                            "task": language_instruction,
                        }
                    )
                
                # 保存这一集
                dataset.save_episode()
                total_episodes += 1
    
    print(f"\n转换完成！总共转换了 {total_episodes} 个演示")
    print(f"数据集保存在: {output_path}")
    
    # 恢复原始环境变量（如果修改过）
    if output_dir is not None:
        if original_hf_home is not None:
            os.environ["HF_LEROBOT_HOME"] = original_hf_home
        else:
            # 如果原来没有设置，删除环境变量
            os.environ.pop("HF_LEROBOT_HOME", None)
    
    # 可选：推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "hdf5", "spatial"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(convert_hdf5_to_lerobot)

