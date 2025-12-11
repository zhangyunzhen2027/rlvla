"""
查看 LeRobot parquet 文件中的图像数据。

Usage:
poetry run python data/view_lerobot_parquet.py data/lerobot/libero_spatial/data/chunk-000/episode_000000.parquet
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import io


def load_image_from_parquet(image_data):
    """
    从 parquet 中的图像数据加载图像
    
    Args:
        image_data: parquet 中的图像数据（可能是 dict 或 bytes）
    
    Returns:
        numpy array 格式的图像
    """
    if isinstance(image_data, dict):
        # LeRobot 格式：{'bytes': b'...', 'path': '...'}
        if 'bytes' in image_data:
            image_bytes = image_data['bytes']
        elif 'path' in image_data:
            # 如果有路径，尝试从文件系统加载
            path = image_data['path']
            if Path(path).exists():
                return np.array(Image.open(path))
            else:
                print(f"警告: 图像路径不存在: {path}")
                return None
        else:
            print(f"警告: 无法解析图像数据: {image_data.keys()}")
            return None
    elif isinstance(image_data, bytes):
        image_bytes = image_data
    else:
        print(f"警告: 未知的图像数据类型: {type(image_data)}")
        return None
    
    # 从 bytes 加载图像
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    except Exception as e:
        print(f"错误: 无法加载图像: {e}")
        return None


def view_parquet_images(parquet_path: str, start_frame: int = 0):
    """
    查看 parquet 文件中的图像并标注 reward
    
    Args:
        parquet_path: parquet 文件路径
        start_frame: 起始帧索引
    """
    parquet_file = Path(parquet_path)
    
    if not parquet_file.exists():
        print(f"错误: 文件不存在: {parquet_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"查看和标注: {parquet_file.name}")
    print(f"{'='*60}\n")
    
    # 读取 parquet 文件
    try:
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
    except Exception as e:
        print(f"错误: 无法读取 parquet 文件: {e}")
        return
    
    # 检查是否有 reward 列
    if 'reward' not in df.columns:
        print("错误: 数据集中没有 reward 列")
        print("请先运行: poetry run python data/add_rewards_to_lerobot.py --dataset-dir data/lerobot/libero_spatial")
        return
    
    # 将 reward 转换为 numpy 数组以便修改
    rewards = df['reward'].values.copy()
    if isinstance(rewards[0], (list, np.ndarray)):
        rewards = np.array([r[0] if len(r) > 0 else 0.0 for r in rewards], dtype=np.float32)
    else:
        rewards = rewards.astype(np.float32)
    
    # 标记是否有未保存的修改
    has_unsaved_changes = False
    
    print(f"总帧数: {len(df)}")
    print(f"列名: {list(df.columns)}")
    print(f"\n{'='*60}")
    print("控制:")
    print("  [←/→] 或 [a/d]: 上一帧/下一帧")
    print("  [1]: 标记当前帧 reward = 1")
    print("  [0]: 标记当前帧 reward = 0")
    print("  [g]: 跳转到指定帧")
    print("  [i]: 显示当前帧信息")
    print("  [s]: 保存修改到文件")
    print("  [r]: 重置所有 reward 为 0")
    print("  [q]: 退出（会提示保存）")
    print(f"{'='*60}\n")
    
    current_frame = start_frame
    n_frames = len(df)
    
    while True:
        if current_frame < 0:
            current_frame = 0
        if current_frame >= n_frames:
            current_frame = n_frames - 1
        
        # 获取当前帧数据
        row = df.iloc[current_frame]
        
        # 加载图像
        images_to_show = {}
        
        if 'image' in df.columns:
            img = load_image_from_parquet(row['image'])
            if img is not None:
                images_to_show['image'] = img
        
        if 'wrist_image' in df.columns:
            wrist_img = load_image_from_parquet(row['wrist_image'])
            if wrist_img is not None:
                images_to_show['wrist_image'] = wrist_img
        
        if not images_to_show:
            print("错误: 未找到图像数据")
            break
        
        # 显示图像
        for img_name, img in images_to_show.items():
            # 转换为 BGR 用于 OpenCV 显示
            if img.shape[-1] == 3:
                display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                display_img = img
            
            # 添加帧信息
            info_text = f"{img_name} - Frame: {current_frame}/{n_frames-1}"
            cv2.putText(display_img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示 reward 信息（使用内存中的值）
            current_reward = rewards[current_frame]
            reward_text = f"Reward: {current_reward:.2f}"
            reward_color = (0, 255, 0) if current_reward > 0 else (255, 255, 255)
            cv2.putText(display_img, reward_text, (10, display_img.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, reward_color, 2)
            
            # 如果 reward = 1，添加绿色边框
            if current_reward > 0:
                cv2.rectangle(display_img, (0, 0), (display_img.shape[1]-1, display_img.shape[0]-1), (0, 255, 0), 3)
            
            # 显示标注统计
            num_annotated = np.sum(rewards > 0)
            stats_text = f"已标注: {num_annotated}/{n_frames}"
            cv2.putText(display_img, stats_text, (10, display_img.shape[0] - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 显示未保存提示
            if has_unsaved_changes:
                unsaved_text = "未保存的修改!"
                cv2.putText(display_img, unsaved_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 显示其他信息
            if 'episode_index' in df.columns:
                ep_text = f"Episode: {row['episode_index']}"
                cv2.putText(display_img, ep_text, (10, display_img.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(img_name, display_img)
        
        # 等待按键
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            if has_unsaved_changes:
                response = input("\n有未保存的修改，是否保存？(y/n): ")
                if response.lower() == 'y':
                    save_rewards_to_parquet(parquet_file, table, rewards)
            break
        elif key == ord('a') or key == 81:  # Left arrow
            current_frame = max(0, current_frame - 1)
        elif key == ord('d') or key == 83:  # Right arrow
            current_frame = min(n_frames - 1, current_frame + 1)
        elif key == ord('1'):
            # 标记 reward = 1
            old_reward = rewards[current_frame]
            rewards[current_frame] = 1.0
            has_unsaved_changes = True
            print(f"帧 {current_frame}: reward {old_reward} -> 1.0")
        elif key == ord('0'):
            # 标记 reward = 0
            old_reward = rewards[current_frame]
            rewards[current_frame] = 0.0
            has_unsaved_changes = True
            print(f"帧 {current_frame}: reward {old_reward} -> 0.0")
        elif key == ord('s'):
            # 保存修改
            save_rewards_to_parquet(parquet_file, table, rewards)
            has_unsaved_changes = False
            print("✓ 已保存修改")
        elif key == ord('r'):
            # 重置所有 reward
            response = input("\n确定要重置所有 reward 为 0 吗？(y/n): ")
            if response.lower() == 'y':
                rewards[:] = 0.0
                has_unsaved_changes = True
                print("已重置所有 reward 为 0")
        elif key == ord('g'):
            try:
                frame_num = int(input(f"\n跳转到帧 (0-{n_frames-1}): "))
                if 0 <= frame_num < n_frames:
                    current_frame = frame_num
            except ValueError:
                print("无效的帧号")
        elif key == ord('i'):
            print(f"\n帧 {current_frame} 信息:")
            print(f"  Episode index: {row.get('episode_index', 'N/A')}")
            print(f"  Frame index: {row.get('frame_index', 'N/A')}")
            print(f"  Reward: {rewards[current_frame]}")
            if 'state' in df.columns:
                state = row['state']
                if isinstance(state, (list, np.ndarray)):
                    print(f"  State shape: {len(state)}")
            if 'actions' in df.columns:
                actions = row['actions']
                if isinstance(actions, (list, np.ndarray)):
                    print(f"  Actions shape: {len(actions)}")
            print()
    
    cv2.destroyAllWindows()
    
    # 显示最终统计
    num_annotated = np.sum(rewards > 0)
    print(f"\n标注统计:")
    print(f"  总帧数: {n_frames}")
    print(f"  已标注 (reward=1): {num_annotated}")
    print(f"  未标注 (reward=0): {n_frames - num_annotated}")


def save_rewards_to_parquet(parquet_file: Path, original_table: pa.Table, rewards: np.ndarray):
    """
    保存 reward 修改到 parquet 文件
    
    Args:
        parquet_file: parquet 文件路径
        original_table: 原始的 Arrow Table
        rewards: 修改后的 reward 数组
    """
    # 备份原文件
    backup_file = parquet_file.with_suffix('.parquet.backup')
    if not backup_file.exists():
        shutil.copy2(parquet_file, backup_file)
    
    # 创建新的 reward 数组
    reward_array = pa.array(rewards, type=pa.float32())
    
    # 获取原始列
    original_columns = {col: original_table[col] for col in original_table.column_names}
    
    # 更新 reward 列
    original_columns['reward'] = reward_array
    
    # 创建新的 table
    new_arrays = [original_columns[col] for col in original_table.column_names]
    new_table = pa.Table.from_arrays(new_arrays, schema=original_table.schema)
    
    # 写入文件
    pq.write_table(new_table, parquet_file, compression='snappy')
    print(f"已保存到: {parquet_file}")


def inspect_parquet(parquet_path: str):
    """
    检查 parquet 文件的结构信息
    
    Args:
        parquet_path: parquet 文件路径
    """
    parquet_file = Path(parquet_path)
    
    if not parquet_file.exists():
        print(f"错误: 文件不存在: {parquet_file}")
        return
    
    print(f"\n{'='*60}")
    print(f"检查: {parquet_file.name}")
    print(f"{'='*60}\n")
    
    # 读取 parquet 文件
    try:
        table = pq.read_table(parquet_file)
        df = table.to_pandas()
    except Exception as e:
        print(f"错误: 无法读取 parquet 文件: {e}")
        return
    
    print(f"总帧数: {len(df)}")
    print(f"\n列名和类型:")
    for col in df.columns:
        col_type = df[col].dtype
        print(f"  - {col}: {col_type}")
    
    # 检查图像列
    print(f"\n图像列信息:")
    for img_col in ['image', 'wrist_image']:
        if img_col in df.columns:
            first_img = df[img_col].iloc[0]
            if isinstance(first_img, dict):
                print(f"  {img_col}:")
                print(f"    - 类型: dict")
                print(f"    - 键: {list(first_img.keys())}")
                if 'bytes' in first_img:
                    print(f"    - Bytes 大小: {len(first_img['bytes'])} bytes")
                if 'path' in first_img:
                    print(f"    - 路径: {first_img['path']}")
            else:
                print(f"  {img_col}: {type(first_img)}")
    
    # 显示前几行数据摘要
    print(f"\n前3行数据摘要:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"\n  帧 {idx}:")
        if 'episode_index' in df.columns:
            print(f"    Episode: {row['episode_index']}")
        if 'reward' in df.columns:
            reward = row['reward']
            if isinstance(reward, (list, np.ndarray)):
                reward = reward[0] if len(reward) > 0 else 0.0
            print(f"    Reward: {reward}")
        if 'state' in df.columns:
            state = row['state']
            if isinstance(state, (list, np.ndarray)):
                print(f"    State shape: {len(state)}")
        if 'actions' in df.columns:
            actions = row['actions']
            if isinstance(actions, (list, np.ndarray)):
                print(f"    Actions shape: {len(actions)}")


def main():
    parser = argparse.ArgumentParser(
        description="查看 LeRobot parquet 文件中的图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看 parquet 文件中的图像
  poetry run python data/view_lerobot_parquet.py data/lerobot/libero_spatial/data/chunk-000/episode_000000.parquet
  
  # 从指定帧开始查看
  poetry run python data/view_lerobot_parquet.py data/lerobot/libero_spatial/data/chunk-000/episode_000000.parquet --start-frame 50
  
  # 只检查文件结构（不显示图像）
  poetry run python data/view_lerobot_parquet.py data/lerobot/libero_spatial/data/chunk-000/episode_000000.parquet --inspect
        """
    )
    parser.add_argument(
        "parquet_path",
        type=str,
        help="Parquet 文件路径"
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="起始帧索引（默认: 0）"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="只检查文件结构，不显示图像"
    )
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_parquet(args.parquet_path)
    else:
        view_parquet_images(args.parquet_path, args.start_frame)


if __name__ == "__main__":
    main()

