"""
为已转换的 LeRobot 数据集添加 reward 字段。

这个脚本会：
1. 读取所有 parquet 文件
2. 为每一帧添加 reward 字段（初始值为 0.0）
3. 更新数据集元数据

Usage:
poetry run python data/add_rewards_to_lerobot.py --dataset-dir data/lerobot/libero_spatial
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def add_rewards_to_dataset(dataset_dir: str, initial_reward: float = 0.0):
    """
    为 LeRobot 数据集添加 reward 字段
    
    Args:
        dataset_dir: 数据集目录路径
        initial_reward: 初始 reward 值（默认 0.0）
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        raise ValueError(f"数据集目录不存在: {dataset_path}")
    
    # 读取元数据
    meta_dir = dataset_path / "meta"
    info_file = meta_dir / "info.json"
    
    if not info_file.exists():
        raise ValueError(f"找不到元数据文件: {info_file}")
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # 检查是否已经有 reward 字段
    if "reward" in info.get("features", {}):
        print("⚠️  数据集已经包含 reward 字段")
        response = input("是否要覆盖现有的 reward 字段？(y/n): ")
        if response.lower() != 'y':
            print("取消操作")
            return
    
    # 更新 info.json，添加 reward 特征定义
    if "features" not in info:
        info["features"] = {}
    
    info["features"]["reward"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["reward"]
    }
    
    # 备份原始 info.json
    backup_file = info_file.with_suffix('.json.backup')
    shutil.copy2(info_file, backup_file)
    print(f"已备份元数据文件到: {backup_file}")
    
    # 处理所有 parquet 文件
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        raise ValueError(f"在 {data_dir} 中未找到 parquet 文件")
    
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    total_frames = 0
    for parquet_file in tqdm(parquet_files, desc="处理 parquet 文件"):
        # 读取 parquet 文件
        table = pq.read_table(parquet_file)
        
        # 检查是否已经有 reward 列
        if "reward" in table.column_names:
            print(f"⚠️  {parquet_file.name} 已经包含 reward 列，跳过")
            continue
        
        # 获取行数
        num_frames = len(table)
        total_frames += num_frames
        
        # 创建 reward 数组
        reward_array = pa.array([initial_reward] * num_frames, type=pa.float32())
        
        # 添加 reward 字段到 schema
        reward_field = pa.field("reward", pa.float32())
        new_schema = table.schema.append(reward_field)
        
        # 创建新的 table，包含所有原有列和新的 reward 列
        new_arrays = [table[col] for col in table.column_names]
        new_arrays.append(reward_array)
        
        new_table = pa.Table.from_arrays(new_arrays, schema=new_schema)
        
        # 备份原始文件
        backup_parquet = parquet_file.with_suffix('.parquet.backup')
        shutil.copy2(parquet_file, backup_parquet)
        
        # 写入新的 parquet 文件
        pq.write_table(new_table, parquet_file, compression='snappy')
    
    # 更新 info.json
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"\n✓ 完成！")
    print(f"  - 处理了 {len(parquet_files)} 个 parquet 文件")
    print(f"  - 添加了 {total_frames} 个 reward 字段（初始值: {initial_reward})")
    print(f"  - 备份文件保存在原文件同目录，扩展名为 .backup")
    print(f"\n现在你可以使用标注工具来修改 reward 值")


def main():
    parser = argparse.ArgumentParser(
        description="为 LeRobot 数据集添加 reward 字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为数据集添加 reward 字段（初始值为 0.0）
  poetry run python data/add_rewards_to_lerobot.py --dataset-dir data/lerobot/libero_spatial
  
  # 指定初始 reward 值
  poetry run python data/add_rewards_to_lerobot.py --dataset-dir data/lerobot/libero_spatial --initial-reward 0.0
        """
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="LeRobot 数据集目录路径"
    )
    parser.add_argument(
        "--initial-reward",
        type=float,
        default=0.0,
        help="初始 reward 值（默认: 0.0）"
    )
    
    args = parser.parse_args()
    
    add_rewards_to_dataset(args.dataset_dir, args.initial_reward)


if __name__ == "__main__":
    main()

