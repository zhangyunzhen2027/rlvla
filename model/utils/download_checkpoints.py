"""
Pi0 Checkpoints下载工具
自动下载Pi0模型checkpoints到指定目录
"""
import sys
from pathlib import Path
from typing import List, Optional
import os

# 添加openpi路径
current_file = Path(__file__).resolve()
# model/utils/download_checkpoints.py -> model -> project root
project_root = current_file.parent.parent.parent
openpi_src = project_root / "pi0_interface" / "openpi" / "src"
if openpi_src.exists():
    if str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))

try:
    from openpi.shared import download
    OPENPI_AVAILABLE = True
except ImportError:
    OPENPI_AVAILABLE = False
    download = None

# 备用方案：直接使用 fsspec 下载（不需要完整的 openpi 依赖）
try:
    import fsspec
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False
    fsspec = None

# 备用方案：直接使用 fsspec 下载（不需要完整的 openpi 依赖）
try:
    import fsspec
    FSSPEC_AVAILABLE = True
except ImportError:
    FSSPEC_AVAILABLE = False
    fsspec = None


# Pi0 checkpoints配置
PI0_CHECKPOINTS = {
    "pi05_libero": "gs://openpi-assets/checkpoints/pi05_libero",
    "pi05_base": "gs://openpi-assets/checkpoints/pi05_base",
    "pi0_base": "gs://openpi-assets/checkpoints/pi0_base",
}

# HuggingFace checkpoints配置
HF_CHECKPOINTS = {
    "rlvla": "yunzhenzhang/rlvla",
    # 可以添加更多 HuggingFace checkpoints
}


def download_pi0_checkpoints(
    checkpoint_names: Optional[List[str]] = None,
    download_dir: Optional[str] = None,
    force_download: bool = False,
) -> dict[str, Path]:
    """
    下载Pi0 checkpoints到指定目录
    
    Args:
        checkpoint_names: 要下载的checkpoint名称列表，如果为None则下载所有
        download_dir: 下载目录，如果为None则使用默认目录
        force_download: 是否强制重新下载
    
    Returns:
        dict: checkpoint名称到本地路径的映射
    """
    # 优先使用 openpi 的下载功能，如果没有则使用 fsspec
    if not OPENPI_AVAILABLE and not FSSPEC_AVAILABLE:
        raise ImportError(
            "无法下载checkpoints：需要 openpi 或 fsspec[gcs]。"
            "请安装: pip install fsspec[gcs]"
        )
    
    # 设置下载目录
    if download_dir is None:
        # 默认使用项目根目录下的checkpoints/pi0目录
        download_dir = project_root / "checkpoints" / "pi0"
    else:
        download_dir = Path(download_dir)
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置环境变量，让Pi0下载到我们的目录
    original_env = os.environ.get("OPENPI_DATA_HOME")
    os.environ["OPENPI_DATA_HOME"] = str(download_dir)
    
    print(f"Pi0 checkpoints将下载到: {download_dir}")
    print(f"设置 OPENPI_DATA_HOME={download_dir}")
    print()
    
    # 确定要下载的checkpoints
    if checkpoint_names is None:
        checkpoint_names = list(PI0_CHECKPOINTS.keys())
    
    downloaded_paths = {}
    
    try:
        for checkpoint_name in checkpoint_names:
            if checkpoint_name not in PI0_CHECKPOINTS:
                print(f"警告: 未知的checkpoint名称: {checkpoint_name}")
                print(f"可用的checkpoints: {list(PI0_CHECKPOINTS.keys())}")
                continue
            
            checkpoint_url = PI0_CHECKPOINTS[checkpoint_name]
            print(f"下载 {checkpoint_name}...")
            print(f"  URL: {checkpoint_url}")
            
            try:
                if OPENPI_AVAILABLE:
                    # 使用 openpi 的下载功能
                    local_path = download.maybe_download(
                        checkpoint_url,
                        force_download=force_download,
                    )
                elif FSSPEC_AVAILABLE:
                    # 使用 fsspec 直接下载
                    fs = fsspec.filesystem('gs')
                    checkpoint_local_dir = download_dir / checkpoint_name
                    
                    if checkpoint_local_dir.exists() and not force_download:
                        print(f"  ✓ checkpoint 已存在: {checkpoint_local_dir}")
                        local_path = checkpoint_local_dir
                    else:
                        if checkpoint_local_dir.exists():
                            import shutil
                            shutil.rmtree(checkpoint_local_dir)
                        checkpoint_local_dir.mkdir(parents=True, exist_ok=True)
                        
                        print(f"  使用 fsspec 下载...")
                        fs.get(checkpoint_url, str(checkpoint_local_dir), recursive=True)
                        local_path = checkpoint_local_dir
                else:
                    raise ImportError("无法下载：需要 openpi 或 fsspec[gcs]")
                
                downloaded_paths[checkpoint_name] = local_path
                print(f"  ✓ 下载完成: {local_path}")
                print()
            except Exception as e:
                print(f"  ✗ 下载失败: {e}")
                print()
    
    finally:
        # 恢复原始环境变量
        if original_env is not None:
            os.environ["OPENPI_DATA_HOME"] = original_env
        elif "OPENPI_DATA_HOME" in os.environ:
            del os.environ["OPENPI_DATA_HOME"]
    
    return downloaded_paths


def get_checkpoint_path(checkpoint_name: str, download_dir: Optional[str] = None) -> Path:
    """
    获取checkpoint的本地路径，如果不存在则下载
    
    Args:
        checkpoint_name: checkpoint名称
        download_dir: 下载目录，如果为None则使用默认目录
    
    Returns:
        checkpoint的本地路径
    """
    if checkpoint_name not in PI0_CHECKPOINTS:
        raise ValueError(
            f"未知的checkpoint名称: {checkpoint_name}. "
            f"可用的checkpoints: {list(PI0_CHECKPOINTS.keys())}"
        )
    
    if download_dir is None:
        download_dir = project_root / "checkpoints" / "pi0"
    else:
        download_dir = Path(download_dir)
    
    # 设置环境变量
    original_env = os.environ.get("OPENPI_DATA_HOME")
    os.environ["OPENPI_DATA_HOME"] = str(download_dir)
    
    try:
        checkpoint_url = PI0_CHECKPOINTS[checkpoint_name]
        local_path = download.maybe_download(checkpoint_url)
        return local_path
    finally:
        if original_env is not None:
            os.environ["OPENPI_DATA_HOME"] = original_env
        elif "OPENPI_DATA_HOME" in os.environ:
            del os.environ["OPENPI_DATA_HOME"]


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载Pi0模型checkpoints")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="要下载的checkpoint名称（默认：下载所有）",
        choices=list(PI0_CHECKPOINTS.keys()) + ["all"],
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="下载目录（默认：checkpoints/pi0）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（即使已存在）",
    )
    
    args = parser.parse_args()
    
    # 处理checkpoints参数
    checkpoint_names = args.checkpoints
    if checkpoint_names is not None and "all" in checkpoint_names:
        checkpoint_names = None
    
    # 下载checkpoints
    downloaded = download_pi0_checkpoints(
        checkpoint_names=checkpoint_names,
        download_dir=args.download_dir,
        force_download=args.force,
    )
    
    print("\n下载完成！")
    print("下载的checkpoints:")
    for name, path in downloaded.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

