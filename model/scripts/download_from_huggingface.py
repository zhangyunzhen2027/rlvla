#!/usr/bin/env python
"""
从 HuggingFace Hub 下载 checkpoint

使用方法:
    poetry run python -m model.scripts.download_from_huggingface \
        --repo-id yunzhenzhang/rlvla \
        --download-dir checkpoints/pi0/rlvla
"""
import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download, login
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("错误: 需要安装 huggingface_hub")
    print("安装命令: pip install huggingface_hub")
    sys.exit(1)


def download_checkpoint(
    repo_id: str,
    download_dir: Optional[str] = None,
    token: Optional[str] = None,
    resume_download: bool = True,
    local_files_only: bool = False,
):
    """
    从 HuggingFace Hub 下载 checkpoint
    
    Args:
        repo_id: HuggingFace 仓库 ID (格式: username/repo-name)
        download_dir: 下载目录，如果为 None 则使用默认目录
        token: HuggingFace token (如果为 None，会尝试从环境变量或缓存中获取)
        resume_download: 是否支持断点续传
        local_files_only: 是否只使用本地文件（不下载）
    
    Returns:
        下载的 checkpoint 路径
    """
    # 设置下载目录
    if download_dir is None:
        # 默认使用项目根目录下的 checkpoints/pi0 目录
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        download_dir = project_root / "checkpoints" / "pi0" / repo_id.split("/")[-1]
    else:
        download_dir = Path(download_dir)
    
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"准备从 HuggingFace 下载 checkpoint")
    print(f"仓库 ID: {repo_id}")
    print(f"下载目录: {download_dir}")
    print()
    
    # 登录（如果需要）
    if token:
        try:
            login(token=token)
        except Exception as e:
            print(f"警告: 登录失败，尝试继续: {e}")
    
    # 检查是否需要登录（私有仓库）
    try:
        print(f"开始下载...")
        print(f"这可能需要一些时间，请耐心等待...")
        print()
        
        # 使用 snapshot_download 下载整个仓库
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        
        print(f"\n✓ 下载完成！")
        print(f"Checkpoint 位置: {local_path}")
        return Path(local_path)
        
    except Exception as e:
        error_msg = str(e)
        
        # 检查是否是认证问题
        if "401" in error_msg or "Unauthorized" in error_msg:
            print(f"\n✗ 下载失败: 需要认证")
            print(f"请提供 token 或登录:")
            print(f"  1. 使用命令行参数: --token YOUR_TOKEN")
            print(f"  2. 使用环境变量: export HF_TOKEN=YOUR_TOKEN")
            print(f"  3. 使用 hf auth login")
        elif "404" in error_msg or "not found" in error_msg.lower():
            print(f"\n✗ 下载失败: 仓库不存在")
            print(f"请检查仓库 ID 是否正确: {repo_id}")
            print(f"访问 https://huggingface.co/{repo_id} 确认仓库存在")
        else:
            print(f"\n✗ 下载失败: {e}")
        
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace Hub 下载 checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本下载
  poetry run python -m model.scripts.download_from_huggingface \\
      --repo-id yunzhenzhang/rlvla

  # 指定下载目录
  poetry run python -m model.scripts.download_from_huggingface \\
      --repo-id yunzhenzhang/rlvla \\
      --download-dir checkpoints/pi0/rlvla

  # 使用 token 下载私有仓库
  poetry run python -m model.scripts.download_from_huggingface \\
      --repo-id yunzhenzhang/rlvla \\
      --token hf_xxxxxxxxxxxxx

  # 只检查本地文件（不下载）
  poetry run python -m model.scripts.download_from_huggingface \\
      --repo-id yunzhenzhang/rlvla \\
      --local-files-only
        """
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace 仓库 ID (格式: username/repo-name)",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=None,
        help="下载目录（默认: checkpoints/pi0/<repo-name>）",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (可选，也可以使用环境变量 HF_TOKEN 或 hf auth login)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传（重新下载所有文件）",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用本地文件，不下载（用于检查已下载的文件）",
    )
    
    args = parser.parse_args()
    
    download_checkpoint(
        repo_id=args.repo_id,
        download_dir=args.download_dir,
        token=args.token,
        resume_download=not args.no_resume,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()

