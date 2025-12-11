#!/usr/bin/env python
"""
检查依赖是否已安装
"""
import sys
from importlib import util

REQUIRED_PACKAGES = {
    'torch': 'PyTorch',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'pyarrow': 'PyArrow',
    'tqdm': 'tqdm',
    'PIL': 'Pillow',
    'fsspec': 'fsspec (for Pi0 checkpoint download)',
    'filelock': 'filelock (for Pi0 checkpoint download)',
}

OPTIONAL_PACKAGES = {
    'openpi': 'Pi0 integration (requires pi0_interface/openpi)',
}

def check_package(package_name, display_name):
    """检查包是否已安装"""
    try:
        if package_name == 'PIL':
            spec = util.find_spec('PIL')
        else:
            spec = util.find_spec(package_name)
        return spec is not None
    except ImportError:
        return False

def main():
    print("检查依赖...")
    print("=" * 60)
    
    all_ok = True
    
    # 检查必需依赖
    print("\n必需依赖:")
    for package, display in REQUIRED_PACKAGES.items():
        installed = check_package(package, display)
        status = "✓" if installed else "✗"
        print(f"  {status} {display}")
        if not installed:
            all_ok = False
    
    # 检查可选依赖
    print("\n可选依赖:")
    for package, display in OPTIONAL_PACKAGES.items():
        installed = check_package(package, display)
        status = "✓" if installed else "○"
        print(f"  {status} {display}")
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("✓ 所有必需依赖已安装")
        return 0
    else:
        print("✗ 缺少必需依赖")
        print("\n安装命令 (使用Poetry):")
        print("  poetry install")
        print("\n或使用pip:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())

