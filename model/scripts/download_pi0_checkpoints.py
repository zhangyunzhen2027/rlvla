#!/usr/bin/env python
"""
下载Pi0 checkpoints脚本
"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.utils.download_checkpoints import main

if __name__ == "__main__":
    main()

