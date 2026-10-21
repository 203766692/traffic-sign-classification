"""实验随机种子设置工具。"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 的随机种子。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 在 CUDA 可用时同步设置所有 GPU 的随机种子并启用确定性卷积。
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
