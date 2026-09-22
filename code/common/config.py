# ============================================================
# config.py - 统一训练/推理配置（四个模型共用）
# 消除各模型脚本中重复的确定性、设备、并行度配置
#
# 环境变量开关：
#   SEED         随机种子，默认 42
#   NUM_WORKERS  DataLoader 工作进程数，默认 4（多核 CPU 加速数据准备）
#   USE_TF32     是否启用 TF32 矩阵乘法，默认 0（关闭以保证结果可复现；
#                设为 1 可让 RTX 40 系提速 20-40%，但会引入微小数值差异）
# ============================================================

import os
import random

# ---------- OMP 冲突兜底（必须在 import torch/numpy 之前设置） ----------
# 现象：Windows + Anaconda 环境下，PyTorch（MKL）和 conda numpy/scipy 各自带一份
#       libiomp5md.dll，导致 "OMP: Error #15: ... already initialized"。
# 处理：用 setdefault 避免覆盖用户自定义值；缺省时允许重复加载 + 限制线程，
#       保证四个模型脚本（lstm / transformer / kronos / lgbm）都能直接跑。
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

# ---------- 默认值 ----------
SEED = int(os.environ.get('SEED', '42'))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', '4'))
USE_TF32 = os.environ.get('USE_TF32', '0') == '1'


def get_device() -> str:
    """统一设备解析（cuda 优先）"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def configure_determinism(seed: int = SEED, use_tf32: bool = USE_TF32) -> str:
    """
    统一确定性配置：固定全部随机源 + 全套 cuDNN 确定性开关
    必须在 import torch 之后调用（调用方需在此之前设置 CUBLAS_WORKSPACE_CONFIG 环境变量）
    返回解析出的设备字符串
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # TF32：默认关闭以保持与历史实验可复现；需要提速时通过 USE_TF32=1 打开
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    return get_device()
