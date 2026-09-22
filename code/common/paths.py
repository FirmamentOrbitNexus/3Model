# ============================================================
# paths.py - 项目路径与环境变量单点配置
# 全项目所有路径均从此处获取（common 内模块不自行拼接路径），
# 环境变量可覆盖，兼容本地开发与 Docker（/app）两种布局
# ============================================================

import os

# 项目根目录（code/common/paths.py -> 上两级）
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# ---------- 数据目录 ----------
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(BASE_DIR, 'data'))
STOCK_CSV = os.environ.get('DATA_FILE', os.path.join(DATA_DIR, 'stock_data.csv'))
HS300_CSV = os.environ.get('HS300_FILE', os.path.join(DATA_DIR, 'hs300_stock_list.csv'))
CALENDAR_JSON = os.path.join(DATA_DIR, 'trade_calendar.json')

# ---------- 产物目录 ----------
MODEL_ROOT = os.environ.get('MODEL_ROOT', os.path.join(BASE_DIR, 'model'))
OUTPUT_ROOT = os.environ.get('OUTPUT_ROOT', os.path.join(BASE_DIR, 'output'))
LOG_DIR = os.environ.get('LOG_DIR', os.path.join(BASE_DIR, 'logs'))


def model_dir(name: str) -> str:
    """某模型的权重目录（如 model/lstm）；环境变量 MODEL_DIR 可整体覆盖"""
    return os.environ.get('MODEL_DIR', os.path.join(MODEL_ROOT, name))


def output_dir(name: str) -> str:
    """某模型的输出目录（如 output/lstm）；环境变量 OUTPUT_DIR 可整体覆盖"""
    return os.environ.get('OUTPUT_DIR', os.path.join(OUTPUT_ROOT, name))
