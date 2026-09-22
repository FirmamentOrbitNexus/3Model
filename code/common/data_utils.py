# ============================================================
# data_utils.py - 兼容壳（P2 重构）
# 原内容已按职责拆分，本文件仅做 re-export，保证既有
# `from common.data_utils import ...` 调用不受影响：
#   - paths.py       路径与环境变量单点配置
#   - data_io.py     数据读取/清洗/日历对齐（纯 pandas，无 torch）
#   - dataset.py     随机种子/样本生成/K 线 Dataset（torch，训练专用）
# ============================================================

from .paths import (BASE_DIR, DATA_DIR, MODEL_ROOT, OUTPUT_ROOT, LOG_DIR,
                    STOCK_CSV, HS300_CSV, CALENDAR_JSON, model_dir, output_dir)
from .config import SEED, NUM_WORKERS, USE_TF32, configure_determinism, get_device
from .featurework import compute_features, MODEL_COLUMNS, FEATURE_COLUMNS, INPUT_DIM
from .data_io import load_stock_dataframe, load_trade_calendar, align_stock_calendar
from .dataset import (
    seed_everything, seed_worker, make_generator,
    generate_samples, KLineDataset, collate_fn,
)

__all__ = [
    'BASE_DIR', 'DATA_DIR', 'MODEL_ROOT', 'OUTPUT_ROOT', 'LOG_DIR',
    'STOCK_CSV', 'HS300_CSV', 'CALENDAR_JSON', 'model_dir', 'output_dir',
    'SEED', 'NUM_WORKERS', 'USE_TF32', 'configure_determinism', 'get_device',
    'compute_features', 'MODEL_COLUMNS', 'FEATURE_COLUMNS', 'INPUT_DIM',
    'load_stock_dataframe', 'load_trade_calendar', 'align_stock_calendar',
    'seed_everything', 'seed_worker', 'make_generator',
    'generate_samples', 'KLineDataset', 'collate_fn',
]
