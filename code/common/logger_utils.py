# ============================================================
# logger_utils.py - 训练日志统一工具（LSTM / Transformer / Kronos 共用）
# 提供三类实验记录，存放于项目 logs/ 目录：
#   1. setup_logger        双通道训练日志（控制台 + logs/train_{name}_{ts}.log）
#                          记录每轮 loss / lr / 耗时、早停轮次、权重保存事件
#   2. save_config_snapshot 配置快照（logs/train_config/{name}_{ts}.json）
#                          完整超参数 + 随机种子 + 运行环境 + 代码/数据文件 MD5，
#                          保证任何一次训练结果都能溯源到确切的代码与数据版本
#   3. MetricsWriter       逐轮训练指标 CSV（logs/metrics_{name}_{ts}.csv）
#                          直接用于绘制三模型收敛曲线对比
# ============================================================

import os
import sys
import csv
import json
import hashlib
import logging
from datetime import datetime

from .paths import BASE_DIR, LOG_DIR


def timestamp() -> str:
    """统一时间戳：同一次训练的日志/快照/指标文件使用同一时间戳，便于对应"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _md5(path: str, chunk_size: int = 1 << 20):
    """计算文件 MD5（用于配置快照），文件不存在返回 None"""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def setup_logger(name: str, ts: str = None, log_dir: str = LOG_DIR) -> logging.Logger:
    """创建同时输出到屏幕和文件的训练 logger"""
    ts = ts or timestamp()
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'train_{name}_{ts}.log')

    logger = logging.getLogger(f'{name}_{ts}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"日志文件: {log_file}")
    return logger


def save_config_snapshot(name: str, ts: str, config: dict,
                         code_files=(), data_files=(),
                         seed: int = 42, log_dir: str = LOG_DIR) -> str:
    """
    保存训练配置快照到 logs/train_config/{name}_{ts}.json
    内容：超参数、随机种子、运行环境（Python/PyTorch/设备）、
          代码文件与数据文件的 MD5（结果溯源依据）
    返回快照文件路径
    """
    import torch

    snapshot = {
        'name': name,
        'created_at': datetime.now().isoformat(),
        'seed': seed,
        'environment': {
            'python': sys.version.split()[0],
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        },
        'config': config,
        'code_md5': {os.path.basename(f): _md5(f) for f in code_files},
        'data_md5': {os.path.basename(f): _md5(f) for f in data_files},
    }
    out_dir = os.path.join(log_dir, 'train_config')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{name}_{ts}.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return out_file


class MetricsWriter:
    """逐轮训练指标追加写入 CSV（默认列: epoch, avg_loss, lr, epoch_time_sec）"""

    def __init__(self, name: str, ts: str = None,
                 columns=('epoch', 'avg_loss', 'lr', 'epoch_time_sec'),
                 log_dir: str = LOG_DIR):
        ts = ts or timestamp()
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f'metrics_{name}_{ts}.csv')
        self.columns = list(columns)
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(self.columns)

    def write(self, *row):
        """追加一行指标（浮点数保留 6 位有效数字）"""
        values = [f'{v:.6g}' if isinstance(v, float) else v for v in row]
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(values)
