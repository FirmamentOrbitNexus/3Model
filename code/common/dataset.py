# ============================================================
# dataset.py - torch 训练组件：随机种子 / 训练样本生成 / K 线 Dataset
# 仅训练链路使用（依赖 torch）；数据读取部分见 data_io.py
# 数据口径与 kronos/train.py 保持一致，保证对比实验公平
# ============================================================

import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .paths import DATA_DIR
from .config import SEED, configure_determinism
from .data_io import load_stock_dataframe, load_trade_calendar, align_stock_calendar
from .featurework import compute_features, MODEL_COLUMNS, FEATURE_COLUMNS, INPUT_DIM


# ==================== 随机种子与确定性 ====================
def seed_everything(seed: int = SEED):
    """固定所有随机源，确保训练可复现（实现见 common/config.py）"""
    return configure_determinism(seed)


def seed_worker(worker_id: int):
    """DataLoader 工作进程的随机种子设置函数"""
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def make_generator():
    """创建固定种子的 DataLoader 随机数生成器"""
    g = torch.Generator()
    g.manual_seed(SEED)
    return g


# ==================== 训练样本生成 ====================
def generate_samples(lookback: int = 60, pred_window: int = 5, step: int = 5,
                     min_extra: int = 10, use_cache: bool = True) -> str:
    """
    滑动窗口切分训练样本（输入 lookback 天 -> 预测 pred_window 天）
    样本以 dict 形式存储：
        {
          'code': 股票代码,
          'x': (lookback, INPUT_DIM) ndarray,   # 输入：13 维（6 原始 + 7 衍生）
          'y': (pred_window, len(MODEL_COLUMNS)) ndarray,  # 标签：仅 6 维原始 OHLCV
        }
    设计要点：
      - 输入含 7 维衍生特征（log_return / MA 偏离 / 波动率 / 量比），帮助端到端模型学习
      - 标签只取原始 6 维 OHLCV，避免强迫模型重学算术约束（衍生特征可由 OHLCV 推算）
      - 反归一化阶段再用预测的 OHLCV 重新计算衍生特征，做下游选股
    支持时间划分（论文实验用，避免数据泄露）：
        环境变量 TRAIN_START_DATE / TRAIN_END_DATE（如 2024-12-31），默认不限制
    结果缓存为 pkl 文件，重复训练时直接复用
    返回 pkl 文件路径
    """
    # y 维度：标签只取原始 OHLCV（不预测衍生特征，避免冗余）
    output_dim = len(MODEL_COLUMNS)
    # 时间划分参数（影响缓存文件名，保证不同划分互不干扰）
    train_start = os.environ.get('TRAIN_START_DATE', '')
    train_end = os.environ.get('TRAIN_END_DATE', '')
    split_tag = (f"_s{train_start.replace('-', '')}" if train_start else '') + \
                (f"_e{train_end.replace('-', '')}" if train_end else '')
    # 文件名含 y 维度，让旧版 y=INPUT_DIM 的缓存自动失效
    cache_file = os.path.join(DATA_DIR, f'samples_lb{lookback}_pw{pred_window}_step{step}_y{output_dim}{split_tag}.pkl')
    if use_cache and os.path.exists(cache_file):
        print(f"使用缓存样本文件: {cache_file}")
        return cache_file

    min_required = lookback + pred_window + min_extra

    # 读取并清洗数据
    df = load_stock_dataframe()
    # 时间划分：训练数据只使用区间内样本
    if train_start:
        df = df[df['timestamps'] >= pd.to_datetime(train_start)]
    if train_end:
        df = df[df['timestamps'] <= pd.to_datetime(train_end)]
    if df.empty:
        raise ValueError(f"时间划分后无训练数据: [{train_start or '最早'}, {train_end or '最新'}]")
    print(f"训练数据区间: {df['timestamps'].min().date()} ~ {df['timestamps'].max().date()}"
          + ("（注意：请确保不与测试期重叠，避免数据泄露）" if not train_end else ''))

    stock_codes = sorted(df['code'].unique())
    trade_dates = load_trade_calendar()

    samples = []
    valid_stocks = 0
    for code in tqdm(stock_codes, desc="生成样本"):
        # 取单只股票数据并对齐交易日历
        stock = df[df['code'] == code].copy().set_index('timestamps')
        stock = align_stock_calendar(stock, trade_dates).reset_index()
        # 数据量不足则跳过
        if len(stock) < min_required:
            continue
        valid_stocks += 1
        # 整只股票一次性计算工程化特征
        raw = stock[MODEL_COLUMNS].values.astype(np.float32)
        feat = compute_features(raw)                       # (N, INPUT_DIM)
        if np.isnan(feat).any() or np.isinf(feat).any():
            continue
        n = len(feat)
        # 滑动窗口切分样本（在特征矩阵上切分，效率优于逐窗口重算）
        for start_idx in range(0, n - lookback - pred_window + 1, step):
            x = feat[start_idx:start_idx + lookback]                                # (lookback, 13)
            y = feat[start_idx + lookback:start_idx + lookback + pred_window, :output_dim]  # (pred_window, 6)
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            samples.append(dict(code=code, x=x, y=y))

    print(f"有效股票数: {valid_stocks}, 样本总数: {len(samples)}, "
          f"输入维度: {INPUT_DIM} ({len(MODEL_COLUMNS)} 原始 + {len(FEATURE_COLUMNS) - len(MODEL_COLUMNS)} 衍生), "
          f"输出维度: {output_dim}（原始 OHLCV）")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(samples, f)
    print(f"样本已保存: {cache_file}")
    return cache_file


# ==================== 归一化 Dataset ====================
class KLineDataset(Dataset):
    """
    K 线训练数据集：对每个样本用输入窗口自身的均值/标准差做归一化
    （与 Kronos 训练时的标准化口径一致），并截断到 ±3 倍标准差（缩尾）
    标签 y 使用 x 的统计量归一化，保证推理时可用 x 的统计量反归一化
    """

    def __init__(self, pkl_path: str):
        with open(pkl_path, 'rb') as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        xv, yv = s['x'], s['y']
        # x：13 维含衍生特征，用 13 维统计量归一化
        xm, xs = np.mean(xv, axis=0), np.std(xv, axis=0)
        xs = np.where(xs < 1e-4, 1.0, xs)
        xn = np.clip(np.nan_to_num((xv - xm) / xs, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)
        # y：仅前 6 维原始 OHLCV，用 x 在前 6 维上的统计量归一化（保证推理时用 x 反归一化）
        ym, ys = xm[:yv.shape[1]], xs[:yv.shape[1]]
        yn = np.clip(np.nan_to_num((yv - ym) / ys, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)
        return {
            'x': torch.from_numpy(xn.astype(np.float32)),
            'y': torch.from_numpy(yn.astype(np.float32)),
            'xm': torch.from_numpy(ym.astype(np.float32)),  # 仅返回 y 用到的前 6 维均值
            'xs': torch.from_numpy(ys.astype(np.float32)),  # 仅返回 y 用到的前 6 维标准差
        }


def collate_fn(batch):
    """自定义批处理函数：堆叠 x / y 及归一化统计量"""
    return (
        torch.stack([i['x'] for i in batch]),
        torch.stack([i['y'] for i in batch]),
    )
