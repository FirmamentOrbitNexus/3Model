# ============================================================
# featurework.py - LSTM / Vanilla Transformer 共用的特征工程
#
# 设计目标：
#   在保持与 Kronos 一致的 6 维原始 OHLCV 基础上，补充 7 维衍生特征
#   形成 13 维特征向量，作为端到端训练模型的输入/输出
#
# 衍生特征（按交易日计算，依赖历史窗口，全部由 rolling 算子实现）：
#   log_return     对数收益率（首日填 0）
#   range_pct      当日振幅 (high-low)/close
#   close_ma5      收盘价相对 5  日均线偏离
#   close_ma10     收盘价相对 10 日均线偏离
#   close_ma20     收盘价相对 20 日均线偏离
#   volatility_5   5 日对数收益率标准差
#   volume_ratio   当日成交量相对 5 日均量比
#
# 注：
#   - 所有 rolling 使用 min_periods=1，避免首段产生 NaN
#   - 整只股票一次性计算完成，再做窗口切分，避免在窗口切分时重复滚动
#   - compute_features() 内部做了 nan/inf 防御性清洗
# ============================================================

import numpy as np
import pandas as pd

# 原始 6 列 K 线特征（与 Kronos 口径一致）
MODEL_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']

# 衍生 7 列特征
ENGINEERED_COLUMNS = [
    'log_return',      # 对数收益率
    'range_pct',       # 当日振幅
    'close_ma5',       # 收盘 vs MA5 偏离
    'close_ma10',      # 收盘 vs MA10 偏离
    'close_ma20',      # 收盘 vs MA20 偏离
    'volatility_5',    # 5 日波动率
    'volume_ratio',    # 量比
]

# 完整特征列名（输入/输出模型的特征顺序）
FEATURE_COLUMNS = MODEL_COLUMNS + ENGINEERED_COLUMNS
INPUT_DIM = len(FEATURE_COLUMNS)  # 13


def compute_features(values: np.ndarray) -> np.ndarray:
    """
    对单只股票的全序列原始 OHLCV 计算工程化特征
    values: (N, 6) float32 原始 OHLCV 数组
    return: (N, INPUT_DIM) float32 特征矩阵，无 NaN/Inf
    """
    df = pd.DataFrame(values, columns=MODEL_COLUMNS)

    # 1) 对数收益率（首日填 0）
    df['log_return'] = np.log(df['close'] / df['close'].shift(1)).fillna(0.0)

    # 2) 当日振幅 (high-low)/close，避免除零
    safe_close = df['close'].replace(0.0, np.nan)
    df['range_pct'] = ((df['high'] - df['low']) / safe_close).fillna(0.0)

    # 3) 收盘价相对均线偏离（min_periods=1 处理首段）
    for ma in (5, 10, 20):
        ma_close = df['close'].rolling(ma, min_periods=1).mean()
        df[f'close_ma{ma}'] = ((df['close'] - ma_close) / ma_close.replace(0.0, np.nan)).fillna(0.0)

    # 4) 5 日对数收益率波动率
    df['volatility_5'] = df['log_return'].rolling(5, min_periods=1).std().fillna(0.0)

    # 5) 成交量相对 5 日均量比（首日 MA5=volume，比值为 1）
    vol_ma5 = df['volume'].rolling(5, min_periods=1).mean()
    df['volume_ratio'] = (df['volume'] / vol_ma5.replace(0.0, np.nan)).fillna(1.0)

    feat = df[FEATURE_COLUMNS].values.astype(np.float32)
    # 防御：清理极罕见 NaN/Inf（rolling 边缘 + 数据缺失）
    return np.nan_to_num(feat, nan=0.0, posinf=3.0, neginf=-3.0)