# ============================================================
# data_io.py - 数据读取/清洗/交易日历对齐
# 纯 pandas 实现，不依赖 torch——训练与推理脚本共用，
# 保证两条链路的数据口径严格一致
# ============================================================

import json

import numpy as np
import pandas as pd

from .paths import STOCK_CSV, HS300_CSV, CALENDAR_JSON
from .featurework import MODEL_COLUMNS


# ==================== 数据读取与清洗 ====================
def load_stock_dataframe() -> pd.DataFrame:
    """读取原始股票数据，重命名、过滤沪深300成分股后返回统一格式 DataFrame"""
    # 使用 utf-8-sig 读取，自动去除可能存在的 BOM 头
    df = pd.read_csv(STOCK_CSV, encoding='utf-8-sig')
    # 重命名列为统一格式
    df = df.rename(columns={
        '股票代码': 'code', '日期': 'timestamps',
        '开盘': 'open', '收盘': 'close', '最高': 'high',
        '最低': 'low', '成交量': 'volume', '成交额': 'amount'
    })
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    # 填充缺失值
    df['volume'] = df['volume'].fillna(0)
    df['amount'] = df['amount'].fillna(0)

    # 读取沪深300成分股列表，过滤非成分股
    hs300 = pd.read_csv(HS300_CSV, encoding='utf-8-sig')
    hs300['code'] = hs300['code'].astype(str).str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    valid_codes = set(hs300['code'].unique())
    df['code'] = df['code'].astype(str).str.zfill(6)
    df = df[df['code'].isin(valid_codes)]

    # 只保留需要的列
    df = df[['code', 'timestamps'] + MODEL_COLUMNS]
    return df


def load_trade_calendar() -> pd.DatetimeIndex:
    """加载交易日历"""
    with open(CALENDAR_JSON) as f:
        return pd.to_datetime(json.load(f))


def align_stock_calendar(stock: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """对齐交易日历，补全停牌日（价格前向填充，量额补 0）"""
    stock = stock[~stock.index.duplicated(keep='last')]
    sc = trade_dates[(trade_dates >= stock.index.min()) & (trade_dates <= stock.index.max())]
    if len(sc.difference(stock.index)) > 0:
        stock = stock.reindex(sc)
        for col in ['open', 'high', 'low', 'close']:
            if col in stock.columns:
                stock[col] = stock[col].ffill()
        for col in ['volume', 'amount']:
            if col in stock.columns:
                stock[col] = stock[col].fillna(0)
        stock = stock.bfill()
    return stock
