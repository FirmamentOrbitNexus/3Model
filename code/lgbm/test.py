# ============================================================
# test.py - LightGBM 基线推理 + 沪深300 选股（多期滚动）
# 输出与 LSTM/Transformer/Kronos 的 test.py 完全同构：
#   result_<日期>.csv（单期明细）、result_full.csv、result_portfolio.csv
#   （后两者按 pred_date 累积去重，供 common/evaluate.py 统一评估）
# 支持 END_DATES / MODEL_DIR / OUTPUT_DIR 等环境变量覆盖
# ============================================================
import os

os.environ.setdefault('PYTHONHASHSEED', '42')

import sys
import json
import logging
import random

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("缺少 lightgbm，请先安装: pip install lightgbm")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import SEED
from common.data_io import load_stock_dataframe, load_trade_calendar, align_stock_calendar
from common.featurework import compute_features, MODEL_COLUMNS
from common.paths import model_dir, output_dir
from common.strategy import (
    TOP_K, select_portfolio, save_full_predictions, save_portfolio,
)

random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('lgbm_test')

# ==================== 路径与参数 ====================
# 窗口参数的默认值（实际取值以模型 config.json 为准，保证与训练一致）
DEFAULT_LOOKBACK = 60
DEFAULT_HORIZON = 5
LOOKBACK = DEFAULT_LOOKBACK
HORIZON = DEFAULT_HORIZON

MODEL_DIR = model_dir('lgbm')      # 环境变量 MODEL_DIR 可覆盖
OUTPUT_DIR = output_dir('lgbm')    # 环境变量 OUTPUT_DIR 可覆盖

# 预测基准日列表（环境变量 END_DATES 逗号分隔可覆盖）
# 每个基准日预测其后 5 个交易日，三期滚动预测：
#   2026-08-14（周五）-> 08-17 ~ 08-21
#   2026-08-21（周五）-> 08-24 ~ 08-28
#   2026-08-28（周五）-> 08-31 ~ 09-04
END_DATES = [d.strip() for d in
             os.environ.get('END_DATES', '2026-08-14,2026-08-21,2026-08-28').split(',')
             if d.strip()]

_booster = None


# ==================== 模型加载 ====================
def load_model():
    """加载 LightGBM 模型与配置；窗口参数以 config.json 为准"""
    global _booster, LOOKBACK, HORIZON
    if _booster is not None:
        return _booster
    model_file = os.path.join(MODEL_DIR, 'lgbm_model.txt')
    cfg_path = os.path.join(MODEL_DIR, 'config.json')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"未找到模型: {model_file}，请先运行 train.py")
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            lg_cfg = json.load(f)
        LOOKBACK = int(lg_cfg.get('lookback', DEFAULT_LOOKBACK))
        HORIZON = int(lg_cfg.get('horizon', DEFAULT_HORIZON))
    _booster = lgb.Booster(model_file=model_file)
    logger.info(f"模型已加载: {MODEL_DIR}, lookback={LOOKBACK}, horizon={HORIZON}")
    return _booster


# ==================== 特征构建（与训练完全一致） ====================
def build_feature(stock: pd.DataFrame, lookback: int):
    """取最近 lookback 日，构造与训练一致的 39 维特征向量"""
    raw = stock[MODEL_COLUMNS].values.astype(np.float32)
    if len(raw) < lookback:
        return None
    feat = compute_features(raw)
    window = feat[-lookback:]
    x = np.concatenate([window[-1], window.mean(axis=0), window.std(axis=0)])
    return np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0).astype(np.float32)


# ==================== 单期截面预测 ====================
def predict_cross_section(booster, df_all: pd.DataFrame, end_date: str,
                          trade_dates, lookback: int):
    """
    在某个基准日对全市场做截面预测（LightGBM 直接输出 expected_roi）
    returns: (result_df, 候选股票数, 异常股票数)
    """
    df = df_all[df_all['timestamps'] <= pd.to_datetime(end_date)]
    stock_codes = sorted(df['code'].unique())
    results, errors = [], 0
    for code in stock_codes:
        try:
            stock = df[df['code'] == code].copy().set_index('timestamps')
            stock = align_stock_calendar(stock, trade_dates).reset_index()
            x = build_feature(stock, lookback)
            if x is None:
                continue
            roi = float(booster.predict(x.reshape(1, -1))[0])
            results.append({'code': code, 'expected_roi': roi})
        except Exception as e:
            errors += 1
            if errors <= 3:               # 只打印前 3 条，避免刷屏
                logger.warning(f"股票 {code} 预测失败: {e}")
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df['rank'] = result_df['expected_roi'].rank(method='first', ascending=False).astype(int)
    return result_df, len(stock_codes), errors


# ==================== 主流程 ====================
def main():
    logger.info("LightGBM 推理 + 沪深300 选股流程开始")
    logger.info(f"预测基准日 {len(END_DATES)} 期: {', '.join(END_DATES)}（每期预测其后 5 个交易日）")

    df_all = load_stock_dataframe()
    trade_dates = load_trade_calendar()
    booster = load_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for end_date in END_DATES:
        result_df, n_codes, n_err = predict_cross_section(
            booster, df_all, end_date, trade_dates, LOOKBACK)
        if result_df.empty:
            logger.warning(f"[{end_date}] 无有效预测结果（候选 {n_codes} 只，异常 {n_err} 只）")
            continue
        logger.info(f"[{end_date}] 候选 {n_codes} 只，有效预测 {len(result_df)} 只，异常 {n_err} 只")

        # 累积保存（按 pred_date 分组，供 evaluate.py 统一评估）
        save_full_predictions(result_df, OUTPUT_DIR, end_date, logger)

        select_rows = select_portfolio(result_df)
        if len(select_rows) < TOP_K:
            logger.warning(f"[{end_date}] 候选不足，仅选出 {len(select_rows)} 只（上限 {TOP_K}）")
        save_portfolio(select_rows, OUTPUT_DIR, end_date, logger)

        # 单期明细（按基准日命名，便于逐期查看）
        out_df = pd.DataFrame(select_rows)
        period_file = os.path.join(OUTPUT_DIR, f"result_{end_date.replace('-', '')}.csv")
        out_df.to_csv(period_file, index=False)
        logger.info(f"[{end_date}] 结果已保存: {period_file}")

        print(f"\n===== 基准日 {end_date}（预测其后 5 个交易日）=====")
        print(out_df.to_string(index=False) if len(out_df) else "（无满足条件的股票）")

    logger.info("LightGBM 推理 + 选股流程结束")


if __name__ == "__main__":
    main()
