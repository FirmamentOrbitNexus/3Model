# ============================================================
# test.py - LightGBM 基线推理 + 沪深300 选股
# 输出与 LSTM/Transformer/Kronos 的 test.py 完全同构：
#   result.csv（竞赛格式）、result_full.csv、result_portfolio.csv
#   （后两者按 pred_date 累积去重，供 common/evaluate.py 统一评估）
# 支持 END_DATE / MODEL_DIR / OUTPUT_DIR 等环境变量覆盖
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
from common.featurework import compute_features, MODEL_COLUMNS, FEATURE_COLUMNS
from common.paths import model_dir, output_dir
from common.strategy import select_portfolio, save_full_predictions, save_portfolio

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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'result.csv')

END_DATE = os.environ.get('END_DATE', '2026-07-31')
PRED_DATE = END_DATE

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


# ==================== 主流程 ====================
def main():
    logger.info("LightGBM 推理 + 沪深300 选股流程开始")
    df = load_stock_dataframe()
    df = df[df['timestamps'] <= pd.to_datetime(END_DATE)]
    trade_dates = load_trade_calendar()
    booster = load_model()

    results = []
    for code in sorted(df['code'].unique()):
        try:
            stock = df[df['code'] == code].copy().set_index('timestamps')
            stock = align_stock_calendar(stock, trade_dates).reset_index()
            x = build_feature(stock, LOOKBACK)
            if x is None:
                continue
            roi = float(booster.predict(x.reshape(1, -1))[0])
            results.append({'code': code, 'expected_roi': roi})
        except Exception as e:
            logger.debug(f"股票 {code} 预测失败: {e}")
            continue

    result_df = pd.DataFrame(results)
    if result_df.empty:
        logger.warning("无有效预测结果")
        return
    result_df['rank'] = result_df['expected_roi'].rank(method='first', ascending=False).astype(int)

    save_full_predictions(result_df, OUTPUT_DIR, PRED_DATE, logger)

    select_rows = select_portfolio(result_df)
    if len(select_rows) != 3:
        logger.warning(f"部分池子无满足筛选条件的股票，输出 {len(select_rows)} 只（约束≤5）")
    save_portfolio(select_rows, OUTPUT_DIR, PRED_DATE, logger)

    out_df = pd.DataFrame(select_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"结果已保存: {OUTPUT_FILE}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
