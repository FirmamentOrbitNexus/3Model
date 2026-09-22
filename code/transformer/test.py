# ============================================================
# test.py - Vanilla Transformer 模型推理 + 沪深300 选股
# 流程：加载模型与配置 -> 逐只股票读取最近 LOOKBACK 天数据 ->
#       特征工程（与训练口径一致）-> 归一化 -> Transformer 预测 5 天 ->
#       反归一化得 T+1/T+5 开盘价 -> 计算 expected_roi ->
#       Top-K 等权选股（common/strategy.py，默认 K=5，只买预期上涨标的）
#       -> 输出 result.csv + 累积 result_full.csv / result_portfolio.csv
# ============================================================
# ========== 所有环境变量必须在 import torch 之前设置 ==========
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')

import sys
import json
import logging

import numpy as np
import pandas as pd
import torch

# 将项目 code/ 目录加入系统路径，导入共用模块与训练好的模型类
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import configure_determinism, get_device
from common.data_io import load_stock_dataframe
from common.featurework import compute_features, MODEL_COLUMNS
from common.paths import model_dir, output_dir
from common.strategy import (
    TOP_K, select_portfolio, save_full_predictions, save_portfolio,
)
from train import VanillaTransformer

# ========== import torch 之后执行统一确定性配置 ==========
configure_determinism()

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('transformer_test')


# ==================== 路径与参数 ====================
# 窗口参数的默认值（实际取值以模型 config.json 为准，保证与训练一致）
DEFAULT_LOOKBACK = 60
DEFAULT_PRED_LEN = 5
LOOKBACK = DEFAULT_LOOKBACK
PRED_LEN = DEFAULT_PRED_LEN

MODEL_DIR = model_dir('transformer')   # 环境变量 MODEL_DIR 可覆盖
OUTPUT_DIR = output_dir('transformer') # 环境变量 OUTPUT_DIR 可覆盖

# 预测基准日列表（环境变量 END_DATES 逗号分隔可覆盖）
# 每个基准日预测其后 5 个交易日的开盘价，三期滚动预测：
#   2026-08-14（周五）-> 08-17 ~ 08-21
#   2026-08-21（周五）-> 08-24 ~ 08-28
#   2026-08-28（周五）-> 08-31 ~ 09-04
END_DATES = [d.strip() for d in
             os.environ.get('END_DATES', '2026-08-14,2026-08-21,2026-08-28').split(',')
             if d.strip()]

DEVICE = get_device()
_model = None
_cfg = None


# ==================== 模型加载 ====================
def load_model():
    """从 model/transformer/ 加载 config.json + best_model.pt，构建同结构 VanillaTransformer"""
    global _model, _cfg, LOOKBACK, PRED_LEN
    if _model is not None:
        return _model, _cfg

    cfg_path = os.path.join(MODEL_DIR, 'config.json')
    weights_path = os.path.join(MODEL_DIR, 'best_model.pt')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"未找到模型配置: {cfg_path}，请先运行 train.py")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"未找到模型权重: {weights_path}，请先运行 train.py")

    with open(cfg_path, 'r', encoding='utf-8') as f:
        _cfg = json.load(f)
    # 窗口参数以 config.json 为准（避免与训练口径脱钩）
    LOOKBACK = int(_cfg.get('lookback', DEFAULT_LOOKBACK))
    PRED_LEN = int(_cfg.get('pred_window', DEFAULT_PRED_LEN))
    _model = VanillaTransformer(
        input_dim=_cfg['input_dim'],
        d_model=_cfg['d_model'],
        n_heads=_cfg['n_heads'],
        n_layers=_cfg['n_layers'],
        ff_dim=_cfg['ff_dim'],
        output_dim=_cfg['output_dim'],
        pred_len=PRED_LEN,
        max_len=max(LOOKBACK, 512),
        dropout=_cfg['dropout'],
    ).to(DEVICE).eval()
    state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    _model.load_state_dict(state)
    logger.info(f"模型已加载: {MODEL_DIR}, 设备={DEVICE}, "
                f"lookback={LOOKBACK}, pred_len={PRED_LEN}")
    return _model, _cfg


# ==================== 单只股票预测 ====================
def predict_one(model, stock: pd.DataFrame, lookback: int):
    """
    stock: 单只股票按时间升序的全部历史
    returns: (T+1_open, T+5_open) 或 None（数据不足/异常）
    关键：模型只输出 6 维原始 OHLCV（与训练 y 维度一致），
          反归一化必须只用 x 在前 6 维上的均值/标准差，否则 numpy broadcast 失败
    """
    raw = stock[MODEL_COLUMNS].values.astype(np.float32)
    if len(raw) < lookback:
        return None
    feat = compute_features(raw)
    window = feat[-lookback:]
    xm = np.mean(window, axis=0)              # (INPUT_DIM,) = (13,)
    xs = np.std(window, axis=0)
    xs = np.where(xs < 1e-4, 1.0, xs)
    xn = np.clip((window - xm) / xs, -3.0, 3.0).astype(np.float32)
    x = torch.from_numpy(xn).unsqueeze(0).to(DEVICE)            # (1, lookback, INPUT_DIM)
    with torch.no_grad():
        yn = model(x).cpu().numpy()[0]                            # (PRED_LEN, OUTPUT_DIM) = (5, 6)
    # ★ 只用前 6 维统计量反归一化（与 KLineDataset.__getitem__ 口径一致）
    output_dim = yn.shape[-1]
    y = yn * xs[:output_dim] + xm[:output_dim]                    # (PRED_LEN, OUTPUT_DIM)
    # 模型输出的 0 列就是 open（与训练时 MODEL_COLUMNS[0]='open' 一致）
    return float(y[0, 0]), float(y[PRED_LEN - 1, 0])


# ==================== 单期截面预测 ====================
def predict_cross_section(model, df_all: pd.DataFrame, end_date: str, lookback: int):
    """
    在某个基准日对全市场做截面预测
    returns: (result_df, 候选股票数, 异常股票数)
    """
    df = df_all[df_all['timestamps'] <= pd.to_datetime(end_date)]
    stock_codes = sorted(df['code'].unique())
    results, errors = [], 0
    for code in stock_codes:
        try:
            stock = df[df['code'] == code].copy().sort_values('timestamps').reset_index(drop=True)
            r = predict_one(model, stock, lookback)
            if r is None:
                continue
            T1, T5 = r
            if T1 <= 0 or T5 <= 0:        # 异常预测过滤（价格必须为正）
                continue
            results.append({'code': code, 'T+1_open': T1, 'T+5_open': T5})
        except Exception as e:
            errors += 1
            if errors <= 3:               # 只打印前 3 条，避免刷屏
                logger.warning(f"股票 {code} 预测失败: {e}")
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df['expected_roi'] = (result_df['T+5_open'] - result_df['T+1_open']) / result_df['T+1_open']
        result_df['rank'] = result_df['expected_roi'].rank(method='first', ascending=False).astype(int)
    return result_df, len(stock_codes), errors


# ==================== 主流程 ====================
def main():
    """数据加载 -> 逐期预测 -> 选股 -> 输出结果（三期滚动）"""
    logger.info("Vanilla Transformer 推理 + 沪深300 选股流程开始")
    logger.info(f"预测基准日 {len(END_DATES)} 期: {', '.join(END_DATES)}（每期预测其后 5 个交易日）")

    df_all = load_stock_dataframe()
    model, cfg = load_model()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for end_date in END_DATES:
        result_df, n_codes, n_err = predict_cross_section(model, df_all, end_date, LOOKBACK)
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

        print(f"\n===== 基准日 {end_date}（预测其后 5 个交易日开盘）=====")
        print(out_df.to_string(index=False) if len(out_df) else "（无满足条件的股票）")

    logger.info("Vanilla Transformer 推理 + 选股流程结束")


if __name__ == "__main__":
    main()