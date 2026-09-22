# ============================================================
# test.py - Vanilla Transformer 模型推理 + 沪深300 选股
# 流程：加载模型与配置 -> 逐只股票读取最近 LOOKBACK 天数据 -> 经
#       特征工程（与训练口径一致）-> 归一化 -> Transformer 预测 5 天 ->
#       反归一化得 T+1/T+5 开盘价 -> 计算 expected_roi ->
#       三池选股（与 Kronos test.py 口径完全一致，最多 3 只股票，
#       满足"最多不超过 5 只"的约束）-> 输出 result.csv
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
from common.featurework import compute_features, FEATURE_COLUMNS, MODEL_COLUMNS
from common.paths import model_dir, output_dir
from common.strategy import select_portfolio, save_full_predictions, save_portfolio
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'result.csv')

END_DATE = os.environ.get('END_DATE', '2026-07-31')
PRED_DATE = END_DATE

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
    returns: (T+1_open, T+5_open) 或 None（数据不足）
    """
    raw = stock[MODEL_COLUMNS].values.astype(np.float32)
    if len(raw) < lookback:
        return None
    feat = compute_features(raw)
    window = feat[-lookback:]
    xm = np.mean(window, axis=0)
    xs = np.std(window, axis=0)
    xs = np.where(xs < 1e-4, 1.0, xs)
    xn = np.clip((window - xm) / xs, -3.0, 3.0).astype(np.float32)
    x = torch.from_numpy(xn).unsqueeze(0).to(DEVICE)            # (1, lookback, INPUT_DIM)
    with torch.no_grad():
        yn = model(x).cpu().numpy()[0]                            # (PRED_LEN, INPUT_DIM)
    y = yn * xs + xm
    open_idx = FEATURE_COLUMNS.index('open')
    return float(y[0, open_idx]), float(y[PRED_LEN - 1, open_idx])


# ==================== 主流程 ====================
def main():
    """数据加载 -> 模型预测 -> 选股 -> 输出结果"""
    logger.info("Vanilla Transformer 推理 + 沪深300 选股流程开始")
    df = load_stock_dataframe()
    df = df[df['timestamps'] <= pd.to_datetime(END_DATE)]
    stock_codes = sorted(df['code'].unique())
    logger.info(f"候选股票数: {len(stock_codes)}，截止日期: {END_DATE}")

    model, cfg = load_model()
    results = []
    for code in stock_codes:
        try:
            stock = df[df['code'] == code].copy().sort_values('timestamps').reset_index(drop=True)
            r = predict_one(model, stock, LOOKBACK)
            if r is None:
                continue
            T1, T5 = r
            if T1 <= 0 or T5 <= 0:
                continue
            results.append({'code': code, 'T+1_open': T1, 'T+5_open': T5})
        except Exception as e:
            logger.debug(f"股票 {code} 预测失败: {e}")
            continue

    result_df = pd.DataFrame(results)
    if result_df.empty:
        logger.warning("无有效预测结果")
        return

    result_df['expected_roi'] = (result_df['T+5_open'] - result_df['T+1_open']) / result_df['T+1_open']
    result_df['rank'] = result_df['expected_roi'].rank(method='first', ascending=False).astype(int)

    # 保存全量预测与组合记录（累积，供统一评估）
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