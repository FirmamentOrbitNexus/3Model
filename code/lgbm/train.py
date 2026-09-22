# ============================================================
# train.py - LightGBM 表格基线训练
# 思路：与深度模型对比的经典表格方法——
#   对每个 (股票, 日期 t) 样本，用过去 LOOKBACK 日的特征工程统计量
#   （最新截面 13 维 + 窗口均值/标准差 26 维 = 39 维）作为输入，
#   直接回归未来 5 日实际 ROI（(open[T+5]-open[T+1])/open[T+1]，
#   与深度模型的 T+1/T+5 定义一致）
# 训练/验证按时间切分（最后 10% 交易日为验证集，早停），无数据泄露
# 支持 TRAIN_START_DATE / TRAIN_END_DATE 环境变量控制训练区间
# ============================================================
import os

os.environ.setdefault('PYTHONHASHSEED', '42')

import sys
import time
import random

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    raise SystemExit("缺少 lightgbm，请先安装: pip install lightgbm")

# 将项目 code/ 目录加入系统路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import SEED
from common.paths import model_dir
from common.data_utils import (
    DATA_DIR, STOCK_CSV, HS300_CSV, CALENDAR_JSON,
    load_stock_dataframe, load_trade_calendar, align_stock_calendar,
)
from common.featurework import compute_features, MODEL_COLUMNS, FEATURE_COLUMNS
from common.logger_utils import timestamp, setup_logger, save_config_snapshot, MetricsWriter
from tqdm import tqdm

random.seed(SEED)
np.random.seed(SEED)


# ==================== 配置 ====================
class Config:
    """LightGBM 训练配置（字段命名与其他模型对齐）"""
    # 历史窗口大小（与深度模型输入窗口一致）
    lookback = 60
    # 预测区间（交易日）
    horizon = 5
    # 标签缩尾（停牌补价等导致的极端 ROI 截断）
    label_clip = 0.3
    # 按时间切分的验证集比例（最后 10% 交易日）
    val_ratio = 0.1
    # 最大提升轮数
    num_rounds = 1000
    # 早停轮数
    early_stopping = 50
    # 随机种子
    seed = SEED
    # 特征维度（13 截面 + 13 均值 + 13 标准差 = 39）
    n_features = len(FEATURE_COLUMNS) * 3
    # 输入/输出维度（与其他模型字段对齐：LightGBM 输入 39 维表格特征，输出标量 ROI）
    input_dim = n_features
    output_dim = 1
    # 训练设备（LightGBM 走 CPU 直方图算法）
    device = 'cpu'
    # 并行进程数（LightGBM 自身使用 num_threads，不涉及 DataLoader 多进程）
    num_workers = 0
    # TF32 开关（LightGBM 不适用）
    use_tf32 = False
    # 模型保存目录（环境变量 MODEL_DIR 可覆盖）
    save_dir = model_dir('lgbm')
    # LightGBM 参数
    params = dict(
        objective='regression',       # 回归任务
        metric='l2',
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=100,         # 叶子最小样本数，防过拟合
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        seed=SEED,
        deterministic=True,           # 可复现
        force_row_wise=True,
        verbosity=-1,
    )


cfg = Config()

# 模块内既有引用的兼容别名
LOOKBACK, HORIZON = cfg.lookback, cfg.horizon
LABEL_CLIP, VAL_RATIO = cfg.label_clip, cfg.val_ratio
NUM_ROUNDS, EARLY_STOPPING = cfg.num_rounds, cfg.early_stopping
SAVE_DIR, N_FEATURES, LGB_PARAMS = cfg.save_dir, cfg.n_features, cfg.params


# ==================== 表格数据构建 ====================
def build_dataset(df: pd.DataFrame, trade_dates, start_date=None, end_date=None):
    """
    构建 (样本特征 X, 标签 y, 元信息 meta)：
    每个样本 = 某股票在日期 t 的特征（t-LOOKBACK+1 .. t 窗口统计）
    标签 = 未来 5 日 ROI（t+1 与 t+HORIZON 开盘价）
    """
    X, y, meta = [], [], []
    stock_codes = sorted(df['code'].unique())
    for code in tqdm(stock_codes, desc="构建表格数据"):
        stock = df[df['code'] == code].copy().set_index('timestamps')
        stock = align_stock_calendar(stock, trade_dates).reset_index()
        if len(stock) < LOOKBACK + HORIZON + 1:
            continue
        raw = stock[MODEL_COLUMNS].values.astype(np.float32)
        if np.isnan(raw).any():
            continue
        feat = compute_features(raw)
        opens = raw[:, 0].astype(np.float64)
        dates = pd.to_datetime(stock['timestamps'])
        n = len(stock)
        for i in range(LOOKBACK - 1, n - HORIZON):
            t = dates.iloc[i]
            if start_date and t < pd.to_datetime(start_date):
                continue
            if end_date and t > pd.to_datetime(end_date):
                continue
            window = feat[i - LOOKBACK + 1: i + 1]
            # 39 维：最新截面 + 窗口均值 + 窗口标准差
            x = np.concatenate([window[-1], window.mean(axis=0), window.std(axis=0)])
            x = np.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)
            o1, o5 = opens[i + 1], opens[i + HORIZON]
            if o1 <= 0 or o5 <= 0:
                continue
            label = float(np.clip((o5 - o1) / o1, -LABEL_CLIP, LABEL_CLIP))
            X.append(x)
            y.append(label)
            meta.append((code, str(t.date())))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), meta


# ==================== 模型配置写盘 ====================
def write_model_config(best_iteration=None, val_rmse=None):
    """
    写模型配置到 save_dir/config.json（字段与其他模型对齐）
    训练开始时即写一次（中途中断 test.py 也能正常加载），训练结束补写指标
    """
    info = {
        'model': 'LightGBM',
        'lookback': cfg.lookback,
        'pred_window': cfg.horizon,
        'horizon': cfg.horizon,
        'input_dim': cfg.input_dim,
        'output_dim': cfg.output_dim,
        'n_features': cfg.n_features,
        'label_clip': cfg.label_clip,
        'val_ratio': cfg.val_ratio,
        'num_rounds': cfg.num_rounds,
        'early_stopping': cfg.early_stopping,
        'seed': cfg.seed,
        'device': cfg.device,
        'num_workers': cfg.num_workers,
        'use_tf32': cfg.use_tf32,
        'params': cfg.params,
        'train_start': os.environ.get('TRAIN_START_DATE', ''),
        'train_end': os.environ.get('TRAIN_END_DATE', ''),
    }
    if best_iteration is not None:
        info['best_iteration'] = best_iteration
    if val_rmse is not None:
        info['val_rmse'] = val_rmse
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ==================== 训练 ====================
def train_model(X, y, meta, logger, ts):
    """按时间切分训练/验证，早停训练 LightGBM，保存模型与配置"""
    all_dates = sorted(set(m[1] for m in meta))
    n_val = max(1, int(len(all_dates) * VAL_RATIO))
    val_dates = set(all_dates[-n_val:])
    tr_idx = [i for i, m in enumerate(meta) if m[1] not in val_dates]
    va_idx = [i for i, m in enumerate(meta) if m[1] in val_dates]
    logger.info(f"样本总数 {len(y)}，训练 {len(tr_idx)}，验证 {len(va_idx)}"
                f"（验证期: {all_dates[-n_val]} ~ {all_dates[-1]}）")

    # 训练开始即写 config.json：中途中断也能被 test.py 正常加载
    write_model_config()

    dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx])
    dval = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtrain)

    t0 = time.time()
    booster = lgb.train(
        LGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS,
        valid_sets=[dval], valid_names=['valid'],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    dt = time.time() - t0
    val_rmse = float(booster.best_score['valid']['l2'])
    logger.info(f"训练完成: best_iteration={booster.best_iteration}, "
                f"验证 RMSE={val_rmse:.6f}, 耗时 {dt:.1f}s")

    # 指标记录（单行：与深度模型指标 CSV 风格一致）
    metrics = MetricsWriter('lgbm', ts)
    metrics.write(booster.best_iteration, val_rmse, LGB_PARAMS['learning_rate'], dt)

    # 保存模型与配置
    os.makedirs(cfg.save_dir, exist_ok=True)
    model_file = os.path.join(cfg.save_dir, 'lgbm_model.txt')
    booster.save_model(model_file)
    write_model_config(best_iteration=booster.best_iteration, val_rmse=val_rmse)
    logger.info(f"模型已保存: {model_file}")
    return model_file


# ==================== 主流程 ====================
def main():
    import json
    ts = timestamp()
    logger = setup_logger('lgbm', ts)
    logger.info("LightGBM 基线训练流程开始")

    train_start = os.environ.get('TRAIN_START_DATE', '')
    train_end = os.environ.get('TRAIN_END_DATE', '')

    df = load_stock_dataframe()
    if train_start:
        df = df[df['timestamps'] >= pd.to_datetime(train_start)]
    if train_end:
        df = df[df['timestamps'] <= pd.to_datetime(train_end)]
    if df.empty:
        raise ValueError(f"时间划分后无训练数据: [{train_start or '最早'}, {train_end or '最新'}]")
    logger.info(f"训练数据区间: {df['timestamps'].min().date()} ~ {df['timestamps'].max().date()}")

    # 配置快照：超参数 + 时间划分 + 代码/数据版本（MD5），字段与其他模型对齐
    snapshot_config = {k: v for k, v in vars(Config).items() if not k.startswith('_')}
    snapshot_config['train_start'] = train_start
    snapshot_config['train_end'] = train_end
    cfg_file = save_config_snapshot(
        'lgbm', ts,
        config=snapshot_config,
        code_files=[os.path.abspath(__file__)],
        data_files=[STOCK_CSV, HS300_CSV, CALENDAR_JSON],
    )
    logger.info(f"配置快照已保存: {cfg_file}")

    trade_dates = load_trade_calendar()
    X, y, meta = build_dataset(df, trade_dates, train_start or None, train_end or None)
    logger.info(f"表格样本数: {len(y)}, 特征维度: {N_FEATURES}")

    train_model(X, y, meta, logger, ts)
    logger.info("LightGBM 训练流程结束")


if __name__ == "__main__":
    main()
